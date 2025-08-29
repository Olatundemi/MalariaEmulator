# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image
import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt
import hashlib

# Import your inference tools
from src.inference_sequence_creator import create_sequences, create_causal_sequences
from src.inference_model_exp import LSTM_EIR, LSTM_Incidence

# ---------------- Page config ----------------
st.set_page_config(page_title="MARLIN", page_icon="🐟", layout="wide")

ASSETS = Path("assets")
ILL = ASSETS / "illustrative_example.png"
CAGE = ASSETS / "cage.png"
MARLIN_IMG = ASSETS / "marlin.png"

COLORS = {
    "eir": "#f59e0b",
    "inc": "#ef4444",
    "prev": "#8b5cf6"
}

log_transform = lambda x: np.log(x + 1e-8)
inverse_log_transform = lambda x: np.exp(x) - 1e-8

sns.set_style("darkgrid")
plt.style.use("ggplot")

# ---------------- Helpers ----------------
def ts_fig(t, y, title, color, show_markers=False, opacity=1.0):
    fig = go.Figure()
    mode = "markers" if show_markers else "lines"
    fig.add_trace(go.Scatter(
        x=t, y=y, mode=mode,
        line=dict(color=color, width=2),
        marker=dict(color=color, size=5, opacity=opacity),
        opacity=opacity,
        name=title
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=40, r=10, t=30, b=30),
        xaxis_title="Month(s)",
        yaxis_title=title,
        template="simple_white"
    )
    return fig

@st.cache_resource
def load_models(model_eir_path, model_inc_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_eir = LSTM_EIR(input_size=1, architecture=[256, 128, 64, 32])
    model_eir.load_state_dict(torch.load(model_eir_path, map_location=device))
    model_eir.to(device)
    model_eir.eval()

    model_inc = LSTM_Incidence(input_size=2, architecture=[200, 100, 50])
    model_inc.load_state_dict(torch.load(model_inc_path, map_location=device))
    model_inc.to(device)
    model_inc.eval()

    return model_eir, model_inc, device

def preprocess_data(df):
    if not pd.api.types.is_numeric_dtype(df['prev_true']):
        st.error("🚨 The selected prevalence column is invalid. It contains non-numeric values.")
        return None, False  
    
    has_true_values = {'EIR_true', 'incall'}.issubset(df.columns)

    if has_true_values:
        df_scaled = df[['prev_true', 'EIR_true', 'incall']].apply(log_transform)
    else:
        df_scaled = df[['prev_true']].apply(log_transform)
    
    return df_scaled, has_true_values

def adjust_trailing_zero_prevalence(df, prevalence_column='prev_true', min_val=0.0003, max_val=0.001, seed=None):
    df = df.copy()
    zeros_mask = df[prevalence_column] == 0
    num_zeros = zeros_mask.sum()
    if num_zeros > 0:
        rng = np.random.default_rng(seed)
        random_values = rng.uniform(min_val, max_val, size=num_zeros)
        df.loc[zeros_mask, prevalence_column] = random_values
    return df

@st.cache_data
def load_uploaded_csv(file_content):
    return pd.read_csv(file_content)

def get_file_hash(file):
    return hashlib.md5(file.getvalue()).hexdigest()

# ---------------- Inline Emulator Functions ----------------
@st.cache_data(show_spinner="🔄 Running model predictions...")
def generate_predictions_per_run(data, selected_runs, run_column, window_size, _model_eir, _model_inc, _device, has_true_values):
    run_results = {}

    for run in selected_runs:
        run_data = data[data[run_column] == run]
        if run_data.empty:
            continue

        scaled_data, _ = preprocess_data(run_data)
        if scaled_data is None:
            continue

        X_eir_scaled, y_eir = create_sequences(scaled_data, window_size)
        if len(X_eir_scaled) == 0:
            continue

        X_eir_scaled = X_eir_scaled.to(_device)
        with torch.no_grad():
            eir_preds_scaled = _model_eir(X_eir_scaled.unsqueeze(-1))
            eir_preds_unscaled = inverse_log_transform(eir_preds_scaled.cpu().numpy())

        prev_series_scaled = scaled_data['prev_true'].values[:len(eir_preds_scaled)]

        inc_input_df_scaled = pd.DataFrame({
            'prev_true': prev_series_scaled,
            'EIR_true': scaled_data['EIR_true'].values[:len(eir_preds_scaled)] if 'EIR_true' in scaled_data.columns else eir_preds_scaled[:, 0].cpu().numpy()
        })

        X_inc_input, _ = create_causal_sequences(inc_input_df_scaled, window_size, features=['prev_true', 'EIR_true'])
        X_inc_input = X_inc_input.to(_device)

        with torch.no_grad():
            inc_preds_scaled = _model_inc(X_inc_input)
            inc_preds_unscaled = inverse_log_transform(inc_preds_scaled.cpu().numpy())

        run_results[run] = {
            "eir_preds_scaled": eir_preds_scaled.cpu().numpy(),
            "eir_preds_unscaled": eir_preds_unscaled,
            "inc_preds_unscaled": inc_preds_unscaled,
            "scaled_data": scaled_data,
            "original_data": run_data,
            "y_eir_true": y_eir.cpu().numpy() if y_eir is not None else None
        }

    return run_results

@st.cache_data
def compute_global_yaxis_limits(run_results):
    all_prev, all_eir, all_inc = [], [], []
    for result in run_results.values():
        run_data = result["original_data"]
        all_prev.extend(run_data['prev_true'].values)
        all_inc.extend(result['inc_preds_unscaled'][:, 0])
        all_eir.extend(result['eir_preds_unscaled'][:, 0])

    prev_min, prev_max = 0, max(all_prev) * 1.1 if all_prev else (0, 1)
    eir_min, eir_max = 0, max(all_eir) * 1.1 if all_eir else (0, 1)
    inc_min, inc_max = 0, max(all_inc) * 1.1 if all_inc else (0, 1)

    return (prev_min, prev_max), (eir_min, eir_max), (inc_min, inc_max)

def plot_predictions(run_results, run_column, time_column, selected_runs, 
                     log_eir, log_inc, log_all, prev_limits, eir_limits, inc_limits):
    is_string_time = not pd.api.types.is_numeric_dtype(
        next(iter(run_results.values()))['original_data'][time_column]
    )

    if is_string_time:
        time_labels = next(iter(run_results.values()))['original_data'][time_column].unique()
        time_values = np.arange(len(time_labels))
    else:
        time_values = next(iter(run_results.values()))['original_data'][time_column].astype(float) / 365.25
        time_labels = None

    num_plots = len(selected_runs)
    fig, axes = plt.subplots(num_plots, 3, figsize=(18, 5 * num_plots), sharex=True)
    if num_plots == 1:
        axes = np.expand_dims(axes, axis=0)

    colors = sns.color_palette("muted", 3)
    data_to_download = []

    prev_min, prev_max = prev_limits
    eir_min, eir_max = eir_limits
    inc_min, inc_max = inc_limits

    for i, run in enumerate(selected_runs):
        result = run_results.get(run)
        if result is None:
            st.warning(f"⚠️ No prediction result available for run '{run}'")
            continue

        eir_preds_unscaled = result["eir_preds_unscaled"]
        inc_preds_unscaled = result["inc_preds_unscaled"]
        run_data = result["original_data"]
        y_eir_unscaled = inverse_log_transform(result["y_eir_true"]) if result["y_eir_true"] is not None else None

        time_values_plot = time_values[:len(eir_preds_unscaled)]
        min_len = len(eir_preds_unscaled)

        plot_data = [
            {"title": "Prevalence", "pred": None, "true": run_data['prev_true'].values[:min_len]},
            {"title": "EIR", "pred": eir_preds_unscaled[:min_len, 0], "true": y_eir_unscaled[:min_len, 0] if y_eir_unscaled is not None else None},
            {"title": "Incidence", "pred": inc_preds_unscaled[:min_len, 0], "true": None}
        ]

        log_scales = {
            "Prevalence": log_all,
            "EIR": log_eir or log_all,
            "Incidence": log_inc or log_all
        }

        y_limits = {
            "Prevalence": (prev_min, prev_max),
            "EIR": (eir_min, eir_max),
            "Incidence": (inc_min, inc_max)
        }

        for ax, data, color in zip(axes[i], plot_data, colors):
            title = data["title"]
            pred = data["pred"]
            true = data["true"]
            x_vals = time_values_plot

            if true is not None:
                ax.plot(x_vals, true, color="black", linestyle="-", label=f"True {title}", linewidth=2)
            if title != "Prevalence" and pred is not None:
                ax.plot(x_vals, pred, linestyle="--", color=color, label=f"Estimated {title}", linewidth=2.5)

            if log_scales[title]:
                ax.set_yscale('log')

            ax.set_ylim(*y_limits[title])
            ax.set_title(f"{run} - {title}", fontsize=14, color="#FF4B4B")
            ax.set_ylabel(title, fontsize=12)
            ax.legend()

            df_export = pd.DataFrame({
                "run": run,
                "time": run_data[time_column].values[:min_len],
                "Prevalence": plot_data[0]["true"],
                "Estimated EIR": plot_data[1]["pred"],
                "Estimated Incidence": plot_data[2]["pred"]
            })
            data_to_download.append(df_export)

    for ax in axes[-1]:
        if is_string_time:
            tick_indices = np.arange(0, len(time_values_plot), step=6, dtype=int)
            ax.set_xticks(time_values_plot[tick_indices])
            ax.set_xticklabels(np.array(time_labels)[tick_indices], rotation=45, fontsize=10)
        else:
            ax.set_xlabel("Years", fontsize=12)


    plt.tight_layout()
    st.pyplot(fig)

    if data_to_download:
        combined_data = pd.concat(data_to_download, ignore_index=True)
        csv_data = combined_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Estimates as CSV",
            data=csv_data,
            file_name="predictions.csv",
            mime="text/csv"
        )

# ---------------- Data sources ----------------
@st.cache_data
def load_remote_bank():
    remote_url = "https://raw.githubusercontent.com/Olatundemi/MalariaEmulator/main/test/ANC_Simulation_1000_test_runs.csv"
    df = pd.read_csv(remote_url)
    if 'prev_true' not in df.columns:
        raise ValueError("remote_url must contain 'prev_true' column")
    return df

REMOTE_DF = load_remote_bank()
UNIQUE_RUNS = REMOTE_DF['run'].unique()

def pick_remote_simulation(n):
    idx = abs(int(n)) % len(UNIQUE_RUNS)
    run = UNIQUE_RUNS[idx]
    sim = REMOTE_DF[REMOTE_DF['run'] == run]
    return sim, run

# ---------------- UI ----------------
# Custom title with responsive logo centered
MARLIN_IMG = ASSETS / "marlin.png"
if MARLIN_IMG.exists():
    import base64
    img_bytes = MARLIN_IMG.read_bytes()
    img_b64 = base64.b64encode(img_bytes).decode()

    st.markdown(
        f"""
        <style>
        .marlin-header {{
            display: flex;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap; /* allow wrapping on smaller screens */
            margin-bottom: 20px;
            text-align: center;
        }}
        .marlin-header img {{
            max-width: 80px;
            height: auto;
            margin-right: 15px;
        }}
        .marlin-header h1 {{
            font-size: 3em;
            margin: 0;
        }}
        @media (max-width: 600px) {{
            .marlin-header h1 {{
                font-size: 2em;
            }}
            .marlin-header img {{
                max-width: 50px;
                margin-bottom: 10px;
            }}
        }}
        </style>
        <div class="marlin-header">
            <img src="data:image/png;base64,{img_b64}" alt="MARLIN Logo"/>
            <h1>MARLIN</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown("<h1 style='text-align:center;'>MARLIN</h1>", unsafe_allow_html=True)

st.subheader("Malaria ANC-based Reconstructions with Learning-based **Inference using Neural networks**")
st.caption("Fast, validated insights from ANC prevalence to malaria transmission and burden.")

tab1, tab2, tab3 = st.tabs([
    "✨ Introducing MARLIN",
    "🚀 Try MARLIN",
    "💡 FAQ"
])

# ---------------- Landing Page ----------------
with tab1:
    st.markdown("---")
    left, right = st.columns([1, 2])

    with left:
        st.header("💡 The Big Idea")
        st.markdown("**1) The promise of ANC data**  \n"
                    "ANC testing is continuous and widespread, giving a dense, routine prevalence signal for program-relevant decisions.")
        st.markdown("**2) The Challenge**  \n"
                    "Prevalence lags & smooths upstream dynamics. You cannot read it as real-time transmission.")
        if ILL.exists():
            st.image(str(ILL))  # no caption here

            st.markdown(
                """
                <div style="text-align: justify; font-size: 0.9em; color: rgba(0,0,0,0.5); margin-bottom: 1em;">
                Illustrative example: Prevalence is a smoothed, lagged indicator. In seasonal settings, the same prevalence level can occur both while incidence is rapidly rising during the transmission season and when incidence has fallen back to near zero in the off-season. Identical prevalence values (purple dots) can therefore correspond to very different underlying incidence (red dots).
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("**3) Our approach**  \n"
                    "Mechanistic inference (pMCMC) runs thousands of equations hundreds of thousands of times — many hours per site.  \n"
                    "**MARLIN hunts down our target much more efficiently** using a sequence-to-sequence neural network to perform "
                    "learning-based inference from the **entire prevalence trajectory** — in seconds.")

    with right:
        st.header("How it Works ⚙️")

        # ------------------- SESSION STATE -------------------
        if "picked_run" not in st.session_state:
            st.session_state["picked_run"] = None
        if "released" not in st.session_state:
            st.session_state["released"] = False
        if "buttons_moved" not in st.session_state:
            st.session_state["buttons_moved"] = False

        # ------------------- INITIAL LAYOUT -------------------
        if not st.session_state["buttons_moved"]:
            # Left controls, right placeholder
            colA, colB = st.columns([1, 2])
            with colA:
                n = st.number_input("🎲 Pick any number:", value=7, step=1)
                if st.button("Pick simulation"):
                    _, run = pick_remote_simulation(n)
                    st.session_state["picked_run"] = run
                    st.session_state["released"] = False
                    st.session_state["buttons_moved"] = True
                    st.rerun()  # 🔑 immediate rerun to full-width mode

                if not st.session_state["released"] and CAGE.exists():
                    st.image(str(CAGE), caption="(MARLIN is ready...)", use_container_width=True)

            # Placeholder for instructions before pick
            if st.session_state["picked_run"] is None:
                colB.info("Pick a number to select a run from remote dataset.")

        else:
            # After first pick → full-width
            colB = st.container()
            with colB:
                n = st.number_input("🎲 Pick any number:", value=7, step=1, key="n_bottom")

            # ------------------- PLOT DISPLAYS -------------------
            if st.session_state["picked_run"] is not None:
                sim = REMOTE_DF[REMOTE_DF['run'] == st.session_state["picked_run"]]
                t = np.arange(len(sim))
                eir = sim['EIR_true'].values if 'EIR_true' in sim else np.zeros_like(t)
                inc = sim['incall'].values if 'incall' in sim else np.zeros_like(t)
                prev = sim['prev_true'].values

                # ----------- Display 1 -----------
                st.markdown("**How transmission, burden and prevalence are linked**")
                st.caption("We use the malariasimulation framework to attempt to capture how transmission (here expressed as the entomological inoculation rate (EIR)) drives clinical incidence, which shapes infection prevalence.")
                c1, c2, c3 = st.columns(3)
                with c2: st.plotly_chart(ts_fig(t, eir, "EIR", COLORS["eir"]), use_container_width=True)
                with c3: st.plotly_chart(ts_fig(t, inc, "Incidence", COLORS["inc"]), use_container_width=True)
                with c1: st.plotly_chart(ts_fig(t, prev, "Prevalence", COLORS["prev"]), use_container_width=True)
                

                # ----------- Display 2 -----------
                st.markdown("**What we actually see**")
                st.caption("Here we assume we only observe prevalence (e.g. ANC) and aim to reconstruct transmission and burden using our mechanistic understanding of these relationships.")
                c1, c2, c3 = st.columns(3)
                with c2: st.plotly_chart(ts_fig(t, eir, "EIR (to estimate)", COLORS["eir"], show_markers=True, opacity=0.25), use_container_width=True)
                with c3: st.plotly_chart(ts_fig(t, inc, "Incidence (to estimate)", COLORS["inc"], show_markers=True, opacity=0.25), use_container_width=True)
                with c1: st.plotly_chart(ts_fig(t, prev, "Prevalence (observed)", COLORS["prev"], show_markers=True, opacity=1.0), use_container_width=True)
                

                # ----------- Display 3 -----------
                if st.session_state["released"]:
                    st.markdown("**MARLIN reconstruction (from ANC prevalence only)**")

                    # Load models
                    window_size = 10
                    model_eir_path = "src/trained_model/shifting_sequences/LSTM_EIR_4_layers_10000run_W10.pth"
                    model_inc_path = "src/trained_model/causal/LSTM_Incidence_3_layers_15000run_causal_W10.pth"
                    model_eir, model_inc, device = load_models(model_eir_path, model_inc_path)

                    # Prepare data
                    filtered_data = adjust_trailing_zero_prevalence(sim, prevalence_column='prev_true', seed=42)
                    df_scaled, has_true_values = preprocess_data(filtered_data)

                    run_results = generate_predictions_per_run(
                        filtered_data, [st.session_state["picked_run"]], "run", window_size,
                        model_eir, model_inc, device, has_true_values
                    )

                    result = run_results[st.session_state["picked_run"]]
                    eir_pred = result["eir_preds_unscaled"][:, 0]
                    inc_pred = result["inc_preds_unscaled"][:, 0]

                    # Overlay plots with fixed-size legends 
                    c1, c2, c3 = st.columns(3)

                    # EIR
                    with c2:
                        fig_eir = go.Figure()
                        fig_eir.add_trace(go.Scatter(x=t, y=eir, mode="lines", name="True",
                                                    line=dict(color=COLORS["eir"], width=2)))
                        fig_eir.add_trace(go.Scatter(x=t[:len(eir_pred)], y=eir_pred, mode="lines", name="Inferred",
                                                    line=dict(color="black", dash="dash")))
                        fig_eir.update_layout(
                            template="simple_white", height=300, margin=dict(l=40, r=10, t=30, b=30),
                            xaxis_title="Month(s)", yaxis_title="EIR",
                            yaxis=dict(range=[0, max(max(eir), max(eir_pred)) * 1.1]),  # lock range with buffer
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                                        bgcolor="rgba(255,255,255,0.7)", bordercolor="gray", borderwidth=1)
                        )
                        st.plotly_chart(fig_eir, use_container_width=True)

                    # Incidence
                    with c3:
                        fig_inc = go.Figure()
                        fig_inc.add_trace(go.Scatter(x=t, y=inc, mode="lines", name="True",
                                                    line=dict(color=COLORS["inc"], width=2)))
                        fig_inc.add_trace(go.Scatter(x=t[:len(inc_pred)], y=inc_pred, mode="lines", name="Inferred",
                                                    line=dict(color="black", dash="dash")))
                        fig_inc.update_layout(
                            template="simple_white", height=300, margin=dict(l=40, r=10, t=30, b=30),
                            xaxis_title="Month(s)", yaxis_title="Incidence",
                            yaxis=dict(range=[0, max(max(inc), max(inc_pred)) * 1.1]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                                        bgcolor="rgba(255,255,255,0.7)", bordercolor="gray", borderwidth=1)
                        )
                        st.plotly_chart(fig_inc, use_container_width=True)

                    # Prevalence
                    with c1:
                        fig_prev = ts_fig(t, prev, "Prevalence (observed.)", COLORS["prev"], show_markers=True)
                        fig_prev.update_layout(
                            height=300, margin=dict(l=40, r=10, t=30, b=30),
                            yaxis=dict(range=[0, max(prev) * 1.1]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                            showlegend=False
                        )
                        st.plotly_chart(fig_prev, use_container_width=True)

                    st.success("✅ MARLIN inferred EIR and Incidence from prevalence alone — overlaid with ground truth for validation.")



            # ------------------- BOTTOM CONTROLS -------------------
            with colB:
                st.markdown("---")
                colX, colY, colZ = st.columns(3)

                with colX:
                    if st.button("Pick simulation", key="pick_bottom"):
                        _, run = pick_remote_simulation(st.session_state.get("n_bottom", n))
                        st.session_state["picked_run"] = run
                        st.session_state["released"] = False
                        st.rerun()  # 🔑 immediate refresh

                with colY:
                    if st.session_state["picked_run"] is not None:
                        if st.button("Unleash MARLIN 🐟", key="unleash_bottom"):
                            st.session_state["released"] = True
                            st.rerun()  # 🔑 immediate refresh

                with colZ:
                    if st.session_state["picked_run"] is not None:
                        if st.button("Reset", key="reset_bottom"):
                            st.session_state["picked_run"] = None
                            st.session_state["released"] = False
                            st.session_state["buttons_moved"] = False
                            st.rerun()  # 🔑 immediate refresh



    st.markdown("---")
    st.header("What this is — and isn’t")
    st.markdown("**MARLIN is:** an emulator trained on mechanistic models; a fast, accurate way to turn ANC prevalence into transmission & burden; scalable and decision-relevant.")
    st.markdown("**MARLIN isn’t:** a replacement for mechanistic research; a universal forecaster; a substitute for expert interpretation.")

# ---------------- Upload & Run Predictions ----------------
with tab2:
    uploaded_file = st.file_uploader("📂 Upload prevalence data to estimate (CSV or Parquet)", type=["csv", "parquet"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                test_data = load_uploaded_csv(uploaded_file)
            elif uploaded_file.name.endswith(".parquet"):
                test_data = pd.read_parquet(uploaded_file)
            else:
                st.error("❌ Unsupported file format.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to load uploaded data: {e}")
            st.stop()

        columns = test_data.columns.tolist()
        run_column = st.selectbox("🔄 Select geographical unit(s)", columns) if 'run' not in columns else 'run'
        time_column = st.selectbox("🕒 Select time column", columns) if 't' not in columns else 't'
        unique_runs = test_data[run_column].unique()
        selected_runs = st.multiselect(f"📊 Select {run_column}(s) to estimate", unique_runs, default=unique_runs[:0])

        if 'prev_true' not in columns:
            prevalence_column = st.selectbox("🩸 Select the column corresponding to prevalence", columns)
            test_data = test_data.rename(columns={prevalence_column: 'prev_true'})

        if selected_runs:
            window_size = 10
            model_eir_path = "src/trained_model/shifting_sequences/LSTM_EIR_4_layers_10000run_W10.pth"
            model_inc_path = "src/trained_model/causal/LSTM_Incidence_3_layers_15000run_causal_W10.pth"
            model_eir, model_inc, device = load_models(model_eir_path, model_inc_path)

            filtered_data = test_data[test_data[run_column].isin(selected_runs)]
            filtered_data = adjust_trailing_zero_prevalence(filtered_data, prevalence_column='prev_true', seed=42)
            df_scaled, has_true_values = preprocess_data(filtered_data)

            if df_scaled is None:
                st.stop()

            log_eir = st.checkbox("📈 View EIR on Log Scale", value=False)
            log_inc = st.checkbox("📉 View Incidence on Log Scale", value=False)
            log_all = st.checkbox("🔍 View All Plots on Log Scale", value=False)

            if st.button("Release MARLIN 🐟"):
                start_time = time.time()
                run_results = generate_predictions_per_run(
                    filtered_data, selected_runs, run_column, window_size,
                    model_eir, model_inc, device, has_true_values
                )
                st.info(f"✅ Predictions computed in {time.time() - start_time:.2f} seconds")

                if not run_results:
                    st.warning("No valid predictions could be generated.")
                    st.stop()

                prev_limits, eir_limits, inc_limits = compute_global_yaxis_limits(run_results)
                plot_predictions(
                    run_results, run_column, time_column, selected_runs,
                    log_eir, log_inc, log_all, prev_limits, eir_limits, inc_limits
                )
    else:
        st.info("Please upload a dataset to proceed.")

with tab3:
    st.header("💡 Frequently Asked Questions")

    with st.expander("1️⃣ What is MARLIN?"):
        st.write(
            "MARLIN is an emulator trained on mechanistic models, "
            "designed to infer malaria transmission dynamics from ANC prevalence data."
        )

    with st.expander("2️⃣ What data do I need?"):
        st.write(
            "You need ANC prevalence time series. "
            "The emulator is flexible but works best with clean, routine prevalence data across time."
        )

    with st.expander("3️⃣ How accurate is MARLIN?"):
        st.write(
            "It has been benchmarked against mechanistic models and shown to capture "
            "transmission and incidence dynamics efficiently in seconds."
        )

    with st.expander("4️⃣ Is MARLIN a replacement for mechanistic models?"):
        st.write(
            "No. MARLIN complements mechanistic models by providing rapid, approximate "
            "inference that supports decision-making at scale."
        )
