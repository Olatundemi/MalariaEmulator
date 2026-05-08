import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import torch
import time
from streamlit_echarts import st_echarts

from src.inference_sequence_creator import create_multistream_sequences
from src.inference_model_exp import MultiHeadModel

# ---------------- Page config ----------------
st.set_page_config(page_title="MARLIN", page_icon="🐟", layout="wide",
                   initial_sidebar_state="expanded")

ASSETS = Path("assets")
ILL = ASSETS / "illustrative_example.png"
CAGE = ASSETS / "cage.png"
MARLIN_IMG = ASSETS / "marlin.png"

COLORS = {
    "eir": "#f59e0b",
    "inc": "#ef4444",
    "prev": "#8b5cf6",
    "phi": "#9d4edd",
}

log_transform = lambda x: np.log(x + 1e-8)
inverse_log_transform = lambda x: np.exp(x) - 1e-8


# ---------------- OOD stats — lazy loaded ONLY when OOD is requested ----------------
@st.cache_resource
def load_stats():
    """Loaded lazily — only invoked when the user actually wants OOD overlays."""
    stats = torch.load("latent_stats.pt")
    return stats["mu_latent"], stats["cov_inv_latent"], stats["ood_threshold"]


# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
        h1 { color: #FF4B4B; text-align: center; }
        .stButton>button {
            background-color: #4CAF50; color: white;
            border-radius: 8px; font-size: 16px;
        }
        .stButton>button:hover { background-color: #45a049; }
        [data-testid="stSidebar"] { background-color: #2E3B4E; color: white; }
    </style>
""", unsafe_allow_html=True)


# ---------------- Plotly small intro figure ----------------
def ts_fig(t, y, title, color, show_markers=False, opacity=1.0):
    fig = go.Figure()
    mode = "markers" if show_markers else "lines"
    fig.add_trace(go.Scatter(
        x=t, y=y, mode=mode,
        line=dict(color=color, width=2),
        marker=dict(color=color, size=5, opacity=opacity),
        opacity=opacity, name=title,
    ))
    fig.update_layout(
        height=240, margin=dict(l=40, r=10, t=30, b=30),
        xaxis_title="Month(s)", yaxis_title=title,
        template="simple_white",
        transition=dict(duration=600, easing="cubic-in-out"),
    )
    return fig

# Function to convert time column → always returns a Series in fractional years
def convert_time_column(df, time_column):
    try:
        if pd.api.types.is_numeric_dtype(df[time_column]):
            return df[time_column].astype(float) / 365.25
        # Try month-year string (e.g. "Jan-16") then fall back to general dateutil parsing
        parsed = pd.to_datetime(df[time_column], errors="coerce", format="%b-%y")
        if parsed.isna().all():
            parsed = pd.to_datetime(df[time_column], errors="coerce")
        if parsed.isna().all():
            st.error("Could not parse the time column. Ensure it uses a date format (e.g. Jan-16 or 2016-01-01).")
            return None
        start_year = parsed.dt.year.min()
        return (parsed.dt.year + (parsed.dt.month - 1) / 12 - start_year).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error converting time column: {e}")
        return None

# ---------------- ECharts time-series helper (animated) ----------------
def echarts_timeseries(
    t,
    series,
    title,
    yaxis_title="Value",
    height=340,
    log_y=False,
    ood_mask=None,
    animation_duration=1800,
    key=None,
):
    """
    series: list of {name, data, color, dash ("solid"|"dashed")}
    ood_mask: bool array same length as t — orange shaded regions
    """
    t_list = [float(x) for x in t]

    # ---- OOD shaded regions via markArea ----
    mark_area_data = []
    if ood_mask is not None:
        start = None
        for i, m in enumerate(ood_mask):
            if m and start is None:
                start = t_list[i]
            elif not m and start is not None:
                mark_area_data.append([{"xAxis": start}, {"xAxis": t_list[i]}])
                start = None
        if start is not None:
            mark_area_data.append([{"xAxis": start}, {"xAxis": t_list[-1]}])

    chart_series = []
    legend_names = []
    first_visible = True
    for s in series:
        if s.get("data") is None:
            continue
        n = min(len(t_list), len(s["data"]))
        item = {
            "name": s["name"],
            "type": "line",
            "data": [[t_list[i], float(s["data"][i])] for i in range(n)],
            "smooth": True,
            "showSymbol": False,
            "lineStyle": {
                "color": s["color"],
                "width": 2.2,
                "type": s.get("dash", "solid"),
            },
            "itemStyle": {"color": s["color"]},
            "emphasis": {"focus": "series"},
            "animationDuration": animation_duration,
            "animationEasing": "cubicOut",
        }
        if first_visible and mark_area_data:
            item["markArea"] = {
                "itemStyle": {"color": "rgba(255,165,0,0.18)"},
                "data": mark_area_data,
                "silent": True,
            }
        first_visible = False
        chart_series.append(item)
        legend_names.append(s["name"])

    options = {
        "title": {"text": title, "left": "center", "textStyle": {"fontSize": 13}},
        "tooltip": {"trigger": "axis"},
        "legend": {"data": legend_names, "bottom": 0},
        "grid": {"top": 40, "bottom": 55, "left": 60, "right": 20},
        "xAxis": {"type": "value", "name": "Time (Years)", "scale": True},
        "yAxis": {
            "type": "log" if log_y else "value",
            "name": yaxis_title,
            "scale": True,
        },
        "series": chart_series,
        "animationDuration": animation_duration,
        "animationDurationUpdate": animation_duration,
        "animationEasing": "cubicOut",
    }
    st_echarts(options=options, height=f"{height}px", key=key)


# ---------------- Observed vs Predicted scatter (Plotly) ----------------
def scatter_obs_vs_pred(result, show_phi=True):
    """
    Returns a Plotly figure with one scatter panel per output (EIR, Phi, Incidence).
    Each panel shows predicted vs ground-truth with the identity line and R².
    """
    metric_map = [
        ("eir", "EIR",        "#1f77b4"),
        ("phi", "Phi (φ)",    "#d62728"),
        ("inc", "Incidence",  "#2ca02c"),
    ]
    panels = []
    for key, label, color in metric_map:
        if key == "phi" and not show_phi:
            continue
        y_true, y_pred = result[key]
        if y_true is None or y_pred is None:
            continue
        n = min(len(y_true), len(y_pred))
        yt, yp = y_true[:n], y_pred[:n]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        panels.append((label, color, yt, yp, r2))

    if not panels:
        return None

    titles = [f"<b>{lbl}</b>  —  Overall R² = {r2:.3f}" for lbl, _, _, _, r2 in panels]
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=titles,
                        horizontal_spacing=0.08)

    show_legend = True
    for i, (label, color, yt, yp, r2) in enumerate(panels, 1):
        mn = float(min(yt.min(), yp.min()))
        mx = float(max(yt.max(), yp.max()))
        pad = (mx - mn) * 0.04
        mn -= pad; mx += pad

        fig.add_trace(go.Scatter(
            x=yt, y=yp, mode="markers",
            marker=dict(color=color, size=4, opacity=0.45),
            name="Predictions", showlegend=show_legend,
        ), row=1, col=i)

        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color="black", width=1.5, dash="dash"),
            name="Perfect Fit (x=y)", showlegend=show_legend,
        ), row=1, col=i)

        show_legend = False
        # No per-panel x title — shared annotation below handles it
        fig.update_xaxes(title_text="", range=[mn, mx],
                         showgrid=True, gridcolor="#eee", row=1, col=i)
        # "Predicted" label only on the leftmost panel to avoid repetition
        fig.update_yaxes(title_text="Predicted" if i == 1 else "",
                         range=[mn, mx], showgrid=True, gridcolor="#eee", row=1, col=i)

    # Single shared x-axis label centred below all panels
    fig.add_annotation(
        text="Observed (Ground Truth)",
        xref="paper", yref="paper",
        x=0.5, y=-0.23,
        showarrow=False,
        font=dict(size=12, color="#444"),
    )

    fig.update_layout(
        height=380,
        template="simple_white",
        margin=dict(l=55, r=20, t=65, b=85),
        # Legend sits below the shared x-label
        legend=dict(orientation="h", yanchor="top", y=-0.22,
                    xanchor="center", x=0.5),
    )
    return fig


# ---------------- Multi-panel ECharts row (shared legend + x-axis label) ----------------
def echarts_panel_row(t, panels, height=420, ood_mask=None, animation_duration=1800,
                      show_legend=True, key=None):
    """
    Renders N panels side-by-side in a single ECharts instance.
    Shared legend at bottom-centre (shown only when show_legend=True).
    Single 'Time (Years)' label below all panels.
    panels: list of {title, yaxis_title, series:[{name,data,color,dash}], log_y}
    """
    n = len(panels)
    t_list = [float(x) for x in t]

    mark_area_data = []
    if ood_mask is not None:
        start = None
        for i, m in enumerate(ood_mask):
            if m and start is None:
                start = t_list[i]
            elif not m and start is not None:
                mark_area_data.append([{"xAxis": start}, {"xAxis": t_list[i]}])
                start = None
        if start is not None:
            mark_area_data.append([{"xAxis": start}, {"xAxis": t_list[-1]}])

    # Grid layout: tight margins — y-axis tick labels only need ~4% gap between panels
    first_left = 5.0
    right_margin = 1.0
    inter_gap = 4.0
    available = 100.0 - first_left - right_margin
    panel_width = (available - (n - 1) * inter_gap) / n

    # Top: 65px to fit title + optional run-name subtitle
    # Bottom: x-tick labels + "Time (Years)" label + legend
    grid_top = "65px"
    grid_bottom = "90px"

    grids, x_axes, y_axes = [], [], []
    for i in range(n):
        left = first_left + i * (panel_width + inter_gap)
        grids.append({"left": f"{left:.1f}%", "width": f"{panel_width:.1f}%",
                       "top": grid_top, "bottom": grid_bottom, "containLabel": False})
        x_axes.append({
            "type": "value", "gridIndex": i, "scale": True,
            "axisLabel": {"fontSize": 11},
        })
        # No y-axis name — panel title above already labels it
        y_axes.append({
            "type": "log" if panels[i].get("log_y") else "value",
            "gridIndex": i,
            "name": "",
            "scale": True,
            "axisLabel": {"fontSize": 10},
        })

    all_series = []
    legend_names_ordered = []
    seen_legend = set()
    ood_attached = False

    for i, panel in enumerate(panels):
        first_in_panel = True
        for s in panel["series"]:
            if s.get("data") is None:
                continue
            n_pts = min(len(t_list), len(s["data"]))
            item = {
                "name": s["name"],
                "type": "line",
                "xAxisIndex": i,
                "yAxisIndex": i,
                "data": [[t_list[j], float(s["data"][j])] for j in range(n_pts)],
                "smooth": True,
                "showSymbol": False,
                "lineStyle": {"color": s["color"], "width": 2.2, "type": s.get("dash", "solid")},
                "itemStyle": {"color": s["color"]},
                "emphasis": {"focus": "series"},
                "animationDuration": animation_duration,
                "animationEasing": "cubicOut",
            }
            if first_in_panel and not ood_attached and mark_area_data:
                item["markArea"] = {"itemStyle": {"color": "rgba(255,165,0,0.18)"},
                                    "data": mark_area_data, "silent": True}
                ood_attached = True
            first_in_panel = False
            all_series.append(item)
            if s["name"] not in seen_legend:
                legend_names_ordered.append(s["name"])
                seen_legend.add(s["name"])

    title_objs = []
    for i, panel in enumerate(panels):
        cx = first_left + i * (panel_width + inter_gap) + panel_width / 2
        t_obj = {
            "text": panel["title"],
            "left": f"{cx:.1f}%",
            "top": "8px",
            "textAlign": "center",
            "textStyle": {"fontSize": 13, "fontWeight": "bold"},
        }
        if panel.get("subtitle"):
            t_obj["subtext"] = panel["subtitle"]
            t_obj["subtextStyle"] = {"fontSize": 10, "color": "#666", "align": "center"}
        title_objs.append(t_obj)

    # "Time (Years)" sits between x-tick labels and the legend
    x_label_graphic = {"type": "text", "bottom": 42, "left": "center",
                        "style": {"text": "Time (Years)", "fontSize": 12, "fill": "#666"}}

    legend_cfg = {
        "show": show_legend,
        "data": legend_names_ordered,
        "bottom": "8px",
        "left": "center",
        "orient": "horizontal",
        "textStyle": {"fontSize": 12},
    }

    options = {
        "title": title_objs,
        # tooltip scoped to each grid — shows only the hovered panel's series
        "tooltip": {
            "trigger": "axis",
            "formatter": None,
        },
        "legend": legend_cfg,
        "grid": grids,
        "xAxis": x_axes,
        "yAxis": y_axes,
        "series": all_series,
        "graphic": [x_label_graphic],
        "animationDuration": animation_duration,
        "animationDurationUpdate": animation_duration,
        "animationEasing": "cubicOut",
    }
    st_echarts(options=options, height=f"{height}px", key=key)


# ---------------- Model loading ----------------
@st.cache_resource
def load_models(model_eir_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_eir = MultiHeadModel()
    model_eir.load_state_dict(torch.load(model_eir_path, map_location=device))
    model_eir.to(device)
    model_eir.eval()
    return model_eir, device


def preprocess_data(df):
    if not pd.api.types.is_numeric_dtype(df['prev_true']):
        st.error("🚨 The selected prevalence column is invalid.")
        return None
    df_scaled = df.copy()
    df_scaled['prev_true'] = log_transform(df_scaled['prev_true'])
    if 'EIR_true' in df.columns:
        df_scaled['EIR_true'] = log_transform(df_scaled['EIR_true'])
    if 'incall' in df.columns:
        df_scaled['incall'] = log_transform(df_scaled['incall'])
    if 'phi' in df.columns:
        df_scaled['phi'] = log_transform(df_scaled['phi'])
    return df_scaled


# ---------------- Inference ----------------
def infer_chained_models(
    model,
    run_df,
    win_eir,
    win_phi,
    device,
    compute_ood: bool = False,
    _mu_latent=None,
    _cov_inv_latent=None,
    _ood_threshold=None,
):
    """
    OOD detection is fully gated by `compute_ood`. When False:
      - The two extra latent forward passes are skipped (faster)
      - latent stats are not touched (allows lazy loading upstream)
    """
    model.eval()

    streams = create_multistream_sequences(run_df, win_eir=win_eir, win_phi=win_phi)

    with torch.no_grad():
        batch = {
            k: tuple(v.to(device) if v is not None else None for v in streams[k])
            for k in streams
        }

        pred_eir_log, pred_phi_log, pred_inc_log = model(batch)

        # ----- OOD only when explicitly requested -----
        ood_distance = None
        ood_mask = None
        if compute_ood and _mu_latent is not None and _cov_inv_latent is not None:
            h_eir = model.eir(batch["eir"][0], batch["eir"][1], return_sequence=False)
            h_phi = model.phi(batch["phi"][0], batch["phi"][1], return_sequence=False)
            z = torch.cat([h_eir, h_phi], dim=-1).cpu()
            diff = z - _mu_latent
            d = torch.sum(diff @ _cov_inv_latent * diff, dim=1)
            ood_distance = d.numpy()
            if _ood_threshold is not None:
                ood_mask = ood_distance > _ood_threshold.item()

        p_eir = inverse_log_transform(pred_eir_log.squeeze(-1).cpu().numpy())
        p_phi = inverse_log_transform(pred_phi_log.squeeze(-1).cpu().numpy())
        p_inc = inverse_log_transform(pred_inc_log.squeeze(-1).cpu().numpy())

    n_preds = len(p_eir)
    if "t_years" in run_df.columns:
        t = run_df["t_years"].values[:n_preds]          # already in fractional years
    elif "t" in run_df.columns:
        t = run_df["t"].values[:n_preds] / 365.25       # numeric days → years
    else:
        t = np.arange(n_preds)

    y_prev_true = inverse_log_transform(run_df["prev_true"].values[:n_preds])
    y_eir_true = inverse_log_transform(streams["eir"][2].squeeze(-1).cpu().numpy()) if streams["eir"][2] is not None else None
    y_phi_true = inverse_log_transform(streams["phi"][2].squeeze(-1).cpu().numpy()) if streams["phi"][2] is not None else None
    y_inc_true = inverse_log_transform(streams["inc"][1].squeeze(-1).cpu().numpy()) if streams["inc"][1] is not None else None

    return {
        "t": t,
        "prev": (y_prev_true, None),
        "eir": (y_eir_true, p_eir),
        "phi": (y_phi_true, p_phi),
        "inc": (y_inc_true, p_inc),
        "ood_distance": ood_distance,
        "ood_mask": ood_mask,
    }


@st.cache_data(show_spinner="🔄 Running model predictions...")
def generate_predictions_per_run(
    data,
    selected_runs,
    run_column,
    _model,
    _device,
    compute_ood: bool = False,
    _mu_latent=None,
    _cov_inv_latent=None,
    _ood_threshold=None,
):
    """OOD is opt-in. Pass compute_ood=True ONLY when the user has ticked the box."""
    run_results = {}
    for run in selected_runs:
        run_df = data[data[run_column] == run].reset_index(drop=True)
        if run_df.empty:
            continue
        res = infer_chained_models(
            _model, run_df,
            win_eir=20, win_phi=300,
            device=_device,
            compute_ood=compute_ood,
            _mu_latent=_mu_latent if compute_ood else None,
            _cov_inv_latent=_cov_inv_latent if compute_ood else None,
            _ood_threshold=_ood_threshold if compute_ood else None,
        )
        run_results[run] = res
    return run_results


# ---------------- Plot predictions (ECharts version, replaces matplotlib) ----------------
def plot_predictions(
    run_results,
    selected_runs,
    log_eir,
    log_inc,
    log_all,
    show_ood=True,
    show_phi=True,
):
    metric_colors = {
        "prev": COLORS["prev"],
        "eir": "#1f77b4",
        "phi": "#d62728",
        "inc": "#2ca02c",
    }
    titles  = ["Prevalence", "EIR", "Phi", "Incidence"]
    metrics = ["prev",       "eir", "phi", "inc"]

    data_to_download = []

    for run_idx, run in enumerate(selected_runs):
        result = run_results[run]
        t = result["t"]
        ood_mask = result.get("ood_mask", None) if show_ood else None

        run_export = pd.DataFrame({"run": run, "time_years": t})

        panels = []
        for j, metric in enumerate(metrics):
            if metric == "phi" and not show_phi:
                continue
            y_true, y_pred = result[metric]
            log_y = log_all or (metric == "eir" and log_eir) or (metric == "inc" and log_inc)
            series = []
            if y_true is not None:
                series.append({"name": "True",      "data": y_true,
                               "color": "#000000",  "dash": "solid"})
                run_export[f"Actual_{metric}"] = y_true
            if y_pred is not None:
                series.append({"name": "Estimated", "data": y_pred,
                               "color": metric_colors[metric], "dash": "dashed"})
                run_export[f"Estimated_{metric}"] = y_pred
            # Embed run name as subtitle on the Prevalence panel — removes need for a separate header
            subtitle = str(run) if metric == "prev" else None
            panels.append({"title": titles[j], "subtitle": subtitle,
                           "yaxis_title": titles[j], "series": series, "log_y": log_y})

        # Legend shown only on the first run — universal reference for all subsequent runs
        echarts_panel_row(t, panels, height=420, ood_mask=ood_mask,
                          show_legend=(run_idx == 0), key=f"chart_{run}")
        data_to_download.append(run_export)

    if data_to_download:
        combined = pd.concat(data_to_download, ignore_index=True)
        st.download_button(
            "📥 Download Estimates as CSV",
            data=combined.to_csv(index=False).encode("utf-8"),
            file_name="model_predictions.csv",
            mime="text/csv",
        )


# ---------------- Misc helpers ----------------
@st.cache_data
def compute_global_yaxis_limits(run_results):
    all_prev, all_eir, all_inc = [], [], []
    for result in run_results.values():
        prev_true, _ = result["prev"]
        if prev_true is not None: all_prev.extend(prev_true)
        eir_true, eir_pred = result["eir"]
        if eir_true is not None: all_eir.extend(eir_true)
        if eir_pred is not None: all_eir.extend(eir_pred)
        inc_true, inc_pred = result["inc"]
        if inc_true is not None: all_inc.extend(inc_true)
        if inc_pred is not None: all_inc.extend(inc_pred)

    def safe(values):
        return (0, 1) if not values else (0, max(values) * 1.1)
    return safe(all_prev), safe(all_eir), safe(all_inc)


def adjust_trailing_zero_prevalence(df, prevalence_column='prev_true',
                                    min_val=0.0003, max_val=0.001, seed=None):
    df = df.copy()
    zeros_mask = df[prevalence_column] == 0
    n = zeros_mask.sum()
    if n > 0:
        rng = np.random.default_rng(seed)
        df.loc[zeros_mask, prevalence_column] = rng.uniform(min_val, max_val, size=n)
    return df


@st.cache_data
def load_uploaded_csv(file_content):
    return pd.read_csv(file_content)


@st.cache_data
def load_remote_bank():
    remote_url = "https://raw.githubusercontent.com/Olatundemi/MalariaEmulator/main/test/ANC_Simulation_test_samples_20_runs_with_under5.csv"
    df = pd.read_csv(remote_url)
    if 'prev_true' not in df.columns:
        raise ValueError("remote_url must contain 'prev_true' column")
    return df


REMOTE_DF = load_remote_bank()
UNIQUE_RUNS = REMOTE_DF['run'].unique()


def pick_remote_simulation(n):
    idx = abs(int(n)) % len(UNIQUE_RUNS)
    run = UNIQUE_RUNS[idx]
    return REMOTE_DF[REMOTE_DF['run'] == run], run


# ---------------- Header ----------------
if MARLIN_IMG.exists():
    import base64
    img_b64 = base64.b64encode(MARLIN_IMG.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        .marlin-header {{
            display:flex; justify-content:center; align-items:center;
            flex-wrap:wrap; margin-bottom:20px; text-align:center;
        }}
        .marlin-header img {{ max-width:80px; height:auto; margin-right:15px; }}
        .marlin-header h1 {{ font-size:3em; margin:0; }}
        @media (max-width:600px) {{
            .marlin-header h1 {{ font-size:2em; }}
            .marlin-header img {{ max-width:50px; margin-bottom:10px; }}
        }}
        </style>
        <div class="marlin-header">
            <img src="data:image/png;base64,{img_b64}" alt="MARLIN Logo"/>
            <h1>MARLIN</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("<h1 style='text-align:center;'>MARLIN</h1>", unsafe_allow_html=True)

st.subheader("Malaria ANC-based Reconstructions with Learning-based **Inference using Neural networks**")
st.caption("Fast, validated insights from ANC prevalence to malaria transmission and burden.")

tab1, tab2, tab3 = st.tabs([
    "✨ Introducing MARLIN",
    "🚀 Try MARLIN",
    "💡 FAQ",
])


# =============================================================================
# TAB 1 — Landing page
# =============================================================================
with tab1:
    st.markdown("---")

    # ---- session state ----
    if "picked_run" not in st.session_state: st.session_state["picked_run"] = None
    if "released" not in st.session_state: st.session_state["released"] = False
    if "buttons_moved" not in st.session_state: st.session_state["buttons_moved"] = False

    left, right = st.columns([1, 2])

    # ---------- LEFT: explainer ----------
    with left:
        st.header("💡 The Big Idea")
        st.markdown("**1) The promise of ANC data**  \n"
                    "ANC testing is continuous and widespread, giving a dense, routine prevalence signal for program-relevant decisions.")
        st.markdown("**2) The challenge**  \n"
                    "Prevalence lags and smooths upstream dynamics. You cannot read it as real-time transmission.")
        if ILL.exists():
            st.image(str(ILL), use_container_width=True)
            st.markdown(
                """
                <div style="text-align:justify; font-size:0.9em; color:rgba(0,0,0,0.5); margin-bottom:1em;">
                Illustrative example: Prevalence is a smoothed, lagged indicator. In seasonal settings, the same prevalence level can occur both while incidence is rapidly rising during the transmission season and when incidence has fallen back to near zero in the off-season. Identical prevalence values (purple dots) can therefore correspond to very different underlying incidence (red dots).
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("**3) Our approach**  \n"
                    "Mechanistic inference (pMCMC) runs thousands of equations hundreds of thousands of times - many hours per site.  \n"
                    "**MARLIN hunts down our target much more efficiently** using a sequence-to-sequence neural network to perform "
                    "learning-based inference from the **entire prevalence trajectory** - in seconds.")

    # ---------- RIGHT: picker + intro plots + buttons ----------
    with right:
        st.header("How it Works ⚙️")

        if not st.session_state["buttons_moved"]:
            colA, colB = st.columns([1, 2])
            with colA:
                n = st.number_input("🎲 Select any number:", value=7, step=1)
                if st.button("Load simulation"):
                    _, run = pick_remote_simulation(n)
                    st.session_state["picked_run"] = run
                    st.session_state["released"] = False
                    st.session_state["buttons_moved"] = True
                    st.rerun()
                if not st.session_state["released"] and CAGE.exists():
                    st.image(str(CAGE), caption="(MARLIN is ready...)", use_container_width=True)
            if st.session_state["picked_run"] is None:
                colB.info("Pick a number to select a run from remote dataset.")
        else:
            n = st.number_input("🎲 Select a  number:", value=7, step=1, key="n_bottom")

            if st.session_state["picked_run"] is not None:
                sim = REMOTE_DF[REMOTE_DF['run'] == st.session_state["picked_run"]]
                t = np.arange(len(sim))
                eir = sim['EIR_true'].values if 'EIR_true' in sim else np.zeros_like(t)
                inc = sim['incall'].values if 'incall' in sim else np.zeros_like(t)
                prev = sim['prev_true'].values

                st.markdown("**How transmission, burden and prevalence are linked**")
                st.caption("We use the malariasimulation framework to attempt to capture how transmission (here expressed as the entomological inoculation rate (EIR)) drives clinical incidence, which shapes infection prevalence.")
                c1, c2, c3 = st.columns(3)
                with c1: st.plotly_chart(ts_fig(t, eir, "EIR", COLORS["eir"]), use_container_width=True)
                with c2: st.plotly_chart(ts_fig(t, inc, "Incidence", COLORS["inc"]), use_container_width=True)
                with c3: st.plotly_chart(ts_fig(t, prev, "Prevalence", COLORS["prev"]), use_container_width=True)

                st.markdown("**What we actually see**")
                st.caption("Here we assume we only observe prevalence (e.g. ANC) and aim to reconstruct transmission and burden using our mechanistic understanding of these relationships.")
                c1, c2, c3 = st.columns(3)
                with c2: st.plotly_chart(ts_fig(t, eir, "EIR (to estimate)", COLORS["eir"], show_markers=True, opacity=0.25), use_container_width=True)
                with c3: st.plotly_chart(ts_fig(t, inc, "Incidence (to estimate)", COLORS["inc"], show_markers=True, opacity=0.25), use_container_width=True)
                with c1: st.plotly_chart(ts_fig(t, prev, "Prevalence (observed)", COLORS["prev"], show_markers=True, opacity=1.0), use_container_width=True)

            # bottom controls (still in right column)
            st.markdown("---")
            colX, colY, colZ = st.columns(3)
            with colX:
                if st.button("Load simulation", key="pick_bottom"):
                    _, run = pick_remote_simulation(st.session_state.get("n_bottom", n))
                    st.session_state["picked_run"] = run
                    st.session_state["released"] = False
                    st.rerun()
            with colY:
                if st.session_state["picked_run"] is not None:
                    if st.button("Unleash MARLIN 🐟", key="unleash_bottom"):
                        st.session_state["released"] = True
                        st.rerun()
            with colZ:
                if st.session_state["picked_run"] is not None:
                    if st.button("Reset", key="reset_bottom"):
                        st.session_state["picked_run"] = None
                        st.session_state["released"] = False
                        st.session_state["buttons_moved"] = False
                        st.rerun()

    # ---------- FULL-WIDTH RECONSTRUCTION (was squeezed inside `right`) ----------
    if st.session_state["buttons_moved"] and st.session_state["released"] \
            and st.session_state["picked_run"] is not None:

        st.markdown("---")
        st.markdown("### MARLIN reconstruction (from ANC prevalence only)")

        sim = REMOTE_DF[REMOTE_DF['run'] == st.session_state["picked_run"]]
        model_path = "src/trained_model/shifting_sequences/multitask_model_improvedMSConv_HPE_EIR_phi_with_incidence.pth"
        model, device = load_models(model_path)

        filtered_data = adjust_trailing_zero_prevalence(sim, prevalence_column='prev_true', seed=42).reset_index(drop=True)
        df_scaled = preprocess_data(filtered_data)

        # ⚡ Tab 1 NEVER displays OOD → compute_ood=False (skips two extra forward passes)
        run_results = generate_predictions_per_run(
            df_scaled,
            [st.session_state["picked_run"]],
            "run",
            model, device,
            compute_ood=False,
        )
        result = run_results.get(st.session_state["picked_run"])

        if result is None:
            st.warning("No valid predictions could be generated for this run.")
        else:
            t_model = result["t"]

            eir_series = []
            if result["eir"][0] is not None:
                eir_series.append({"name": "True",     "data": result["eir"][0],
                                   "color": "#000000", "dash": "solid"})
            eir_series.append(    {"name": "Inferred", "data": result["eir"][1],
                                   "color": "#1f77b4", "dash": "dashed"})

            phi_series = []
            if result["phi"][0] is not None:
                phi_series.append({"name": "True",     "data": result["phi"][0],
                                   "color": "#000000", "dash": "solid"})
            phi_series.append(    {"name": "Inferred", "data": result["phi"][1],
                                   "color": "#d62728", "dash": "dashed"})

            inc_series = []
            if result["inc"][0] is not None:
                inc_series.append({"name": "True",     "data": result["inc"][0],
                                   "color": "#000000", "dash": "solid"})
            inc_series.append(    {"name": "Inferred", "data": result["inc"][1],
                                   "color": "#2ca02c", "dash": "dashed"})

            _opt1, _opt2 = st.columns(2)
            with _opt1:
                show_phi_tab1    = st.checkbox("Show Immunity Function (Phi)", value=True,  key="show_phi_tab1")
            with _opt2:
                show_scatter_tab1 = st.checkbox("📊 Show Observed vs Predicted scatter",    value=False, key="show_scatter_tab1")

            recon_panels = [
                {"title": "Prevalence (observed)", "yaxis_title": "Prevalence", "log_y": False,
                 "series": [{"name": "Prevalence (observed)", "data": result["prev"][0],
                              "color": COLORS["prev"], "dash": "solid"}]},
                {"title": "EIR",       "yaxis_title": "EIR",       "log_y": False, "series": eir_series},
                {"title": "Phi",       "yaxis_title": "Phi",       "log_y": False, "series": phi_series},
                {"title": "Incidence", "yaxis_title": "Incidence", "log_y": False, "series": inc_series},
            ]
            visible_panels = [p for p in recon_panels if p["title"] != "Phi" or show_phi_tab1]
            echarts_panel_row(t_model, visible_panels, height=430, key="recon_panel")

            
            if show_scatter_tab1:
                scatter_fig = scatter_obs_vs_pred(result, show_phi=show_phi_tab1)
                if scatter_fig:
                    st.plotly_chart(scatter_fig, use_container_width=True)

            st.success("✅ MARLIN inferred EIR, Immunity and Incidence from prevalence alone — overlaid with ground truth.")


    st.markdown("---")
    st.header("What this is - and isn’t")
    st.markdown("**MARLIN is:** an emulator trained on mechanistic models; a fast, accurate way to turn ANC prevalence into transmission & burden; scalable and decision-relevant.")
    st.markdown("**MARLIN isn’t:** a replacement for mechanistic research; a universal forecaster; a substitute for expert interpretation.")


# =============================================================================
# TAB 2 — Upload & Run
# =============================================================================
with tab2:
    uploaded_file = st.file_uploader("📂 Upload prevalence data to estimate (CSV or Parquet)",
                                     type=["csv", "parquet"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                test_data = load_uploaded_csv(uploaded_file)
            elif uploaded_file.name.endswith(".parquet"):
                test_data = pd.read_parquet(uploaded_file)
            else:
                st.error("❌ Unsupported file format."); st.stop()
        except Exception as e:
            st.error(f"Failed to load uploaded data: {e}"); st.stop()

        columns = test_data.columns.tolist()
        run_column  = st.selectbox("🔄 Select geographical unit(s)", columns) if 'run' not in columns else 'run'
        time_column = st.selectbox("🕒 Select time column", columns) if 't' not in columns else 't'

        # Convert time to fractional years and attach as t_years
        converted_t = convert_time_column(test_data, time_column)
        if converted_t is not None:
            test_data = test_data.copy()
            test_data["t_years"] = converted_t.values

        unique_runs = test_data[run_column].unique()
        selected_runs = st.multiselect(f"📊 Select {run_column}(s) to estimate",
                                       unique_runs, default=unique_runs[:0])

        if 'prev_true' not in columns:
            prevalence_column = st.selectbox("🩸 Select the column corresponding to prevalence", columns)
            test_data = test_data.rename(columns={prevalence_column: 'prev_true'})

        if selected_runs:
            model_path = "src/trained_model/shifting_sequences/multitask_model_improvedMSConv_HPE_EIR_phi_with_incidence.pth"
            model, device = load_models(model_path)

            filtered_data = test_data[test_data[run_column].isin(selected_runs)]
            filtered_data = adjust_trailing_zero_prevalence(filtered_data, prevalence_column='prev_true', seed=42)
            df_scaled = preprocess_data(filtered_data)
            if df_scaled is None: st.stop()

            _c1, _c2, _c3 = st.columns(3)
            with _c1:
                log_eir  = st.checkbox("📈 EIR on Log Scale", value=False)
                show_phi = st.checkbox("🧬 View Immunity plot", value=True,
                                       help="Uncheck to hide the Immunity function panel and give more space to the remaining plots.")
            with _c2:
                log_inc  = st.checkbox("📉 Incidence on Log Scale", value=False)
                show_ood = st.checkbox("🔴 Show OOD Regions", value=False,
                                       help="Adds an extra latent forward pass per run.")
            with _c3:
                log_all  = st.checkbox("🔍 All Plots on Log Scale", value=False)

            if st.button("Release MARLIN 🐟"):
                # 🔑 OOD stats are loaded ONLY when the user wants OOD
                mu_latent = cov_inv_latent = ood_threshold = None
                if show_ood:
                    mu_latent, cov_inv_latent, ood_threshold = load_stats()

                start_time = time.time()
                run_results = generate_predictions_per_run(
                    df_scaled, selected_runs, run_column,
                    model, device,
                    compute_ood=show_ood,
                    _mu_latent=mu_latent,
                    _cov_inv_latent=cov_inv_latent,
                    _ood_threshold=ood_threshold,
                )
                st.info(f"✅ Predictions computed in {time.time() - start_time:.2f} seconds")

                if not run_results:
                    st.warning("No valid predictions could be generated."); st.stop()

                plot_predictions(
                    run_results, selected_runs,
                    log_eir, log_inc, log_all,
                    show_ood=show_ood,
                    show_phi=show_phi,
                )
    else:
        st.info("Please upload a dataset to proceed.")


# =============================================================================
# TAB 3 — FAQ
# =============================================================================
with tab3:
    st.header("💡 Frequently Asked Questions")

    with st.expander("1️⃣ What is MARLIN?"):
        st.write("MARLIN is an emulator trained on mechanistic models, "
                 "designed to infer malaria transmission dynamics from ANC prevalence data.")
    with st.expander("2️⃣ What data do I need?"):
        st.write("You need ANC prevalence time series. "
                 "The emulator is flexible but works best with clean, routine prevalence data across time.")
    with st.expander("3️⃣ How accurate is MARLIN?"):
        st.write("It has been benchmarked against mechanistic models and shown to capture "
                 "transmission and incidence dynamics efficiently in seconds.")
    with st.expander("4️⃣ Is MARLIN a replacement for mechanistic models?"):
        st.write("No. MARLIN complements mechanistic models by providing rapid, approximate "
                 "inference that supports decision-making at scale.")