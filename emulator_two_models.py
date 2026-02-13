import streamlit as st 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.inference_sequence_creator import create_multistream_sequences
from src.inference_model_exp import MultiHeadModel
import time
import hashlib


log_transform = lambda x: np.log(x + 1e-8)
inverse_log_transform = lambda x: np.exp(x) - 1e-8


# Set page configuration
st.set_page_config(
    page_title="Malaria Estimator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# # Apply a modern Seaborn theme
# sns.set_style("darkgrid")  
# plt.style.use("ggplot")

# Custom CSS for better UI
st.markdown("""
    <style>
        /* Customize headers */
        h1 {
            color: #FF4B4B;
            text-align: center;
        }
        /* Improve widgets */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        /* Adjust sidebar */
        [data-testid="stSidebar"] {
            background-color: #2E3B4E;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Function to load model
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
        return None, False

    has_true_values = {'EIR_true', 'incall', 'phi'}.issubset(df.columns)

    df_scaled = df.copy()

    # Log-transform ONLY model columns
    df_scaled['prev_true'] = log_transform(df_scaled['prev_true'])

    if has_true_values:
        df_scaled['EIR_true'] = log_transform(df_scaled['EIR_true'])
        df_scaled['incall'] = log_transform(df_scaled['incall'])
        df_scaled['phi'] = log_transform(df_scaled['phi'])

    return df_scaled, has_true_values


# Function to convert time column
def convert_time_column(df, time_column):
    try:
        if pd.api.types.is_numeric_dtype(df[time_column]):
            return df[time_column].astype(float) / 365.25  # Convert days to years
        else:
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce', format='%b-%y')  # Example: "Jan-16"
            
            if df[time_column].isna().all():
                st.error("Could not parse the time column. Ensure it's a proper date format (e.g., Jan-16).")
                return None
        
            start_year = df[time_column].dt.year.min()
            df['time_in_years'] = df[time_column].dt.year + (df[time_column].dt.month - 1) / 12 - start_year

        if len(df['time_in_years']) != len(df):
            st.error(f"🚨 Mismatch error, possibly due to invalid time column selection'{time_column}' – expected {len(df)} values but found {len(df['time_in_years'])}. "
                     "Please verify your dataset.")
            return None
        return df['time_in_years']
    
    except Exception as e:
        st.error(f"Error in converting time column: {e}")
        return None
    
def infer_chained_models(model, run_df, win_eir, win_phi, device, has_true_values):

    model.eval()

    streams = create_multistream_sequences(
        run_df,
        win_eir=win_eir,
        win_phi=win_phi
    )

    with torch.no_grad():
        batch = {k: tuple(v.to(device) if v is not None else None for v in streams[k]) for k in streams}

        pred_eir_log, pred_phi_log, pred_inc_log = model(batch)

        # Convert to natural scale
        p_eir = inverse_log_transform(
        pred_eir_log.squeeze(-1).cpu().numpy()
        )

        p_phi = inverse_log_transform(
            pred_phi_log.squeeze(-1).cpu().numpy()
        )

        p_inc = inverse_log_transform(
            pred_inc_log.squeeze(-1).cpu().numpy()
        )


    n_preds = len(p_eir)

    # ---- Time Alignment ----
    if "t" in run_df.columns:
        t = run_df["t"].values[:n_preds] / 365.25
    else:
        t = np.arange(n_preds)

    # ---- Ground Truth (only if available) ----
    if has_true_values:
        y_prev_true = inverse_log_transform(run_df["prev_true"].values[:n_preds])
        y_eir_true = inverse_log_transform(streams["eir"][2].squeeze(-1).cpu().numpy())
        y_phi_true = inverse_log_transform(streams["phi"][2].squeeze(-1).cpu().numpy())
        y_inc_true = inverse_log_transform(streams["inc"][1].squeeze(-1).cpu().numpy())

    else:
        y_prev_true = run_df["prev_true"].values[:n_preds]
        y_eir_true = None
        y_phi_true = None
        y_inc_true = None

    return {
        "t": t,
        "prev": (y_prev_true, None),
        "eir": (y_eir_true, p_eir),
        "phi": (y_phi_true, p_phi),
        "inc": (y_inc_true, p_inc)
    }

    
@st.cache_data(show_spinner="🔄 Running model predictions...")
def generate_predictions_per_run(data, selected_runs, run_column,
                                 _model, _device, has_true_values):

    run_results = {}

    for run in selected_runs:

        run_df = data[data[run_column] == run].reset_index(drop=True)
        if run_df.empty:
            continue

        res = infer_chained_models(
            _model,
            run_df,
            win_eir=20,
            win_phi=300,
            device=_device,
            has_true_values=has_true_values
        )

        run_results[run] = res

    return run_results


@st.cache_data
def compute_global_yaxis_limits(run_results):

    all_prev = []
    all_eir = []
    all_inc = []

    for result in run_results.values():

        # ---- PREVALENCE ----
        prev_true, _ = result["prev"]
        if prev_true is not None:
            all_prev.extend(prev_true)

        # ---- EIR ----
        eir_true, eir_pred = result["eir"]

        if eir_true is not None:
            all_eir.extend(eir_true)

        if eir_pred is not None:
            all_eir.extend(eir_pred)

        # ---- INCIDENCE ----
        inc_true, inc_pred = result["inc"]

        if inc_true is not None:
            all_inc.extend(inc_true)

        if inc_pred is not None:
            all_inc.extend(inc_pred)

    # ---- Compute Limits Safely ----
    def safe_limits(values):
        if len(values) == 0:
            return (0, 1)
        return (0, max(values) * 1.1)

    prev_limits = safe_limits(all_prev)
    eir_limits = safe_limits(all_eir)
    inc_limits = safe_limits(all_inc)

    return prev_limits, eir_limits, inc_limits


def plot_predictions(run_results, selected_runs,
                     log_eir, log_inc, log_all):
    
    metric_colors = {
    "eir": "#1f77b4",   # blue
    "phi":  "#d62728",   # red
    "inc":  "#2ca02c"
    }


    num_plots = len(selected_runs)

    fig, axes = plt.subplots(num_plots, 4,
                             figsize=(22, 5 * num_plots),
                             sharex=False)

    if num_plots == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ["Prevalence", "EIR", "Phi", "Incidence"]
    metrics = ["prev", "eir", "phi", "inc"]

    data_to_download = []

    for i, run in enumerate(selected_runs):

        result = run_results[run]
        t = result["t"]

        run_export = pd.DataFrame({
            "run": run,
            "time_years": t
        })

        for j, metric in enumerate(metrics):

            y_true, y_pred = result[metric]
            ax = axes[i, j]

            # ---- Plot Ground Truth ----
            if y_true is not None:
                ax.plot(t, y_true,
                        color="black",
                        linewidth=2,
                        label="True")

                run_export[f"Actual_{metric}"] = y_true

            # ---- Plot Prediction ----
            if y_pred is not None:
                ax.plot(t, y_pred,
                        linestyle="--",
                        linewidth=2.5,
                        color=metric_colors[metric],
                        label="Estimated")

                run_export[f"Estimated_{metric}"] = y_pred

            # ---- Log Scaling ----
            if log_all:
                ax.set_yscale("log")
            elif metric == "eir" and log_eir:
                ax.set_yscale("log")
            elif metric == "inc" and log_inc:
                ax.set_yscale("log")

            ax.set_title(f"{run} - {titles[j]}", fontsize=13)
            ax.set_xlabel("Time (Years)")
            ax.grid(alpha=0.3)

            if j == 0:
                ax.set_ylabel("Value")

            ax.legend()

        data_to_download.append(run_export)

    plt.tight_layout()
    st.pyplot(fig)

    # ---- DOWNLOAD BUTTON ----
    if data_to_download:
        combined_data = pd.concat(data_to_download, ignore_index=True)
        csv_data = combined_data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Estimates as CSV",
            data=csv_data,
            file_name="model_predictions.csv",
            mime="text/csv"
        )


def adjust_trailing_zero_prevalence(df, prevalence_column='prev_true', min_val=0.0003, max_val=0.001, seed=None):
    df = df.copy()
    zeros_mask = df[prevalence_column] == 0
    num_zeros = zeros_mask.sum()

    if num_zeros > 0:
        #st.warning(f"⚠️ Found {num_zeros} zero prevalence value(s); replacing with random values.")
        rng = np.random.default_rng(seed)
        random_values = rng.uniform(min_val, max_val, size=num_zeros)
        df.loc[zeros_mask, prevalence_column] = random_values

    return df

@st.cache_data
def load_uploaded_csv(file_content):
    return pd.read_csv(file_content)

def get_file_hash(file):
    return hashlib.md5(file.getvalue()).hexdigest()

# Streamlit UI
st.title("🔬 Testing New MultiHead Model")


# Choose data source
data_source = st.radio("📊 Select data source", ("Upload my own data", "Use preloaded test data"))

# Load the data accordingly
if data_source == "Use preloaded test data":
    remote_url = "https://raw.githubusercontent.com/Olatundemi/MalariaEmulator/main/test/ANC_Simulation_test_samples_20_runs_with_under5.csv"

    try:
        test_data = pd.read_csv(remote_url)
        st.success("✅ Preloaded test data loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load preloaded data: {e}")
        st.stop()
else:
    uploaded_file = st.file_uploader("📂 Upload prevalence data to estimate (CSV or Parquet)", type=["csv", "parquet"])

    if uploaded_file is not None:
        file_hash = get_file_hash(uploaded_file)

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

    else:
        st.warning("Please upload a CSV or Parquet file to continue.")
        st.stop()

columns = test_data.columns.tolist()

run_column = st.selectbox("🔄 Select geographical unit(s) e.g. Region, District, Province...", columns) if 'run' not in columns else 'run'

time_column = st.selectbox("🕒 Select time column", columns) if 't' not in columns else 't'

unique_runs = test_data[run_column].unique()
selected_runs = st.multiselect(f"📊 Select {run_column}(s) to estimate", unique_runs, default=unique_runs[:0])

# Filter the data based on selected runs
filtered_data = test_data[test_data[run_column].isin(selected_runs)]

if 'prev_true' not in columns:
    
    prevalence_column = st.selectbox("🩸 Select the column corresponding to prevalence", columns)#, key=f"prevalence_select_{key_suffix}")
    filtered_data = filtered_data.rename(columns={prevalence_column: 'prev_true'})

test_data = adjust_trailing_zero_prevalence(test_data, prevalence_column='prev_true', seed=42)

#model_path = #"src/trained_model/4_layers_model.pth"
window_size = 10
model_eir_path = "src/trained_model/shifting_sequences/multitask_model_improvedMSConv_HPE_EIR_phi_with_incidence.pth"#LSTM_EIR_4_layers_10000run_W10.pth"
model_eir, device = load_models(model_eir_path)

if filtered_data.empty:
    st.warning("No valid data found. Select necessary items to proceed")
    st.stop()

# Adjust zero prevalence in the filtered data only
filtered_data = adjust_trailing_zero_prevalence(filtered_data, prevalence_column='prev_true', seed=42)

# Preprocess the filtered data
df_scaled, has_true_values = preprocess_data(filtered_data)

if df_scaled is None:
    st.stop() # Stop execution if preprocessing fails
    
log_eir = st.checkbox("📈 View EIR on Log Scale", value=False)
log_inc = st.checkbox("📉 View Incidence on Log Scale", value=False)
log_all = st.checkbox("🔍 View All Plots on Log Scale", value=False)


if selected_runs:
    if st.button("🚀 Run Predictions"):
        start_time = time.time()
        run_results = generate_predictions_per_run(
            df_scaled, selected_runs, run_column,
            model_eir, device, has_true_values
        )
        st.info(f"✅ Predictions computed in {time.time() - start_time:.2f} seconds")

        if not run_results:
            st.warning("No valid predictions could be generated.")
            st.stop()

        start_time = time.time()
        prev_limits, eir_limits, inc_limits = compute_global_yaxis_limits(run_results)
        st.info(f"✅ Axis limits calculated in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        plot_predictions(
            run_results, selected_runs,
            log_eir, log_inc, log_all
        )
        st.info(f"✅ Plots generated in {time.time() - start_time:.2f} seconds")