import streamlit as st 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.inference_sequence_creator import create_sequences, create_shifting_sequences
from src.inference_model_exp import LSTM_EIR, LSTM_Incidence
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

# Apply a modern Seaborn theme
sns.set_style("darkgrid")  
plt.style.use("ggplot")

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
def load_models(model_eir_path, model_inc_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_eir = LSTM_EIR(input_size=1, architecture=[256, 128, 64, 32])
    model_eir.load_state_dict(torch.load(model_eir_path, map_location=device))
    model_eir.to(device)
    model_eir.eval()

    model_inc = LSTM_Incidence(input_size=2, architecture=[256, 128, 64, 32])
    model_inc.load_state_dict(torch.load(model_inc_path, map_location=device))
    model_inc.to(device)
    model_inc.eval()

    return model_eir, model_inc, device

# Function to preprocess data
def preprocess_data(df):

    # Sanity Check for selected prevalence column
    if not pd.api.types.is_numeric_dtype(df['prev_true']):
        st.error("🚨 The selected prevalence column is invalid. It contains non-numeric values.")
        return None, False  # Return None to indicate failure
    
    has_true_values = {'EIR_true', 'incall'}.issubset(df.columns)

    if has_true_values:
        df_scaled = df[['prev_true', 'EIR_true', 'incall']].apply(log_transform)
    else:
        df_scaled = df[['prev_true']].apply(log_transform)
    
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
        })# This is not absolutely right (at the moment) interms of actual versus predicted EIR

        X_inc_input, _ = create_shifting_sequences(inc_input_df_scaled, window_size)
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
        #scaled_data = result['scaled_data']
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
        scaled_run_data = result["scaled_data"]
        run_data = result["original_data"]
        y_eir_unscaled = inverse_log_transform(result["y_eir_true"]) if result["y_eir_true"] is not None else None

        has_inc_true = 'incall' in scaled_run_data.columns
        y_inc_unscaled = None
        if has_inc_true:
            X_inc_all, y_inc = create_shifting_sequences(
                scaled_run_data[['prev_true', 'EIR_true', 'incall']], window_size=10
            )
            y_inc_unscaled = inverse_log_transform(y_inc.numpy())

        time_values_plot = time_values[:len(eir_preds_unscaled)-window_size] # I took off the last rough predictions
        min_len = len(eir_preds_unscaled) - window_size#len(time_va

        plot_data = []

        # Prevalence (true only, no prediction)
        plot_data.append({
            "title": "Prevalence",
            "pred": None,  # No estimated prevalence
            "true": run_data['prev_true'].values[:min_len]
        })

        # EIR
        plot_data.append({
            "title": "EIR",
            "pred": eir_preds_unscaled[:min_len, 0], #
            "true": y_eir_unscaled[:min_len, 0] if y_eir_unscaled is not None else None
        })

        # Incidence
        plot_data.append({
            "title": "Incidence",
            "pred": inc_preds_unscaled[:min_len, 0],
            "true": y_inc_unscaled[:min_len, 0] if y_inc_unscaled is not None else None
        })

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

            x_vals = time_values_plot#[:len(true)]

            # Always plot true values (black)
            if true is not None:
                ax.plot(x_vals, true, color="black", linestyle="-",
                        label=f"True {title}", linewidth=2)

            # Only plot prediction for EIR and Incidence
            if title != "Prevalence" and pred is not None:
                ax.plot(x_vals, pred, linestyle="--", color=color,
                        label=f"Estimated {title}", linewidth=2.5)

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

            # Add Actual EIR if available
            if plot_data[1]["true"] is not None:
                df_export["Actual EIR"] = plot_data[1]["true"]

            # Add Actual Incidence if available
            if plot_data[2]["true"] is not None:
                df_export["Actual Incidence"] = plot_data[2]["true"]

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
        combined_data = pd.concat(data_to_download, keys=selected_runs, names=[run_column, "Index"])
        csv_data = combined_data.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Estimates as CSV", data=csv_data, file_name="predictions.csv", mime="text/csv")


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
st.title("🔬 Malaria Incidence and EIR Estimator with AI")


# Choose data source
data_source = st.radio("📊 Select data source", ("Upload my own data", "Use preloaded test data"))

# Load the data accordingly
if data_source == "Use preloaded test data":
    remote_url = "https://raw.githubusercontent.com/Olatundemi/MalariaEmulator/main/test/ANC_Simulation_1000_test_runs.csv"

    try:
        test_data = pd.read_csv(remote_url)
        st.success("✅ Preloaded test data loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load preloaded data: {e}")
        st.stop()
else:
    uploaded_file = st.file_uploader("📂 Upload prevalence data to estimate (CSV)", type=["csv"])

    if uploaded_file is not None:
        file_hash = get_file_hash(uploaded_file)
        test_data = load_uploaded_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file to continue.")
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

#test_data = adjust_trailing_zero_prevalence(test_data, prevalence_column='prev_true', seed=42)

#model_path = #"src/trained_model/4_layers_model.pth"
window_size = 10
model_eir_path = "src/trained_model/shifting_sequences/LSTM_EIR_4_layers_10000run_W10.pth"
model_inc_path = "src/trained_model/shifting_sequences/LSTM_Incidence_4_layer_10000run_W10_shifting_sequence.pth"
model_eir, model_inc, device = load_models(model_eir_path, model_inc_path)

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
            filtered_data, selected_runs, run_column, window_size,
            model_eir, model_inc, device, has_true_values
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
            run_results, run_column, time_column, selected_runs,
            log_eir, log_inc, log_all, prev_limits, eir_limits, inc_limits
        )
        st.info(f"✅ Plots generated in {time.time() - start_time:.2f} seconds")



# def main():
#     import streamlit.web.bootstrap
#     from pathlib import Path
#     script_path = Path(__file__).resolve()
#     streamlit.web.bootstrap.run(script_path, '', [])

# if __name__ == "__main__":
#     main()