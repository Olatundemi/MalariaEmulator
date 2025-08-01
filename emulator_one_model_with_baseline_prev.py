import streamlit as st 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.inference_sequence_creator import create_sequences_with_separate_scalar
from src.inference_model_exp import LSTMModel_baseline
import time

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

# Custom CSS
st.markdown("""
    <style>
        h1 { color: #FF4B4B; text-align: center; }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-size: 16px;
        }
        .stButton>button:hover { background-color: #45a049; }
        [data-testid="stSidebar"] { background-color: #2E3B4E; color: white; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel_baseline(input_size=1, architecture=[200, 100, 50])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


@st.cache_data
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


def convert_time_column(df, time_column):
    try:
        if pd.api.types.is_numeric_dtype(df[time_column]):
            return df[time_column].astype(float) / 365.25
        else:
            df[time_column] = pd.to_datetime(df[time_column], errors='coerce', format='%b-%y')
            if df[time_column].isna().all():
                st.error("Could not parse the time column. Ensure it's a proper date format (e.g., Jan-16).")
                return None

            start_year = df[time_column].dt.year.min()
            df['time_in_years'] = df[time_column].dt.year + (df[time_column].dt.month - 1) / 12 - start_year
        return df['time_in_years']

    except Exception as e:
        st.error(f"Error in converting time column: {e}")
        return None

@st.cache_data(show_spinner="🔄 Running model predictions...")
def get_predictions(test_data, selected_runs, run_column, window_size, _model, _device, has_true_values):

    all_preds = {}

    for run in selected_runs:
        run_data = test_data[test_data[run_column] == run]
        if run_data.empty:
            continue

        scaled_run_data, _ = preprocess_data(run_data)
        X_test_scaled, X_test_scalar, y_test_scaled = create_sequences_with_separate_scalar(scaled_run_data, window_size) #A sequential input and a scalar

        if len(X_test_scaled) == 0:
            continue

        X_test_scaled = X_test_scaled.to(_device)
        X_test_scalar = X_test_scalar.to(device)

        with torch.no_grad():
            test_predictions_scaled = _model(X_test_scaled, X_test_scalar).cpu().numpy()

        test_predictions_unscaled = inverse_log_transform(test_predictions_scaled)
        X_test_unscaled = inverse_log_transform(X_test_scaled.cpu().numpy()) # Consider using the original values rather than unscaling again
        y_test_unscaled = inverse_log_transform(y_test_scaled.numpy()) if has_true_values else None ## Think this in better way

        all_preds[run] = {
            "X_test_unscaled": X_test_unscaled,
            "predictions": test_predictions_unscaled,
            "true_values": y_test_unscaled,
        }

    return all_preds

@st.cache_data
def compute_global_yaxis_limits(predictions_dict, test_data, selected_runs, run_column):
    all_prev, all_eir, all_inc = [], [], []

    for run in selected_runs:
        run_data = test_data[test_data[run_column] == run]
        if run_data.empty or run not in predictions_dict:
            continue

        run_preds = predictions_dict[run]
        pred = run_preds["predictions"]

        all_prev.extend(run_data['prev_true'].values[:len(pred)])
        all_eir.extend(pred[:, 0])
        if pred.shape[1] > 1:
            all_inc.extend(pred[:, 1])

    prev_min, prev_max = 0, max(all_prev) * 1.1 if all_prev else (0, 1)
    eir_min, eir_max = 0, max(all_eir) * 1.2 if all_eir else (0, 550)
    inc_min, inc_max = 0, max(all_inc) * 1.1 if all_inc else (0, 1)

    return (prev_min, prev_max), (eir_min, eir_max), (inc_min, inc_max)

#@st.cache_resource
def plot_predictions(test_data, run_column, time_column, selected_runs,
                     predictions_dict, log_eir, log_inc, log_all, has_true_values,
                     prev_limits, eir_limits, inc_limits):

    is_string_time = not pd.api.types.is_numeric_dtype(test_data[time_column])

    if is_string_time:
        time_labels = test_data[time_column].unique()
        time_values = np.arange(len(time_labels))
    else:
        time_values = test_data[time_column].astype(float) / 365.25
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
        run_data = test_data[test_data[run_column] == run]

        if run_data.empty or run not in predictions_dict:
            st.warning(f"⚠️ Run data not found or predictions missing for: {run}")
            continue

        run_preds = predictions_dict[run]
        X_test_unscaled = run_preds["X_test_unscaled"]
        test_predictions_unscaled = run_preds["predictions"]
        y_test_unscaled = run_preds["true_values"]

        time_values_plot = time_values[:len(test_predictions_unscaled)]

        # 🧠 Fixed time_vals based on whether it's string or numeric
        if is_string_time:
            time_vals = np.array(time_labels)[time_values_plot.astype(int)]
        else:
            time_vals = np.ravel(time_values_plot)

        run_vals = np.array([run] * len(time_vals))

        titles = ["Prevalence", "EIR", "Incidence"]
        predictions = [
            X_test_unscaled[:, -1],
            test_predictions_unscaled[:, 0],
            test_predictions_unscaled[:, 1] if test_predictions_unscaled.shape[1] > 1 else None
        ]
        true_values = [
            None,
            y_test_unscaled[:, 0] if has_true_values and y_test_unscaled is not None else None,
            y_test_unscaled[:, 1] if has_true_values and y_test_unscaled is not None and y_test_unscaled.shape[1] > 1 else None
        ]
        log_scales = [log_all, log_eir or log_all, log_inc or log_all]

        for ax, title, color, pred, true_val, log_scale in zip(axes[i], titles, colors, predictions, true_values, log_scales):
            if pred is None:
                continue

            if true_val is not None:
                ax.plot(time_values_plot, true_val, color="black", linestyle="-", label=f"True {title}", linewidth=2)

            if title == "Prevalence":
                prev_true_adjusted = run_data['prev_true'].values[:len(pred)]
                ax.plot(time_values[:len(prev_true_adjusted)], prev_true_adjusted, linestyle="--", color=color, label="Prevalence", linewidth=2.5)
            else:
                ax.plot(time_values_plot, pred, linestyle="--", color=color, label=f"Estimated {title}", linewidth=2.5)

            if log_scale:
                ax.set_yscale('log')

            if title == "Prevalence":
                ax.set_ylim(prev_min, prev_max)
            elif title == "EIR":
                ax.set_ylim(eir_min, eir_max)
            elif title == "Incidence":
                ax.set_ylim(inc_min, inc_max)

            ax.set_title(f"{run} - {title}", fontsize=14, color="#FF4B4B")
            ax.set_ylabel(title, fontsize=12)
            ax.legend()

        # Create downloadable dataframe
        prevalence_vals = np.ravel(predictions[0])
        eir_vals = np.ravel(predictions[1])
        inc_vals = np.ravel(predictions[2]) if predictions[2] is not None else np.full_like(eir_vals, np.nan)

        df_to_append = pd.DataFrame({
            "Run": run_vals,
            "Time": time_vals,
            "Prevalence": prevalence_vals,
            "Estimated EIR": eir_vals,
            "Estimated Incidence": inc_vals,
        })

        if has_true_values and true_values[1] is not None:
            df_to_append["Actual EIR"] = np.ravel(true_values[1])
        if has_true_values and true_values[2] is not None:
            df_to_append["Actual Incidence"] = np.ravel(true_values[2])

        data_to_download.append(df_to_append)

    # Set x-axis tick labels on final row of plots
    for ax in axes[-1]:
        if is_string_time:
            tick_indices = np.arange(0, len(time_values_plot), step=6, dtype=int)
            ax.set_xticks(time_values_plot[tick_indices])
            ax.set_xticklabels(np.array(time_labels)[tick_indices], rotation=45, fontsize=10)
        else:
            ax.set_xlabel("Years", fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)

    # 📥 Combine and download
    if data_to_download:
        combined_data = pd.concat(data_to_download, ignore_index=True)
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


# Streamlit UI
st.title("🔬 Malaria Incidence and EIR Estimator with AI - With Baseline Prev")

data_source = st.radio("📊 Select data source", ("Upload my own data", "Use preloaded test data"))

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
    if uploaded_file:
        test_data = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

columns = test_data.columns.tolist()
run_column = st.selectbox("🔄 Select geographical unit(s)...", columns) if 'run' not in columns else 'run'
time_column = st.selectbox("🕒 Select time column", columns) if 't' not in columns else 't'

unique_runs = test_data[run_column].unique()
selected_runs = st.multiselect(f"📊 Select {run_column}(s)", unique_runs, default=unique_runs[:0])
filtered_data = test_data[test_data[run_column].isin(selected_runs)]

if 'prev_true' not in columns:
    prevalence_column = st.selectbox("🩸 Select prevalence column", columns)
    test_data = test_data.rename(columns={prevalence_column: 'prev_true'})

test_data = adjust_trailing_zero_prevalence(test_data, prevalence_column='prev_true', seed=42)

model_path = "C:/Users/oibrahim/Documents/MalariaEmulator/src/trained_model/3_layers_model_15000runs_W10_trial_for_runs_baseline.pth"
window_size = 10
model, device = load_model(model_path)

df_scaled, has_true_values = preprocess_data(test_data)
if df_scaled is None:
    st.stop()

log_eir = st.checkbox("📈 View EIR on Log Scale", value=False)
log_inc = st.checkbox("📉 View Incidence on Log Scale", value=False)
log_all = st.checkbox("🔍 View All Plots on Log Scale", value=False)

if selected_runs:
    # 1. Run predictions once
    start_time = time.time()
    predictions_dict = get_predictions(
        test_data, selected_runs, run_column, window_size,
        _model=model, _device=device, has_true_values=has_true_values
    )
    st.info(f"✅ Predictions computed in {time.time() - start_time:.2f} seconds")

    # 2. Compute y-axis limits using the already available predictions
    start_time = time.time()
    prev_limits, eir_limits, inc_limits = compute_global_yaxis_limits(
        predictions_dict, test_data, selected_runs, run_column
    )
    st.info(f"✅ Axis limits calculated in {time.time() - start_time:.2f} seconds")


    # 3. Plot using predictions and limits
    start_time = time.time()
    plot_predictions(
        test_data, run_column, time_column, selected_runs, predictions_dict,
        log_eir, log_inc, log_all, has_true_values, prev_limits, eir_limits, inc_limits
    )
    st.info(f"✅ Plots generated in {time.time() - start_time:.2f} seconds")


def main():
    import streamlit.web.bootstrap
    from pathlib import Path
    script_path = Path(__file__).resolve()
    streamlit.web.bootstrap.run(script_path, '', [])

if __name__ == "__main__":
    main()