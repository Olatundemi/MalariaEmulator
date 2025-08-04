import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


#from src.model_exp import LSTMModel

from src.interface_util import load_model, preprocess_data, convert_time_column, compute_global_yaxis_limits, plot_predictions, adjust_trailing_zero_prevalence
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
    uploaded_file = st.file_uploader("📂 Upload prevalence data to estimate (CSV or Excel)", type=["csv", "xls", "xlsx"])
    
    if uploaded_file:
        file_name = uploaded_file.name.lower()

        try:
            if file_name.endswith(".csv"):
                test_data = pd.read_csv(uploaded_file)
            elif file_name.endswith((".xls", ".xlsx")):
                test_data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                st.stop()
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()

    else:
        st.warning("Please upload a CSV or Excel file to continue.")
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
    test_data = test_data.rename(columns={prevalence_column: 'prev_true'})
test_data = adjust_trailing_zero_prevalence(test_data, prevalence_column='prev_true', seed=42)


model_path = "src/trained_model/4_layers_model.pth"
window_size = 10
model, device = load_model(model_path)

df_scaled, has_true_values = preprocess_data(test_data)

if df_scaled is None:
    st.stop()  # Stop further execution if preprocessing fails
    
log_eir = st.checkbox("📈 View EIR on Log Scale", value=False)
log_inc = st.checkbox("📉 View Incidence on Log Scale", value=False)
log_all = st.checkbox("🔍 View All Plots on Log Scale", value=False)


if selected_runs:
    prev_limits, eir_limits = compute_global_yaxis_limits(
        test_data, selected_runs, run_column, window_size, model, device, has_true_values
    )
    plot_predictions(
        test_data, run_column, time_column, selected_runs, model, device, window_size,
        log_eir, log_inc, log_all, has_true_values, prev_limits, eir_limits
    )


def main():
    import streamlit.web.bootstrap
    from pathlib import Path
    script_path = Path(__file__).resolve()
    streamlit.web.bootstrap.run(script_path, '', [])

if __name__ == "__main__":
    main()