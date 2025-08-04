#EIR and Incidence Prediction Analysis

This project aims to provide tool for estimating malaria incidence and transmission intensity from timeseries prevalence data through emulation of mechanistic Malaria transmission model. It includes helper functions, data processing, and machine learning models pipeline to create predictive systems using simulation data from malsimgem.

## Repository Overview

- Primary Jupyter Notebook: `ANC_Emulator_PyTorch.ipynb` and other variants for model training
- Helper scripts for sequence creation, model development and Inferencing:
  - `model_exp.py`
  - `sequence_creator.py` etc
- Dashboard for estimating incidence and transmission intensities/EIR 
  - `emulator.py` Deployed to [estimate](https://estimatemalariaparameters.streamlit.app/)
  - `emulator_one_model.py`
  - `emulator_two_models.py` #Uses one model to predict each variable

---
## Folder Structure

```
project_root/
│── src/                  # Source code/functions directory
│   ├── preprocessing.py  # Handles data loading and preprocessing
│   ├── sequence_creator.py  # Creates birectional sequences of tensors
│   ├── model_exp.py      # Defines PyTorch model and training
│   ├── test.py           # Model testing
│   ├── utils.py          # Utility functions (metrics, visualization, etc.)
|   ├── inference.py      # Specifically used for inferencing within training notebook
|   ├── inference_util.py # Specifically used for dashboard inferencing

│── test/                # Unit tests
│   ├── test_data         # Contains a thousand test runs across different transmission intensities
│── notebooks/            # Jupyter notebooks (with recent experiments)
│── requirements.txt      # Dependencies
│── README.md             # Project overview and instructions
│── emulator.py           # Python script containing deployed streamlit dashboard
│── emulator_one_model.py # Emulator variant predicting with one model   
│── emulator_one_model_with_baseline_prev.py # Emulator variant predicting with one model 
│── emulator_two_model.py # Emulator variant predicting with two models
│── .gitignore            # Ignored files (e.g., __pycache__, .venv)
│── setup.py              # Setup script (to be updated)
│── pyproject.toml        # Modern package management (to be updated)
├── ANC_Emulator_PyTorch.ipynb  # Main analysis notebook
├── data/                  # simulated data from mamasante/malsimgem
├── plots/                 # Saved visuals


##Installtion and Set-up

- Install Python of Desired Version - This project uses Python 3.12.6
- Clone repo and navigate to project root directory using command prompt (cd /path/to/MalariaEmulator)
- Install Dependecies with "pip install -r requirements.txt" (create python virtual environment in the project directory before installing dependencies if desired. That is the standard practice)
- Run the ANC_Emulator_PyTorch.ipynb notebook or other variants for the training process
- Model Weights and plots are saved in src and plots folder respectively. This can be modified in the respective function creating the plot
- To run the dashboard, while still in the project root (in command prompt), run "streamlit run emulator.py" or the other variants

## Key Training Pipeline

### Main Components
1. **Annual Averages Calculation**
   - Analyzes simulation data to compute annual averages for key metrics (e.g., true prevalence, transmission intensity) for years 2, 5, and 8.

2. **Monthly Data Filtering**
   - Extracts monthly data for years 10 to 20 from simulation results.

3. **Sequence Creation**
   - Processes time-series data to generate input-output pairs suitable for model training, handling padding for initial time steps.

4. **Machine Learning Model**
   - Implements an LSTM-based architecture to predict malaria metrics (`EIR_true`, `incall`) using simulation data.