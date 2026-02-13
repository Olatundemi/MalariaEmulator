import pandas as pd
import numpy as np
import torch
import streamlit as st

st.cache_data
def create_sequences_2outputs(data, window_size):
    xs, ys = [], []
    has_targets = all(col in data.columns for col in ['EIR_true', 'incall'])  # Check if target columns exist

    for i in range(len(data) - window_size):
        if i < window_size:
            # Pad beginning of sequence
            pad_size = window_size - i
            first_values = data.iloc[0][['prev_true']].values
            replicated_values = np.tile(first_values, (pad_size, 1))
            x_values = np.concatenate((replicated_values, data.iloc[0:i + window_size + 1][['prev_true']].values), axis=0)
        else:
            x_values = data.iloc[i - window_size:i + window_size + 1][['prev_true']].values

        xs.append(x_values.flatten())

        if has_targets:
            y = data.iloc[i][['EIR_true', 'incall']].values
            ys.append(y)

    xs = np.array(xs, dtype=np.float32)

    if has_targets:
        ys = np.array(ys, dtype=np.float32)
        return torch.tensor(xs), torch.tensor(ys)
    else:
        return torch.tensor(xs), None  # Return None for ys if targets are missing
    
st.cache_data
def create_sequences(data, window_size):
    xs, ys = [], []
    has_targets = all(col in data.columns for col in ['EIR_true'])  #, 'incall'# Check if target columns exist

    for i in range(len(data) - window_size):
        if i < window_size:
            # Pad beginning of sequence
            pad_size = window_size - i
            first_values = data.iloc[0][['prev_true']].values
            replicated_values = np.tile(first_values, (pad_size, 1))
            x_values = np.concatenate((replicated_values, data.iloc[0:i + window_size + 1][['prev_true']].values), axis=0)
        else:
            x_values = data.iloc[i - window_size:i + window_size + 1][['prev_true']].values

        xs.append(x_values.flatten())

        if has_targets:
            y = data.iloc[i][['EIR_true']].values#, 'incall'
            ys.append(y)

    xs = np.array(xs, dtype=np.float32)

    if has_targets:
        ys = np.array(ys, dtype=np.float32)
        return torch.tensor(xs), torch.tensor(ys)
    else:
        return torch.tensor(xs), None  # Return None for ys if targets are missing


def create_causal_sequences(data, window_size, features=['prev_true', 'EIR_true']):
    xs, ys = [], []
    feature_data = data[features].to_numpy()

    has_target = 'incall' in data.columns  # check if incidence is in the data

    for i in range(len(data)):
        if i < window_size:
            # Pad with the first row
            pad_size = window_size - i
            first_row = feature_data[0].reshape(1, -1)
            padding = np.tile(first_row, (pad_size, 1))
            actual = feature_data[0:i+1]  # from 0 to i (inclusive)
            x_values = np.concatenate((padding, actual), axis=0)
        else:
            x_values = feature_data[i - window_size:i + 1]

        xs.append(x_values)

        # Only append target if it's available
        if has_target:
            y = data.iloc[i]['incall']
            ys.append([y])

    xs = torch.tensor(np.array(xs), dtype=torch.float32)  

    if has_target:
        ys = torch.tensor(np.array(ys), dtype=torch.float32)
        return xs, ys
    else:
        return xs, None

st.cache_data
def create_sequences_assymetric(data, window_size):
    xs, ys = [], []
    has_targets = all(col in data.columns for col in ['EIR_true'])  #, 'incall'# Check if target columns exist
    half_window_size = int(np.ceil(window_size / 2))

    for i in range(len(data)-half_window_size):
        # if i + half_window_size >= len(data):
        #     break  # Not enough future steps
        if i < window_size:
            # Pad beginning of sequence
            pad_size = window_size - i
            first_values = data.iloc[0][['prev_true']].values
            replicated_values = np.tile(first_values, (pad_size, 1))
            x_values = np.concatenate((replicated_values, data.iloc[0:i + half_window_size + 1][['prev_true']].values), axis=0)
        else:
            x_values = data.iloc[i - window_size:i + half_window_size + 1][['prev_true']].values

        xs.append(x_values.flatten())

        if has_targets:
            y = data.iloc[i][['EIR_true']].values#, 'incall'
            ys.append(y)

    xs = np.array(xs, dtype=np.float32)

    if has_targets:
        ys = np.array(ys, dtype=np.float32)
        return torch.tensor(xs), torch.tensor(ys)
    else:
        return torch.tensor(xs), None  # Return None for ys if targets are missing

def build_windows_1d(x, past, future):
    """
    x: (T,) numpy array
    Returns:
      windows: (T, past+future+1)
      mask:    (T, past+future+1)
    """

    T = len(x)
    W = past + future + 1

    # Pad signal
    x_pad = np.pad(x, (past, future), mode="constant")
    mask_pad = np.pad(np.ones(T, dtype=np.float32), (past, future), mode="constant")

    # Build windows by slicing (vectorized)
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, W)
    mask    = np.lib.stride_tricks.sliding_window_view(mask_pad, W)

    return windows.astype(np.float32), mask.astype(np.float32)
def create_multistream_sequences(
    run_df,
    win_eir=20,
    win_phi=300,
    prev_col="prev_true",
    eir_col="EIR_true"
):
    """
    Creates windowed sequences for one run dataframe.

    Returns dict:
        {
            "eir": (X_eir, M_eir, y_eir or None),
            "phi": (X_phi, M_phi, y_phi or None),
            "inc": (M_phi, y_inc or None)
        }
    """

    # ---- Detect if targets exist ----
    has_targets = all(col in run_df.columns for col in [eir_col, "phi", "incall"])

    # ---- Extract input ONCE ----
    prev = run_df[prev_col].to_numpy(dtype=np.float32)

    # ---- Targets (only if available) ----
    if has_targets:
        eir = run_df[eir_col].to_numpy(dtype=np.float32)
        phi = run_df["phi"].to_numpy(dtype=np.float32)
        inc = run_df["incall"].to_numpy(dtype=np.float32)
    else:
        eir = phi = inc = None

    # ---- Define window structure ----
    past_eir = int(0.75 * win_eir)
    future_eir = win_eir - past_eir

    # ---- Build windows ----
    X_eir, M_eir = build_windows_1d(prev, past_eir, future_eir)
    X_phi, M_phi = build_windows_1d(prev, win_phi, 0)

    # ---- Expand feature dimension ----
    X_eir = X_eir[..., None]   # (T, win_eir+1, 1)
    X_phi = X_phi[..., None]   # (T, win_phi+1, 1)

    # ---- Convert inputs to torch ----
    X_eir = torch.from_numpy(X_eir)
    M_eir = torch.from_numpy(M_eir)
    X_phi = torch.from_numpy(X_phi)
    M_phi = torch.from_numpy(M_phi)

    # ---- Targets ----
    if has_targets:
        y_eir = torch.from_numpy(eir[:, None])
        y_phi = torch.from_numpy(phi[:, None])
        y_inc = torch.from_numpy(inc[:, None])
    else:
        y_eir = y_phi = y_inc = None

    return {
        "eir": (X_eir, M_eir, y_eir),
        "phi": (X_phi, M_phi, y_phi),
        "inc": (M_phi, y_inc)
    }

