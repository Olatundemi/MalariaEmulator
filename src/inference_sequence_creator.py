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

st.cache_data
def create_sequences_assymetric(data, window_size):
    xs, ys = [], []
    has_targets = all(col in data.columns for col in ['EIR_true'])  #, 'incall'# Check if target columns exist

    sequence_col = ['prev_true']
    half_window_size = int(np.ceil(window_size / 2)) 

    for i in range(len(data)):
        if i + half_window_size >= len(data):
            break  # Not enough future steps

        # Prepare padding near the beginning
        if i < window_size:
            pad_size = window_size - i
            first_value = data.iloc[0][sequence_col].values.reshape(1, -1)
            pad_values = np.tile(first_value, (pad_size, 1))

            # actual values up to current + half_window_size
            actual_values = data.iloc[0:i + half_window_size + 1][sequence_col].values
            x_values = np.concatenate((pad_values, actual_values), axis=0)
        else:
            # Extract window before, current, and after
            start_idx = i - window_size
            end_idx = i + half_window_size + 1  # exclusive
            x_values = data.iloc[start_idx:end_idx][sequence_col].values

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

def create_sequences_in_parallel(features, targets, window_size):
    xs, ys = [], []
    has_targets = all(col in targets.columns for col in ['EIR_true'])
    
    for i in range(len(features) - window_size):
        if i < window_size:
            pad_size = window_size - i
            first_values = features.iloc[0].values
            replicated_values = np.tile(first_values, (pad_size, 1))
            x_values = np.concatenate((replicated_values, features.iloc[0:i + window_size + 1].values), axis=0)
        else:
            x_values = features.iloc[i - window_size:i + window_size + 1].values
        
        xs.append(x_values.flatten())

        if has_targets:
            y = targets.iloc[i][['EIR_true']].values
            ys.append(y)
            
    xs = np.array(xs, dtype=np.float32)
    
    if has_targets:
        ys = np.array(ys, dtype=np.float32)
        return torch.tensor(xs), torch.tensor(ys)
    else:
        return torch.tensor(xs), None  # Return None for ys if targets are missing

st.cache_data
def create_shifting_sequences(data, window_size):
    xs = []   # For combined sequential input [prev_true, EIR_true]
    ys = []   # For target output (incall), if available

    full_seq_len = 2 * window_size + 1
    has_target = 'incall' in data.columns  # check if incidence is in the data

    for i in range(len(data)):
        start_idx = i - window_size
        end_idx = i + window_size + 1

        if end_idx <= len(data):
            if start_idx < 0:
                pad_size = abs(start_idx)
                first_values = data.iloc[0][['prev_true', 'EIR_true']].values
                replicated_values = np.tile(first_values, (pad_size, 1))
                actual_values = data.iloc[0:end_idx][['prev_true', 'EIR_true']].values
                x_values = np.concatenate((replicated_values, actual_values), axis=0)
            else:
                x_values = data.iloc[start_idx:end_idx][['prev_true', 'EIR_true']].values
        else:
            # causal padding for end of series
            causal_start_idx = i - (2 * window_size)
            if causal_start_idx < 0:
                pad_size = abs(causal_start_idx)
                first_values = data.iloc[0][['prev_true', 'EIR_true']].values
                replicated_values = np.tile(first_values, (pad_size, 1))
                actual_values = data.iloc[0:i + 1][['prev_true', 'EIR_true']].values
                x_values = np.concatenate((replicated_values, actual_values), axis=0)
            else:
                x_values = data.iloc[causal_start_idx:i + 1][['prev_true', 'EIR_true']].values

        # Ensure correct sequence length
        if x_values.shape[0] != full_seq_len:
            raise ValueError(f"Sequence length mismatch at index {i}: got {x_values.shape[0]}, expected {full_seq_len}")

        xs.append(x_values)  # shape: (seq_len, 2)

        # Only append target if it's available
        if has_target:
            y = data.iloc[i]['incall']
            ys.append([y])

    xs = torch.tensor(np.array(xs), dtype=torch.float32)  # (N, seq_len, 2)

    if has_target:
        ys = torch.tensor(np.array(ys), dtype=torch.float32)  # (N, 1)
        return xs, ys
    else:
        return xs, None


def create_sequences_with_separate_scalar(data, window_size):
    sequences = []
    scalars = []
    targets = []

    # Check if target columns exist in the dataframe
    has_targets = all(col in data.columns for col in ['EIR_true', 'incall'])

    for i in range(len(data) - window_size):
        if i < window_size:
            pad_size = window_size - i
            pad_value = data.iloc[0][['prev_true']].values
            pad_seq = np.tile(pad_value, (pad_size, 1))
            sequence = np.concatenate(
                (pad_seq, data.iloc[0:i + window_size + 1][['prev_true']].values),
                axis=0
            )
        else:
            sequence = data.iloc[i - window_size:i + window_size + 1][['prev_true']].values

        sequence = sequence.reshape(-1, 1)

        # Fixed scalar averaging over the last 24 points (or less if near start)
        if i >= 1:
            window_start = max(i - 24, 0)
            if window_start == i:  # empty slice fallback
                avg_prev = data.iloc[0]['prev_true']
            else:
                avg_prev = np.mean(data['prev_true'][window_start:i])
        else:
            avg_prev = data.iloc[0]['prev_true']

        sequences.append(sequence.astype(np.float32))
        scalars.append(np.array([avg_prev], dtype=np.float32))

        if has_targets:
            targets.append(data.iloc[i][['EIR_true', 'incall']].values.astype(np.float32))

    x_seq = torch.tensor(np.array(sequences))  # (N, seq_len, 1)
    x_scalar = torch.tensor(np.array(scalars))  # (N, 1)

    if has_targets:
        y = torch.tensor(np.array(targets))  # (N, 2)
        return x_seq, x_scalar, y
    else:
        return x_seq, x_scalar, None

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