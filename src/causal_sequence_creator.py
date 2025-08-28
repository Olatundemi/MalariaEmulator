import pandas as pd
import numpy as np
import torch

def create_sequences_1output(data, window_size):
    xs, ys = [], []

    for i in range(len(data)):
        start_idx = i - window_size
        end_idx = i + 1  # Inclusive of i

        if start_idx < 0:
            # pad with the first value
            pad_size = abs(start_idx)
            first_value = data.iloc[0][['prev_true']].values.reshape(1, -1)
            replicated_values = np.tile(first_value, (pad_size, 1))
            actual_values = data.iloc[0:end_idx][['prev_true']].values
            x_values = np.concatenate((replicated_values, actual_values), axis=0)
        else:
            x_values = data.iloc[start_idx:end_idx][['prev_true']].values

        # Target variable: EIR_true at time i
        y = data.iloc[i][['EIR_true']].values

        xs.append(x_values.flatten())
        ys.append(y)

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    return torch.tensor(xs), torch.tensor(ys)
    

def create_causal_sequences(data, window_size, features=['prev_true', 'EIR_true'], target='incall'):
    xs, ys = [], []
    feature_data = data[features].to_numpy()

    for i in range(len(data)):
        if i < window_size:
            # Pad with the first row
            pad_size = window_size - i - 1
            first_row = feature_data[0].reshape(1, -1)
            padding = np.tile(first_row, (pad_size, 1))
            actual = feature_data[0:i+1]  # from 0 to i (inclusive)
            x_values = np.concatenate((padding, actual), axis=0)
        else:
            x_values = feature_data[i - window_size + 1:i + 1]

        y = data.iloc[i][[target]].values
        xs.append(x_values)
        ys.append(y)

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    return torch.tensor(xs), torch.tensor(ys)

