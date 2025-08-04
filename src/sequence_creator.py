import pandas as pd
import numpy as np
import torch

"""This script contains variants of sequences creation approach. Summary of each has been maded in few lines near the respective functions. The Right hand window of non-bidirection indices have been commented out"""  

def create_sequences(data, window_size): # This pairs two outputs to one input
    xs, ys = [], []

    for i in range(len(data)):
        end_idx = i + window_size + 1

        # Full bidirectional sequence
        if end_idx <= len(data):
            if i < window_size:
                # Pad the start with the first value
                pad_size = window_size - i
                first_value = data.iloc[0][['prev_true']].values.reshape(1, -1)
                replicated_values = np.tile(first_value, (pad_size, 1))
                actual_values = data.iloc[0:end_idx][['prev_true']].values
                x_values = np.concatenate((replicated_values, actual_values), axis=0)
            else:
                x_values = data.iloc[i - window_size:end_idx][['prev_true']].values
        else:
            continue
            # # Fallback to causal window with padding if needed
            # causal_start_idx = i - (2 * window_size)
            # if causal_start_idx < 0:
            #     pad_size = abs(causal_start_idx)
            #     first_value = data.iloc[0][['prev_true']].values.reshape(1, -1)
            #     replicated_values = np.tile(first_value, (pad_size, 1))
            #     actual_values = data.iloc[0:i + 1][['prev_true']].values
            #     x_values = np.concatenate((replicated_values, actual_values), axis=0)
            # else:
            #     x_values = data.iloc[causal_start_idx:i + 1][['prev_true']].values

        # Target variables: EIR_true and incall
        y = data.iloc[i][['EIR_true', 'incall']].values

        xs.append(x_values.flatten())
        ys.append(y)

    # Convert to tensors
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    return torch.tensor(xs), torch.tensor(ys)

def create_sequences_1output(data, window_size): # This adopts one input and one ouput approach
    xs, ys = [], []

    for i in range(len(data)):
        end_idx = i + window_size + 1

        # Full bidirectional sequence
        if end_idx <= len(data):
            if i < window_size:
                # Pad the start with the first value
                pad_size = window_size - i
                first_value = data.iloc[0][['prev_true']].values.reshape(1, -1)
                replicated_values = np.tile(first_value, (pad_size, 1))
                actual_values = data.iloc[0:end_idx][['prev_true']].values
                x_values = np.concatenate((replicated_values, actual_values), axis=0)
            else:
                x_values = data.iloc[i - window_size:end_idx][['prev_true']].values
        else:
            continue
            # # Fallback to causal window with padding if needed
            # causal_start_idx = i - (2 * window_size)
            # if causal_start_idx < 0:
            #     pad_size = abs(causal_start_idx)
            #     first_value = data.iloc[0][['prev_true']].values.reshape(1, -1)
            #     replicated_values = np.tile(first_value, (pad_size, 1))
            #     actual_values = data.iloc[0:i + 1][['prev_true']].values
            #     x_values = np.concatenate((replicated_values, actual_values), axis=0)
            # else:
            #     x_values = data.iloc[causal_start_idx:i + 1][['prev_true']].values

        # Target variable: EIR_true
        y = data.iloc[i][['EIR_true']].values

        xs.append(x_values.flatten())
        ys.append(y)

    # Convert to tensors
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    return torch.tensor(xs), torch.tensor(ys)
    

def create_sequences_in_parallel(features, targets, window_size):
    xs, ys = [], []
    
    for i in range(len(features) - window_size):
        start_idx = i - window_size
        end_idx = i + window_size + 1
        if i < window_size:
            pad_size = window_size - i
            first_values = features.iloc[0].values
            replicated_values = np.tile(first_values, (pad_size, 1))
            x_values = np.concatenate((replicated_values, features.iloc[0:i + window_size + 1].values), axis=0)
        else:
            x_values = features.iloc[i - window_size:i + window_size + 1].values
        
        y = np.array([targets.iloc[i]], dtype=np.float32)  # Ensure it's an array

        
        xs.append(x_values.flatten())
        ys.append(y)
    
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    
    return torch.tensor(xs), torch.tensor(ys)
    



def create_sequences_multi_inputs(data, window_size, features=['prev_true, EIR_true'], target='incall'): #This adopts multiple inputs to create one output
    xs, ys = [], []

    # # Group the data by 'run'
    # for run, run_df in data.groupby('run'):
    #     run_df = run_df.reset_index(drop=True)
    feature_data = data[features].to_numpy()

    for i in range(len(data)):
        end_idx = i + window_size + 1

        # Bidirectional window
        if end_idx <= len(data):
            if i < window_size:
                # Pad with first row
                pad_size = window_size - i
                first_row = feature_data[0].reshape(1, -1)
                padding = np.tile(first_row, (pad_size, 1))
                actual = feature_data[0:end_idx]
                x_values = np.concatenate((padding, actual), axis=0)
            else:
                x_values = feature_data[i - window_size:end_idx]
        else:
            continue
            # # Fallback to causal-only window
            # causal_start_idx = i - (2 * window_size)
            # if causal_start_idx < 0:
            #     pad_size = abs(causal_start_idx)
            #     first_row = feature_data[0].reshape(1, -1)
            #     padding = np.tile(first_row, (pad_size, 1))
            #     actual = feature_data[0:i + 1]
            #     x_values = np.concatenate((padding, actual), axis=0)
            # else:
            #     x_values = feature_data[causal_start_idx:i + 1]

        # Target variable
        y = data.iloc[i][[target]].values

        xs.append(x_values)#.flatten())
        ys.append(y)

    # Convert to tensors
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    return torch.tensor(xs), torch.tensor(ys)


def create_sequences_with_initial_value(data, window_size): #Creates sequential inputs of timeseies prevalence and scalar series of run first prev
    xs, ys = [], []

    for run, run_df in data.groupby('run'):
        run_df = run_df.reset_index(drop=True)
        prev_true = run_df['prev_true'].to_numpy()

        # This is the scalar to use as the constant co-feature
        first_prev_value = run_df.iloc[0]['prev_true']  # for co-input

        for i in range(len(prev_true)):
            end_idx = i + window_size + 1

            # Constructing the main sequence
            if end_idx <= len(prev_true):
                if i < window_size:
                    pad_size = window_size - i
                    pad_value = run_df.iloc[0][['prev_true']].values.reshape(1, -1)[0][0]  # scalar
                    pad_seq = np.full((pad_size,), pad_value)
                    actual_seq = prev_true[0:end_idx]
                    sequence = np.concatenate((pad_seq, actual_seq))
                else:
                    sequence = prev_true[i - window_size:end_idx]
            else:
                causal_start_idx = i - (2 * window_size)
                if causal_start_idx < 0:
                    pad_size = abs(causal_start_idx)
                    pad_value = run_df.iloc[0][['prev_true']].values.reshape(1, -1)[0][0]
                    pad_seq = np.full((pad_size,), pad_value)
                    actual_seq = prev_true[0:i + 1]
                    sequence = np.concatenate((pad_seq, actual_seq))
                else:
                    sequence = prev_true[causal_start_idx:i + 1]

            # Now sequence is (seq_len,)
            sequence = sequence.reshape(-1, 1)  # (seq_len, 1)

            # Add co-input: constant first value
            first_val_column = np.full((sequence.shape[0], 1), first_prev_value)  # (seq_len, 1)
            combined = np.concatenate((sequence, first_val_column), axis=1)  # (seq_len, 2)

            xs.append(combined)
            ys.append(run_df.iloc[i][['EIR_true']].values)

    xs = np.array(xs, dtype=np.float32)  # (batch, seq_len, input_size)
    ys = np.array(ys, dtype=np.float32)  # (batch, 1)

    return torch.tensor(xs), torch.tensor(ys)