import torch 
import torch.nn as nn
import torch.optim as optim
import time


#Function to define the ML model
class LSTMModel(nn.Module):
    def __init__(self, input_size, architecture):
        super(LSTMModel, self).__init__()
        self.lstm_layers = nn.ModuleList()
        for i, hidden_size in enumerate(architecture):
            self.lstm_layers.append(nn.LSTM(input_size if i == 0 else architecture[i - 1], hidden_size, batch_first=True))
        self.fc = nn.Linear(architecture[-1], 2)
    
    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = self.fc(x[:, -1, :])
        return x
    

class LSTM_EIR(nn.Module):
    def __init__(self, input_size, architecture):
        super(LSTM_EIR, self).__init__()
        self.lstm_layers = nn.ModuleList()
        for i, hidden_size in enumerate(architecture):
            self.lstm_layers.append(nn.LSTM(input_size if i == 0 else architecture[i - 1], hidden_size, batch_first=True))
        self.fc = nn.Linear(architecture[-1], 1)  # Predicting only EIR
    
    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = self.fc(x[:, -1, :])
        return x

class LSTM_Incidence(nn.Module):
    def __init__(self, input_size, architecture):
        super(LSTM_Incidence, self).__init__()
        self.lstm_layers = nn.ModuleList()
        for i, hidden_size in enumerate(architecture):
            self.lstm_layers.append(nn.LSTM(input_size if i == 0 else architecture[i - 1], hidden_size, batch_first=True))
        self.fc = nn.Linear(architecture[-1], 1)  # Predicting only Incidence
    
    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = self.fc(x[:, -1, :])
        return x
    
class LSTMModel_baseline(nn.Module):
    def __init__(self, input_size, architecture):
        super(LSTMModel_baseline, self).__init__()
        self.lstm_layers = nn.ModuleList()
        for i, hidden_size in enumerate(architecture):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size if i == 0 else architecture[i - 1],
                    hidden_size,
                    batch_first=True
                )
            )
        self.fc = nn.Linear(architecture[-1] + 1, 2)  # output 2 values

    def forward(self, x_seq, x_scalar):
        x = x_seq
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        last_hidden = x[:, -1, :]  # (batch_size, hidden_size)
        combined = torch.cat((last_hidden, x_scalar), dim=1)  # (batch_size, hidden_size + 1)
        return self.fc(combined)  # (batch_size, 2)