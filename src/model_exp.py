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
    

#Function to train the model

def train_model(model, train_loader, eval_loader, model_name, epochs=25, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_history = []
    eval_loss_history = []

    start_time = time.time()  # Track time

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch.unsqueeze(-1))
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Evaluation phase
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for X_val, y_val in eval_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val.unsqueeze(-1))
                eval_loss += criterion(val_outputs, y_val).item()

        avg_eval_loss = eval_loss / len(eval_loader)
        eval_loss_history.append(avg_eval_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")

    duration = time.time() - start_time  # Training duration

    # Save the trained model
    torch.save(model.state_dict(), f"src/{model_name}_model.pth")
    print(f"Model saved as src/{model_name}_model.pth")

    return model, loss_history, eval_loss_history, duration


def train_models_in_parallel(model_EIR, model_Incidence, train_loader_EIR, train_loader_Incidence, 
                             eval_loader_EIR, eval_loader_Incidence, model_name="model", 
                             epochs=20, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_EIR.to(device)
    model_Incidence.to(device)
    
    optimizer_EIR = optim.Adam(model_EIR.parameters(), lr=lr)
    optimizer_Incidence = optim.Adam(model_Incidence.parameters(), lr=lr)
    
    criterion = nn.MSELoss()
    loss_history_EIR = []
    loss_history_Incidence = []
    eval_loss_history_EIR = []
    eval_loss_history_Incidence = []
    
    start_time = time.time()

    for epoch in range(epochs):
        model_EIR.train()
        model_Incidence.train()
        epoch_loss_EIR = 0
        epoch_loss_Incidence = 0
        
        for X_batch, y_batch in train_loader_EIR:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer_EIR.zero_grad()
            outputs = model_EIR(X_batch.unsqueeze(-1))
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer_EIR.step()
            epoch_loss_EIR += loss.item()
        
        for X_batch, y_batch in train_loader_Incidence:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer_Incidence.zero_grad()
            outputs = model_Incidence(X_batch)#.unsqueeze(-1))
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer_Incidence.step()
            epoch_loss_Incidence += loss.item()
        
        avg_loss_EIR = epoch_loss_EIR / len(train_loader_EIR)
        avg_loss_Incidence = epoch_loss_Incidence / len(train_loader_Incidence)
        loss_history_EIR.append(avg_loss_EIR)
        loss_history_Incidence.append(avg_loss_Incidence)
        
        model_EIR.eval()
        model_Incidence.eval()
        eval_loss_EIR = 0
        eval_loss_Incidence = 0
        
        with torch.no_grad():
            for X_val, y_val in eval_loader_EIR:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model_EIR(X_val.unsqueeze(-1))
                eval_loss_EIR += criterion(val_outputs, y_val).item()
            
            for X_val, y_val in eval_loader_Incidence:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model_Incidence(X_val)#.unsqueeze(-1))
                eval_loss_Incidence += criterion(val_outputs, y_val).item()
        
        avg_eval_loss_EIR = eval_loss_EIR / len(eval_loader_EIR)
        avg_eval_loss_Incidence = eval_loss_Incidence / len(eval_loader_Incidence)
        eval_loss_history_EIR.append(avg_eval_loss_EIR)
        eval_loss_history_Incidence.append(avg_eval_loss_Incidence)
        
        print(f"Epoch {epoch+1}/{epochs}, EIR Loss: {avg_loss_EIR:.4f}, Incidence Loss: {avg_loss_Incidence:.8f}, Eval EIR Loss: {avg_eval_loss_EIR:.4f}, Eval Incidence Loss: {avg_eval_loss_Incidence:.8f}")
    
    duration = time.time() - start_time
    
    # Save models using the model name
    eir_path = f"src/LSTM_EIR_{model_name}.pth"
    incidence_path = f"src/LSTM_Incidence_{model_name}.pth"
    torch.save(model_EIR.state_dict(), eir_path)
    torch.save(model_Incidence.state_dict(), incidence_path)
    print(f"Models saved as {eir_path} and {incidence_path}")
    
    return model_EIR, model_Incidence, loss_history_EIR, loss_history_Incidence, eval_loss_history_EIR, eval_loss_history_Incidence, duration