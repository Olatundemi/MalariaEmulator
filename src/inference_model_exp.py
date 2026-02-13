import torch 
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np


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
    
class MSConv(nn.Module):
    """
    Multi-branch temporal conv:
      • ultra-low dilation branch  --> micro structure
      • low dilation branch        --> short-term trends
      • high dilation branch       --> long-range structure
    """
    def __init__(
        self,
        in_channels=1,
        channels_per_branch=8
    ):
        super().__init__()

        # -------- ULTRA-LOW dilation branch --------
        self.ultra = nn.Sequential(
            nn.Conv1d(
                in_channels, channels_per_branch,
                kernel_size=3, dilation=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels_per_branch),

            nn.Conv1d(
                channels_per_branch, channels_per_branch,
                kernel_size=3, dilation=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels_per_branch)
        )

        # -------- LOW dilation branch (medium receptive field) --------
        self.low = nn.Sequential(
            nn.Conv1d(
                in_channels, channels_per_branch,
                kernel_size=5, dilation=2,
                padding=4    # (5-1)/2 * 2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels_per_branch),

            nn.Conv1d(
                channels_per_branch, channels_per_branch,
                kernel_size=5, dilation=2,
                padding=4
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels_per_branch)
        )

        # -------- HIGH dilation branch (long range) --------
        self.high = nn.Sequential(
            nn.Conv1d(
                in_channels, channels_per_branch,
                kernel_size=7, dilation=8,
                padding=24   # SAME
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels_per_branch),

            nn.Conv1d(
                channels_per_branch, channels_per_branch,
                kernel_size=7, dilation=16,
                padding=48
            ),
            nn.ReLU(),
            nn.BatchNorm1d(channels_per_branch)
        )

    def forward(self, x):
        # x: (B,1,T)
        u = self.ultra(x)
        l = self.low(x)
        h = self.high(x)
        return torch.cat([u, l, h], dim=1)   # (B, 3*channels, T)


class FourierPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512, num_freqs=8):
        super().__init__()
        self.dim = dim
        self.num_freqs = num_freqs

        freqs = 2 ** torch.arange(num_freqs).float()
        self.register_buffer("freqs", freqs)

        self.proj = nn.Linear(2 * num_freqs, dim)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, _ = x.shape
        device = x.device

        t = torch.linspace(0, 1, T, device=device)  # normalized time
        angles = 2 * np.pi * t[:, None] * self.freqs[None, :]

        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        pe = self.proj(pe)                          # (T, D)
        pe = pe.unsqueeze(0).expand(B, -1, -1)      # (B, T, D)

        return x + pe

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        B, T, _ = x.shape
        positions = torch.arange(T, device=x.device)
        pe = self.pe(positions)          # (T, D)
        return pe.unsqueeze(0).expand(B, -1, -1)

class HybridPositionalEncoding(nn.Module):
    def __init__(
        self,
        dim,
        max_len=512,
        num_freqs=8,
        alpha_init=1.0,
        beta_init=1.0
    ):
        super().__init__()

        self.fourier_pe = FourierPositionalEncoding(
            dim=dim,
            max_len=max_len,
            num_freqs=num_freqs
        )

        self.learned_pe = LearnedPositionalEmbedding(
            max_len=max_len,
            dim=dim
        )

        # Learnable scaling factors
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta  = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x):
        """
        x: (B, T, D)
        """
        pe_fourier = self.fourier_pe(x) - x
        pe_learned = self.learned_pe(x)

        return x + self.alpha * pe_fourier + self.beta * pe_learned


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x, mask):
        energy = torch.tanh(self.attn(x))
        scores = self.v(energy).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context

class TemporalEncoder(nn.Module):
    def __init__(
        self,
        input_channels=1,
        conv_channels=8,
        hidden_dim=128,
        num_layers=2,
        use_attention=True,
        positional_encoding: nn.Module | None = None,
    ):
        super().__init__()

        self.use_attention = use_attention
        self.pe = positional_encoding

        self.conv = MSConv(
            in_channels=input_channels,
            channels_per_branch=conv_channels
        )
        conv_dim = conv_channels * 3
        self.input_proj = nn.Linear(conv_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        if use_attention:
            self.attn = SelfAttention(hidden_dim)

    def forward(self, x, mask, return_sequence=False):
        """
        x:    (B, T, 1)
        mask: (B, T)
        """

        # Conv
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.input_proj(x)

        if self.pe is not None:
            x = self.pe(x)

        x = x * mask.unsqueeze(-1)

        x, _ = self.lstm(x)   # (B, T, H)

        if return_sequence:
            return x

        if self.use_attention:
            return self.attn(x, mask)
        else:
            # last *valid* timestep
            idx = mask.sum(dim=1).long() - 1
            return x[torch.arange(len(x)), idx]

class IncidenceHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attn_eir = SelfAttention(hidden_dim)
        self.attn_phi = SelfAttention(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, h_eir_seq, h_phi_seq, mask_eir, mask_phi):
        z_eir = self.attn_eir(h_eir_seq, mask_eir)
        z_phi = self.attn_phi(h_phi_seq, mask_phi)

        z = torch.cat([z_eir, z_phi], dim=-1)
        return self.mlp(z)

class MultiHeadModel(nn.Module):
    def __init__(self, max_len=256):
        super().__init__()

        #Positional encoding ONLY for EIR
#         eir_pe = HybridPositionalEncoding(
#             dim=128,
#             max_len=max_len,
#             num_freqs=8,
#             alpha_init=1.0,
#             beta_init=0.5
#         )

        self.eir = TemporalEncoder(
            hidden_dim=128,
            num_layers=3,
            use_attention=True,
            positional_encoding=None#eir_pe
        )

        self.phi = TemporalEncoder(
            hidden_dim=128,
            num_layers=3,
            use_attention=True,
            positional_encoding=None
        )

        self.head_eir = nn.Linear(128, 1)
        self.head_phi = nn.Linear(128, 1)

        self.incidence = IncidenceHead(hidden_dim=128)

    def forward(self, batch):
        # ---- pooled latents ----
        h_eir = self.eir(
            batch["eir"][0],
            batch["eir"][1],
            return_sequence=False
        )

        h_phi = self.phi(
            batch["phi"][0],
            batch["phi"][1],
            return_sequence=False
        )

        out_eir = self.head_eir(h_eir)
        out_phi = self.head_phi(h_phi)
        
        # ---- sequence latents ----
        h_eir_seq = self.eir(
            batch["eir"][0],
            batch["eir"][1],
            return_sequence=True
        )
        
        h_phi_seq = self.phi(
            batch["phi"][0],
            batch["phi"][1],
            return_sequence=True
        )
        
        # ---- incidence ----
        out_inc = self.incidence(
            h_eir_seq,
            h_phi_seq,
            batch["eir"][1],
            batch["phi"][1]
        )


        return out_eir, out_phi, out_inc
