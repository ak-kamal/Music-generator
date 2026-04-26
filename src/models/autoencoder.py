import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=49, hidden_dim=256, latent_dim=64, num_layers=2):
        super(LSTMEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim * 2, latent_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        hidden_concat = self.dropout(hidden_concat)
        z = self.fc(hidden_concat)
        
        return z

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=256, output_dim=49, seq_len=128, num_layers=2):
        super(LSTMDecoder, self).__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_cell = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        self.lstm = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        batch_size = z.shape[0]
        
        hidden = self.fc_hidden(z).view(batch_size, self.num_layers, self.hidden_dim)
        hidden = hidden.transpose(0, 1).contiguous()
        
        cell = self.fc_cell(z).view(batch_size, self.num_layers, self.hidden_dim)
        cell = cell.transpose(0, 1).contiguous()
        
        x = torch.zeros(batch_size, self.seq_len, 49).to(z.device)
        
        lstm_out, _ = self.lstm(x, (hidden, cell))
        x_recon = torch.sigmoid(self.fc_out(lstm_out))
        
        return x_recon

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=49, hidden_dim=256, latent_dim=64, seq_len=128, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers)
        
    def forward(self, x):
        if len(x.shape) == 4:
            x = x.squeeze(1)
        
        z = self.encoder(x)
        x_recon = self.decoder(z)
        x_recon = x_recon.unsqueeze(1)
        
        return x_recon, z
    
    def generate(self, num_samples, latent_dim=64, device='cpu'):
        z = torch.randn(num_samples, latent_dim).to(device)
        with torch.no_grad():
            generated = self.decoder(z)
        return generated.unsqueeze(1)