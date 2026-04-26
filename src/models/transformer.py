import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4, dropout=0.1, max_seq_len=512):
        super(MusicTransformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        self.register_buffer("causal_mask", self.generate_causal_mask(max_seq_len))
        
    def generate_causal_mask(self, max_len):
        mask = torch.triu(torch.ones(max_len, max_len) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        
        output = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            memory_mask=None
        )
        
        logits = self.output_layer(output)
        
        return logits
    
    def compute_loss(self, logits, targets):
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        loss = loss_fn(logits.view(-1, self.vocab_size), targets.view(-1))
        return loss
    
    def compute_perplexity(self, logits, targets):
        loss = self.compute_loss(logits, targets)
        perplexity = torch.exp(loss)
        return perplexity
    
    def generate(self, tokenizer, seed_tokens=None, max_len=256, temperature=1.0, device='cpu'):
        self.eval()
        
        if seed_tokens is None:
            generated = torch.tensor([[0]]).to(device)
        else:
            generated = torch.tensor([seed_tokens]).to(device)
        
        with torch.no_grad():
            for _ in range(max_len - generated.shape[1]):
                logits = self.forward(generated)
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if (generated[:, -20:] == 0).all() and generated.shape[1] > 50:
                    break
        
        generated_tokens = generated.squeeze(0).cpu().tolist()
        piano_roll = tokenizer.decode(generated_tokens)
        
        return piano_roll, generated_tokens