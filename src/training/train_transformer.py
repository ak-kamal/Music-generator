import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import MusicTransformer

class TransformerTrainer:
    def __init__(self, model, tokenizer, device='cpu', learning_rate=0.0001):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []
    
    def prepare_batch(self, token_sequences, max_len=None):
        if max_len is None:
            max_len = max(len(seq) for seq in token_sequences)
        
        batch = []
        for seq in token_sequences:
            if len(seq) > max_len:
                seq = seq[:max_len]
            else:
                seq = seq + [0] * (max_len - len(seq))
            batch.append(seq)
        
        return torch.tensor(batch, dtype=torch.long).to(self.device)
    
    def train_epoch(self, train_sequences, batch_size=16, max_seq_len=128):
        self.model.train()
        total_loss = 0
        total_ppl = 0
        n_batches = 0
        
        for i in range(0, len(train_sequences), batch_size):
            batch_seqs = train_sequences[i:i+batch_size]
            batch = self.prepare_batch(batch_seqs, max_seq_len)
            
            logits = self.model(batch)
            loss = self.model.compute_loss(logits, batch)
            ppl = self.model.compute_perplexity(logits, batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_ppl += ppl.item()
            n_batches += 1
            
            if n_batches % 50 == 0:
                print(f"  Batch {n_batches}, Loss: {loss.item():.4f}")
        
        return total_loss / n_batches, total_ppl / n_batches
    
    def validate(self, val_sequences, batch_size=16, max_seq_len=128):
        self.model.eval()
        total_loss = 0
        total_ppl = 0
        n_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_sequences), batch_size):
                batch_seqs = val_sequences[i:i+batch_size]
                batch = self.prepare_batch(batch_seqs, max_seq_len)
                
                logits = self.model(batch)
                loss = self.model.compute_loss(logits, batch)
                ppl = self.model.compute_perplexity(logits, batch)
                
                total_loss += loss.item()
                total_ppl += ppl.item()
                n_batches += 1
        
        return total_loss / n_batches, total_ppl / n_batches
    
    def train(self, train_sequences, val_sequences, epochs=30, 
              batch_size=16, max_seq_len=128,
              save_path='../outputs/models/transformer_maestro.pth'):
        
        print(f"\n{'='*50}")
        print(f"Training Transformer on Tokenized MAESTRO")
        print(f"{'='*50}")
        print(f"Vocabulary size: {self.model.vocab_size}")
        print(f"Training sequences: {len(train_sequences)}")
        print(f"Validation sequences: {len(val_sequences)}")
        print(f"{'='*50}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            train_loss, train_ppl = self.train_epoch(train_sequences, batch_size, max_seq_len)
            self.train_losses.append(train_loss)
            self.train_perplexities.append(train_ppl)
            
            val_loss, val_ppl = self.validate(val_sequences, batch_size, max_seq_len)
            self.val_losses.append(val_loss)
            self.val_perplexities.append(val_ppl)
            
            print(f"Epoch {epoch}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Perplexity: {train_ppl:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Perplexity: {val_ppl:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_perplexity': train_ppl,
                    'val_perplexity': val_ppl,
                }, save_path)
                print(f"  -> Saved best model")
        
        return self.train_losses, self.val_losses
    
    def plot_curves(self, save_path='../outputs/plots/transformer_curves.png'):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Autoregressive Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.train_perplexities, label='Train Perplexity')
        axes[1].plot(self.val_perplexities, label='Validation Perplexity')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Perplexity')
        axes[1].set_title('Perplexity = exp(L_TR / T)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()