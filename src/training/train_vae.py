import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae import VAE
from training.data_loader import get_data_loaders


class VAETrainer:
    def __init__(self, model, beta=1.0, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        self.train_recon_losses = []
        self.train_kl_losses = []
        self.train_total_losses = []
        self.val_recon_losses = []
        self.val_kl_losses = []
        self.val_total_losses = []
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_recon = 0
        total_kl = 0
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            batch = batch.to(self.device)
            
            recon_batch, mu, log_var, z = self.model(batch)
            
            # Debug: print log_var statistics
            # print(f"mu mean: {mu.mean().item():.4f}, mu std: {mu.std().item():.4f}")
            # print(f"log_var mean: {log_var.mean().item():.4f}, log_var std: {log_var.std().item():.4f}")
            # print(f"log_var min: {log_var.min().item():.4f}, log_var max: {log_var.max().item():.4f}")
            # print(f"exp(log_var) min: {log_var.exp().min().item():.6f}")
            
            # recon_loss = self.criterion(recon_batch, batch)
            # kl_loss = self.model.kl_divergence(mu, log_var)
            
            # print(f"recon_loss: {recon_loss.item():.6f}, kl_loss: {kl_loss.item():.10f}")
        
            recon_loss = self.criterion(recon_batch, batch)
            kl_loss = self.model.kl_divergence(mu, log_var)
            loss = recon_loss + self.beta * kl_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_loss += loss.item()
     
        n = len(train_loader)
        return total_recon / n, total_kl / n, total_loss / n
    
    def validate(self, val_loader):
        self.model.eval()
        total_recon = 0
        total_kl = 0
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                batch = batch.to(self.device)
                recon_batch, mu, log_var, z = self.model(batch)
                
                recon_loss = self.criterion(recon_batch, batch)
                kl_loss = self.model.kl_divergence(mu, log_var)
                loss = recon_loss + self.beta * kl_loss
                
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                total_loss += loss.item()
        
        n = len(val_loader)
        return total_recon / n, total_kl / n, total_loss / n
    
    def train(self, train_loader, val_loader, epochs=100, save_path='../outputs/models/vae_maestro.pth'):
        print(f"\n{'='*50}")
        print(f"Training VAE on MAESTRO (beta={self.beta})")
        print(f"{'='*50}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"{'='*50}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            train_recon, train_kl, train_total = self.train_epoch(train_loader)
            self.train_recon_losses.append(train_recon)
            self.train_kl_losses.append(train_kl)
            self.train_total_losses.append(train_total)
            
            val_recon, val_kl, val_total = self.validate(val_loader)
            self.val_recon_losses.append(val_recon)
            self.val_kl_losses.append(val_kl)
            self.val_total_losses.append(val_total)
            
            print(f"Epoch {epoch}/{epochs}")
            print(f"  Train - Recon: {train_recon:.6f}, KL: {train_kl:.6f}, Total: {train_total:.6f}")
            print(f"  Val   - Recon: {val_recon:.6f}, KL: {val_kl:.6f}, Total: {val_total:.6f}")
            
            if val_total < best_val_loss:
                best_val_loss = val_total
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_total,
                    'val_loss': val_total,
                    'beta': self.beta,
                }, save_path)
                print(f"  -> Saved best model")
        
        print(f"\nBest Validation Total Loss: {best_val_loss:.6f}")
        return self.train_total_losses, self.val_total_losses
    
    def plot_losses(self, save_path='../outputs/plots/vae_loss_curve_maestro.png'):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        axes[0].plot(self.train_recon_losses, label='Train')
        axes[0].plot(self.val_recon_losses, label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE Loss')
        axes[0].set_title('Reconstruction Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(self.train_kl_losses, label='Train')
        axes[1].plot(self.val_kl_losses, label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('KL Divergence')
        axes[1].set_title(f'KL Divergence (beta={self.beta})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(self.train_total_losses, label='Train')
        axes[2].plot(self.val_total_losses, label='Val')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Total Loss')
        axes[2].set_title('Total VAE Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()