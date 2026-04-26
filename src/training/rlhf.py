import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from generation.midi_export import numpy_to_midi

class RLHFTrainer:
    def __init__(self, model, tokenizer, device='cpu', learning_rate=0.0001):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
    def load_human_scores(self, feedback_path):
        df = pd.read_excel(feedback_path, index_col=0)
        print(f"Loaded feedback from {feedback_path}")
        print(f"  Participants: {df.shape[0]}, Samples: {df.shape[1]}")
        print(f"  Average score: {df.mean().mean():.3f}")
        return df
    
    def compute_reward(self, feedback_df):
        return feedback_df.mean().mean()
    
    def compute_log_probs_for_token_sequence(self, token_sequence):
        """
        Compute log p_theta(X) for a token sequence (the one that was rated)
        This is the key: we reuse the EXACT sequence, not generate a new one.
        """
        self.model.train()
        token_tensor = torch.tensor([token_sequence], dtype=torch.long).to(self.device)
        
        # Forward pass
        logits = self.model(token_tensor)
        
        # Calculate log probability of the entire sequence
        # cross_entropy = -sum(log p(x_t | x_<t))
        nll = torch.nn.functional.cross_entropy(
            logits.view(-1, self.model.vocab_size),
            token_tensor.view(-1),
            reduction='sum'
        )
        
        # log p_theta(X) = -nll
        log_prob = -nll
        
        return log_prob
    
    def policy_gradient_update(self, rated_sequences, rewards, epochs=10, batch_size=5):
        """
        True policy gradient update:
        θ ← θ + η ∇θ J(θ)
        where ∇θ J(θ) = (1/N) * Σ [r_i * ∇θ log p_theta(X_i)]
        
        Args:
            rated_sequences: List of token sequences that were rated by humans
            rewards: List of rewards (average human scores) for each sequence
            epochs: Number of training epochs
            batch_size: Batch size for gradient updates
        """
        print("\n" + "="*50)
        print("POLICY GRADIENT UPDATE")
        print("="*50)
        
        print(f"Training on {len(rated_sequences)} rated sequences")
        print(f"Rewards: {[f'{r:.3f}' for r in rewards]}")
        
        # Normalize rewards for stability
        rewards = np.array(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            total_loss = 0
            
            # Shuffle indices
            indices = np.random.permutation(len(rated_sequences))
            
            for batch_start in range(0, len(rated_sequences), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                
                batch_loss = 0
                for idx in batch_indices:
                    tokens = rated_sequences[idx]
                    reward = rewards[idx]
                    
                    # Compute log p_theta(X) for this EXACT sequence
                    log_prob = self.compute_log_probs_for_token_sequence(tokens)
                    
                    # Loss = -reward * log_prob
                    # This gives gradient = -reward * ∇θ log_prob
                    # So optimizer.step() does: θ ← θ - η * (-reward * ∇θ log_prob) = θ + η * reward * ∇θ log_prob
                    loss_term = -reward * log_prob / len(batch_indices)
                    batch_loss = batch_loss + loss_term
                
                batch_loss.backward()
                total_loss += batch_loss.item()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {total_loss:.4f}")
        
        print(f"\nPolicy gradient update complete.")
    
    def save_model(self, save_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {load_path}")


def generate_rlhf_samples(model, tokenizer, num_samples=10, output_dir='../outputs/generated_midis/task4_rlhf/', 
                          iteration=1, max_len=256, temperature=0.8, device='cpu', return_sequences=False):
    """Generate samples and return token sequences for next iteration"""
    os.makedirs(output_dir, exist_ok=True)
    iteration_dir = os.path.join(output_dir, f'iteration_{iteration}')
    os.makedirs(iteration_dir, exist_ok=True)
    
    samples = []
    token_sequences = []
    model.eval()
    
    print(f"\nGenerating {num_samples} samples for iteration {iteration}...")
    with torch.no_grad():
        for i in range(num_samples):
            piano_roll, tokens = model.generate(
                tokenizer=tokenizer,
                seed_tokens=None,
                max_len=max_len,
                temperature=temperature,
                device=device
            )
            
            output_path = os.path.join(iteration_dir, f'rlhf_sample_{i+1}.mid')
            numpy_to_midi(piano_roll, output_path, tempo=120, velocity_scale=100)
            samples.append(piano_roll)
            token_sequences.append(tokens)
            
            density = piano_roll.mean() * 100
            print(f"  Sample {i+1}: density={density:.1f}%")
    
    print(f"Samples saved to {iteration_dir}")
    
    if return_sequences:
        return samples, token_sequences
    return samples