import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder import LSTMAutoencoder

def generate_music_samples(model_path='../outputs/models/autoencoder.pth', 
                           num_samples=5, 
                           latent_dim=32, 
                           seq_len=64,
                           output_dir='../outputs/generated_midis/'):
    """
    Generate new music samples by sampling latent vectors
    
    Following project guideline:
    "Generate new music by sampling latent codes z"
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = LSTMAutoencoder(
        input_dim=49,
        hidden_dim=128,
        latent_dim=latent_dim,
        seq_len=seq_len,
        num_layers=2
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    
    # Generate samples
    print(f"\nGenerating {num_samples} music samples...")
    
    samples = []
    with torch.no_grad():
        for i in range(num_samples):
            # Sample random latent vector from N(0, I)
            z = torch.randn(1, latent_dim).to(device)
            
            # Decode to piano roll
            generated = model.decoder(z)
            generated = generated.squeeze(0).cpu().numpy()  # (seq_len, 49)
            
            # Convert to binary (threshold at 0.5)
            generated_binary = (generated > 0.5).astype(np.float32)
            
            samples.append(generated_binary)
            
            print(f"  Sample {i+1}: Generated shape {generated_binary.shape}, "
                  f"Note density: {generated_binary.mean()*100:.1f}%")
    
    # Save as numpy arrays for now (will convert to MIDI later)
    os.makedirs(output_dir, exist_ok=True)
    for i, sample in enumerate(samples):
        np.save(os.path.join(output_dir, f'generated_sample_{i+1}.npy'), sample)
    
    print(f"\n✅ Generated {num_samples} samples saved to {output_dir}")
    
    return samples

def interpolate_latent_space(model_path='../outputs/models/autoencoder.pth',
                            num_steps=10,
                            output_dir='../outputs/generated_midis/'):
    """
    Interpolate between two random latent vectors to show smooth transitions
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = LSTMAutoencoder(input_dim=49, hidden_dim=128, latent_dim=32, seq_len=64, num_layers=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Sample two random latent vectors
    z1 = torch.randn(1, 32).to(device)
    z2 = torch.randn(1, 32).to(device)
    
    # Create interpolations
    interpolations = []
    alphas = np.linspace(0, 1, num_steps)
    
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            generated = model.decoder(z_interp)
            generated = generated.squeeze(0).cpu().numpy()
            generated_binary = (generated > 0.5).astype(np.float32)
            interpolations.append(generated_binary)
    
    # Save interpolations
    os.makedirs(output_dir, exist_ok=True)
    for i, interp in enumerate(interpolations):
        np.save(os.path.join(output_dir, f'interpolation_step_{i+1:03d}.npy'), interp)
    
    print(f"✅ Generated {num_steps} interpolation steps saved to {output_dir}")
    
    return interpolations

if __name__ == "__main__":
    # Generate 5 samples as required by project
    samples = generate_music_samples(num_samples=5)
    
    # Optional: Generate interpolations
    interpolations = interpolate_latent_space(num_steps=10)