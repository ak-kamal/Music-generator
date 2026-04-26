import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.vae import VAE
from generation.midi_export import numpy_to_midi


def generate_vae_samples(model_path='../outputs/models/vae.pth',
                         num_samples=8,  # Project requires 8 samples
                         latent_dim=32,
                         seq_len=64,
                         output_dir='../outputs/generated_midis/vae_samples/'):
    """
    Step 13: Generate diverse multi-genre music by sampling z ∼ N(0, I)
    Following project algorithm exactly
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = VAE(
        input_dim=49,
        hidden_dim=128,
        latent_dim=latent_dim,
        seq_len=seq_len,
        num_layers=2
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded VAE model from {model_path}")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    print(f"Beta used: {checkpoint.get('beta', 1.0)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples by sampling from N(0, I)
    print(f"\nGenerating {num_samples} samples by sampling z ∼ N(0, I)...")
    
    midi_paths = []
    with torch.no_grad():
        for i in range(num_samples):
            # Sample from standard normal distribution
            z = torch.randn(1, latent_dim).to(device)
            
            # Decode
            generated = model.decoder(z)
            generated = generated.squeeze(0).cpu().numpy()
            
            # Convert to binary (threshold 0.5)
            generated_binary = (generated > 0.5).astype(np.float32)
            
            note_density = generated_binary.mean() * 100
            print(f"  Sample {i+1}: Note density = {note_density:.1f}%")
            
            # Save as MIDI
            output_path = os.path.join(output_dir, f'vae_sample_{i+1}.mid')
            numpy_to_midi(generated_binary, output_path)
            midi_paths.append(output_path)
    
    print(f"\n Generated {num_samples} MIDI files in {output_dir}")
    return midi_paths


def latent_interpolation(model_path='../outputs/models/vae.pth',
                         num_steps=10,
                         output_dir='../outputs/generated_midis/interpolations/'):
    """
    Optional: Interpolate between two latent vectors
    Shows smooth transitions in latent space
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = VAE(input_dim=49, hidden_dim=128, latent_dim=32, seq_len=64).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample two random latent vectors
    z1 = torch.randn(1, 32).to(device)
    z2 = torch.randn(1, 32).to(device)
    
    print(f"\nGenerating latent space interpolation...")
    
    with torch.no_grad():
        for i, alpha in enumerate(np.linspace(0, 1, num_steps)):
            z_interp = (1 - alpha) * z1 + alpha * z2
            generated = model.decoder(z_interp)
            generated = generated.squeeze(0).cpu().numpy()
            generated_binary = (generated > 0.5).astype(np.float32)
            
            output_path = os.path.join(output_dir, f'interp_step_{i+1:03d}.mid')
            numpy_to_midi(generated_binary, output_path)
    
    print(f" Generated {num_steps} interpolation steps in {output_dir}")
    return output_dir


if __name__ == "__main__":
    # Generate 8 samples as required by project
    samples = generate_vae_samples(num_samples=8)
    
    # Optional: Generate interpolations
    interpolations = latent_interpolation(num_steps=10)