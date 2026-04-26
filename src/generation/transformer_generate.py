import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import MusicTransformer
from generation.midi_export import numpy_to_midi


def generate_long_compositions(model_path='../outputs/models/transformer.pth',
                               num_samples=10,  # Project requires 10
                               max_len=256,  # Longer sequences
                               temperature=1.0,
                               output_dir='../outputs/generated_midis/transformer_samples/'):
    """
    Generate long coherent sequences using Transformer
    Following: x_t ~ p(x_t | x_<t)
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = MusicTransformer(
        n_pitches=49,
        d_model=256,
        n_heads=8,
        n_layers=4,
        max_seq_len=512
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded Transformer model from {model_path}")
    print(f"Best validation loss: {checkpoint['val_loss']:.6f}")
    print(f"Best validation perplexity: {checkpoint['val_perplexity']:.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating {num_samples} long compositions (max_len={max_len} steps)...")
    
    midi_paths = []
    for i in range(num_samples):
        # Generate autoregressively
        generated = model.generate(
            seed=None,
            max_len=max_len,
            temperature=temperature,
            device=device
        )
        
        # Convert to numpy
        generated_np = generated.squeeze(0).cpu().numpy()  # (seq_len, n_pitches)
        
        # Calculate statistics
        note_density = generated_np.mean() * 100
        active_steps = (generated_np.sum(axis=1) > 0).sum()
        
        print(f"  Sample {i+1}: {generated_np.shape[0]} steps, "
              f"note density={note_density:.1f}%, "
              f"active steps={active_steps}")
        
        # Save as MIDI
        output_path = os.path.join(output_dir, f'transformer_sample_{i+1}.mid')
        numpy_to_midi(generated_np, output_path, tempo=120)
        midi_paths.append(output_path)
        
        # Also save as numpy
        np.save(os.path.join(output_dir, f'transformer_sample_{i+1}.npy'), generated_np)
    
    print(f"\n Generated {num_samples} MIDI files in {output_dir}")
    return midi_paths


def generate_with_seed(model_path='../outputs/models/transformer.pth',
                       seed_path=None,
                       max_len=256,
                       temperature=0.8,
                       output_dir='../outputs/generated_midis/'):
    """
    Generate continuation of a seed sequence
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = MusicTransformer(n_pitches=49, d_model=256, n_heads=8, n_layers=4).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load seed
    if seed_path:
        seed = np.load(seed_path)
        seed_tensor = torch.FloatTensor(seed).unsqueeze(0).to(device)
    else:
        # Use a random real sample from test set
        seed_tensor = None
    
    # Generate
    generated = model.generate(seed=seed_tensor, max_len=max_len, temperature=temperature, device=device)
    generated_np = generated.squeeze(0).cpu().numpy()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'continuation.mid')
    numpy_to_midi(generated_np, output_path)
    
    print(f" Continuation generated: {output_path}")
    return output_path


if __name__ == "__main__":
    # Generate 10 samples as required by project
    samples = generate_long_compositions(num_samples=10, max_len=256, temperature=0.8)