import numpy as np
import pretty_midi
import os

def numpy_to_midi(piano_roll, output_path, tempo=120, fs=8, velocity_scale=100):
    """
    Convert continuous velocity piano roll to playable MIDI file
    
    Args:
        piano_roll: numpy array of shape (time_steps, pitches) with values in [0, 1]
        output_path: path to save .mid file
        tempo: BPM (default 120)
        fs: frames per second (must match preprocessing: 8)
        velocity_scale: multiply velocity by this to get MIDI velocity (0-127)
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)
    
    step_duration = 1.0 / fs
    min_pitch = 36
    time_steps, n_pitches = piano_roll.shape
    
    for t in range(time_steps):
        for pitch_idx in range(n_pitches):
            velocity_value = piano_roll[t, pitch_idx]
            
            # Only create note if velocity is meaningful (above threshold)
            if velocity_value > 0.05:  # Lower threshold for continuous data
                midi_note = min_pitch + pitch_idx
                start_time = t * step_duration
                
                # Find end time (when note stops or velocity drops below threshold)
                end_time = start_time + step_duration
                next_t = t + 1
                while (next_t < time_steps and 
                       piano_roll[next_t, pitch_idx] > 0.05):
                    end_time = (next_t + 1) * step_duration
                    next_t += 1
                
                # Skip if note already added in previous step
                if t > 0 and piano_roll[t-1, pitch_idx] > 0.05:
                    continue
                
                # Scale velocity to MIDI range (0-127)
                midi_velocity = int(velocity_value * velocity_scale)
                midi_velocity = max(1, min(127, midi_velocity))  # Clamp to valid range
                
                note = pretty_midi.Note(
                    velocity=midi_velocity,
                    pitch=midi_note,
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)
    
    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"MIDI saved: {output_path}")
    
    return output_path

def generate_and_save_midi(model, num_samples=5, output_dir='../outputs/generated_midis/', 
                           latent_dim=64, device='cpu', tempo=120, velocity_scale=100):
    """
    Generate samples from model and save as MIDI files (continuous velocities)
    """
    import torch
    
    os.makedirs(output_dir, exist_ok=True)
    
    midi_paths = []
    
    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, latent_dim).to(device)
            generated = model.decoder(z)
            generated = generated.squeeze(0).cpu().numpy()
            
            # Do NOT threshold. Pass continuous values.
            output_path = os.path.join(output_dir, f'generated_sample_{i+1}.mid')
            numpy_to_midi(generated, output_path, tempo=tempo, velocity_scale=velocity_scale)
            midi_paths.append(output_path)
            
            np.save(os.path.join(output_dir, f'generated_sample_{i+1}.npy'), generated)
    
    print(f"\nGenerated {num_samples} MIDI files in {output_dir}")
    return midi_paths

def reconstruct_and_save(model, original_piano_roll, output_path, tempo=120, velocity_scale=100):
    """
    Reconstruct a sample and save as MIDI for comparison (continuous velocities)
    """
    import torch
    
    model.eval()
    with torch.no_grad():
        if len(original_piano_roll.shape) == 2:
            original_tensor = torch.FloatTensor(original_piano_roll).unsqueeze(0).unsqueeze(0)
        else:
            original_tensor = original_piano_roll
        
        recon, _ = model(original_tensor)
        recon = recon.squeeze(0).squeeze(0).cpu().numpy()
        
        # Do NOT threshold. Pass continuous values.
        numpy_to_midi(recon, output_path, tempo=tempo, velocity_scale=velocity_scale)