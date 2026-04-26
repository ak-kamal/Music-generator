# src/preprocessing/piano_roll.py
import numpy as np
import pretty_midi
import os

CONFIG = {
    'fs': 8,
    'min_pitch': 36,
    'max_pitch': 84,
    'window_size': 128,
    'hop_size': 64,
    'min_velocity': 0.05,  # Changed from 0.01 to 0.05
    'binary': False,
}

def midi_to_piano_roll(midi_path, verbose=False):
    """
    Convert MAESTRO MIDI to continuous velocity piano roll
    Values in [0, 1] range
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        
        if len(pm.instruments) == 0:
            return None
        
        # MAESTRO files are solo piano, take first instrument
        pm.instruments = [pm.instruments[0]]
        
        # Get piano roll with velocities (0-127)
        piano_roll = pm.get_piano_roll(fs=CONFIG['fs'])
        
        # Trim to pitch range
        piano_roll = piano_roll[CONFIG['min_pitch']:CONFIG['max_pitch']+1]
        
        # Normalize to [0, 1]
        piano_roll = piano_roll / 127.0
        
        # Transpose to (time, pitches)
        piano_roll = piano_roll.T
        
        return piano_roll.astype(np.float32)
    
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        return None

def segment_piano_roll(piano_roll):
    """
    Split into windows, filter silent segments
    """
    if piano_roll is None or piano_roll.shape[0] < CONFIG['window_size']:
        return []
    
    segments = []
    for start in range(0, piano_roll.shape[0] - CONFIG['window_size'] + 1, CONFIG['hop_size']):
        segment = piano_roll[start:start + CONFIG['window_size']]
        
        # Filter silent segments
        if segment.mean() >= CONFIG['min_velocity']:
            segments.append(segment)
    
    return segments