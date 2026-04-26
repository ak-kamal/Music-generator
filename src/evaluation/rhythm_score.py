import numpy as np

def extract_note_durations(piano_roll, fs=8):
    """
    Extract durations of notes from piano roll
    
    Args:
        piano_roll: numpy array of shape (time_steps, pitches)
        fs: frames per second (default 8)
    
    Returns:
        List of note durations in seconds
    """
    time_steps, n_pitches = piano_roll.shape
    step_duration = 1.0 / fs
    
    durations = []
    
    for pitch_idx in range(n_pitches):
        t = 0
        while t < time_steps:
            if piano_roll[t, pitch_idx] > 0.05:  # Note is active
                start = t
                # Find end of note
                while t < time_steps and piano_roll[t, pitch_idx] > 0.05:
                    t += 1
                end = t
                duration = (end - start) * step_duration
                durations.append(duration)
            else:
                t += 1
    
    return durations

def compute_rhythm_diversity(piano_roll, fs=8):
    """
    Compute rhythm diversity score: D_rhythm = unique_durations / total_notes
    Following project guideline formula
    
    Args:
        piano_roll: numpy array of shape (time_steps, pitches)
        fs: frames per second (default 8)
    
    Returns:
        Rhythm diversity score (higher = more diverse)
    """
    durations = extract_note_durations(piano_roll, fs)
    
    if len(durations) == 0:
        return 0.0
    
    # Round durations to 2 decimal places for uniqueness
    rounded_durations = [round(d, 2) for d in durations]
    unique_durations = len(set(rounded_durations))
    total_notes = len(durations)
    
    diversity = unique_durations / total_notes
    
    return diversity

def compute_repetition_ratio(piano_roll, pattern_length=8):
    """
    Compute repetition ratio: R = repeated_patterns / total_patterns
    Following project guideline formula
    
    Args:
        piano_roll: numpy array of shape (time_steps, pitches)
        pattern_length: Length of pattern to check for repetition (in time steps)
    
    Returns:
        Repetition ratio (lower = less repetitive)
    """
    time_steps, n_pitches = piano_roll.shape
    
    if time_steps < pattern_length * 2:
        return 0.0
    
    # Convert each time step to a string representation for pattern matching
    patterns = []
    for i in range(0, time_steps - pattern_length + 1, pattern_length):
        pattern = piano_roll[i:i+pattern_length, :]
        # Flatten and round to create hashable representation
        pattern_hash = tuple(np.round(pattern.flatten(), 2))
        patterns.append(pattern_hash)
    
    if len(patterns) < 2:
        return 0.0
    
    # Count repeated patterns
    unique_patterns = set(patterns)
    repeated_count = len(patterns) - len(unique_patterns)
    
    repetition_ratio = repeated_count / len(patterns)
    
    return repetition_ratio