import numpy as np

def compute_pitch_histogram(piano_roll, n_bins=12):
    """
    Compute pitch class histogram (C, C#, D, ..., B)
    
    Args:
        piano_roll: numpy array of shape (time_steps, pitches)
        n_bins: Number of pitch classes (default 12 for chromatic scale)
    
    Returns:
        Normalized histogram of length n_bins
    """
    n_pitches = piano_roll.shape[1]
    pitches_per_bin = n_pitches // n_bins
    
    histogram = np.zeros(n_bins)
    
    for pitch_idx in range(n_pitches):
        # Determine which pitch class this pitch belongs to
        bin_idx = pitch_idx % n_bins
        # Sum velocities for this pitch across all time steps
        total_velocity = np.sum(piano_roll[:, pitch_idx])
        histogram[bin_idx] += total_velocity
    
    # Normalize
    if np.sum(histogram) > 0:
        histogram = histogram / np.sum(histogram)
    
    return histogram

def compute_pitch_histogram_similarity(piano_roll1, piano_roll2, n_bins=12):
    """
    Compute L1 distance between pitch histograms of two piano rolls
    Following project guideline: H(p,q) = sum |p_i - q_i|
    
    Args:
        piano_roll1: First piano roll (time_steps, pitches)
        piano_roll2: Second piano roll (time_steps, pitches)
        n_bins: Number of pitch classes
    
    Returns:
        Similarity score (lower = more similar)
    """
    hist1 = compute_pitch_histogram(piano_roll1, n_bins)
    hist2 = compute_pitch_histogram(piano_roll2, n_bins)
    
    # L1 distance as per formula
    similarity = np.sum(np.abs(hist1 - hist2))
    
    return similarity