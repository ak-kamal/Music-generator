import numpy as np
from .pitch_histogram import compute_pitch_histogram_similarity
from .rhythm_score import compute_rhythm_diversity, compute_repetition_ratio

def compute_all_metrics(generated_samples, real_samples=None):
    """
    Compute all evaluation metrics for generated music samples
    
    Args:
        generated_samples: List of piano roll arrays (time_steps, pitches)
        real_samples: Optional list of real piano rolls for comparison
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Rhythm Diversity Score
    diversity_scores = [compute_rhythm_diversity(sample) for sample in generated_samples]
    metrics['rhythm_diversity'] = np.mean(diversity_scores)
    metrics['rhythm_diversity_std'] = np.std(diversity_scores)
    
    # Repetition Ratio
    repetition_ratios = [compute_repetition_ratio(sample) for sample in generated_samples]
    metrics['repetition_ratio'] = np.mean(repetition_ratios)
    metrics['repetition_ratio_std'] = np.std(repetition_ratios)
    
    # Pitch Histogram Similarity (if real samples provided)
    if real_samples is not None:
        similarities = []
        for gen in generated_samples:
            # Compare each generated sample to a random real sample
            real = real_samples[np.random.randint(0, len(real_samples))]
            similarity = compute_pitch_histogram_similarity(gen, real)
            similarities.append(similarity)
        metrics['pitch_histogram_similarity'] = np.mean(similarities)
        metrics['pitch_histogram_similarity_std'] = np.std(similarities)
    
    return metrics

def print_metrics(metrics, model_name="Model"):
    """Pretty print metrics"""
    print(f"\n{'='*50}")
    print(f"EVALUATION METRICS - {model_name}")
    print(f"{'='*50}")
    print(f"Rhythm Diversity: {metrics['rhythm_diversity']:.4f} (+- {metrics.get('rhythm_diversity_std', 0):.4f})")
    print(f"Repetition Ratio: {metrics['repetition_ratio']:.4f} (+- {metrics.get('repetition_ratio_std', 0):.4f})")
    if 'pitch_histogram_similarity' in metrics:
        print(f"Pitch Histogram Similarity: {metrics['pitch_histogram_similarity']:.4f} (+- {metrics.get('pitch_histogram_similarity_std', 0):.4f})")
    print(f"{'='*50}")