import pandas as pd
import numpy as np
import os

def load_human_scores(file_path):
    """
    Load human feedback scores from Excel file
    
    Expected format:
        First row: Sample names (Sample 1, Sample 2, ...)
        First column: Participant names (Participant 1, Participant 2, ...)
        Values: Scores from 1 to 5
    
    Args:
        file_path: Path to .xlsx file
    
    Returns:
        DataFrame of scores, average score per sample, overall average
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feedback file not found: {file_path}")
    
    df = pd.read_excel(file_path, index_col=0)
    
    print(f"Loaded human feedback from {file_path}")
    print(f"  Participants: {df.shape[0]}")
    print(f"  Samples: {df.shape[1]}")
    print(f"  Score range: {df.min().min():.1f} - {df.max().max():.1f}")
    
    return df

def compute_human_score(df):
    """
    Compute human listening score from feedback DataFrame
    
    Args:
        df: DataFrame with participants as rows, samples as columns
    
    Returns:
        Dictionary with overall average, per-sample averages, and statistics
    """
    # Overall average across all participants and samples
    overall_avg = df.mean().mean()
    
    # Per-sample average
    per_sample_avg = df.mean(axis=0).to_dict()
    
    # Per-participant average
    per_participant_avg = df.mean(axis=1).to_dict()
    
    # Standard deviation
    overall_std = df.stack().std()
    
    return {
        'overall_score': overall_avg,
        'overall_std': overall_std,
        'per_sample_scores': per_sample_avg,
        'per_participant_scores': per_participant_avg,
        'min_score': df.min().min(),
        'max_score': df.max().max(),
        'num_participants': df.shape[0],
        'num_samples': df.shape[1]
    }

def print_human_score(scores, model_name="Model"):
    """Pretty print human score results"""
    print(f"\n{'='*50}")
    print(f"HUMAN LISTENING SCORE - {model_name}")
    print(f"{'='*50}")
    print(f"Participants: {scores['num_participants']}")
    print(f"Samples evaluated: {scores['num_samples']}")
    print(f"Overall Score: {scores['overall_score']:.2f} / 5.00 (+- {scores['overall_std']:.2f})")
    print(f"Score Range: {scores['min_score']:.1f} - {scores['max_score']:.1f}")
    print(f"\nPer-Sample Scores:")
    for sample, score in scores['per_sample_scores'].items():
        print(f"  {sample}: {score:.2f}")
    print(f"{'='*50}")

def save_human_score_summary(scores, output_path):
    """Save human score summary to CSV"""
    summary_df = pd.DataFrame([
        {'Metric': 'Overall Score', 'Value': scores['overall_score']},
        {'Metric': 'Overall Std', 'Value': scores['overall_std']},
        {'Metric': 'Min Score', 'Value': scores['min_score']},
        {'Metric': 'Max Score', 'Value': scores['max_score']},
        {'Metric': 'Participants', 'Value': scores['num_participants']},
        {'Metric': 'Samples', 'Value': scores['num_samples']}
    ])
    summary_df.to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")