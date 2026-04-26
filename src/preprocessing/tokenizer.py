import numpy as np
from collections import Counter
import pickle
import os

class MusicTokenizer:
    """
    Convert piano roll to tokens and back
    Each token represents a unique combination of active pitches
    """
    
    def __init__(self, max_vocab_size=1000):
        self.max_vocab_size = max_vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.pad_token_id = 0
        self.unk_token_id = 1
        
    def build_vocabulary(self, piano_roll_segments, min_frequency=5):
        """
        Build vocabulary from piano roll segments
        Each unique pitch combination becomes a token
        """
        print("Building vocabulary from piano roll segments...")
        
        # Count all unique pitch combinations (rows in piano roll)
        token_counts = Counter()
        
        for segment in piano_roll_segments:
            # Each time step is a token (vector of 49 pitches)
            for time_step in segment:
                # Convert to bytes for hashing
                token_bytes = time_step.astype(np.uint8).tobytes()
                token_counts[token_bytes] += 1
        
        print(f"Found {len(token_counts)} unique tokens")
        
        # Add special tokens
        self.token_to_id = {
            '<PAD>': 0,
            '<UNK>': 1
        }
        
        # Add most common tokens
        token_id = 2
        for token_bytes, count in token_counts.most_common(self.max_vocab_size - 2):
            if count >= min_frequency:
                self.token_to_id[token_bytes] = token_id
                self.id_to_token[token_id] = token_bytes
                token_id += 1
        
        print(f"Vocabulary size: {len(self.token_to_id)} (including <PAD>, <UNK>)")
        return self
    
    def encode(self, piano_roll):
        """
        Convert piano roll to token IDs
        Args:
            piano_roll: (seq_len, n_pitches) numpy array
        Returns:
            List of token IDs
        """
        tokens = []
        for time_step in piano_roll:
            token_bytes = time_step.astype(np.uint8).tobytes()
            token_id = self.token_to_id.get(token_bytes, self.unk_token_id)
            tokens.append(token_id)
        return tokens
    
    def decode(self, token_ids, n_pitches=49):
        """
        Convert token IDs back to piano roll
        Args:
            token_ids: List of token IDs
            n_pitches: Number of pitches (49)
        Returns:
            Piano roll numpy array (seq_len, n_pitches)
        """
        piano_roll = []
        for token_id in token_ids:
            if token_id == self.pad_token_id or token_id == self.unk_token_id:
                # Return zeros for padding/unknown
                piano_roll.append(np.zeros(n_pitches, dtype=np.float32))
            else:
                token_bytes = self.id_to_token[token_id]
                time_step = np.frombuffer(token_bytes, dtype=np.uint8).astype(np.float32)
                piano_roll.append(time_step)
        
        return np.array(piano_roll)
    
    def save(self, save_path):
        """Save tokenizer to disk"""
        with open(save_path, 'wb') as f:
            pickle.dump({
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token,
                'max_vocab_size': self.max_vocab_size
            }, f)
        print(f"Tokenizer saved to {save_path}")
    
    def load(self, load_path):
        """Load tokenizer from disk"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        self.token_to_id = data['token_to_id']
        self.id_to_token = data['id_to_token']
        self.max_vocab_size = data['max_vocab_size']
        print(f"Tokenizer loaded from {load_path}")
        return self


def tokenize_dataset(piano_roll_segments, max_vocab_size=1000, min_frequency=5):
    """
    Complete pipeline: Build tokenizer and tokenize dataset
    """
    # Build tokenizer
    tokenizer = MusicTokenizer(max_vocab_size)
    tokenizer.build_vocabulary(piano_roll_segments, min_frequency)
    
    # Tokenize all segments
    tokenized_sequences = []
    for segment in piano_roll_segments:
        tokens = tokenizer.encode(segment)
        tokenized_sequences.append(tokens)
    
    return tokenizer, tokenized_sequences