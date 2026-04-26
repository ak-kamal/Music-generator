import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class MusicDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)
        print(f"Loaded {len(self.data)} segments from {data_path}")
        print(f"Shape: {self.data.shape}, Memory: {self.data.nbytes / 1024 / 1024:.1f} MB")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        segment = self.data[idx]
        # Add channel dimension: (time, pitches) -> (1, time, pitches)
        segment = torch.FloatTensor(segment).unsqueeze(0)
        return segment

def get_data_loaders(batch_size=32, data_dir='../data/train_test_split/'):
    train_dataset = MusicDataset(os.path.join(data_dir, 'train.npy'))
    val_dataset = MusicDataset(os.path.join(data_dir, 'val.npy'))
    test_dataset = MusicDataset(os.path.join(data_dir, 'test.npy'))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    return train_loader, val_loader, test_loader