# src/preprocessing/midi_parser.py
import os
import glob
from pathlib import Path

class MIDIParser:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.midi_files = []
    
    def scan_all_midi_files(self, extension='.midi'):
        """Find all .midi files in MAESTRO folder structure"""
        pattern = str(self.root_path / '**' / f'*{extension}')
        self.midi_files = glob.glob(pattern, recursive=True)
        return self.midi_files
    
    def get_year_from_path(self, file_path):
        """Extract year from folder name (2004, 2006, etc.)"""
        parts = Path(file_path).parts
        for part in parts:
            if part.isdigit() and len(part) == 4:
                return int(part)
        return None
    
    def split_by_year(self, train_years=None, val_years=None, test_years=None):
        """
        Split by year (recommended for temporal generalization)
        Default: train on older years, test on newer years
        """
        if train_years is None:
            train_years = [2004, 2006, 2008, 2009, 2011]
        if val_years is None:
            val_years = [2013, 2014]
        if test_years is None:
            test_years = [2015, 2017, 2018]
        
        train_files = []
        val_files = []
        test_files = []
        
        for f in self.midi_files:
            year = self.get_year_from_path(f)
            if year in train_years:
                train_files.append(f)
            elif year in val_years:
                val_files.append(f)
            elif year in test_years:
                test_files.append(f)
        
        return train_files, val_files, test_files