"""
Data processing and loading module for ICESat classification.

This module provides utilities for preprocessing ICESat data, including:
- Loading and validating CSV data files
- Feature scaling and normalization
- Sequence generation with no data leakage
- Cloud-based feature masking
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List


class FastSequenceDataset(Dataset):
    """Efficient PyTorch Dataset for sequence data.
    
    Converts numpy arrays to PyTorch tensors for training.
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """Initialize dataset.
        
        Args:
            features: Shape (N, seq_length, n_features)
            labels: Shape (N,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class DataPreprocessor:
    """Handles preprocessing of raw ICESat data files."""
    
    # Base feature names
    BASE_FEATURES = [
        'photon_density_per_m', 'photon_rate', 'height_segment_height',
        'height_segment_w_gaussian', 'asr_25', 'hist_w', 'background_r_norm'
    ]
    
    # Physical feature ranges for validation
    FEATURE_RANGES = {
        0: (0, 100),      # photon_density_per_m
        1: (0, 100),      # photon_rate
        2: (-10, 10),     # height_segment_height
        3: (0, 1),        # height_segment_w_gaussian
        4: (0, 10),       # asr_25
        5: (0, 10),       # hist_w
        6: (0, 1e8)       # background_r_norm
    }
    
    def __init__(self, use_season_mask: bool = True, verbose: bool = False):
        """Initialize preprocessor.
        
        Args:
            use_season_mask: If True, mask background_r_norm in summer (May-Aug)
            verbose: Print processing information
        """
        self.use_season_mask = use_season_mask
        self.verbose = verbose
        self.scaler = RobustScaler()
        self.scaler_fitted = False
        self.file_data_cache = {}
        
    def preprocess_single_file(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.Index]]:
        """Preprocess a single CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (features, labels, valid_indices) or (None, None, None) if invalid
        """
        try:
            # Read header to validate columns
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    header = f.readline().strip().split(',')
            except:
                return None, None, None

            # Check required columns
            required_cols = set(self.BASE_FEATURES + ['classify'])
            if not required_cols.issubset(header):
                if self.verbose:
                    print(f"Skipping {os.path.basename(file_path)}: Missing required columns")
                return None, None, None

            # Read data
            cols_to_use = self.BASE_FEATURES + ['classify']
            df = pd.read_csv(file_path, usecols=lambda c: c in cols_to_use, low_memory=False)

            # Apply seasonal masking for background_r_norm
            if self.use_season_mask:
                self._apply_seasonal_mask(df, file_path)

            # Convert to numeric
            for col in self.BASE_FEATURES:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Apply feature range filtering
            mask = pd.Series(True, index=df.index)
            for feat_idx, col_name in enumerate(self.BASE_FEATURES):
                if feat_idx in self.FEATURE_RANGES:
                    min_v, max_v = self.FEATURE_RANGES[feat_idx]
                    mask &= (df[col_name] >= min_v) & (df[col_name] <= max_v)

            df = df[mask]
            valid_indices = df.index

            # Clean NaNs
            df = df.dropna(subset=self.BASE_FEATURES + ['classify'])

            if len(df) == 0:
                return None, None, None

            # Extract features and labels
            feats = df[self.BASE_FEATURES].values.astype(np.float32)
            
            # Validate finite values
            if not np.isfinite(feats).all():
                return None, None, None

            # Apply cloud weighting if available
            if 'cloud_flag_asr' in df.columns:
                weights = 1.0 + (df['cloud_flag_asr'].fillna(0).values == 1) * 1.0
                feats *= weights[:, None]

            labels = df['classify'].values.astype(np.float32)

            return feats, labels, valid_indices

        except Exception as e:
            if self.verbose:
                print(f"Error processing {os.path.basename(file_path)}: {e}")
            return None, None, None

    def _apply_seasonal_mask(self, df: pd.DataFrame, file_path: str) -> None:
        """Apply seasonal masking to background_r_norm feature.
        
        Masks background_r_norm in summer (May-Aug) to reduce solar background noise.
        
        Args:
            df: DataFrame to modify in place
            file_path: Path to file (used to extract date)
        """
        filename = os.path.basename(file_path)
        # Match 8-digit numbers starting with 20 (YYYYMMDD)
        match = re.search(r'(20\d{2})(\d{2})(\d{2})', filename)
        if match:
            try:
                month = int(match.group(2))
                if 5 <= month <= 8:  # May to August
                    if 'background_r_norm' in df.columns:
                        df['background_r_norm'] = 0.0
            except ValueError:
                pass

    def create_sequences(self, feats: np.ndarray, labels: np.ndarray, 
                        sequence_length: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Create sliding window sequences.
        
        Args:
            feats: Features array (N, n_features)
            labels: Labels array (N,)
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (sequences, labels) or (None, None)
        """
        if feats is None or labels is None:
            return None, None

        if len(feats) < sequence_length:
            return None, None

        # Create sliding windows
        seq_wins = sliding_window_view(feats, window_shape=sequence_length, axis=0)
        
        # Align labels to window centers
        center_idx = sequence_length // 2
        label_wins = labels[center_idx: center_idx + len(seq_wins)]

        # Ensure matching lengths
        min_len = min(len(seq_wins), len(label_wins))
        if min_len == 0:
            return None, None

        return seq_wins[:min_len], label_wins[:min_len]


class DataLoader:
    """Loads data from files without leakage across train/test splits."""
    
    def __init__(self, preprocessor: DataPreprocessor, sequence_length: int = 7, verbose: bool = False):
        """Initialize data loader.
        
        Args:
            preprocessor: DataPreprocessor instance
            sequence_length: Length of sequences
            verbose: Print loading information
        """
        self.preprocessor = preprocessor
        self.sequence_length = sequence_length
        self.verbose = verbose
        
    def load_files_without_leakage(self, file_paths: List[str], 
                                   fit_scaler: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load data from multiple files without leakage.
        
        Args:
            file_paths: List of file paths
            fit_scaler: If True, fit scaler on this data
            
        Returns:
            Tuple of (X, y) or (None, None)
        """
        all_seqs = []
        all_labels = []

        iterator = tqdm(file_paths, desc="Loading files") if self.verbose else file_paths
        files_loaded = 0
        files_skipped = 0

        for fp in iterator:
            # Load from cache or disk
            if fp in self.preprocessor.file_data_cache:
                feats, labels = self.preprocessor.file_data_cache[fp]
            else:
                feats, labels, _ = self.preprocessor.preprocess_single_file(fp)
                if feats is not None:
                    self.preprocessor.file_data_cache[fp] = (feats, labels)

            if feats is None:
                files_skipped += 1
                continue

            # Create sequences
            seqs, lbls = self.preprocessor.create_sequences(feats, labels, self.sequence_length)

            if seqs is not None and len(seqs) > 0:
                all_seqs.append(seqs)
                all_labels.append(lbls)
                files_loaded += 1
            else:
                files_skipped += 1

        if not all_seqs:
            if self.verbose:
                print("Warning: No valid data found")
            return None, None

        # Fit scaler if needed
        if fit_scaler:
            all_features = []
            for seqs in all_seqs:
                N, S, F = seqs.shape
                all_features.append(seqs.reshape(-1, F))
            features = np.concatenate(all_features, axis=0)
            self.preprocessor.scaler.fit(features)
            self.preprocessor.scaler_fitted = True

        # Apply scaler
        scaled_seqs = []
        for seqs in all_seqs:
            N, S, F = seqs.shape
            seqs_flat = seqs.reshape(-1, F)
            if self.preprocessor.scaler_fitted:
                seqs_scaled = self.preprocessor.scaler.transform(seqs_flat).reshape(N, S, F)
            else:
                seqs_scaled = seqs.copy()
            scaled_seqs.append(seqs_scaled)

        X = np.concatenate(scaled_seqs, axis=0)
        y = np.concatenate(all_labels, axis=0)

        if self.verbose:
            print(f"Loaded {len(X)} samples from {files_loaded}/{len(file_paths)} files")

        return X, y
