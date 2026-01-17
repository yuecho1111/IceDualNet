"""
ICESat sea ice classifier using dual-path neural network.

Provides the main ICESatClassifierNoLeak class for training and inference.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader as TorchDataLoader
from typing import List, Optional, Tuple
import warnings

from .models import DualPathNetwork
from .data import DataPreprocessor, DataLoader as CustomDataLoader, FastSequenceDataset

warnings.filterwarnings('ignore')


class ICESatClassifierNoLeak(BaseEstimator, ClassifierMixin):
    """Sea ice classification model with no data leakage.
    
    This classifier:
    - Processes files as atomic units (no leakage across files)
    - Uses dual-path neural network architecture
    - Implements season-aware feature masking
    - Supports model save/load functionality
    """
    
    def __init__(self, input_dim: int = 7, sequence_length: int = 7, 
                 hidden_dim: int = 128, dropout_rate: float = 0.3,
                 learning_rate: float = 0.001, batch_size: int = 64,
                 epochs: int = 20, device: Optional[torch.device] = None,
                 verbose: bool = False, random_state: int = 42,
                 use_season_mask: bool = True):
        """Initialize classifier.
        
        Args:
            input_dim: Number of input features (default: 7)
            sequence_length: Length of input sequences (default: 7)
            hidden_dim: Hidden layer dimension (default: 128)
            dropout_rate: Dropout rate (default: 0.3)
            learning_rate: Adam optimizer learning rate (default: 0.001)
            batch_size: Training batch size (default: 64)
            epochs: Number of training epochs (default: 20)
            device: torch device (auto-detect if None)
            verbose: Print training information
            random_state: Random seed
            use_season_mask: Apply seasonal masking to background features
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.random_state = random_state
        self.use_season_mask = use_season_mask

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.preprocessor = DataPreprocessor(use_season_mask=use_season_mask, verbose=verbose)
        self.data_loader = CustomDataLoader(self.preprocessor, sequence_length, verbose)
        
        self.model: Optional[DualPathNetwork] = None
        self.best_threshold_ = 0.5
        self.stats_ = {
            'total_files_processed': 0,
            'files_skipped': 0,
            'total_samples': 0
        }

    def fit(self, X: List[str], y=None) -> 'ICESatClassifierNoLeak':
        """Train the model.
        
        Args:
            X: List of file paths to training data
            y: Ignored (for sklearn compatibility)
            
        Returns:
            self
        """
        if not isinstance(X, list) or not all(isinstance(x, str) for x in X):
            raise ValueError("X should be a list of file paths")

        # Clear cache for fresh training
        self.preprocessor.file_data_cache = {}
        self.preprocessor.scaler_fitted = False

        # Split files (not samples) for training/validation
        from sklearn.model_selection import train_test_split
        train_files, val_files = train_test_split(
            X, test_size=0.2, random_state=self.random_state
        )

        # Load training data
        X_train, y_train = self.data_loader.load_files_without_leakage(
            train_files, fit_scaler=True
        )
        if X_train is None or len(X_train) == 0:
            raise ValueError(f"No valid training data found in {len(train_files)} files")

        # Load validation data
        X_val, y_val = self.data_loader.load_files_without_leakage(
            val_files, fit_scaler=False
        )
        if X_val is None or len(X_val) == 0:
            if self.verbose:
                print("Warning: No validation data, using training data")
            X_val, y_val = X_train, y_train

        if self.verbose:
            print(f"Training: {len(X_train)} samples, Validation: {len(X_val)} samples")

        # Prepare data loaders
        train_loader = TorchDataLoader(
            FastSequenceDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = TorchDataLoader(
            FastSequenceDataset(X_val, y_val),
            batch_size=self.batch_size * 2,
            shuffle=False
        )

        # Initialize model
        self.model = DualPathNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            sequence_length=self.sequence_length,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # Setup training
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )

        # Class weights for imbalanced data
        num_pos = y_train.sum()
        num_neg = len(y_train) - num_pos
        pos_weight = torch.tensor(num_neg / (num_pos + 1e-6), dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_f1 = 0
        patience = 5
        counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0
            for seqs, lbls in train_loader:
                seqs, lbls = seqs.to(self.device), lbls.to(self.device)
                optimizer.zero_grad()
                out = self.model(seqs)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0
            all_probs, all_targets = [], []
            with torch.no_grad():
                for seqs, lbls in val_loader:
                    seqs, lbls = seqs.to(self.device), lbls.to(self.device)
                    out = self.model(seqs)
                    val_loss += criterion(out, lbls).item()
                    probs = torch.sigmoid(out)
                    all_probs.extend(probs.cpu().numpy())
                    all_targets.extend(lbls.cpu().numpy())

            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            # Compute metrics
            all_probs = np.array(all_probs)
            all_targets = np.array(all_targets)

            if len(np.unique(all_targets)) < 2:
                current_threshold = 0.5
                val_f1 = 0.0
            else:
                current_threshold = self._find_optimal_threshold(all_targets, all_probs)
                predictions = (all_probs >= current_threshold).astype(int)
                val_f1 = f1_score(all_targets, predictions, zero_division=0)

            if self.verbose and epoch % 2 == 0:
                print(f"Epoch {epoch}: Loss={train_loss:.4f}, Val_Loss={val_loss:.4f}, "
                      f"F1={val_f1:.4f}, Threshold={current_threshold:.3f}")

            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.best_threshold_ = current_threshold
                best_model_state = self.model.state_dict().copy()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        if self.verbose:
            print(f"Training completed. Best F1: {best_val_f1:.4f}, Threshold: {self.best_threshold_:.3f}")

        return self

    def predict(self, X: List[str]) -> np.ndarray:
        """Predict labels for files.
        
        Args:
            X: List of file paths
            
        Returns:
            Predicted labels (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if not isinstance(X, list) or not all(isinstance(x, str) for x in X):
            raise ValueError("X should be a list of file paths")

        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.best_threshold_).astype(int)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict probabilities for files.
        
        Args:
            X: List of file paths
            
        Returns:
            Probability array of shape (n_samples, 2)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if not isinstance(X, list) or not all(isinstance(x, str) for x in X):
            raise ValueError("X should be a list of file paths")

        X_data, _ = self.data_loader.load_files_without_leakage(X, fit_scaler=False)
        if X_data is None or len(X_data) == 0:
            return np.zeros((0, 2))

        dataset = FastSequenceDataset(X_data, np.zeros(len(X_data)))
        loader = TorchDataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False)

        self.model.eval()
        probs = []
        with torch.no_grad():
            for seqs, _ in loader:
                seqs = seqs.to(self.device)
                out = self.model(seqs)
                batch_probs = torch.sigmoid(out).cpu().numpy()
                probs.extend(batch_probs)

        probs = np.array(probs)
        return np.vstack([1 - probs, probs]).T

    def save(self, path: str) -> None:
        """Save model to file.
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        state = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.preprocessor.scaler,
            'best_threshold': self.best_threshold_,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'sequence_length': self.sequence_length,
            'dropout_rate': self.dropout_rate,
            'use_season_mask': self.use_season_mask
        }
        torch.save(state, path)
        if self.verbose:
            print(f"Model saved to {path}")

    def load(self, path: str) -> 'ICESatClassifierNoLeak':
        """Load model from file.
        
        Args:
            path: Path to model file
            
        Returns:
            self
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        state = torch.load(path, map_location=self.device)

        self.input_dim = state['input_dim']
        self.hidden_dim = state['hidden_dim']
        self.sequence_length = state['sequence_length']
        self.dropout_rate = state['dropout_rate']
        self.use_season_mask = state.get('use_season_mask', True)
        self.best_threshold_ = state['best_threshold']
        self.preprocessor.scaler = state['scaler']
        self.preprocessor.scaler_fitted = True

        self.model = DualPathNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            sequence_length=self.sequence_length,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()

        if self.verbose:
            print(f"Model loaded from {path}")

        return self

    def _find_optimal_threshold(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """Find threshold that maximizes F1 score.
        
        Args:
            y_true: True labels
            y_probs: Predicted probabilities
            
        Returns:
            Optimal threshold
        """
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.1, 0.9, 0.01):
            f1 = f1_score(y_true, (y_probs >= threshold).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold
