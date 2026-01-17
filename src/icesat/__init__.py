"""
ICESat Sea Ice Classification Package.

A deep learning framework for classifying sea ice using ICESat-2 data
with no data leakage across train/test splits.

Main components:
- DualPathNetwork: Neural network architecture
- ICESatClassifierNoLeak: Main classifier with no leakage
- DataPreprocessor: Data loading and preprocessing
"""

from .models import DualPathNetwork
from .data import DataPreprocessor, DataLoader, FastSequenceDataset
from .classifier import ICESatClassifierNoLeak

__version__ = "1.0.0"
__author__ = "ICESat Research Team"
__all__ = [
    'DualPathNetwork',
    'ICESatClassifierNoLeak',
    'DataPreprocessor',
    'DataLoader',
    'FastSequenceDataset'
]
