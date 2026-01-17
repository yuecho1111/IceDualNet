"""Unit tests for ICESat classification package."""

import pytest
import numpy as np
import torch
import tempfile
import os
from pathlib import Path

from icesat.models import DualPathNetwork
from icesat.data import DataPreprocessor, FastSequenceDataset


class TestDualPathNetwork:
    """Test cases for DualPathNetwork model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = DualPathNetwork(input_dim=7, hidden_dim=128, sequence_length=7)
        assert model.input_dim == 7
        assert model.hidden_dim == 128
        assert model.sequence_length == 7
    
    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        model = DualPathNetwork(input_dim=7, hidden_dim=128, sequence_length=7)
        
        # Create dummy input: (batch_size=4, seq_len=7, features=7)
        x = torch.randn(4, 7, 7)
        
        # Forward pass
        output = model(x)
        
        # Check output shape: (batch_size,)
        assert output.shape == (4,)
        assert torch.isfinite(output).all()
    
    def test_different_batch_sizes(self):
        """Test model with different batch sizes."""
        model = DualPathNetwork()
        
        for batch_size in [1, 8, 32, 64]:
            x = torch.randn(batch_size, 7, 7)
            output = model(x)
            assert output.shape == (batch_size,)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        model = DualPathNetwork()
        x = torch.randn(4, 7, 7, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestDataPreprocessor:
    """Test cases for DataPreprocessor."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor(use_season_mask=True, verbose=False)
        assert preprocessor.use_season_mask is True
        assert preprocessor.verbose is False
    
    def test_feature_ranges(self):
        """Test feature range definitions."""
        preprocessor = DataPreprocessor()
        
        # Check all 7 features have ranges defined
        assert len(preprocessor.FEATURE_RANGES) == 7
        
        # Check range values are sensible
        for feat_idx, (min_v, max_v) in preprocessor.FEATURE_RANGES.items():
            assert min_v < max_v
            assert isinstance(min_v, (int, float))
            assert isinstance(max_v, (int, float))
    
    def test_seasonal_masking(self):
        """Test seasonal masking logic."""
        preprocessor = DataPreprocessor(use_season_mask=True)
        
        # Create dummy dataframe
        import pandas as pd
        df = pd.DataFrame({
            'background_r_norm': [1000, 2000, 3000]
        })
        
        # Test summer date (August)
        preprocessor._apply_seasonal_mask(df, "S2_20190815.csv")
        # Should be masked (set to 0) in summer
        assert all(df['background_r_norm'] == 0.0)
        
        # Test winter date (January)
        df['background_r_norm'] = [1000, 2000, 3000]
        preprocessor._apply_seasonal_mask(df, "S2_20190115.csv")
        # Should NOT be masked in winter
        assert all(df['background_r_norm'] == [1000, 2000, 3000])


class TestFastSequenceDataset:
    """Test cases for FastSequenceDataset."""
    
    def test_initialization(self):
        """Test dataset initialization."""
        features = np.random.randn(100, 7, 7).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(np.float32)
        
        dataset = FastSequenceDataset(features, labels)
        
        assert len(dataset) == 100
    
    def test_getitem(self):
        """Test getting items from dataset."""
        features = np.random.randn(100, 7, 7).astype(np.float32)
        labels = np.random.randint(0, 2, 100).astype(np.float32)
        
        dataset = FastSequenceDataset(features, labels)
        
        # Get first item
        feat, label = dataset[0]
        
        assert feat.shape == (7, 7)
        assert label.shape == ()
        assert isinstance(feat, torch.Tensor)
        assert isinstance(label, torch.Tensor)
    
    def test_tensor_conversion(self):
        """Test that data is properly converted to tensors."""
        features = np.arange(100).reshape(10, 5, 2).astype(np.float32)
        labels = np.arange(10).astype(np.float32)
        
        dataset = FastSequenceDataset(features, labels)
        
        for i in range(10):
            feat, label = dataset[i]
            assert isinstance(feat, torch.FloatTensor)
            assert isinstance(label, torch.FloatTensor)


def test_imports():
    """Test that all major classes can be imported."""
    from icesat import (
        DualPathNetwork,
        ICESatClassifierNoLeak,
        DataPreprocessor,
        DataLoader,
        FastSequenceDataset
    )
    
    assert DualPathNetwork is not None
    assert ICESatClassifierNoLeak is not None
    assert DataPreprocessor is not None
    assert DataLoader is not None
    assert FastSequenceDataset is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
