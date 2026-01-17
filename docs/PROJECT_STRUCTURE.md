# Project Structure

## Architecture

`
src/icesat/
 models.py           # DualPathNetwork
 classifier.py       # ICeSatClassifierNoLeak
 data.py             # Data loading
 __init__.py         # Exports
`

## Key Components

### models.py
- DualPathNetwork: Neural network architecture with local and context paths

### classifier.py
- ICeSatClassifierNoLeak: Main API for training and prediction
- No data leakage: Files processed independently

### data.py
- DataPreprocessor: Feature extraction and preprocessing
- DataLoader: File loading without leakage
- FastSequenceDataset: PyTorch dataset for sequences

## Design Principles

1. **No Data Leakage**: Each file is processed independently
2. **Seasonal Masking**: Background rate masked May-August
3. **Production Ready**: Type hints, error handling, documentation
4. **Simple API**: Easy to use for training and prediction
