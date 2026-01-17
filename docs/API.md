# API Reference

## ICeSatClassifierNoLeak

Main classifier class.

### Parameters
```python
ICeSatClassifierNoLeak(
    input_dim=7,           # Input features
    sequence_length=7,     # Window size
    hidden_dim=128,        # Hidden dimension
    dropout_rate=0.3,      # Dropout
    learning_rate=0.001,   # Learning rate
    batch_size=64,         # Batch size
    epochs=20,             # Training epochs
    use_season_mask=True,  # Seasonal masking
    verbose=False          # Print info
)
```

### Methods

#### fit(train_files)
Train on list of CSV file paths.

#### predict(test_files)
Returns binary predictions (0 or 1).

#### predict_proba(test_files)
Returns class probabilities of shape (n_samples, 2).

#### save(path)
Save trained model to .pth file.

#### load(path)
Load trained model from .pth file.

## DualPathNetwork

Neural network with dual processing paths.

Two parallel paths:
- **Local Path**: Dense layers on center point
- **Context Path**: Conv1D layers on full sequence
- **Fusion**: Combines both for prediction

## DataPreprocessor

Handles data loading and preprocessing.

```python
preprocessor = DataPreprocessor(use_season_mask=True)
features, labels = preprocessor.load_and_preprocess(files)
```
