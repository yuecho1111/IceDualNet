"""
Example: Training and Using ICESat Classifier

This script demonstrates:
1. Training the model on ICESat-2 data files
2. Saving the trained model
3. Loading and making predictions
4. Evaluating performance
"""

import os
from pathlib import Path
from icesat import ICESatClassifierNoLeak
from sklearn.metrics import f1_score, accuracy_score


def example_train():
    """Example: Train a classifier on ICESat-2 data."""
    
    print("=" * 60)
    print("Example: Training ICESat Classifier")
    print("=" * 60)
    
    # Path to training data
    train_dir = Path("data/ATL07_split_by_strength")
    
    # Collect training files
    train_files = []
    if train_dir.exists():
        for subfolder in ["strong", "weak"]:
            folder = train_dir / subfolder
            if folder.exists():
                files = list(folder.glob("*.csv"))
                train_files.extend([str(f) for f in files])
    
    if not train_files:
        print("No training files found. Please check the data directory.")
        return
    
    print(f"\nFound {len(train_files)} training files")
    
    # Initialize classifier
    print("\nInitializing classifier...")
    clf = ICESatClassifierNoLeak(
        input_dim=7,
        sequence_length=7,
        hidden_dim=128,
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=64,
        epochs=20,
        use_season_mask=True,
        verbose=True
    )
    
    # Train model
    print("\nStarting training...")
    clf.fit(train_files)
    
    # Save model
    model_path = "model_trained.pth"
    print(f"\nSaving model to {model_path}...")
    clf.save(model_path)
    print("✓ Model saved successfully")
    
    return clf, model_path


def example_predict(clf=None, model_path=None):
    """Example: Load model and make predictions."""
    
    print("\n" + "=" * 60)
    print("Example: Making Predictions")
    print("=" * 60)
    
    # Load model if not provided
    if clf is None:
        if model_path is None:
            model_path = "model_trained.pth"
        
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found.")
            return
        
        print(f"\nLoading model from {model_path}...")
        clf = ICESatClassifierNoLeak(verbose=True)
        clf.load(model_path)
        print("✓ Model loaded successfully")
    
    # Path to test data
    test_dir = Path("data/ATL07_split_by_strength_test")
    
    # Collect test files
    test_files = []
    if test_dir.exists():
        for subfolder in ["strong", "weak"]:
            folder = test_dir / subfolder
            if folder.exists():
                files = list(folder.glob("*.csv"))
                test_files.extend([str(f) for f in files[:2]])  # Use first 2 files
    
    if not test_files:
        print("No test files found. Please check the data directory.")
        return
    
    print(f"\nFound {len(test_files)} test files")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = clf.predict(test_files)
    probabilities = clf.predict_proba(test_files)
    
    print(f"\nResults:")
    print(f"  Total samples: {len(predictions)}")
    print(f"  Positive predictions: {predictions.sum()}")
    print(f"  Negative predictions: {(1-predictions).sum()}")
    print(f"  Mean probability: {probabilities[:, 1].mean():.4f}")
    
    return clf, predictions, probabilities


def example_evaluate():
    """Example: Evaluate model performance."""
    
    print("\n" + "=" * 60)
    print("Example: Model Evaluation")
    print("=" * 60)
    
    model_path = "model_trained.pth"
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Train a model first.")
        return
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    clf = ICESatClassifierNoLeak(verbose=True)
    clf.load(model_path)
    
    # Load test data
    test_dir = Path("data/ATL07_split_by_strength_test")
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Collect test files
    test_files = []
    labels = []
    
    for subfolder in ["strong", "weak"]:
        folder = test_dir / subfolder
        if folder.exists():
            for csv_file in folder.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    if "classify" in df.columns:
                        test_files.append(str(csv_file))
                        labels.extend(df["classify"].values)
                except:
                    pass
    
    if not test_files:
        print("No test files with labels found.")
        return
    
    print(f"\nEvaluating on {len(test_files)} files...")
    
    # Make predictions
    predictions = clf.predict(test_files)
    
    # Evaluate (assuming equal sample counts)
    if len(predictions) == len(labels):
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 Score: {f1:.4f}")


if __name__ == "__main__":
    # Uncomment to run examples
    
    # Example 1: Train a model
    # clf, model_path = example_train()
    
    # Example 2: Load model and make predictions
    # clf, predictions, probs = example_predict()
    
    # Example 3: Evaluate model
    # example_evaluate()
    
    print("Uncomment examples in __main__ to run them.")
    print("\nExample code available in this file.")
