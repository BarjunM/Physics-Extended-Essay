"""
Basic tests for the stellar spectral classification pipeline.

These tests verify that the main components work correctly.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from download_data import create_synthetic_data, extract_spectral_type
from preprocess import compute_color_features, prepare_features_and_labels, split_and_scale_data
from model import build_model


def test_extract_spectral_type():
    """Test spectral type extraction from subclass."""
    assert extract_spectral_type('G5') == 'G'
    assert extract_spectral_type('M2') == 'M'
    assert extract_spectral_type('A0') == 'A'
    assert extract_spectral_type('') == 'Unknown'
    assert extract_spectral_type(None) == 'Unknown'
    print("✓ test_extract_spectral_type passed")


def test_synthetic_data_generation():
    """Test synthetic data generation."""
    df = create_synthetic_data('/tmp/test_stars.csv', n_samples=100)
    
    assert len(df) == 100, "Should generate 100 samples"
    assert 'spectral_type' in df.columns, "Should have spectral_type column"
    assert 'u' in df.columns and 'z' in df.columns, "Should have ugriz magnitudes"
    
    # Check spectral types are valid
    valid_types = set(['O', 'B', 'A', 'F', 'G', 'K', 'M'])
    assert set(df['spectral_type'].unique()).issubset(valid_types), "Should only have valid spectral types"
    
    print("✓ test_synthetic_data_generation passed")


def test_color_features():
    """Test color feature computation."""
    # Create simple test data
    data = {
        'u': [20.0, 19.0],
        'g': [19.0, 18.5],
        'r': [18.5, 18.0],
        'i': [18.0, 17.5],
        'z': [17.5, 17.0]
    }
    df = pd.DataFrame(data)
    
    df_with_colors = compute_color_features(df)
    
    # Check that color features are computed correctly
    assert 'u-g' in df_with_colors.columns, "Should have u-g color"
    assert 'g-r' in df_with_colors.columns, "Should have g-r color"
    
    # Check values
    assert np.isclose(df_with_colors.iloc[0]['u-g'], 1.0), "u-g should be 1.0 for first row"
    assert np.isclose(df_with_colors.iloc[0]['g-r'], 0.5), "g-r should be 0.5 for first row"
    
    print("✓ test_color_features passed")


def test_feature_preparation():
    """Test feature and label preparation."""
    df = create_synthetic_data('/tmp/test_stars2.csv', n_samples=200)
    df = compute_color_features(df)
    
    X, y, feature_names, label_encoder = prepare_features_and_labels(df)
    
    assert X.shape[0] == len(df), "Number of samples should match"
    assert X.shape[1] == len(feature_names), "Number of features should match feature_names"
    assert len(y) == len(df), "Number of labels should match"
    assert len(label_encoder.classes_) <= 7, "Should have at most 7 classes"
    
    # Check y is properly encoded as integers
    assert y.dtype in [np.int32, np.int64], "Labels should be integers"
    assert y.min() >= 0, "Labels should be non-negative"
    assert y.max() < len(label_encoder.classes_), "Labels should be valid indices"
    
    print("✓ test_feature_preparation passed")


def test_data_splitting():
    """Test data splitting and scaling."""
    # Create random data
    X = np.random.randn(100, 13)
    y = np.random.randint(0, 7, 100)
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale_data(
        X, y, test_size=0.2, val_size=0.1
    )
    
    # Check sizes
    assert len(X_train) + len(X_val) + len(X_test) == 100, "All data should be split"
    assert len(X_test) == 20, "Test set should be 20% (20 samples)"
    
    # Check scaling (mean should be close to 0, std close to 1)
    assert np.abs(X_train.mean()) < 0.5, "Scaled training data should have mean close to 0"
    assert np.abs(X_train.std() - 1.0) < 0.5, "Scaled training data should have std close to 1"
    
    print("✓ test_data_splitting passed")


def test_model_building():
    """Test model building."""
    model = build_model(input_dim=13, n_classes=7, hidden_layers=[64, 32])
    
    # Check model structure
    assert model.input_shape[1] == 13, "Input dimension should be 13"
    assert model.output_shape[1] == 7, "Output dimension should be 7"
    
    # Check model can make predictions
    X_dummy = np.random.randn(10, 13)
    predictions = model.predict(X_dummy, verbose=0)
    
    assert predictions.shape == (10, 7), "Predictions should have correct shape"
    assert np.allclose(predictions.sum(axis=1), 1.0), "Softmax outputs should sum to 1"
    
    print("✓ test_model_building passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60 + "\n")
    
    test_extract_spectral_type()
    test_synthetic_data_generation()
    test_color_features()
    test_feature_preparation()
    test_data_splitting()
    test_model_building()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60 + "\n")


if __name__ == '__main__':
    run_all_tests()
