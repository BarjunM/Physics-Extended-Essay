"""
Data preprocessing module for stellar spectral classification.

This module handles loading, cleaning, and feature engineering for the stellar data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath='data/sdss_stars.csv'):
    """
    Load stellar data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} samples from {filepath}")
    return df


def compute_color_features(df):
    """
    Compute color-based features from ugriz magnitudes.
    
    Color indices are differences between adjacent filters and are the primary
    features used for stellar classification.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with u, g, r, i, z magnitude columns
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional color index columns
    """
    df = df.copy()
    
    # Basic color indices (differences between adjacent bands)
    df['u-g'] = df['u'] - df['g']
    df['g-r'] = df['g'] - df['r']
    df['r-i'] = df['r'] - df['i']
    df['i-z'] = df['i'] - df['z']
    
    # Additional useful color indices
    df['u-r'] = df['u'] - df['r']
    df['g-i'] = df['g'] - df['i']
    df['r-z'] = df['r'] - df['z']
    df['u-i'] = df['u'] - df['i']
    
    # Include individual magnitudes as features
    # (they provide information about absolute brightness)
    
    print("Computed color-based features")
    return df


def prepare_features_and_labels(df, feature_cols=None):
    """
    Prepare feature matrix X and label vector y from the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with computed features and spectral_type column
    feature_cols : list, optional
        List of column names to use as features. If None, uses default colors + magnitudes
        
    Returns:
    --------
    tuple
        (X, y, feature_names, label_encoder)
        - X: numpy array of features
        - y: numpy array of encoded labels
        - feature_names: list of feature column names
        - label_encoder: fitted LabelEncoder object
    """
    if feature_cols is None:
        # Use color indices and magnitudes as features
        feature_cols = [
            'u-g', 'g-r', 'r-i', 'i-z',  # Basic colors
            'u-r', 'g-i', 'r-z', 'u-i',  # Additional colors
            'u', 'g', 'r', 'i', 'z'       # Raw magnitudes
        ]
    
    # Remove any rows with missing values in features or labels
    original_size = len(df)
    df_clean = df.dropna(subset=feature_cols + ['spectral_type'])
    dropped_rows = original_size - len(df_clean)
    
    if dropped_rows > 0:
        print(f"Warning: Dropped {dropped_rows} rows ({100*dropped_rows/original_size:.1f}%) due to missing values")
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    print(f"Clean dataset size: {len(df_clean)} samples")
    
    # Extract features
    X = df_clean[feature_cols].values
    
    # Encode labels (spectral types) as integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_clean['spectral_type'])
    
    print(f"Label encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    print(f"Class distribution:")
    for cls, count in zip(*np.unique(y, return_counts=True)):
        print(f"  {label_encoder.classes_[cls]}: {count} ({100*count/len(y):.1f}%)")
    
    return X, y, feature_cols, label_encoder


def split_and_scale_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split data into train/validation/test sets and scale features.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Label vector
    test_size : float
        Proportion of data to use for test set
    val_size : float
        Proportion of training data to use for validation set
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Second split: separate validation set from training set
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    # Scale features using StandardScaler (mean=0, std=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"\nData split:")
    print(f"  Training set: {len(X_train)} samples ({100*len(X_train)/len(X):.1f}%)")
    print(f"  Validation set: {len(X_val)} samples ({100*len(X_val)/len(X):.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({100*len(X_test)/len(X):.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def preprocess_pipeline(filepath='data/sdss_stars.csv', test_size=0.2, val_size=0.1):
    """
    Complete preprocessing pipeline: load, compute features, prepare, split, and scale.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV data file
    test_size : float
        Proportion for test set
    val_size : float
        Proportion for validation set
        
    Returns:
    --------
    dict
        Dictionary containing all preprocessed data and metadata
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Load data
    df = load_data(filepath)
    
    # Compute color features
    df = compute_color_features(df)
    
    # Prepare features and labels
    X, y, feature_names, label_encoder = prepare_features_and_labels(df)
    
    # Split and scale
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale_data(
        X, y, test_size=test_size, val_size=val_size
    )
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'n_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_
    }


if __name__ == '__main__':
    # Test the preprocessing pipeline
    data = preprocess_pipeline()
    print("\nPreprocessed data shapes:")
    print(f"  X_train: {data['X_train'].shape}")
    print(f"  X_val: {data['X_val'].shape}")
    print(f"  X_test: {data['X_test'].shape}")
    print(f"\nNumber of classes: {data['n_classes']}")
    print(f"Class names: {data['class_names']}")
