#!/usr/bin/env python3
"""
Quick start example for stellar spectral classification.

This script demonstrates how to:
1. Generate synthetic data
2. Train a classifier
3. Make predictions on new data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from download_data import create_synthetic_data
from preprocess import preprocess_pipeline
from model import build_model, train_model, save_model_info
from evaluate import evaluate_model
import numpy as np

def main():
    print("\n" + "="*70)
    print("STELLAR SPECTRAL CLASSIFICATION - QUICK START EXAMPLE")
    print("="*70 + "\n")
    
    # Step 1: Generate data
    print("Step 1: Generating synthetic stellar data...")
    data_path = 'data/sdss_stars.csv'
    df = create_synthetic_data(data_path, n_samples=2000)
    
    # Step 2: Preprocess
    print("\nStep 2: Preprocessing and feature engineering...")
    data = preprocess_pipeline(filepath=data_path)
    
    # Step 3: Build model
    print("\nStep 3: Building neural network...")
    model = build_model(
        input_dim=data['X_train'].shape[1],
        n_classes=data['n_classes'],
        hidden_layers=[64, 32],
        dropout_rate=0.3
    )
    
    # Step 4: Train (fewer epochs for quick demo)
    print("\nStep 4: Training model (quick mode - 20 epochs)...")
    history = train_model(
        model,
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        epochs=20,
        batch_size=32
    )
    
    # Step 5: Evaluate
    print("\nStep 5: Evaluating on test set...")
    metrics = evaluate_model(
        model,
        data['X_test'],
        data['y_test'],
        data['class_names']
    )
    
    # Step 6: Make predictions on a few examples
    print("\nStep 6: Example predictions...")
    print("\nShowing 10 random predictions:\n")
    print(f"{'True Type':<12} {'Predicted':<12} {'Confidence':<12} {'Status'}")
    print("-" * 55)
    
    indices = np.random.choice(len(data['X_test']), 10, replace=False)
    for idx in indices:
        X_sample = data['X_test'][idx:idx+1]
        y_true = data['y_test'][idx]
        
        pred_proba = model.predict(X_sample, verbose=0)[0]
        y_pred = np.argmax(pred_proba)
        confidence = pred_proba[y_pred]
        
        true_class = data['class_names'][y_true]
        pred_class = data['class_names'][y_pred]
        status = "✓ Correct" if y_true == y_pred else "✗ Wrong"
        
        print(f"{true_class:<12} {pred_class:<12} {confidence:<12.4f} {status}")
    
    print("\n" + "="*70)
    print(f"FINAL ACCURACY: {metrics['accuracy']:.2%}")
    print("="*70 + "\n")
    
    print("For full training with more data and epochs, run:")
    print("  python src/train.py --sample-size 10000 --epochs 100\n")


if __name__ == '__main__':
    main()
