"""
Main training script for stellar spectral classification.

This script orchestrates the complete pipeline: data download, preprocessing,
model training, and evaluation.
"""

import os
import sys
import argparse
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download_data import download_sdss_sample_data, create_synthetic_data
from preprocess import preprocess_pipeline
from model import build_model, train_model, save_model_info
from evaluate import (
    evaluate_model, plot_confusion_matrix, plot_learning_curves,
    plot_class_performance, create_evaluation_report
)


def main(args):
    """
    Main training pipeline.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    print("\n" + "=" * 70)
    print("STELLAR SPECTRAL CLASSIFICATION USING NEURAL NETWORKS")
    print("=" * 70 + "\n")
    
    # Step 1: Download/load data
    print("Step 1: Loading data...")
    if not os.path.exists(args.data_path):
        print(f"Data file not found at {args.data_path}")
        if args.download:
            print("Downloading data...")
            download_sdss_sample_data(args.data_path, args.sample_size)
        else:
            print("Creating synthetic data...")
            create_synthetic_data(args.data_path, args.sample_size)
    else:
        print(f"Using existing data from {args.data_path}")
    
    # Step 2: Preprocess data
    print("\nStep 2: Preprocessing data...")
    data = preprocess_pipeline(
        filepath=args.data_path,
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    # Step 3: Build model
    print("\nStep 3: Building neural network model...")
    model = build_model(
        input_dim=data['X_train'].shape[1],
        n_classes=data['n_classes'],
        hidden_layers=args.hidden_layers,
        dropout_rate=args.dropout
    )
    
    print("\nModel architecture:")
    model.summary()
    print(f"\nTotal trainable parameters: {model.count_params():,}")
    
    # Step 4: Train model
    print("\nStep 4: Training model...")
    history = train_model(
        model,
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Step 5: Save model and metadata
    print("\nStep 5: Saving model and metadata...")
    save_model_info(
        model,
        data['feature_names'],
        data['label_encoder'],
        data['scaler'],
        save_dir=args.model_dir
    )
    
    # Step 6: Evaluate model
    print("\nStep 6: Evaluating model on test set...")
    metrics = evaluate_model(
        model,
        data['X_test'],
        data['y_test'],
        data['class_names']
    )
    
    # Step 7: Create visualizations
    print("\nStep 7: Creating visualizations...")
    
    # Learning curves
    plot_learning_curves(history, save_path=os.path.join(args.output_dir, 'learning_curves.png'))
    
    # Confusion matrix
    plot_confusion_matrix(
        metrics['y_true'],
        metrics['y_pred'],
        data['class_names'],
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Per-class performance
    plot_class_performance(
        metrics,
        data['class_names'],
        save_path=os.path.join(args.output_dir, 'class_performance.png')
    )
    
    # Step 8: Create evaluation report
    print("\nStep 8: Creating evaluation report...")
    create_evaluation_report(
        metrics,
        data['class_names'],
        save_path=os.path.join(args.output_dir, 'evaluation_report.txt')
    )
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to:")
    print(f"  - Model: {args.model_dir}/best_model.keras")
    print(f"  - Visualizations: {args.output_dir}/")
    print(f"  - Report: {args.output_dir}/evaluation_report.txt")
    print(f"\nFinal Test Accuracy: {metrics['accuracy']:.4f}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a neural network for stellar spectral classification'
    )
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='data/sdss_stars.csv',
                       help='Path to the data CSV file')
    parser.add_argument('--download', action='store_true',
                       help='Download real SDSS data (requires internet)')
    parser.add_argument('--sample-size', type=int, default=10000,
                       help='Number of samples to download/generate')
    
    # Preprocessing arguments
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for test set')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Proportion of training data for validation set')
    
    # Model arguments
    parser.add_argument('--hidden-layers', type=int, nargs='+', default=[128, 64, 32],
                       help='Sizes of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate for regularization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    
    # Output arguments
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory to save the trained model')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results and visualizations')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the pipeline
    main(args)
