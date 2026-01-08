"""
Model evaluation and visualization module.

This module provides functions for evaluating the trained model and creating
visualizations including confusion matrices, learning curves, and performance metrics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support
)


def evaluate_model(model, X_test, y_test, class_names):
    """
    Evaluate model performance on test set.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    X_test : numpy.ndarray
        Test features
    y_test : numpy.ndarray
        Test labels
    class_names : list
        List of class names
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Make predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_test, y_pred, average=None)
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (weighted)")
    print(f"  Recall:    {recall:.4f} (weighted)")
    print(f"  F1-Score:  {f1:.4f} (weighted)")
    
    print(f"\nPer-Class Performance:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: Precision={precision_per_class[i]:.3f}, "
              f"Recall={recall_per_class[i]:.3f}, F1={f1_per_class[i]:.3f}, "
              f"Support={support_per_class[i]}")
    
    print("\n" + "=" * 60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='results/confusion_matrix.png'):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    class_names : list
        List of class names
    save_path : str
        Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('Confusion Matrix (Counts)')
    
    # Plot normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_learning_curves(history, save_path='results/learning_curves.png'):
    """
    Plot training and validation learning curves.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history object
    save_path : str
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy over Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss over Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Learning curves saved to {save_path}")
    plt.close()


def plot_class_performance(metrics, class_names, save_path='results/class_performance.png'):
    """
    Plot per-class performance metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing per-class metrics
    class_names : list
        List of class names
    save_path : str
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    ax.bar(x - width, metrics['precision_per_class'], width, 
           label='Precision', alpha=0.8)
    ax.bar(x, metrics['recall_per_class'], width,
           label='Recall', alpha=0.8)
    ax.bar(x + width, metrics['f1_per_class'], width,
           label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Spectral Type')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Class performance plot saved to {save_path}")
    plt.close()


def create_evaluation_report(metrics, class_names, save_path='results/evaluation_report.txt'):
    """
    Create a comprehensive evaluation report and save to file.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing evaluation metrics
    class_names : list
        List of class names
    save_path : str
        Path to save the report
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("STELLAR SPECTRAL CLASSIFICATION - EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERALL PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f} (weighted average)\n")
        f.write(f"Recall:    {metrics['recall']:.4f} (weighted average)\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f} (weighted average)\n\n")
        
        f.write("PER-CLASS PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
        f.write("-" * 70 + "\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<8} "
                   f"{metrics['precision_per_class'][i]:<12.4f} "
                   f"{metrics['recall_per_class'][i]:<12.4f} "
                   f"{metrics['f1_per_class'][i]:<12.4f} "
                   f"{metrics['support_per_class'][i]:<10.0f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"Evaluation report saved to {save_path}")


if __name__ == '__main__':
    print("This module provides evaluation and visualization functions.")
    print("Use it with a trained model and test data.")
