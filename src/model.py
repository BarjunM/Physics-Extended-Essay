"""
Neural network model for stellar spectral type classification.

This module defines and trains a TensorFlow/Keras neural network classifier.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import json


def build_model(input_dim, n_classes, hidden_layers=[128, 64, 32], dropout_rate=0.3):
    """
    Build a feedforward neural network for classification.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    n_classes : int
        Number of output classes (spectral types)
    hidden_layers : list
        List of integers specifying the size of each hidden layer
    dropout_rate : float
        Dropout rate for regularization
        
    Returns:
    --------
    keras.Model
        Compiled Keras model
    """
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(input_dim,)))
    
    # Hidden layers with BatchNormalization and Dropout
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(layers.BatchNormalization(name=f'bn_{i+1}'))
        model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
    
    # Output layer with softmax activation for multi-class classification
    model.add(layers.Dense(n_classes, activation='softmax', name='output'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_callbacks(model_path='models/best_model.keras', patience=15):
    """
    Create callbacks for training.
    
    Parameters:
    -----------
    model_path : str
        Path to save the best model
    patience : int
        Number of epochs to wait for improvement before early stopping
        
    Returns:
    --------
    list
        List of Keras callbacks
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    callback_list = [
        # Save the best model based on validation loss
        callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when validation loss plateaus
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callback_list


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Train the neural network model.
    
    Parameters:
    -----------
    model : keras.Model
        Compiled Keras model
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    X_val : numpy.ndarray
        Validation features
    y_val : numpy.ndarray
        Validation labels
    epochs : int
        Maximum number of training epochs
    batch_size : int
        Batch size for training
        
    Returns:
    --------
    keras.callbacks.History
        Training history object
    """
    print("=" * 60)
    print("TRAINING NEURAL NETWORK")
    print("=" * 60)
    
    # Create callbacks
    callback_list = create_callbacks()
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return history


def save_model_info(model, feature_names, label_encoder, scaler, 
                    save_dir='models'):
    """
    Save model architecture and preprocessing information.
    
    Parameters:
    -----------
    model : keras.Model
        Trained model
    feature_names : list
        List of feature names
    label_encoder : sklearn.preprocessing.LabelEncoder
        Fitted label encoder
    scaler : sklearn.preprocessing.StandardScaler
        Fitted feature scaler
    save_dir : str
        Directory to save model info
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model architecture summary to text file
    with open(os.path.join(save_dir, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Save metadata as JSON
    metadata = {
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'class_names': label_encoder.classes_.tolist(),
        'n_classes': len(label_encoder.classes_),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }
    
    with open(os.path.join(save_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model information saved to {save_dir}/")


def load_trained_model(model_path='models/best_model.keras'):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
        
    Returns:
    --------
    keras.Model
        Loaded Keras model
    """
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


if __name__ == '__main__':
    # Example usage
    print("Building example model...")
    model = build_model(input_dim=13, n_classes=7)
    print("\nModel architecture:")
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
