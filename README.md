# Stellar Spectral Classification using Machine Learning

This project uses machine learning (neural networks) to classify stellar spectral types (O, B, A, F, G, K, M) from multi-band photometric data. The classifier is trained on SDSS ugriz magnitudes and uses color-based features derived from photometry.

## Project Overview

Stellar spectral classification is traditionally done through spectroscopy, which analyzes the detailed spectrum of a star. However, photometric classification using broadband colors (like SDSS ugriz) offers a faster alternative. This project demonstrates:

- **Data Collection**: Downloads labeled stellar data from SDSS or generates synthetic data
- **Feature Engineering**: Computes color indices (u-g, g-r, r-i, i-z, etc.) from ugriz magnitudes
- **Neural Network**: Trains a feedforward neural network classifier using TensorFlow/Keras
- **Evaluation**: Comprehensive performance metrics including accuracy, precision, recall, confusion matrix, and learning curves

## Spectral Types

The classifier distinguishes between seven main spectral types:
- **O**: Hot blue stars (>30,000 K)
- **B**: Blue-white stars (10,000-30,000 K)
- **A**: White stars (7,500-10,000 K)
- **F**: Yellow-white stars (6,000-7,500 K)
- **G**: Yellow stars like the Sun (5,200-6,000 K)
- **K**: Orange stars (3,700-5,200 K)
- **M**: Red stars (<3,700 K)

## Project Structure

```
.
├── src/
│   ├── download_data.py    # Download/generate stellar data from SDSS
│   ├── preprocess.py        # Data preprocessing and feature engineering
│   ├── model.py             # Neural network model definition
│   ├── evaluate.py          # Model evaluation and visualization
│   └── train.py             # Main training pipeline script
├── data/                    # Data directory (created automatically)
├── models/                  # Saved models (created automatically)
├── results/                 # Evaluation results and plots (created automatically)
├── notebooks/               # Jupyter notebooks for exploration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/BarjunM/Physics-Extended-Essay.git
cd Physics-Extended-Essay
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete pipeline with default settings:
```bash
python src/train.py
```

This will:
1. Generate synthetic stellar data (10,000 samples)
2. Preprocess and create color-based features
3. Train a neural network classifier
4. Evaluate performance on test set
5. Generate visualizations and reports

### Custom Training

You can customize various parameters:

```bash
python src/train.py \
    --sample-size 20000 \
    --hidden-layers 256 128 64 \
    --epochs 150 \
    --batch-size 64 \
    --dropout 0.4
```

### Download Real SDSS Data

To download real data from SDSS (requires internet connection):
```bash
python src/train.py --download --sample-size 5000
```

### Command-line Arguments

- `--data-path`: Path to data CSV file (default: `data/sdss_stars.csv`)
- `--download`: Download real SDSS data instead of synthetic
- `--sample-size`: Number of samples (default: 10000)
- `--test-size`: Test set proportion (default: 0.2)
- `--val-size`: Validation set proportion (default: 0.1)
- `--hidden-layers`: Hidden layer sizes (default: 128 64 32)
- `--dropout`: Dropout rate (default: 0.3)
- `--epochs`: Maximum training epochs (default: 100)
- `--batch-size`: Training batch size (default: 32)
- `--model-dir`: Model save directory (default: `models`)
- `--output-dir`: Results output directory (default: `results`)

## Features

### Color-Based Features

The model uses 13 features derived from ugriz photometry:

**Color Indices** (8 features):
- u-g, g-r, r-i, i-z (adjacent band colors)
- u-r, g-i, r-z, u-i (wider baseline colors)

**Raw Magnitudes** (5 features):
- u, g, r, i, z bands

These features are standardized (zero mean, unit variance) before training.

### Neural Network Architecture

Default architecture:
- Input layer: 13 features
- Hidden layer 1: 128 neurons + BatchNorm + Dropout
- Hidden layer 2: 64 neurons + BatchNorm + Dropout
- Hidden layer 3: 32 neurons + BatchNorm + Dropout
- Output layer: 7 neurons (softmax)

Total parameters: ~15,000 (varies with architecture)

### Training Features

- **Early Stopping**: Stops training if validation loss doesn't improve
- **Learning Rate Reduction**: Reduces learning rate when loss plateaus
- **Model Checkpointing**: Saves best model based on validation loss
- **Batch Normalization**: Stabilizes training
- **Dropout Regularization**: Prevents overfitting

## Results

After training, you'll find:

1. **Model Files** (`models/`):
   - `best_model.keras`: Trained model
   - `model_summary.txt`: Model architecture
   - `model_metadata.json`: Feature names and preprocessing info

2. **Visualizations** (`results/`):
   - `learning_curves.png`: Training/validation accuracy and loss curves
   - `confusion_matrix.png`: Confusion matrices (counts and normalized)
   - `class_performance.png`: Per-class precision, recall, and F1-score
   - `evaluation_report.txt`: Detailed performance metrics

### Expected Performance

With synthetic data (10,000 samples):
- **Overall Accuracy**: ~95-98%
- **Precision/Recall**: >0.93 for most classes
- Training time: 2-5 minutes on CPU

Performance varies by spectral type:
- Best: G, K, M types (most common, distinct colors)
- Moderate: A, F types (intermediate properties)
- Challenging: O, B types (rarer, can overlap with A)

## Comparison with Traditional Methods

Traditional photometric classification using color-color diagrams:
- Simple decision boundaries in color space
- Accuracy: ~70-85% for main sequence stars
- Limited ability to handle complex cases

This neural network approach:
- Learns complex, non-linear decision boundaries
- Accuracy: ~95-98% with good data
- Better handles edge cases and borderline classifications
- Can incorporate more features simultaneously

## Scientific Background

### Why Color-Based Classification Works

Different spectral types have different temperatures and surface properties, which affect their spectral energy distribution. This creates characteristic colors:

- Hot O/B stars: Blue colors (negative u-g, g-r)
- Medium A/F/G stars: White to yellow (intermediate colors)
- Cool K/M stars: Red colors (large positive color indices)

The neural network learns to map these color patterns to spectral types more effectively than simple color cuts.

### SDSS Photometry

The SDSS uses five broadband filters:
- **u** (ultraviolet): 354 nm
- **g** (green): 477 nm  
- **r** (red): 623 nm
- **i** (near-infrared): 763 nm
- **z** (infrared): 913 nm

These filters are designed to sample the optical/near-IR spectrum efficiently.

## Limitations

1. **Spectral Type Only**: Classifies main spectral type (O-M), not luminosity class
2. **Photometric Redshift**: Doesn't account for redshift effects (assumes nearby stars)
3. **Interstellar Extinction**: Doesn't correct for dust reddening
4. **Binary Systems**: May misclassify unresolved binary stars
5. **Peculiar Stars**: May struggle with chemically peculiar or exotic objects

## Future Improvements

- Add extinction correction using Galactic dust maps
- Include luminosity class prediction (I-V)
- Incorporate proper motion and parallax data
- Use recurrent/attention mechanisms for spectrophotometry
- Implement uncertainty estimation (Bayesian neural networks)
- Add explainability analysis (feature importance, SHAP values)

## References

- SDSS: https://www.sdss.org/
- Stellar Classification: https://en.wikipedia.org/wiki/Stellar_classification
- SDSS Photometry: https://www.sdss.org/dr17/algorithms/photometry/

## License

MIT License - feel free to use for educational and research purposes.

## Author

Physics Extended Essay Project - Stellar Spectral Classification