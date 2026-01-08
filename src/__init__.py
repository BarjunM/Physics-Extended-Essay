"""
Stellar Spectral Classification Package

This package provides tools for classifying stellar spectral types using
machine learning on multi-band photometric data.
"""

__version__ = '1.0.0'
__author__ = 'Physics Extended Essay Project'

from . import download_data
from . import preprocess
from . import model
from . import evaluate

__all__ = ['download_data', 'preprocess', 'model', 'evaluate']
