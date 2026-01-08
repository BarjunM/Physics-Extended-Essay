"""
Download and prepare stellar spectral data from SDSS.

This script downloads labeled stellar data with ugriz magnitudes and spectral types
from the SDSS (Sloan Digital Sky Survey) database.
"""

import os
import pandas as pd
import numpy as np
from astroquery.sdss import SDSS
from astropy import coordinates as coords
import requests

def download_sdss_sample_data(output_path='data/sdss_stars.csv', sample_size=10000):
    """
    Download a sample of stellar data from SDSS with spectral classifications.
    
    Parameters:
    -----------
    output_path : str
        Path where the CSV file will be saved
    sample_size : int
        Number of samples to download
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the stellar data
    """
    print(f"Downloading {sample_size} stellar samples from SDSS...")
    
    # SDSS SQL query to get stars with spectral type information
    # We'll use the SpecObj and PhotoObj tables
    query = f"""
    SELECT TOP {sample_size}
        p.objid, p.ra, p.dec,
        p.u, p.g, p.r, p.i, p.z,
        p.err_u, p.err_g, p.err_r, p.err_i, p.err_z,
        s.class, s.subclass, s.z as redshift,
        s.zWarning, s.sciencePrimary
    FROM PhotoObj AS p
    JOIN SpecObj AS s ON s.bestobjid = p.objid
    WHERE s.class = 'STAR'
        AND s.zWarning = 0
        AND p.u > 0 AND p.g > 0 AND p.r > 0 AND p.i > 0 AND p.z > 0
        AND p.u < 30 AND p.g < 30 AND p.r < 30 AND p.i < 30 AND p.z < 30
        AND s.subclass != ''
        AND s.sciencePrimary = 1
    """
    
    try:
        # Query SDSS
        result = SDSS.query_sql(query, timeout=300)
        
        if result is None or len(result) == 0:
            print("No data returned from SDSS. Using synthetic data instead.")
            return create_synthetic_data(output_path, sample_size)
        
        # Convert to pandas DataFrame
        df = result.to_pandas()
        
        # Map subclass to spectral type
        df['spectral_type'] = df['subclass'].apply(extract_spectral_type)
        
        # Filter to only keep O, B, A, F, G, K, M types
        valid_types = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
        df = df[df['spectral_type'].isin(valid_types)]
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"Downloaded {len(df)} stellar samples")
        print(f"Spectral type distribution:")
        print(df['spectral_type'].value_counts().sort_index())
        
        return df
        
    except Exception as e:
        print(f"Error downloading from SDSS: {e}")
        print("Creating synthetic dataset instead...")
        return create_synthetic_data(output_path, sample_size)


def extract_spectral_type(subclass):
    """
    Extract the main spectral type letter from SDSS subclass.
    
    Parameters:
    -----------
    subclass : str
        SDSS subclass string (e.g., 'G5', 'K0', 'M2')
        
    Returns:
    --------
    str
        Main spectral type letter (O, B, A, F, G, K, M) or 'Unknown'
    """
    if not isinstance(subclass, str) or len(subclass) == 0:
        return 'Unknown'
    
    first_char = subclass[0].upper()
    if first_char in ['O', 'B', 'A', 'F', 'G', 'K', 'M']:
        return first_char
    return 'Unknown'


def create_synthetic_data(output_path='data/sdss_stars.csv', n_samples=10000):
    """
    Create synthetic stellar data based on typical SDSS photometric properties.
    This is used as a fallback when SDSS data cannot be downloaded.
    
    Parameters:
    -----------
    output_path : str
        Path where the CSV file will be saved
    n_samples : int
        Number of synthetic samples to generate
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the synthetic stellar data
    """
    print("Creating synthetic stellar dataset...")
    
    np.random.seed(42)
    
    # Define typical colors and magnitudes for each spectral type
    # Based on actual SDSS photometric properties
    spectral_properties = {
        'O': {'u-g': -1.0, 'g-r': -0.5, 'r-i': -0.3, 'i-z': -0.3, 'r_mag': 18.0, 'std': 0.3},
        'B': {'u-g': -0.5, 'g-r': -0.3, 'r-i': -0.2, 'i-z': -0.2, 'r_mag': 18.5, 'std': 0.3},
        'A': {'u-g': 0.5, 'g-r': 0.0, 'r-i': -0.1, 'i-z': -0.1, 'r_mag': 17.5, 'std': 0.3},
        'F': {'u-g': 1.2, 'g-r': 0.3, 'r-i': 0.1, 'i-z': 0.0, 'r_mag': 17.0, 'std': 0.3},
        'G': {'u-g': 1.5, 'g-r': 0.6, 'r-i': 0.2, 'i-z': 0.1, 'r_mag': 16.5, 'std': 0.3},
        'K': {'u-g': 2.2, 'g-r': 1.1, 'r-i': 0.4, 'i-z': 0.2, 'r_mag': 16.0, 'std': 0.3},
        'M': {'u-g': 2.8, 'g-r': 1.5, 'r-i': 0.8, 'i-z': 0.5, 'r_mag': 17.5, 'std': 0.4},
    }
    
    # Distribution of spectral types (roughly based on stellar population)
    type_weights = {'O': 0.01, 'B': 0.05, 'A': 0.08, 'F': 0.12, 'G': 0.20, 'K': 0.30, 'M': 0.24}
    
    data = []
    for spectral_type, props in spectral_properties.items():
        n = int(n_samples * type_weights[spectral_type])
        
        # Generate color indices with some scatter
        u_g = np.random.normal(props['u-g'], props['std'], n)
        g_r = np.random.normal(props['g-r'], props['std'], n)
        r_i = np.random.normal(props['r-i'], props['std'], n)
        i_z = np.random.normal(props['i-z'], props['std'], n)
        
        # Generate r-band magnitude
        r_mag = np.random.normal(props['r_mag'], 1.5, n)
        
        # Calculate individual magnitudes from r and color indices
        g = r_mag + g_r
        u = g + u_g
        i = r_mag - r_i
        z = i - i_z
        
        # Add measurement errors (typical SDSS errors)
        err_base = 0.02
        err_u = np.random.uniform(err_base, 0.1, n)
        err_g = np.random.uniform(err_base, 0.05, n)
        err_r = np.random.uniform(err_base, 0.05, n)
        err_i = np.random.uniform(err_base, 0.05, n)
        err_z = np.random.uniform(err_base, 0.08, n)
        
        # Random sky positions
        ra = np.random.uniform(0, 360, n)
        dec = np.random.uniform(-10, 70, n)
        
        for idx in range(n):
            data.append({
                'objid': int(1000000000000 + len(data)),
                'ra': ra[idx],
                'dec': dec[idx],
                'u': u[idx],
                'g': g[idx],
                'r': r_mag[idx],
                'i': i[idx],
                'z': z[idx],
                'err_u': err_u[idx],
                'err_g': err_g[idx],
                'err_r': err_r[idx],
                'err_i': err_i[idx],
                'err_z': err_z[idx],
                'spectral_type': spectral_type,
                'subclass': f"{spectral_type}{np.random.randint(0, 10)}"
            })
    
    df = pd.DataFrame(data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Created {len(df)} synthetic stellar samples")
    print(f"Spectral type distribution:")
    print(df['spectral_type'].value_counts().sort_index())
    
    return df


if __name__ == '__main__':
    # Download or create the dataset
    df = download_sdss_sample_data(sample_size=10000)
    print(f"\nDataset saved to data/sdss_stars.csv")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
