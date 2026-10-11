"""
Data loader module for order book snapshot data.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
import glob
import os
import pandas as pd
import numpy as np


def get_default_tick_sizes() -> Dict[str, float]:
    """
    Returns empirical minimum price increment (tick size) for each crypto instrument.
    """
    return {
        'ADA': 0.000005,
        'APT': 0.00005,
        'ARB': 0.000005,
        'ATOM': 0.00005,
        'AVAX': 0.00005,
        'BNB': 0.005,
        'BTC': 1.0,
        'DOGE': 0.000001,
        'DOT': 0.00005,
        'ETC': 0.00005,
        'ETH': 0.05,
        'FIL': 0.000005,
        'INJ': 0.00005,
        'LINK': 0.00005,
        'LTC': 0.0005,
        'NEAR': 0.00002,
        'OP': 0.000005,
        'SEI': 0.000001,
        'SOL': 0.0005,
        'SUI': 0.000005,
        'TIA': 0.000005,
        'WIF': 0.000005,
        'XLM': 0.000005,
        'XRP': 0.00005,
    }


def load_raw_dataset(
    file_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Loads raw CSV or Parquet file and standardizes column formats.
    """
    path = Path(file_path)
    if not path.exists():
        if path.suffix == '.parquet' and path.with_suffix('.csv').exists():
            path = path.with_suffix('.csv')
        else:
            raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == '.parquet':
        try:
            df = pd.read_parquet(path)
        except BaseException:
            try:
                df = pd.read_parquet(path, engine='fastparquet')
            except BaseException:
                csv_path = path.with_suffix('.csv')
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                else:
                    raise
    elif path.suffix in ('.csv', '.gz'):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    # Standardize timestamp
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], utc=True)
    df = df.sort_values('timestamp_utc').reset_index(drop=True)
    return df


def list_available_instruments(data_dir: Union[str, Path]) -> List[str]:
    """
    Returns list of ticker symbols available in data directory.
    """
    p = Path(data_dir)
    files = list(p.glob('*_sample_7d.parquet'))
    if not files:
        files = list(p.glob('*_sample_7d.csv'))
    symbols = sorted([f.name.split('_sample_7d')[0] for f in files])
    return symbols
