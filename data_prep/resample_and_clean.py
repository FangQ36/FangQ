"""
resample_and_clean.py

Utilities for resampling to 0.05deg, removing extreme values and gap-filling.
This is a template: adapt to your raster I/O (rasterio/xarray) and conventions.
"""
import numpy as np
import pandas as pd

def remove_outliers(series, lowq=0.01, highq=0.99):
    lo = series.quantile(lowq)
    hi = series.quantile(highq)
    return series.clip(lo, hi)

# Add more functions as needed
