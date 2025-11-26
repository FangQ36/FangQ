"""
prepare_inputs.py

Expected to produce a CSV: data/processed/features_with_year.csv
Columns:
  - year (int)
  - county_fips or county_id (str/int)   # used for county aggregation/CV
  - feature columns (as in config/features.yaml)
  - yield (target, e.g. county-level yield or pixel-level yield aggregated to county)

This is a template. Modify file paths and variable names to match your data.
"""
import os
import yaml
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
cfg_path = os.path.join(ROOT, "config", "features.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
feature_list = cfg['features']

def main():
    # This template assumes you already have per-pixel WOFOST outputs, NDVI, and API indices
    # merged into a single DataFrame or NetCDF. Replace below with your actual I/O.
    # Example: load a prepared CSV (from prior processing)
    src = os.path.join(ROOT, "data", "raw", "merged_per_pixel.csv")
    if not os.path.exists(src):
        print("Please place your merged_per_pixel.csv at", src)
        return
    df = pd.read_csv(src)
    # Ensure required columns exist
    missing = [c for c in feature_list + ['year', 'county_id', 'yield'] if c not in df.columns]
    if missing:
        raise ValueError("Missing columns in source data: " + ", ".join(missing))
    # Optionally: aggregate pixel -> county-year by mean for predictors, and sum/mean for yield as needed
    agg_cols = ['year', 'county_id'] + feature_list + ['yield']
    df = df[agg_cols]
    processed_dir = os.path.join(ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    out = os.path.join(processed_dir, "features_with_year.csv")
    df.to_csv(out, index=False)
    print("Wrote processed features to", out)

if __name__ == "__main__":
    main()
