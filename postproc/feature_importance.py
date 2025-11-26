"""
feature_importance.py

Compute feature importance using RandomForest (Mean Decrease Impurity - MDI).
Produces a CSV with feature names and importance scores.
"""
import os
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "features_with_year.csv"))
    cfg_path = os.path.join(ROOT, "config", "features.yaml")
    with open(cfg_path) as f:
        features = yaml.safe_load(f)['features']
    feature_cols = [c for c in features if c in df.columns]
    # aggregate to county-year
    agg = df.groupby(['county_id','year'])[feature_cols + ['yield']].mean().reset_index()
    X = agg[feature_cols].values
    y = agg['yield'].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(Xs, y)
    importances = rf.feature_importances_
    out = pd.DataFrame({'feature':feature_cols, 'importance':importances})
    out = out.sort_values('importance', ascending=False)
    out.to_csv(os.path.join(ROOT, "postproc", "feature_importance_mdi.csv"), index=False)
    print("Saved feature importance to postproc/feature_importance_mdi.csv")

if __name__ == "__main__":
    main()
