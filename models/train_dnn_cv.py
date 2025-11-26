"""
train_dnn_cv.py

Train DNN with 5-fold forward-chained county-level CV.

Usage:
    python train_dnn_cv.py --data data/processed/features_with_year.csv --output saved_models/
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# reproducibility
SEED = 42
os.environ.setdefault('PYTHONHASHSEED', str(SEED))
random.seed(SEED)
np.random.seed(SEED)

from models.dnn_model import build_dnn_3layer

def forward_chained_year_splits(years, n_splits=5):
    years = sorted(years)
    n = len(years)
    # create roughly equal folds of years for test windows
    fold_size = max(1, n // n_splits)
    splits = []
    for i in range(n_splits):
        train_end_idx = i * fold_size + fold_size - 1
        if train_end_idx >= n-1:
            train_end_idx = n-2
        train_years = years[:train_end_idx+1]
        test_years = years[train_end_idx+1:train_end_idx+1+fold_size]
        if len(test_years)==0:
            break
        splits.append((train_years, test_years))
    return splits

def evaluate(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse':rmse, 'mae':mae, 'r2':r2}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="saved_models")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if 'year' not in df.columns or 'county_id' not in df.columns or 'yield' not in df.columns:
        raise ValueError("Input CSV must contain 'year', 'county_id', and 'yield' columns.")

    years = sorted(df['year'].unique().tolist())
    splits = forward_chained_year_splits(years, n_splits=5)

    feature_cfg = os.path.join(os.path.dirname(__file__), "..", "config", "features.yaml")
    with open(feature_cfg) as f:
        features = yaml.safe_load(f)['features']

    feature_cols = [c for c in features if c in df.columns]
    X_all = df[feature_cols + ['year', 'county_id', 'yield']]

    os.makedirs(args.output, exist_ok=True)
    results = []

    for i, (train_years, test_years) in enumerate(splits):
        train_df = X_all[X_all['year'].isin(train_years)].copy()
        test_df  = X_all[X_all['year'].isin(test_years)].copy()

        # aggregate to county-year level (mean of predictors)
        agg_train = train_df.groupby(['county_id','year'])[feature_cols].mean().reset_index()
        agg_train['yield'] = train_df.groupby(['county_id','year'])['yield'].mean().values
        agg_test = test_df.groupby(['county_id','year'])[feature_cols].mean().reset_index()
        agg_test['yield'] = test_df.groupby(['county_id','year'])['yield'].mean().values

        X_train = agg_train[feature_cols].values
        y_train = agg_train['yield'].values
        X_test  = agg_test[feature_cols].values
        y_test  = agg_test['yield'].values

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = build_dnn_3layer(X_train_s.shape[1], units=(128,64,32), activation='relu', dropout=0.2, lr=1e-3)

        # simple training
        history = model.fit(X_train_s, y_train, validation_split=0.1, epochs=200, batch_size=64, verbose=2,
                            callbacks=[__import__('tensorflow').keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
        preds = model.predict(X_test_s).ravel()
        metrics = evaluate(y_test, preds)
        print(f"Fold {i} metrics:", metrics)
        results.append(metrics)

        # save model and scaler
        model.save(os.path.join(args.output, f"dnn_fold{i}.h5"))
        joblib.dump(scaler, os.path.join(args.output, f"scaler_fold{i}.joblib"))

    # aggregate metrics
    import numpy as _np
    mean_metrics = {k:_np.mean([r[k] for r in results]) for k in results[0]}
    print("Mean cross-validated metrics:", mean_metrics)

if __name__ == "__main__":
    main()
