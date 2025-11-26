# CropYield-DNN-GA

Repository template for the paper *Integrating Remote Sensing, Deep Learning with Crop Model for Enhanced Yield Estimation Under Air Pollution Stress*.

**Key points**
- Model inputs per-pixel: WOFOST output yield, NDVI, and user-defined API indices.
- Main model: Deep Neural Network (DNN) with 3 hidden layers.
- Optimization: Genetic Algorithm (GA) for joint feature selection + DNN hyperparameters.
- Validation: 5-fold forward-chained / county-level cross-validation using USDA county yield.
- Feature importance: Mean Decrease Impurity (RandomForest) for MDI.

This repository contains code templates to preprocess data, run experiments, and reproduce the analyses in the paper. Replace the `data/processed/features_with_year.csv` with your processed dataset and adjust configs as needed.

## Repo structure

```
CropYield-DNN-GA/
├── README.md
├── LICENSE
├── CITATION
├── requirements.txt
├── environment.yml
├── config/
│   └── features.yaml
├── data_prep/
│   ├── prepare_inputs.py
│   └── resample_and_clean.py
├── models/
│   ├── dnn_model.py
│   └── train_dnn_cv.py
├── ga/
│   └── ga_search.py
├── postproc/
│   └── feature_importance.py
├── notebooks/
│   └── demo.ipynb
└── Manuscript(农业遥感)-9.3.docx
```

## Quick start

1. Create a conda environment:

```bash
conda env create -f environment.yml
conda activate cropyield
```

or with pip:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Prepare your `data/processed/features_with_year.csv` (see `data_prep/prepare_inputs.py` for expected format).

3. Run a demo training (single fold):

```bash
python models/train_dnn_cv.py --data data/processed/features_with_year.csv --output saved_models/
```

4. Run GA search (this can be slow):

```bash
python ga/ga_search.py --data data/processed/features_with_year.csv --output ga_results/
```

## Notes for reproducibility

- StandardScaler is fit on training folds only.
- The forward-chained CV splits by year and aggregated at county level to avoid leakage.

Please edit `config/features.yaml` to list your feature names (order matters for GA mask).
