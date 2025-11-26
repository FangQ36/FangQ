"""
ga_search.py

Genetic Algorithm for joint feature selection and DNN hyperparameter tuning.
This template uses DEAP. For speed, GA evaluates individuals with short training (few epochs).
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
import yaml
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from models.dnn_model import build_dnn_3layer
from models.train_dnn_cv import forward_chained_year_splits

def load_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="ga_results")
    return p.parse_args()

def fitness_eval(individual, df, years_splits, feature_cols, target_col):
    # decode individual: first N bits are feature mask, then three hyperparams indices
    N = len(feature_cols)
    mask = np.array(individual[:N], dtype=bool)
    if mask.sum() == 0:
        return 1e6,
    hidden_idx = individual[N]
    lr_idx = individual[N+1]
    # map discrete indices
    HIDDEN_OPTIONS = [(64,32,16),(128,64,32),(256,128,64)]
    LR_OPTIONS = [1e-4, 3e-4, 1e-3]
    units = HIDDEN_OPTIONS[hidden_idx]
    lr = LR_OPTIONS[lr_idx]

    rmses = []
    for train_years, test_years in years_splits:
        train = df[df['year'].isin(train_years)]
        test = df[df['year'].isin(test_years)]
        agg_train = train.groupby(['county_id','year'])[feature_cols].mean().reset_index()
        agg_train['yield'] = train.groupby(['county_id','year'])['yield'].mean().values
        agg_test = test.groupby(['county_id','year'])[feature_cols].mean().reset_index()
        agg_test['yield'] = test.groupby(['county_id','year'])['yield'].mean().values

        X_train = agg_train[feature_cols].values[:, mask]
        y_train = agg_train['yield'].values
        X_test = agg_test[feature_cols].values[:, mask]
        y_test = agg_test['yield'].values

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = build_dnn_3layer(X_train_s.shape[1], units=units, activation='relu', dropout=0.2, lr=lr)
        # short training for GA evaluation
        model.fit(X_train_s, y_train, epochs=40, batch_size=64, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),])
        y_pred = model.predict(X_test_s).ravel()
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        rmses.append(rmse)
    mean_rmse = np.mean(rmses)
    # penalize by number of features
    penalty = 0.01 * mask.sum()
    return mean_rmse + penalty,

def main():
    args = load_args()
    df = pd.read_csv(args.data)
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "features.yaml")
    with open(cfg_path) as f:
        features = yaml.safe_load(f)['features']
    feature_cols = [c for c in features if c in df.columns]
    years = sorted(df['year'].unique().tolist())
    years_splits = forward_chained_year_splits(years, n_splits=5)

    N = len(feature_cols)
    # DEAP setup
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_feat", lambda: random.randint(0,1))
    toolbox.register("attr_hidden", lambda: random.randint(0,2))
    toolbox.register("attr_lr", lambda: random.randint(0,2))
    def init_ind():
        ind = [toolbox.attr_feat() for _ in range(N)]
        ind += [toolbox.attr_hidden(), toolbox.attr_lr()]
        return creator.Individual(ind)
    toolbox.register("individual", init_ind)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", fitness_eval, df=df, years_splits=years_splits, feature_cols=feature_cols, target_col='yield')
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean); stats.register("min", np.min)
    algorithms.eaSimple(pop, toolbox, cxpb=0.8, mutpb=0.1, ngen=40, stats=stats, halloffame=hof, verbose=True)

    os.makedirs(args.output, exist_ok=True)
    # save hof individuals
    import json
    with open(os.path.join(args.output, "hall_of_fame.json"), "w") as f:
        json.dump([list(ind) for ind in hof], f)
    print("GA finished. Best individuals saved to", args.output)

if __name__ == "__main__":
    main()
