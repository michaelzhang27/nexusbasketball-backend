"""
predictor.py — helper module for loading saved XGBoost models and running inference.
"""

import os
import pickle
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'xgboost_models_no_team_stats')


def load_models(model_dir: str = MODEL_DIR) -> tuple[dict, list]:
    """
    Load all saved XGBoost models and the expected feature columns.

    Returns:
        models       : dict mapping target column name → XGBRegressor
        feature_cols : list of feature column names (in training order)
    """
    feature_path = os.path.join(model_dir, 'feature_columns.pkl')
    with open(feature_path, 'rb') as f:
        feature_cols = pickle.load(f)

    models = {}
    for fname in os.listdir(model_dir):
        if fname.startswith('model_') and fname.endswith('.pkl'):
            target = fname[len('model_'):-len('.pkl')]
            with open(os.path.join(model_dir, fname), 'rb') as f:
                models[target] = pickle.load(f)

    return models, feature_cols


def predict(
    input_df: pd.DataFrame,
    models: dict | None = None,
    feature_cols: list | None = None,
    model_dir: str = MODEL_DIR,
) -> pd.DataFrame:
    """
    Run all loaded models on input_df and return a DataFrame of predictions.

    Args:
        input_df     : DataFrame whose columns include the feature columns.
        models       : Pre-loaded models dict (loaded via load_models if None).
        feature_cols : Expected feature columns (loaded via load_models if None).
        model_dir    : Directory containing saved models (used only if models is None).

    Returns:
        predictions_df : DataFrame with one column per target, one row per input row.
    """
    if models is None or feature_cols is None:
        models, feature_cols = load_models(model_dir)

    # Select and order feature columns; fill missing with column median
    X = input_df[feature_cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    predictions = {}
    for target, model in sorted(models.items()):
        predictions[target] = model.predict(X)

    return pd.DataFrame(predictions, index=input_df.index)
