#!/usr/bin/env python3
"""
main.py — run XGBoost player stat predictions from a feature list.

Edit FEATURES below with a player's current-year stats, then run:
    python3 main.py
"""

import pandas as pd
from predictor import load_models, predict

# ---------------------------------------------------------------------------
# Input: provide one value per feature (in order)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    'weight',
    'height',
    'position_id',
    'experience_years',
    'defensive_avg_defensive_rebounds',
    'defensive_avg_blocks',
    'defensive_avg_steals',
    'general_avg_minutes',
    'general_avg_rebounds',
    'general_avg_fouls',
    'offensive_field_goal_pct',
    'offensive_free_throws',
    'offensive_avg_field_goals_made',
    'offensive_avg_field_goals_attempted',
    'offensive_avg_three_point_field_goals_made',
    'offensive_avg_three_point_field_goals_attempted',
    'offensive_avg_free_throws_made',
    'offensive_avg_free_throws_attempted',
    'offensive_avg_points',
    'offensive_avg_offensive_rebounds',
    'offensive_avg_assists',
    'offensive_avg_turnovers',
    'offensive_three_point_field_goal_pct',
    'offensive_avg_two_point_field_goals_made',
    'offensive_avg_two_point_field_goals_attempted',
    'offensive_two_point_field_goal_pct',
    'offensive_shooting_efficiency',
    'offensive_scoring_efficiency',
    'general_minutes',
    'next_general_avg_minutes',
]

FEATURES = [
    195,    # weight
    76,     # height
    2,      # position_id
    2,      # experience_years
    3.5,    # defensive_avg_defensive_rebounds
    0.4,    # defensive_avg_blocks
    0.9,    # defensive_avg_steals
    28.0,   # general_avg_minutes
    4.8,    # general_avg_rebounds
    2.1,    # general_avg_fouls
    0.46,   # offensive_field_goal_pct
    0.78,   # offensive_free_throws
    5.2,    # offensive_avg_field_goals_made
    11.3,   # offensive_avg_field_goals_attempted
    1.4,    # offensive_avg_three_point_field_goals_made
    3.8,    # offensive_avg_three_point_field_goals_attempted
    2.1,    # offensive_avg_free_throws_made
    2.7,    # offensive_avg_free_throws_attempted
    13.9,   # offensive_avg_points
    1.3,    # offensive_avg_offensive_rebounds
    3.0,    # offensive_avg_assists
    1.8,    # offensive_avg_turnovers
    0.37,   # offensive_three_point_field_goal_pct
    3.8,    # offensive_avg_two_point_field_goals_made
    7.5,    # offensive_avg_two_point_field_goals_attempted
    0.51,   # offensive_two_point_field_goal_pct
    1.05,   # offensive_shooting_efficiency
    0.92,   # offensive_scoring_efficiency
    812.0,  # general_minutes
    29.0,   # next_general_avg_minutes
]


def run_prediction(features: list) -> list:
    """
    Takes a list of feature values (matching FEATURE_NAMES order),
    returns a list of (target_name, predicted_value) tuples.
    """
    models, feature_cols = load_models()
    input_df = pd.DataFrame([features], columns=feature_cols)
    predictions_df = predict(input_df, models=models, feature_cols=feature_cols)
    return list(predictions_df.iloc[0].items())


if __name__ == '__main__':
    results = run_prediction(FEATURES)

    print('=== Predicted Next-Year Stats ===')
    for target, value in results:
        print(f'  {target}: {value:.4f}')
