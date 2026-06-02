# xG Model — Expected Goals from StatsBomb World Cup 2022 Data

A machine learning pipeline that trains and evaluates multiple xG models on StatsBomb open event data from the 2022 FIFA World Cup, then validates predictions against StatsBomb's own xG values.

## What It Does

- Extracts shot events from all WC 2022 matches via the StatsBomb open data API
- Engineers features from raw coordinate and event data
- Trains three models and compares performance
- Validates against StatsBomb's published xG values
- Generates shot maps and xG timelines per match

## Models

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble with feature importance analysis |
| Gradient Boosting | Best-performing model |

## Features Used

- Shot distance and angle from goal
- Body part (foot / head)
- Shot type (open play, free kick, corner)
- First time shot
- One-on-one with goalkeeper
- Under pressure flag

## Evaluation Metrics

- Log Loss
- Brier Score
- AUC-ROC
- Correlation with StatsBomb xG (validation)

## Outputs

- `shot_map.png` — all WC 2022 shots plotted by xG value
- `xg_timeline_{match_id}.png` — xG flow chart per match
- `xg_validation.png` — predicted vs StatsBomb xG scatter
- `feature_importance.png` — Random Forest feature weights
- `calibration.png` — model calibration curves
- `eda_analysis.png` — exploratory data analysis

## Data

Uses [StatsBomb open data](https://github.com/statsbomb/open-data) — no API key required. Covers all 64 matches from the 2022 FIFA World Cup.

## Stack

Python · scikit-learn · pandas · matplotlib · StatsBombPy
