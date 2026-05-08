# Football Predictor — Project Documentation

## Overview

Machine learning pipeline that predicts Premier League match results (Home Win / Draw / Away Win) using historical data. Built with XGBoost, calibrated probabilities, and Optuna hyperparameter tuning.

👉 **[Execution guide & weekly workflow →](WORKFLOW.md)**

---

## Project Structure

```
football-predictor/
│
├── data/
│   ├── features.csv        # Processed dataset with all features
│   └── model.pkl           # Trained and calibrated XGBoost model
│
├── 01_explore_data.ipynb           # Initial data exploration
├── 02_feature_engineering.ipynb    # Feature construction pipeline
├── 03_model.ipynb                  # Model training and evaluation
├── 04_predict.ipynb                # Match prediction interface
├── 05_tuning.ipynb                 # Hyperparameter optimization
└── 06_update.ipynb                 # Weekly update script
```

---

## Data Source

**football-data.co.uk** — Free historical match data for Premier League.

- No API key required
- Direct CSV download per season
- URL pattern: `https://www.football-data.co.uk/mmz4281/{SEASON_CODE}/E0.csv`

### Seasons loaded

| Code | Season  |
|------|---------|
| 2526 | 2025/26 |
| 2425 | 2024/25 |
| 2324 | 2023/24 |
| 2223 | 2022/23 |
| 2122 | 2021/22 |
| 2021 | 2020/21 |
| 1920 | 2019/20 |
| 1819 | 2018/19 |
| 1718 | 2017/18 |
| 1617 | 2016/17 |
| 1516 | 2015/16 |
| 1415 | 2014/15 |

**Total: ~4,500 matches across 12 seasons**

### Raw columns used

| Column | Description |
|--------|-------------|
| Date | Match date |
| HomeTeam / AwayTeam | Team names |
| Referee | Match referee |
| FTHG / FTAG | Full Time Home/Away Goals |
| FTR | Full Time Result (H/D/A) |
| HTHG / HTAG | Half Time Goals |
| HS / AS | Shots |
| HST / AST | Shots on Target |
| HC / AC | Corners |
| HF / AF | Fouls |
| HY / AY | Yellow Cards |
| HR / AR | Red Cards |

---

## Features

### Target variable
```
FTR encoded as:
  H (Home Win)  → 0
  D (Draw)      → 1
  A (Away Win)  → 2
```

### Feature list

| Feature | Description | Type |
|---------|-------------|------|
| `home_form_last5` | Average points per match, last 5 games (home team) | Rolling |
| `away_form_last5` | Average points per match, last 5 games (away team) | Rolling |
| `home_avg_scored` | Average goals scored, last 5 games (home team) | Rolling |
| `home_avg_conceded` | Average goals conceded, last 5 games (home team) | Rolling |
| `away_avg_scored` | Average goals scored, last 5 games (away team) | Rolling |
| `away_avg_conceded` | Average goals conceded, last 5 games (away team) | Rolling |
| `ref_matches` | Total matches refereed in database | Referee |
| `ref_home_win_rate` | % of matches where home team won with this referee | Referee |
| `ref_yellows_avg` | Average yellow cards per match | Referee |
| `ref_reds_avg` | Average red cards per match | Referee |
| `ref_goals_avg` | Average total goals per match | Referee |
| `elo_home` | ELO rating of home team before match | ELO |
| `elo_away` | ELO rating of away team before match | ELO |
| `elo_diff` | ELO home minus ELO away | ELO |
| `h2h_matches` | Number of past H2H encounters (max 5) | H2H |
| `h2h_home_wins` | % of H2H matches won by home team | H2H |
| `h2h_draws` | % of H2H matches that ended in draw | H2H |
| `h2h_goals_avg` | Average total goals in H2H matches | H2H |

> ⚠️ All features are calculated using only past data (no data leakage).

### Feature importance (final model)

```
elo_diff          13.9%  ← most important
elo_home           6.4%
elo_away           6.1%
h2h_draws          5.4%
h2h_goals_avg      5.2%
ref_yellows_avg    5.1%
ref_home_win_rate  5.0%
home_form_last5    5.0%
...
```

---

## ELO Calculation

ELO is calculated from scratch using all historical matches. No external source needed.

```python
K = 32          # sensitivity constant
BASE_ELO = 1500 # starting rating for all teams

# After each match:
expected_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
elo_home_new = elo_home + K * (actual_result - expected_home)
```

Actual result values: Win = 1, Draw = 0.5, Loss = 0

---

## Model

### Algorithm
**XGBoost Classifier** with isotonic probability calibration (`CalibratedClassifierCV`).

### Why XGBoost over deep learning
- ~4,500 rows is too small for neural networks
- XGBoost handles tabular data natively
- Fast training (seconds, not minutes)
- More interpretable via feature importance

### Validation strategy
**TimeSeriesSplit (5 folds)** — never uses future data to validate past predictions.

```
Fold 1: train [0..800]    val [800..1600]
Fold 2: train [0..1600]   val [1600..2400]
...
```

> ⚠️ Never use random KFold for time series data — it leaks future information.

### Hyperparameter tuning
**Optuna** with 50 trials, minimizing log-loss.

Best parameters found:
```python
{
  'n_estimators':     383,
  'max_depth':        2,
  'learning_rate':    0.199,
  'subsample':        0.782,
  'colsample_bytree': 0.865,
  'min_child_weight': 6,
  'gamma':            4.620,
}
```

### Performance

| Version | Log-loss | Accuracy |
|---------|----------|----------|
| Baseline (no tuning, no H2H) | 1.0687 | 49.5% |
| + Tuning | 0.9826 | — |
| + H2H + Tuning | **0.9748** | **51.6%** |

> Baseline for accuracy comparison: always predicting home win = ~44.6%

---

## Prediction Interface

```python
# Basic prediction
predict_match("Arsenal", "Man City")

# With referee
predict_match("Arsenal", "Man City", referee="M Oliver")
```

Output:
```
⚽ Arsenal vs Man City
─────────────────────────────
🏠 Arsenal wins:  50.5%
🤝 Draw:          24.1%
✈️  Man City wins: 25.4%
─────────────────────────────
📊 Prediction: Arsenal wins
```

### Default values when referee is unknown
```python
ref_home_win_rate = 0.45   # league average
ref_yellows_avg   = 3.5
ref_reds_avg      = 0.1
ref_goals_avg     = 2.7
```

### Default values when no H2H history
```python
h2h_home_wins = 0.33
h2h_draws     = 0.25
h2h_goals_avg = 2.7
```

---

## Weekly Update Workflow

Run `06_update.ipynb` once per week (or after each matchday):

```
1. Downloads latest season CSVs from football-data.co.uk
2. Rebuilds all features from scratch
3. Evaluates CURRENT model on NEW matches (retrospective test)
4. Saves updated features.csv
5. Runs Optuna tuning (20 trials)
6. Retrains and saves model.pkl
```

After running, `04_predict.ipynb` automatically uses the updated model.

---

## Performance Tracking (Test Set)

Each weekly run evaluates the **current model** on matches it has **never seen** before retraining. This acts as a natural, ongoing test set — the matches were never used for training or hyperparameter tuning.

Results are logged to `data/tracking.csv`:

```
eval_date   | new_matches | log_loss | accuracy | period_start | period_end
2026-05-08  | 10          | 0.9821   | 0.5000   | 2026-04-26   | 2026-05-04
2026-05-15  | 10          | 0.9654   | 0.5500   | 2026-05-05   | 2026-05-11
```

> ⚠️ The validation log-loss (from Optuna/TimeSeriesSplit) can be optimistic due to hyperparameter overfitting. The tracking log-loss is the **honest** measure of real-world performance.

---

## Dependencies

```toml
pandas
numpy
xgboost
scikit-learn
optuna
jupyter
ipykernel
matplotlib
```

Install with:
```bash
uv add pandas numpy xgboost scikit-learn optuna jupyter ipykernel matplotlib
```

---

## Target Distribution

```
Home Win  (0): 44.6%
Away Win  (2): 32.0%
Draw      (1): 23.4%
```

Strong home advantage present in the data — consistent with real Premier League statistics.

---

## Known Limitations

- **No injury/suspension data** — a key player missing can invalidate predictions
- **No transfer window data** — squad quality changes mid-season not reflected until results do
- **ELO resets each run** — teams always start at 1500, early seasons have less reliable ratings
- **Single league** — model trained only on Premier League, other leagues need separate pipelines
- **High variance sport** — even the best models top out around 55-57% accuracy in football

---

## Future Improvements

- Add injury/suspension data (Transfermarkt API)
- Add betting odds as a feature (already in raw data: B365H, B365D, B365A)
- Expand to other leagues (La Liga, Bundesliga, Serie A)
- Add shots on target ratio as a form metric
- Build REST API for agent integration