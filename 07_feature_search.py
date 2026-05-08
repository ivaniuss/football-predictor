"""
07_feature_search.py — Automated Feature Search
Concept: try → measure (log-loss) → keep if better → discard if worse → repeat

Usage: uv run python 07_feature_search.py
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from datetime import datetime

# ─── CONFIG ──────────────────────────────────────────────────────────────────

BASELINE_FEATURES = [
    'home_form_last5', 'away_form_last5',
    'home_avg_scored', 'home_avg_conceded',
    'away_avg_scored', 'away_avg_conceded',
    'ref_matches', 'ref_home_win_rate',
    'ref_yellows_avg', 'ref_reds_avg', 'ref_goals_avg',
    'elo_home', 'elo_away', 'elo_diff',
    'h2h_matches', 'h2h_home_wins', 'h2h_draws', 'h2h_goals_avg',
]

# Hyperparams (from your best Optuna run)
MODEL_PARAMS = {
    'n_estimators': 383, 'max_depth': 2, 'learning_rate': 0.199,
    'subsample': 0.782, 'colsample_bytree': 0.865,
    'min_child_weight': 6, 'gamma': 4.620,
    'eval_metric': 'mlogloss', 'random_state': 42,
}

SEASONS = {
    '2526': '2025/26', '2425': '2024/25', '2324': '2023/24',
    '2223': '2022/23', '2122': '2021/22', '2021': '2020/21',
    '1920': '2019/20', '1819': '2018/19', '1718': '2017/18',
    '1617': '2016/17', '1516': '2015/16', '1415': '2014/15',
}

# ─── STEP 1: DOWNLOAD DATA (with odds) ──────────────────────────────────────

print("=" * 60)
print("STEP 1: Downloading data with betting odds...")
print("=" * 60)

USEFUL_COLS = [
    'Date', 'HomeTeam', 'AwayTeam', 'Referee',
    'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG',
    'HS', 'AS', 'HST', 'AST', 'HC', 'AC',
    'HF', 'AF', 'HY', 'AY', 'HR', 'AR',
    # Betting odds (pre-match, NOT leakage)
    'B365H', 'B365D', 'B365A',
    'AvgH', 'AvgD', 'AvgA',
]

dfs = []
for code, name in SEASONS.items():
    url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
    try:
        temp = pd.read_csv(url, usecols=lambda c: c in USEFUL_COLS)
        temp['Season'] = name
        dfs.append(temp)
        print(f"  ✅ {name} → {len(temp)} matches")
    except Exception as e:
        print(f"  ❌ {name} → {e}")

df = pd.concat(dfs, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
df = df.dropna(subset=['FTR']).sort_values('Date').reset_index(drop=True)
print(f"\n  Total: {len(df)} matches")

# ─── STEP 2: FEATURE ENGINEERING (all existing + candidates) ────────────────

print("\n" + "=" * 60)
print("STEP 2: Engineering ALL features (baseline + candidates)...")
print("=" * 60)

# --- FORM LAST 5 (existing) ---
records = []
for _, row in df.iterrows():
    records.append({'Date': row['Date'], 'Team': row['HomeTeam'],
                    'Points': 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0)})
    records.append({'Date': row['Date'], 'Team': row['AwayTeam'],
                    'Points': 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0)})

team_df = pd.DataFrame(records).sort_values('Date').reset_index(drop=True)
team_df['form_last5'] = team_df.groupby('Team')['Points'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean())

home_form = team_df[['Date', 'Team', 'form_last5']].rename(
    columns={'Team': 'HomeTeam', 'form_last5': 'home_form_last5'})
away_form = team_df[['Date', 'Team', 'form_last5']].rename(
    columns={'Team': 'AwayTeam', 'form_last5': 'away_form_last5'})
df = df.merge(home_form, on=['Date', 'HomeTeam'], how='left')
df = df.merge(away_form, on=['Date', 'AwayTeam'], how='left')
print("  ✅ Form last 5")

# --- GOALS AVG (existing) ---
goal_records = []
for _, row in df.iterrows():
    goal_records.append({'Date': row['Date'], 'Team': row['HomeTeam'],
                         'scored': row['FTHG'], 'conceded': row['FTAG']})
    goal_records.append({'Date': row['Date'], 'Team': row['AwayTeam'],
                         'scored': row['FTAG'], 'conceded': row['FTHG']})

goals_df = pd.DataFrame(goal_records).sort_values('Date').reset_index(drop=True)
goals_df['avg_scored_last5'] = goals_df.groupby('Team')['scored'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean())
goals_df['avg_conceded_last5'] = goals_df.groupby('Team')['conceded'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean())

home_goals = goals_df[['Date', 'Team', 'avg_scored_last5', 'avg_conceded_last5']].rename(
    columns={'Team': 'HomeTeam', 'avg_scored_last5': 'home_avg_scored',
             'avg_conceded_last5': 'home_avg_conceded'})
away_goals = goals_df[['Date', 'Team', 'avg_scored_last5', 'avg_conceded_last5']].rename(
    columns={'Team': 'AwayTeam', 'avg_scored_last5': 'away_avg_scored',
             'avg_conceded_last5': 'away_avg_conceded'})
df = df.merge(home_goals, on=['Date', 'HomeTeam'], how='left')
df = df.merge(away_goals, on=['Date', 'AwayTeam'], how='left')
print("  ✅ Goals avg")

# --- REFEREE (existing) ---
referee_stats = []
for idx, row in df.iterrows():
    ref, date = row['Referee'], row['Date']
    past = df[(df['Referee'] == ref) & (df['Date'] < date)]
    if len(past) == 0:
        referee_stats.append({'ref_matches': 0, 'ref_home_win_rate': None,
                              'ref_yellows_avg': None, 'ref_reds_avg': None, 'ref_goals_avg': None})
    else:
        referee_stats.append({
            'ref_matches': len(past),
            'ref_home_win_rate': (past['FTR'] == 'H').sum() / len(past),
            'ref_yellows_avg': (past['HY'] + past['AY']).mean(),
            'ref_reds_avg': (past['HR'] + past['AR']).mean(),
            'ref_goals_avg': (past['FTHG'] + past['FTAG']).mean(),
        })
df = pd.concat([df, pd.DataFrame(referee_stats)], axis=1)
print("  ✅ Referee stats")

# --- ELO (existing) ---
K, BASE_ELO = 32, 1500
elo_dict, elo_records = {}, []
for _, row in df.iterrows():
    home, away = row['HomeTeam'], row['AwayTeam']
    if home not in elo_dict: elo_dict[home] = BASE_ELO
    if away not in elo_dict: elo_dict[away] = BASE_ELO
    elo_home, elo_away = elo_dict[home], elo_dict[away]
    expected_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
    if row['FTR'] == 'H':   ah, aa = 1, 0
    elif row['FTR'] == 'A': ah, aa = 0, 1
    else:                    ah, aa = 0.5, 0.5
    elo_records.append({'Date': row['Date'], 'HomeTeam': home, 'AwayTeam': away,
                        'elo_home': elo_home, 'elo_away': elo_away, 'elo_diff': elo_home - elo_away})
    elo_dict[home] = elo_home + K * (ah - expected_home)
    elo_dict[away] = elo_away + K * (aa - (1 - expected_home))
df = df.merge(pd.DataFrame(elo_records), on=['Date', 'HomeTeam', 'AwayTeam'], how='left')
print("  ✅ ELO ratings")

# --- H2H (existing) ---
h2h_records = []
for _, row in df.iterrows():
    home, away, date = row['HomeTeam'], row['AwayTeam'], row['Date']
    past = df[(df['Date'] < date) &
              (((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) |
               ((df['HomeTeam'] == away) & (df['AwayTeam'] == home)))].tail(5)
    if len(past) == 0:
        h2h_records.append({'h2h_matches': 0, 'h2h_home_wins': None,
                            'h2h_draws': None, 'h2h_goals_avg': None})
    else:
        hw = sum(1 for _, p in past.iterrows()
                 if (p['HomeTeam'] == home and p['FTR'] == 'H') or
                    (p['HomeTeam'] == away and p['FTR'] == 'A'))
        draws = (past['FTR'] == 'D').sum()
        h2h_records.append({
            'h2h_matches': len(past), 'h2h_home_wins': hw / len(past),
            'h2h_draws': draws / len(past),
            'h2h_goals_avg': (past['FTHG'] + past['FTAG']).mean(),
        })
df = pd.concat([df, pd.DataFrame(h2h_records)], axis=1)
print("  ✅ H2H stats")

# ─── NEW CANDIDATE FEATURES ─────────────────────────────────────────────────

# --- BETTING ODDS (implied probabilities, normalized) ---
if 'AvgH' in df.columns and df['AvgH'].notna().sum() > 100:
    overround = (1/df['AvgH']) + (1/df['AvgD']) + (1/df['AvgA'])
    df['odds_prob_home'] = (1/df['AvgH']) / overround
    df['odds_prob_draw'] = (1/df['AvgD']) / overround
    df['odds_prob_away'] = (1/df['AvgA']) / overround
    df['odds_home_away_diff'] = df['odds_prob_home'] - df['odds_prob_away']
    print("  ✅ Betting odds (implied probabilities)")
else:
    print("  ⚠️ No odds data available")

# --- SHOTS ROLLING AVG ---
shot_records = []
for _, row in df.iterrows():
    if pd.notna(row.get('HS')):
        shot_records.append({'Date': row['Date'], 'Team': row['HomeTeam'],
                             'shots': row['HS'], 'sot': row.get('HST', 0),
                             'corners': row.get('HC', 0), 'fouls': row.get('HF', 0)})
        shot_records.append({'Date': row['Date'], 'Team': row['AwayTeam'],
                             'shots': row['AS'], 'sot': row.get('AST', 0),
                             'corners': row.get('AC', 0), 'fouls': row.get('AF', 0)})

if shot_records:
    shots_df = pd.DataFrame(shot_records).sort_values('Date').reset_index(drop=True)
    for col in ['shots', 'sot', 'corners', 'fouls']:
        shots_df[f'{col}_avg5'] = shots_df.groupby('Team')[col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    # Shots on target percentage
    shots_df['sot_pct'] = shots_df['sot'] / shots_df['shots'].replace(0, np.nan)
    shots_df['sot_pct_avg5'] = shots_df.groupby('Team')['sot_pct'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())

    home_shots = shots_df[['Date', 'Team', 'shots_avg5', 'sot_avg5', 'sot_pct_avg5',
                           'corners_avg5', 'fouls_avg5']].rename(
        columns={'Team': 'HomeTeam', 'shots_avg5': 'home_shots_avg',
                 'sot_avg5': 'home_sot_avg', 'sot_pct_avg5': 'home_sot_pct',
                 'corners_avg5': 'home_corners_avg', 'fouls_avg5': 'home_fouls_avg'})
    away_shots = shots_df[['Date', 'Team', 'shots_avg5', 'sot_avg5', 'sot_pct_avg5',
                           'corners_avg5', 'fouls_avg5']].rename(
        columns={'Team': 'AwayTeam', 'shots_avg5': 'away_shots_avg',
                 'sot_avg5': 'away_sot_avg', 'sot_pct_avg5': 'away_sot_pct',
                 'corners_avg5': 'away_corners_avg', 'fouls_avg5': 'away_fouls_avg'})
    df = df.merge(home_shots, on=['Date', 'HomeTeam'], how='left')
    df = df.merge(away_shots, on=['Date', 'AwayTeam'], how='left')
    print("  ✅ Shots, SOT%, corners, fouls (rolling avg)")

# --- GOAL DIFFERENCE ROLLING ---
gd_records = []
for _, row in df.iterrows():
    gd_records.append({'Date': row['Date'], 'Team': row['HomeTeam'],
                       'gd': row['FTHG'] - row['FTAG']})
    gd_records.append({'Date': row['Date'], 'Team': row['AwayTeam'],
                       'gd': row['FTAG'] - row['FTHG']})
gd_df = pd.DataFrame(gd_records).sort_values('Date').reset_index(drop=True)
gd_df['gd_avg5'] = gd_df.groupby('Team')['gd'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean())
home_gd = gd_df[['Date', 'Team', 'gd_avg5']].rename(
    columns={'Team': 'HomeTeam', 'gd_avg5': 'home_gd_avg'})
away_gd = gd_df[['Date', 'Team', 'gd_avg5']].rename(
    columns={'Team': 'AwayTeam', 'gd_avg5': 'away_gd_avg'})
df = df.merge(home_gd, on=['Date', 'HomeTeam'], how='left')
df = df.merge(away_gd, on=['Date', 'AwayTeam'], how='left')
print("  ✅ Goal difference (rolling avg)")

# --- DAYS SINCE LAST MATCH ---
last_match = {}
days_rest_h, days_rest_a = [], []
for _, row in df.iterrows():
    home, away, date = row['HomeTeam'], row['AwayTeam'], row['Date']
    days_rest_h.append((date - last_match[home]).days if home in last_match else None)
    days_rest_a.append((date - last_match[away]).days if away in last_match else None)
    last_match[home] = date
    last_match[away] = date
df['days_rest_home'] = days_rest_h
df['days_rest_away'] = days_rest_a
print("  ✅ Days rest")

# ─── PREPARE MODEL DATA ─────────────────────────────────────────────────────

df['target'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

# All candidate features (baseline + new)
ALL_CANDIDATES = [
    # --- baseline (18) ---
    'home_form_last5', 'away_form_last5',
    'home_avg_scored', 'home_avg_conceded', 'away_avg_scored', 'away_avg_conceded',
    'ref_matches', 'ref_home_win_rate', 'ref_yellows_avg', 'ref_reds_avg', 'ref_goals_avg',
    'elo_home', 'elo_away', 'elo_diff',
    'h2h_matches', 'h2h_home_wins', 'h2h_draws', 'h2h_goals_avg',
    # --- new candidates ---
    'odds_prob_home', 'odds_prob_draw', 'odds_prob_away', 'odds_home_away_diff',
    'home_shots_avg', 'away_shots_avg',
    'home_sot_avg', 'away_sot_avg',
    'home_sot_pct', 'away_sot_pct',
    'home_corners_avg', 'away_corners_avg',
    'home_fouls_avg', 'away_fouls_avg',
    'home_gd_avg', 'away_gd_avg',
    'days_rest_home', 'days_rest_away',
]

# Filter to features that actually exist and have data
available = [f for f in ALL_CANDIDATES if f in df.columns and df[f].notna().sum() > 1000]
print(f"\n  Available features: {len(available)} / {len(ALL_CANDIDATES)}")

# Drop rows with NaN in baseline features
df_model = df.dropna(subset=BASELINE_FEATURES + ['target']).reset_index(drop=True)
print(f"  Model rows: {len(df_model)}")

# ─── STEP 3: EVALUATION FUNCTION ────────────────────────────────────────────


def evaluate(feature_list):
    """Evaluate feature set with TimeSeriesSplit. Returns mean log-loss."""
    data = df_model[feature_list].copy()
    # Fill remaining NaN with median (for candidate features)
    data = data.fillna(data.median())
    X = data.values
    y = df_model['target'].values

    tscv = TimeSeriesSplit(n_splits=5)
    losses = []
    for train_idx, val_idx in tscv.split(X):
        model = XGBClassifier(**MODEL_PARAMS)
        model.fit(X[train_idx], y[train_idx], verbose=False)
        probs = model.predict_proba(X[val_idx])
        losses.append(log_loss(y[val_idx], probs))
    return np.mean(losses)


# ─── STEP 4: AUTOMATED FORWARD FEATURE SELECTION ────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Baseline evaluation...")
print("=" * 60)

baseline_score = evaluate(BASELINE_FEATURES)
print(f"  Baseline log-loss: {baseline_score:.6f} ({len(BASELINE_FEATURES)} features)")

print("\n" + "=" * 60)
print("STEP 4: Forward feature selection...")
print("=" * 60)

current_features = BASELINE_FEATURES.copy()
current_score = baseline_score
remaining = [f for f in available if f not in current_features]

results_log = []
results_log.append({
    'round': 0, 'action': 'baseline', 'feature': '-',
    'log_loss': baseline_score, 'delta': 0, 'status': 'baseline',
    'n_features': len(current_features),
})

round_num = 0
while remaining:
    round_num += 1
    print(f"\n--- Round {round_num} ({len(remaining)} candidates left) ---")
    best_candidate = None
    best_score = current_score

    for candidate in remaining:
        trial = current_features + [candidate]
        score = evaluate(trial)
        delta = score - current_score
        symbol = "↓" if delta < 0 else "↑"
        print(f"  {candidate:25s} → {score:.6f} ({symbol}{abs(delta):.6f})")

        results_log.append({
            'round': round_num, 'action': 'trial', 'feature': candidate,
            'log_loss': score, 'delta': delta,
            'status': 'pending', 'n_features': len(trial),
        })

        if score < best_score:
            best_candidate = candidate
            best_score = score

    if best_candidate:
        delta = best_score - current_score
        current_features.append(best_candidate)
        current_score = best_score
        remaining.remove(best_candidate)
        # Update status
        for r in results_log:
            if r['feature'] == best_candidate and r['round'] == round_num:
                r['status'] = 'keep'
        print(f"\n  ✅ KEEP: {best_candidate} → {current_score:.6f} (Δ {delta:.6f})")
    else:
        print(f"\n  🛑 No improvement found. Stopping.")
        break

# ─── STEP 5: RESULTS ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

improvement = baseline_score - current_score
print(f"\n  Baseline log-loss:  {baseline_score:.6f}")
print(f"  Final log-loss:     {current_score:.6f}")
print(f"  Improvement:        {improvement:.6f} ({improvement/baseline_score*100:.2f}%)")
print(f"  Features:           {len(BASELINE_FEATURES)} → {len(current_features)}")

added = [f for f in current_features if f not in BASELINE_FEATURES]
if added:
    print(f"\n  New features added:")
    for f in added:
        print(f"    + {f}")

print(f"\n  ─── WINNING FEATURE_COLS (copy to your notebooks) ───")
print(f"\n  FEATURE_COLS = [")
for f in current_features:
    print(f"      '{f}',")
print(f"  ]")

# Save results
results_df = pd.DataFrame(results_log)
results_df.to_csv('data/feature_search_results.csv', index=False)
print(f"\n  ✅ Full results saved to data/feature_search_results.csv")
