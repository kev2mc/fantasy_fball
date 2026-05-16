"""
predict_2026.py  —  Fantasy football season-points prediction for 2026.

Methodology
-----------
Target   : points_for (total regular-season fantasy points)
Features : prior-season and career/rolling stats per manager — zero data leakage
           (all features for season Y are computed from seasons < Y only)
CV       : leave-one-season-out (LOSO) — trains on all other seasons, predicts held-out one

Models
------
Ridge, Lasso, ElasticNet, Decision Tree, Random Forest,
Gradient Boosting (sklearn), XGBoost

Usage
-----
  python predict_2026.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor

DATA_DIR     = Path('data')
RANDOM_STATE = 42
MIN_PRIOR_SEASONS = 1   # rows with fewer prior seasons are excluded
KEEPER_START = 2019


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_season_df():
    p = DATA_DIR / 'manager_season_df.csv'
    if not p.exists():
        raise FileNotFoundError(f'{p} not found — run data_science.py first')
    df = pd.read_csv(p)
    df['champion']      = df['champion'].astype(int)
    df['made_playoffs'] = df['made_playoffs'].astype(int)
    df['percentage']    = pd.to_numeric(df['percentage'], errors='coerce')
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _safe(row, col):
    """Get scalar from row; NaN if column absent or null."""
    v = row.get(col, np.nan) if isinstance(row, dict) else (
        row[col] if col in row.index else np.nan)
    return np.nan if pd.isna(v) else float(v)


def _scol(df, col):
    """Return column as Series, or NaN Series if column absent."""
    return df[col] if col in df.columns else pd.Series(np.nan, index=df.index)


def _feature_row(prior, recent3):
    """Build one feature dict from a manager's prior-season DataFrame."""
    last = prior.iloc[-1]
    gp  = lambda c: _safe(last, c)
    ga  = lambda c: _scol(prior, c).mean()
    gr3 = lambda c: _scol(recent3, c).mean()

    ew_sum  = _scol(prior, 'exp_wins').sum()
    rw_sum  = _scol(prior, 'reg_weeks').sum()
    w_sum   = prior['wins'].sum()
    l_sum   = prior['losses'].sum()
    t_sum   = prior['ties'].sum()
    r3w_sum = recent3['wins'].sum()
    r3l_sum = recent3['losses'].sum()
    r3t_sum = recent3['ties'].sum()

    return {
        # previous season
        'prev_points_for':      gp('points_for'),
        'prev_win_pct':         gp('percentage'),
        'prev_made_playoffs':   gp('made_playoffs'),
        'prev_luck_wins':       gp('luck_wins'),
        'prev_moves':           gp('number_of_moves'),
        'prev_trades':          gp('number_of_trades'),
        'prev_pts_left_bench':  gp('avg_pts_left_per_week'),
        'prev_draft_overpf':    gp('draft_overperformance'),
        # career aggregates
        'career_avg_pf':        ga('points_for'),
        'career_playoff_rate':  ga('made_playoffs'),
        'career_championships': float(prior['champion'].sum()),
        'career_win_pct':       w_sum / max(w_sum + l_sum + t_sum, 1),
        'career_exp_win_rate':  ew_sum / max(rw_sum, 1),
        'career_pts_left_bench':ga('avg_pts_left_per_week'),
        'career_draft_overpf':  _scol(prior, 'draft_overperformance').mean(),
        'seasons_played':       float(len(prior)),
        # 3-season rolling
        'roll3_avg_pf':         gr3('points_for'),
        'roll3_win_pct':        r3w_sum / max(r3w_sum + r3l_sum + r3t_sum, 1),
        'roll3_luck_wins':      gr3('luck_wins'),
        'roll3_moves':          gr3('number_of_moves'),
        'roll3_pts_left_bench': gr3('avg_pts_left_per_week'),
    }


FEATURE_COLS = [
    # previous season (8)
    'prev_points_for', 'prev_win_pct', 'prev_made_playoffs',
    'prev_luck_wins', 'prev_moves', 'prev_trades',
    'prev_pts_left_bench', 'prev_draft_overpf',
    # career aggregates (8)
    'career_avg_pf', 'career_playoff_rate', 'career_championships',
    'career_win_pct', 'career_exp_win_rate', 'career_pts_left_bench',
    'career_draft_overpf', 'seasons_played',
    # 3-season rolling (5)
    'roll3_avg_pf', 'roll3_win_pct', 'roll3_luck_wins', 'roll3_moves',
    'roll3_pts_left_bench',
]


def build_feature_matrix(df):
    """One row per (manager, season); features use only prior-season data."""
    records = []
    for manager, grp in df.groupby('manager'):
        grp = grp.sort_values('season').reset_index(drop=True)
        for i in range(len(grp)):
            prior = grp.iloc[:i]
            if len(prior) < MIN_PRIOR_SEASONS:
                continue
            row     = grp.iloc[i]
            recent3 = prior.tail(3)
            feat = {'manager': manager, 'season': int(row['season']),
                    'points_for': float(row['points_for'])}
            feat.update(_feature_row(prior, recent3))
            feat['draft_position'] = _safe(row, 'draft_position')
            records.append(feat)
    return pd.DataFrame(records)


def build_2026_rows(df):
    """Feature rows for 2026 prediction: use ALL historical data as prior."""
    latest = df['season'].max()
    records = []
    for manager, grp in df.groupby('manager'):
        grp = grp.sort_values('season').reset_index(drop=True)
        if grp['season'].max() < latest:
            continue   # manager did not play most recent season
        recent3 = grp.tail(3)
        feat = {'manager': manager, 'season': 2026}
        feat.update(_feature_row(grp, recent3))
        records.append(feat)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def get_models():
    return {
        'Ridge':        Pipeline([('sc', StandardScaler()),
                                  ('m',  Ridge(alpha=10.0))]),
        'Lasso':        Pipeline([('sc', StandardScaler()),
                                  ('m',  Lasso(alpha=5.0, max_iter=10_000))]),
        'ElasticNet':   Pipeline([('sc', StandardScaler()),
                                  ('m',  ElasticNet(alpha=5.0, l1_ratio=0.5, max_iter=10_000))]),
        'DecisionTree': DecisionTreeRegressor(max_depth=3, min_samples_leaf=5,
                                              random_state=RANDOM_STATE),
        'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=4,
                                              min_samples_leaf=3, max_features=0.7,
                                              random_state=RANDOM_STATE),
        'GradBoost':    GradientBoostingRegressor(n_estimators=300, max_depth=2,
                                                  learning_rate=0.05, subsample=0.8,
                                                  min_samples_leaf=3,
                                                  random_state=RANDOM_STATE),
        'XGBoost':      XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     reg_alpha=1.0, reg_lambda=5.0, min_child_weight=3,
                                     random_state=RANDOM_STATE, verbosity=0),
    }


# ---------------------------------------------------------------------------
# Leave-one-season-out cross-validation
# ---------------------------------------------------------------------------

def loso_cv(feat_df, target='points_for', min_test=5, model_features=None):
    """LOSO CV over all seasons. Returns (results_df, predictions_df).

    model_features: optional dict {model_name: [feature_cols]} for per-model
                    feature sets. Falls back to FEATURE_COLS for any missing model.
    """
    seasons  = sorted(feat_df['season'].unique())
    models   = get_models()
    default_fcols = [c for c in FEATURE_COLS if c in feat_df.columns]

    season_metrics = {n: {'rmse': [], 'mae': [], 'spearman': []} for n in models}
    all_preds = []

    for held in seasons:
        train = feat_df[feat_df['season'] != held]
        test  = feat_df[feat_df['season'] == held]
        if len(test) < min_test or len(train) < 20:
            continue

        row_preds = test[['manager', 'season', target]].copy().reset_index(drop=True)
        y_te = test[target].values

        for name, model in models.items():
            fcols = (model_features or {}).get(name, default_fcols)
            fcols = [c for c in fcols if c in feat_df.columns]

            train_med = train[fcols].median()
            X_tr = train[fcols].fillna(train_med).fillna(0)
            y_tr = train[target]
            X_te = test[fcols].fillna(train_med).fillna(0)

            m = clone(model)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)

            rmse = np.sqrt(mean_squared_error(y_te, preds))
            mae  = mean_absolute_error(y_te, preds)
            rho, _ = spearmanr(y_te, preds)

            season_metrics[name]['rmse'].append(rmse)
            season_metrics[name]['mae'].append(mae)
            season_metrics[name]['spearman'].append(float(np.nan_to_num(rho)))
            row_preds[f'pred_{name}'] = np.round(preds, 1)

        all_preds.append(row_preds)

    rows = []
    for name in models:
        m = season_metrics[name]
        rows.append({
            'model':    name,
            'rmse':     round(float(np.mean(m['rmse'])),     1),
            'mae':      round(float(np.mean(m['mae'])),      1),
            'spearman': round(float(np.mean(m['spearman'])), 3),
            'seasons':  len(m['rmse']),
        })

    results_df = pd.DataFrame(rows).sort_values('rmse').reset_index(drop=True)
    pred_df    = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return results_df, pred_df


# ---------------------------------------------------------------------------
# 2026 prediction
# ---------------------------------------------------------------------------

def predict_2026(feat_df, rows_2026, best_model_name, best_features=None, target='points_for'):
    """Train best model on all historical data, predict 2026."""
    fcols = best_features if best_features else [c for c in FEATURE_COLS if c in feat_df.columns]
    fcols = [c for c in fcols if c in feat_df.columns]
    model = clone(get_models()[best_model_name])

    train_med = feat_df[fcols].median()
    model.fit(feat_df[fcols].fillna(train_med).fillna(0), feat_df[target])

    X_2026 = rows_2026[fcols].fillna(train_med).fillna(0)
    preds  = model.predict(X_2026)

    out = pd.DataFrame({
        'manager':               rows_2026['manager'].values,
        'predicted_points_for':  np.round(preds, 1),
    }).sort_values('predicted_points_for', ascending=False).reset_index(drop=True)
    out.index     = out.index + 1
    out.index.name = 'predicted_rank'
    return out


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def print_importance(feat_df, model_name, target='points_for', top_n=12):
    fcols = [c for c in FEATURE_COLS if c in feat_df.columns]
    model = clone(get_models()[model_name])
    model.fit(feat_df[fcols].fillna(feat_df[fcols].median()).fillna(0), feat_df[target])

    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=fcols)
    elif hasattr(model, 'named_steps'):
        inner = model.named_steps.get('m')
        if hasattr(inner, 'coef_'):
            imp = pd.Series(np.abs(inner.coef_), index=fcols)
        else:
            return
    else:
        return

    top = imp.nlargest(top_n)
    width = max(len(c) for c in top.index)
    print(f'\n  Top {top_n} features ({model_name}):')
    for feat, val in top.items():
        bar = '#' * int(val / top.max() * 28)
        print(f'    {feat:{width}s}  {bar:<28s} {val:.4f}')


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def _loso_rmse_single(feat_df, model, fcols, target='points_for', min_test=5):
    """LOSO CV RMSE for one model and one feature subset."""
    seasons = sorted(feat_df['season'].unique())
    rmses = []
    for held in seasons:
        train = feat_df[feat_df['season'] != held]
        test  = feat_df[feat_df['season'] == held]
        if len(test) < min_test or len(train) < 20:
            continue
        train_med = train[fcols].median()
        X_tr = train[fcols].fillna(train_med).fillna(0)
        X_te = test[fcols].fillna(train_med).fillna(0)
        y_tr = train[target].values
        y_te = test[target].values
        m = clone(model)
        m.fit(X_tr, y_tr)
        rmses.append(np.sqrt(mean_squared_error(y_te, m.predict(X_te))))
    return float(np.mean(rmses)) if rmses else float('inf')


def _get_importance_ranking(model, fcols, feat_df, target):
    """Fit model on all features and return features ranked by importance."""
    train_med = feat_df[fcols].median()
    X = feat_df[fcols].fillna(train_med).fillna(0)
    m = clone(model)
    m.fit(X, feat_df[target])
    if hasattr(m, 'feature_importances_'):
        imp = pd.Series(m.feature_importances_, index=fcols)
    elif hasattr(m, 'named_steps'):
        inner = m.named_steps['m']
        imp = pd.Series(np.abs(inner.coef_), index=fcols)
    else:
        imp = pd.Series(np.ones(len(fcols)), index=fcols)
    return imp.sort_values(ascending=False).index.tolist()


def find_optimal_features(feat_df, target='points_for'):
    """Importance-ranked subset sweep: for each model find the N features
    (ranked by full-model importance) that minimise LOSO CV RMSE."""

    # Faster models for the sweep — same hyperparams, fewer trees
    fast_models = {
        'Ridge':        Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=10.0))]),
        'Lasso':        Pipeline([('sc', StandardScaler()), ('m', Lasso(alpha=5.0, max_iter=10_000))]),
        'ElasticNet':   Pipeline([('sc', StandardScaler()), ('m', ElasticNet(alpha=5.0, l1_ratio=0.5, max_iter=10_000))]),
        'DecisionTree': DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE),
        'RandomForest': RandomForestRegressor(n_estimators=80, max_depth=4, min_samples_leaf=3,
                                              max_features=0.7, random_state=RANDOM_STATE),
        'GradBoost':    GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.05,
                                                  subsample=0.8, min_samples_leaf=3, random_state=RANDOM_STATE),
        'XGBoost':      XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
                                     reg_lambda=5.0, min_child_weight=3,
                                     random_state=RANDOM_STATE, verbosity=0),
    }

    all_fcols = [c for c in FEATURE_COLS if c in feat_df.columns]
    results = {}

    for name, model in fast_models.items():
        print(f'  {name}...', end='', flush=True)
        ranked = _get_importance_ranking(model, all_fcols, feat_df, target)

        best_rmse, best_n = float('inf'), len(all_fcols)
        for n in range(1, len(all_fcols) + 1):
            rmse = _loso_rmse_single(feat_df, model, ranked[:n], target)
            if rmse < best_rmse:
                best_rmse, best_n = rmse, n

        results[name] = {
            'features': ranked[:best_n],
            'n':        best_n,
            'rmse':     round(best_rmse, 1),
        }
        print(f' n={best_n}, RMSE={best_rmse:.1f}')

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pd.set_option('display.float_format', '{:.1f}'.format)
    pd.set_option('display.max_columns', None)

    print('Loading data...')
    df = load_season_df()
    print(f'  {len(df)} rows | {df["season"].nunique()} seasons '
          f'({df["season"].min()}–{df["season"].max()}) | '
          f'{df["manager"].nunique()} managers')

    print('\nBuilding feature matrix (no-leakage rolling features)...')
    feat_df   = build_feature_matrix(df).dropna(subset=['points_for'])
    rows_2026 = build_2026_rows(df)
    fcols     = [c for c in FEATURE_COLS if c in feat_df.columns]
    non_null  = feat_df[fcols].notna().any()
    active    = non_null[non_null].index.tolist()
    print(f'  {len(feat_df)} training rows | {len(active)}/{len(fcols)} features with data')
    print(f'  {len(rows_2026)} managers eligible for 2026 prediction')

    print('\nRunning leave-one-season-out cross-validation across all models...')
    results_df, pred_df = loso_cv(feat_df)

    w = 13
    print(f'\n{"="*54}')
    print(f' {"Model":<{w}} {"RMSE":>7}  {"MAE":>7}  {"Spearman":>9}  {"Seasons":>7}')
    print(f'{"-"*54}')
    for _, row in results_df.iterrows():
        flag = ' <- best' if _ == 0 else ''
        print(f' {row["model"]:<{w}} {row["rmse"]:>7.1f}  {row["mae"]:>7.1f}'
              f'  {row["spearman"]:>9.3f}  {row["seasons"]:>7}{flag}')
    print(f'{"="*54}')
    print(' RMSE / MAE in fantasy points  |  Spearman = rank correlation')

    best_model = results_df.iloc[0]['model']

    print('\nFeature importances:')
    print_importance(feat_df, 'XGBoost')
    print_importance(feat_df, 'RandomForest')

    print('\nRunning feature selection (importance-ranked subset sweep)...')
    sel = find_optimal_features(feat_df)
    w2 = 13
    print(f'\n{"="*66}')
    print(f' {"Model":<{w2}} {"Opt RMSE":>9}  {"Full RMSE":>9}  {"Delta":>7}  {"N feats":>7}')
    print(f'{"-"*66}')
    full_rmse = {r["model"]: r["rmse"] for _, r in results_df.iterrows()}
    for name, s in sorted(sel.items(), key=lambda x: x[1]['rmse']):
        delta = s['rmse'] - full_rmse[name]
        print(f' {name:<{w2}} {s["rmse"]:>9.1f}  {full_rmse[name]:>9.1f}  {delta:>+7.1f}  {s["n"]:>7}')
    print(f'{"="*66}')

    print('\nOptimal feature sets per model:')
    for name, s in sorted(sel.items(), key=lambda x: x[1]['rmse']):
        print(f'  {name} (n={s["n"]}, RMSE={s["rmse"]}): {s["features"]}')

    # --- Final CV pass with optimal per-model features (full estimators) ---
    print('\nRunning final CV with optimal features per model (full estimators)...')
    opt_features = {name: s['features'] for name, s in sel.items()}
    opt_results_df, opt_pred_df = loso_cv(feat_df, model_features=opt_features)

    w = 13
    print(f'\n{"="*54}')
    print(f' {"Model":<{w}} {"RMSE":>7}  {"MAE":>7}  {"Spearman":>9}  {"N feats":>7}')
    print(f'{"-"*54}')
    for _, row in opt_results_df.iterrows():
        n = sel[row['model']]['n']
        flag = ' <- best' if _ == 0 else ''
        print(f' {row["model"]:<{w}} {row["rmse"]:>7.1f}  {row["mae"]:>7.1f}'
              f'  {row["spearman"]:>9.3f}  {n:>7}{flag}')
    print(f'{"="*54}')
    print(' RMSE / MAE in fantasy points  |  Spearman = rank correlation')

    best_opt_model = opt_results_df.iloc[0]['model']
    best_opt_features = sel[best_opt_model]['features']

    print(f'\n{"="*54}')
    print(f' 2026 Predicted Points  (model: {best_opt_model})')
    print(f' Features: {best_opt_features}')
    print(f' Trained on {df["season"].min()}-{df["season"].max()} seasons')
    print(f'{"-"*54}')
    preds_2026 = predict_2026(feat_df, rows_2026, best_opt_model, best_opt_features)
    for rank, row in preds_2026.iterrows():
        print(f'  {rank:2d}.  {row["manager"]:<22s}  {row["predicted_points_for"]:>7.1f} pts')
    print(f'{"="*54}')
    print(' Note: draft_position unknown until 2026 draft -')
    print(' predictions will sharpen once keepers + draft order are set.')

    opt_results_df.to_csv(DATA_DIR / 'model_comparison.csv', index=False)
    preds_2026.to_csv(DATA_DIR / 'predictions_2026.csv')
    if not opt_pred_df.empty:
        opt_pred_df.to_csv(DATA_DIR / 'loso_predictions.csv', index=False)
    print('\nSaved: model_comparison.csv, predictions_2026.csv, loso_predictions.csv')
