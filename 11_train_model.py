"""
Train and BACKTEST per-stat projection models from training_data.csv.

    python 11_train_model.py

For each of points / rebounds / assists we train three gradient-boosted models:

    mean  -> the expected value (squared-error loss)
    q16   -> the 16th percentile  (quantile loss)
    q84   -> the 84th percentile  (quantile loss)

The two quantiles straddle roughly +/-1 standard deviation, so for any prediction
we get a per-row spread sigma = (q84 - q16) / 2 instead of one global number. That
spread is what turns a point estimate into a P(over) at inference time.

Backtest discipline
--------------------
For the BACKTEST we train only on past seasons and test on the most recent,
held-out season the model has never seen -- a true forward test reported against
two naive baselines (season-to-date and last-10 averages) plus interval coverage
(share of real outcomes inside [q16, q84]; ~68% if the spread is calibrated).

Then we REFIT each model on ALL rows -- the most recent season and the playoffs
included -- and ship THOSE. So the deployed model has learned from everything we
have, while the printed metrics still come from the honest held-out backtest.

Outputs: models/<stat>_{mean,q16,q84}.joblib and models/metadata.json.

Setup:
    pip install -r requirements.txt
"""

import os
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_CSV = os.path.join(HERE, "training_data.csv")
MODEL_DIR = os.path.join(HERE, "models")

TARGETS = ["points", "rebounds", "assists"]

# Held-out season for the forward backtest (trained on everything before it).
TEST_SEASON = "2025-26"

# Numeric features fed to every model. We use only STATIONARY, rate-based
# features (per-game averages). Cumulative career counters -- games_played_todate,
# opp_games_todate, h2h_games -- are deliberately EXCLUDED: they drift upward every
# season, fall out of the training distribution in later seasons, and make the
# model extrapolate badly (this caused a large upward bias in the v1 model).
# (player_id / opponent / name are also excluded so the model generalizes from
# form & matchup rather than memorizing individual players.)
NUMERIC_FEATURES = [
    "home", "days_rest", "season_games_todate",
    "l5_points", "l10_points", "season_points",
    "l5_rebounds", "l10_rebounds", "season_rebounds",
    "l5_assists", "l10_assists", "season_assists",
    "l5_minutes", "l10_minutes",
    "h2h_points", "h2h_rebounds", "h2h_assists",
    "opp_points_allowed_to_pos", "opp_rebounds_allowed_to_pos",
    "opp_assists_allowed_to_pos",
]
CATEGORICAL_FEATURES = ["position", "season_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Quantiles straddling ~+/-1 sigma, and a floor so sigma never collapses to 0.
Q_LOW, Q_HIGH = 0.159, 0.841
SIGMA_FLOOR = 1.0


def make_model(loss, quantile=None):
    """A HistGradientBoostingRegressor with our standard settings.

    Handles NaN features natively (no imputation) and native categorical splits.
    """
    return HistGradientBoostingRegressor(
        loss=loss,
        quantile=quantile,
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        categorical_features=CATEGORICAL_FEATURES,
    )


def prepare(df):
    """Return X (typed for the model) from a frame; categoricals as category dtype."""
    X = df[FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    return X


def anchor_for(df, stat):
    """The strong recent-form baseline we anchor each prediction to.

    Season-to-date average, falling back to L10 then L5 (one is always present
    because rows require >= MIN_PLAYER_GAMES of prior history). The model learns
    the ADJUSTMENT on top of this, never the absolute level -- which keeps it
    immune to season-to-season scoring drift, the bias that sank the v1 model.
    """
    return (
        df[f"season_{stat}"]
        .fillna(df[f"l10_{stat}"])
        .fillna(df[f"l5_{stat}"])
        .astype(float)
    )


def main():
    if not os.path.exists(TRAINING_CSV):
        raise SystemExit(f"Missing {TRAINING_CSV}. Run 10_build_training_data.py first.")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(TRAINING_CSV)
    print(f"Loaded {len(df)} rows. Seasons: {sorted(df['season'].dropna().unique())}\n")

    train_df = df[df["season"] != TEST_SEASON]
    test_df = df[df["season"] == TEST_SEASON]
    if test_df.empty or train_df.empty:
        raise SystemExit(
            f"Need both a train split and a '{TEST_SEASON}' test split; "
            f"got train={len(train_df)} test={len(test_df)}."
        )
    print(f"Train rows: {len(train_df)} (seasons < {TEST_SEASON})")
    print(f"Test rows:  {len(test_df)} ({TEST_SEASON}, held out)\n")

    X_train = prepare(train_df)
    X_test = prepare(test_df)
    X_full = prepare(df)   # everything, incl. the most recent season + playoffs

    seasons_all = sorted(s for s in df["season"].dropna().unique())
    has_playoffs = bool((df["season_type"] == "Playoffs").any())

    metadata = {
        "test_season": TEST_SEASON,
        # Inference MUST replicate this: prediction = anchor + model(residual),
        # where anchor = season_avg -> l10 -> l5 (first non-null).
        "prediction_scheme": "residual_over_anchor",
        "anchor_order": ["season", "l10", "l5"],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "features": FEATURES,
        "sigma_floor": SIGMA_FLOOR,
        "quantiles": {"low": Q_LOW, "high": Q_HIGH},
        # The SHIPPED models are fit on every row; metrics come from the backtest
        # split (train < TEST_SEASON, test == TEST_SEASON) so they stay honest.
        "production_trained_on": "all_seasons",
        "production_seasons": seasons_all,
        "production_includes_playoffs": has_playoffs,
        "production_rows": len(df),
        "backtest_train_rows": len(train_df),
        "backtest_test_rows": len(test_df),
        "metrics": {},
    }

    for stat in TARGETS:
        target_col = f"target_{stat}"
        y_test = test_df[target_col].to_numpy()

        print("=" * 64)
        print(f"TARGET: {stat.upper()}")
        print("-" * 64)

        # Anchor each row to its recent-form baseline; models learn the residual.
        anchor_train = anchor_for(train_df, stat)
        anchor_test = anchor_for(test_df, stat).to_numpy()
        resid_train = train_df[target_col].to_numpy() - anchor_train.to_numpy()

        # --- Train the three models on the RESIDUAL ------------------------
        mean_model = make_model("squared_error").fit(X_train, resid_train)
        q16_model = make_model("quantile", Q_LOW).fit(X_train, resid_train)
        q84_model = make_model("quantile", Q_HIGH).fit(X_train, resid_train)

        # --- Predict on the held-out season (anchor + predicted residual) --
        pred = anchor_test + mean_model.predict(X_test)
        q16 = anchor_test + q16_model.predict(X_test)
        q84 = anchor_test + q84_model.predict(X_test)
        # Counting stats can't go negative.
        pred = np.clip(pred, 0, None)
        q16 = np.clip(q16, 0, None)

        model_mae = mean_absolute_error(y_test, pred)
        model_rmse = np.sqrt(mean_squared_error(y_test, pred))

        # --- Naive baselines on the same held-out rows ---------------------
        pred_series = pd.Series(pred, index=test_df.index)
        season_base = test_df[f"season_{stat}"].fillna(test_df[f"l10_{stat}"]).fillna(pred_series)
        l10_base = test_df[f"l10_{stat}"].fillna(test_df[f"season_{stat}"]).fillna(pred_series)
        base_season_mae = mean_absolute_error(y_test, season_base)
        base_l10_mae = mean_absolute_error(y_test, l10_base)

        # --- Calibration of the predicted spread ---------------------------
        coverage = float(np.mean((y_test >= q16) & (y_test <= q84)))
        bias = float(np.mean(pred - y_test))

        improve_season = 100 * (base_season_mae - model_mae) / base_season_mae
        improve_l10 = 100 * (base_l10_mae - model_mae) / base_l10_mae

        print(f"  [backtest] Model MAE {model_mae:5.3f} | RMSE {model_rmse:5.3f} | bias {bias:+.3f}")
        print(f"  [backtest] Baseline MAE  season-avg {base_season_mae:5.3f} | L10-avg {base_l10_mae:5.3f}")
        print(f"  [backtest] Improvement vs season-avg: {improve_season:+5.1f}%  | vs L10: {improve_l10:+5.1f}%")
        print(f"  [backtest] Interval [q16,q84] coverage: {coverage*100:4.1f}%  (target ~68.2%)")

        # --- Fit the SHIPPED models on ALL rows (incl. newest season + playoffs)
        # The models above were trained on the backtest split only, to keep the
        # metrics honest. The deployed model should learn from everything.
        anchor_full = anchor_for(df, stat)
        resid_full = df[target_col].to_numpy() - anchor_full.to_numpy()
        prod_mean = make_model("squared_error").fit(X_full, resid_full)
        prod_q16 = make_model("quantile", Q_LOW).fit(X_full, resid_full)
        prod_q84 = make_model("quantile", Q_HIGH).fit(X_full, resid_full)
        print(f"  [production] refit on all {len(df)} rows -> saved")

        # --- Persist the production models ---------------------------------
        joblib.dump(prod_mean, os.path.join(MODEL_DIR, f"{stat}_mean.joblib"))
        joblib.dump(prod_q16, os.path.join(MODEL_DIR, f"{stat}_q16.joblib"))
        joblib.dump(prod_q84, os.path.join(MODEL_DIR, f"{stat}_q84.joblib"))

        metadata["metrics"][stat] = {
            "model_mae": round(model_mae, 4),
            "model_rmse": round(model_rmse, 4),
            "bias": round(bias, 4),
            "baseline_season_mae": round(base_season_mae, 4),
            "baseline_l10_mae": round(base_l10_mae, 4),
            "improvement_vs_season_pct": round(improve_season, 2),
            "improvement_vs_l10_pct": round(improve_l10, 2),
            "interval_coverage": round(coverage, 4),
        }

    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 64)
    print(f"DONE. Models + metadata.json written to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
