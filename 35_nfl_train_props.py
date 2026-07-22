"""
Train and BACKTEST per-stat NFL prop models from nfl_training_data.csv.

    python 35_nfl_train_props.py

The football sibling of 11_train_model.py. For every modeled stat (passing /
rushing / receiving yards, attempts, receptions, targets, TDs, INTs) we train
three gradient-boosted models:

    mean -> expected value (squared-error loss)
    q16  -> 16th percentile   (quantile loss)
    q84  -> 84th percentile   (quantile loss)

so every prediction carries a per-row spread sigma = (q84 - q16) / 2 that turns
a point estimate into a P(over) at inference time.

Residual-over-anchor, exactly like the NBA models: each model learns the
ADJUSTMENT on top of the player's recent-form rate (season -> L5 -> L3), never
the absolute level, so it can't be biased by year-to-year scoring drift.

Backtest: the most recent season present is held out; we train on everything
before it and report MAE / interval coverage vs a recent-form baseline. Then we
REFIT on all rows and ship those. With only one season loaded the backtest is
skipped and the shipped models are simply fit on everything.

Outputs: models/nfl_<stat>_{mean,q16,q84}.joblib + models/nfl_metadata.json.
"""

import os
import json

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_CSV = os.path.join(HERE, "nfl_training_data.csv")
MODEL_DIR = os.path.join(HERE, "models")

TARGETS = [
    "pass_yds", "pass_att", "completions", "pass_td", "interceptions",
    "rush_yds", "rush_att", "rush_td",
    "rec_yds", "receptions", "targets", "rec_td",
]

# Sigma never collapses below a stat-appropriate floor (yards vary a lot,
# touchdowns barely). Anything not listed uses the default.
SIGMA_FLOORS = {
    "pass_yds": 35.0, "rush_yds": 18.0, "rec_yds": 15.0,
    "pass_att": 4.0, "rush_att": 3.0, "completions": 3.0, "targets": 2.0,
    "receptions": 1.5, "pass_td": 0.7, "rush_td": 0.5, "rec_td": 0.5,
    "interceptions": 0.6,
}
DEFAULT_SIGMA_FLOOR = 1.0

Q_LOW, Q_HIGH = 0.159, 0.841


def _numeric_features():
    feats = ["home", "season_games_todate", "opp_games_todate"]
    for s in TARGETS:
        feats += [f"l3_{s}", f"l5_{s}", f"season_{s}", f"opp_{s}_allowed"]
    return feats


NUMERIC_FEATURES = _numeric_features()
CATEGORICAL_FEATURES = ["position", "season_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_model(loss, quantile=None):
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
    X = df[FEATURES].copy()
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].astype("category")
    return X


def anchor_for(df, stat):
    return (df[f"season_{stat}"].fillna(df[f"l5_{stat}"])
            .fillna(df[f"l3_{stat}"]).fillna(0.0).astype(float))


def main():
    if not os.path.exists(TRAINING_CSV):
        raise SystemExit(f"Missing {TRAINING_CSV}. Run 34_nfl_build_props_training.py first.")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(TRAINING_CSV)
    df["season"] = df["season"].astype(str)
    seasons = sorted(df["season"].dropna().unique())
    print(f"Loaded {len(df)} rows. Seasons: {seasons}\n")
    if df.empty:
        raise SystemExit("Training data is empty -- backfill nfl_player_game_logs first.")

    # Hold out the most recent season for a forward backtest when we can.
    test_season = seasons[-1] if len(seasons) >= 2 else None
    if test_season:
        train_df = df[df["season"] != test_season]
        test_df = df[df["season"] == test_season]
        print(f"Backtest: train on {seasons[:-1]} ({len(train_df)} rows), "
              f"test on {test_season} ({len(test_df)} rows)\n")
    else:
        train_df, test_df = df, df.iloc[0:0]
        print("Only one season present -- skipping backtest, shipping on all rows.\n")

    X_full = prepare(df)
    X_train = prepare(train_df)
    X_test = prepare(test_df) if not test_df.empty else None

    metadata = {
        "test_season": test_season,
        "prediction_scheme": "residual_over_anchor",
        "anchor_order": ["season", "l5", "l3"],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "features": FEATURES,
        "sigma_floors": SIGMA_FLOORS,
        "default_sigma_floor": DEFAULT_SIGMA_FLOOR,
        "quantiles": {"low": Q_LOW, "high": Q_HIGH},
        "production_trained_on": "all_seasons",
        "production_seasons": seasons,
        "production_rows": len(df),
        "metrics": {},
    }

    for stat in TARGETS:
        col = f"target_{stat}"
        if col not in df.columns:
            continue
        floor = SIGMA_FLOORS.get(stat, DEFAULT_SIGMA_FLOOR)
        print("=" * 60)
        print(f"TARGET: {stat}")

        if X_test is not None:
            y_test = test_df[col].to_numpy()
            anchor_tr = anchor_for(train_df, stat)
            anchor_te = anchor_for(test_df, stat).to_numpy()
            resid_tr = train_df[col].to_numpy() - anchor_tr.to_numpy()
            mean_m = make_model("squared_error").fit(X_train, resid_tr)
            q16_m = make_model("quantile", Q_LOW).fit(X_train, resid_tr)
            q84_m = make_model("quantile", Q_HIGH).fit(X_train, resid_tr)
            pred = np.clip(anchor_te + mean_m.predict(X_test), 0, None)
            q16 = np.clip(anchor_te + q16_m.predict(X_test), 0, None)
            q84 = anchor_te + q84_m.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            base_mae = mean_absolute_error(y_test, anchor_te)
            coverage = float(np.mean((y_test >= q16) & (y_test <= np.maximum(q84, q16))))
            improve = 100 * (base_mae - mae) / base_mae if base_mae else 0.0
            print(f"  [backtest] MAE {mae:7.3f} | recent-form MAE {base_mae:7.3f} "
                  f"({improve:+.1f}%) | coverage {coverage*100:4.1f}%")
            metadata["metrics"][stat] = {
                "model_mae": round(mae, 4),
                "baseline_recent_mae": round(base_mae, 4),
                "improvement_vs_recent_pct": round(improve, 2),
                "interval_coverage": round(coverage, 4),
            }

        # Ship models fit on ALL rows.
        anchor_full = anchor_for(df, stat)
        resid_full = df[col].to_numpy() - anchor_full.to_numpy()
        joblib.dump(make_model("squared_error").fit(X_full, resid_full),
                    os.path.join(MODEL_DIR, f"nfl_{stat}_mean.joblib"))
        joblib.dump(make_model("quantile", Q_LOW).fit(X_full, resid_full),
                    os.path.join(MODEL_DIR, f"nfl_{stat}_q16.joblib"))
        joblib.dump(make_model("quantile", Q_HIGH).fit(X_full, resid_full),
                    os.path.join(MODEL_DIR, f"nfl_{stat}_q84.joblib"))
        print(f"  [production] refit on all {len(df)} rows -> saved")

    with open(os.path.join(MODEL_DIR, "nfl_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("=" * 60)
    print(f"DONE. NFL prop models + nfl_metadata.json written to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
