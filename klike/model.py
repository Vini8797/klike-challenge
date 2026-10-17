"""Features and the Klike Score model.

Two versions of the same ridge model:
- "creative": only what exists before launch (creative choices + context). Used by the engine.
- "full": adds performance measured after launch (CTR, completion, conversion rate).
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "klike_score"
ALPHAS = np.logspace(-3, 3, 61)
PLATFORMS = ["TikTok", "Meta", "LinkedIn"]

# What the advertiser chooses when making the video: the engine may suggest changing these.
BINARY_LEVERS = ["has_hook", "has_face", "has_cta", "has_subtitle"]
NUMERIC_LEVERS = ["music_voice_ratio", "video_duration_s"]
CATEGORICAL_LEVERS = ["text_density", "format"]
# Who/where/why the ad runs: explains the score, but is never recommended.
CONTEXT = ["platform", "category", "objective", "target_audience_age"]
# Measured after launch: only the full model uses these.
PERFORMANCE = ["log_ctr", "completion_rate", "conv_rate"]

LABELS = {
    "has_hook": "Hook in first 3s", "has_face": "Human face on screen", "has_cta": "Call to action",
    "has_subtitle": "Subtitles", "is_retargeting": "Retargeting audience",
    "music_voice_ratio": "Music/voice ratio", "video_duration_s": "Video duration",
    "text_density": "On-screen text density", "format": "Video format", "platform": "Platform",
    "category": "Advertiser category", "objective": "Campaign objective",
    "target_audience_age": "Audience age", "log_ctr": "CTR", "completion_rate": "Completion rate",
    "conv_rate": "Conversion rate per click",
}


def engineer(df):
    """Add the derived columns. Uses only the row itself, so it's safe before any split."""
    out = df.copy()
    for column in BINARY_LEVERS + ["is_retargeting"]:
        values = out[column].astype(object)
        out[column] = values.where(values.notna(), np.nan).astype(float)
    # Format only means something together with the platform: vertical helps on TikTok and
    # hurts on LinkedIn, so a single "format" column would average the two into nothing.
    out["platform_format"] = out["platform"].astype(str) + "|" + out["format"].astype(str)
    if "ctr" in out:
        out["log_ctr"] = np.log10(out["ctr"].where(out["ctr"] > 0))
    if "conversions" in out and "clicks" in out:
        out["conv_rate"] = out["conversions"] / out["clicks"].where(out["clicks"] > 0)
    return out


def make_pipeline(kind="creative", estimator=None):
    binary = BINARY_LEVERS + ["is_retargeting"]
    numeric = NUMERIC_LEVERS + (PERFORMANCE if kind == "full" else [])
    categorical = CATEGORICAL_LEVERS + CONTEXT + ["platform_format"]
    # Imputation lives inside the pipeline, so in cross-validation each fold only learns
    # from its own training rows.
    preprocess = ColumnTransformer([
        ("binary", Pipeline([("impute", SimpleImputer(strategy="mean", add_indicator=True)),
                             ("scale", StandardScaler())]), binary),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([
        ("engineer", FunctionTransformer(engineer)),
        ("preprocess", preprocess),
        ("model", estimator if estimator is not None else RidgeCV(alphas=ALPHAS)),
    ])


def candidate_models():
    boosting = dict(max_depth=2, learning_rate=0.05, max_iter=400, random_state=0)
    return {
        "Baseline (mean)": make_pipeline("creative", DummyRegressor()),
        "Ridge · creative": make_pipeline("creative"),
        "Gradient boosting · creative": make_pipeline("creative", HistGradientBoostingRegressor(**boosting)),
        "Random forest · creative": make_pipeline("creative", RandomForestRegressor(300, min_samples_leaf=3, random_state=0)),
        "Ridge · full": make_pipeline("full"),
        "Gradient boosting · full": make_pipeline("full", HistGradientBoostingRegressor(**boosting)),
    }


def cross_validated_scores(models, df):
    """RMSE / MAE / R² with 5-fold cross-validation repeated 5 times."""
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
    rows = []
    for name, model in models.items():
        result = cross_validate(model, df, df[TARGET], cv=cv, scoring=scoring)
        rows.append({
            "model": name,
            "RMSE": -result["test_rmse"].mean(), "RMSE sd": result["test_rmse"].std(),
            "MAE": -result["test_mae"].mean(),
            "R²": result["test_r2"].mean(), "R² sd": result["test_r2"].std(),
        })
    return pd.DataFrame(rows).set_index("model")


def evaluate(model, test):
    predicted, actual = model.predict(test), test[TARGET]
    return {
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "MAE": mean_absolute_error(actual, predicted),
        "R²": r2_score(actual, predicted),
    }


def fixed_alpha(model):
    """Same pipeline with the penalty already chosen, so refitting it many times is fast."""
    return clone(model).set_params(model=Ridge(alpha=model.named_steps["model"].alpha_))


def bootstrap(model, df, n=200, seed=0):
    """Refit the model on n resamples of the data (for confidence intervals)."""
    rng = np.random.default_rng(seed)
    fitted = []
    for _ in range(n):
        sample = df.iloc[rng.integers(0, len(df), len(df))]
        fitted.append(clone(model).fit(sample, sample[TARGET]))
    return fitted
