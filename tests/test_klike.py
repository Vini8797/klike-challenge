"""Run with:  python -m pytest"""

import warnings

import numpy as np
import pandas as pd
import pytest

from klike.data import clean, load_with_recovery
from klike.model import make_pipeline
from klike.recommender import OBJECTIVE_METRIC, RecommendationEngine

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def raw():
    return load_with_recovery()


@pytest.fixture(scope="module")
def data(raw):
    return clean(raw[0])


@pytest.fixture(scope="module")
def engine(data):
    return RecommendationEngine(n_bootstrap=100).fit(data)


GOOD_CREATIVE = dict(category="E-commerce", target_audience_age="25-34", is_retargeting=False,
                     video_duration_s=20, has_subtitle=True, has_cta=True, has_hook=True,
                     has_face=True, text_density="medium", music_voice_ratio=0.3)


# --- data cleaning

def test_recovers_the_excel_dates(raw):
    _, recovered = raw
    assert int(recovered.to_numpy().sum()) == 558
    assert int(recovered["klike_score"].sum()) == 16


def test_keeps_every_row(data):
    assert len(data) == 500
    assert data["klike_score"].between(0, 100).all()


def test_rates_match_the_counts(data):
    assert np.allclose(data["ctr"], data["clicks"] / data["impressions"])


def test_spend_repair_matches_the_reported_roas(raw, data):
    # Spend was rebuilt from cpc only, so ROAS matching the original value confirms the repair.
    ids = ["KLK-0109", "KLK-0259"]
    reported = pd.to_numeric(raw[0].set_index("campaign_id").loc[ids, "roas"])
    assert np.allclose(data.set_index("campaign_id").loc[ids, "roas"].round(2), reported)


# --- model

def test_model_beats_the_average(data):
    train, test = data.iloc[:400], data.iloc[400:]
    model = make_pipeline("creative").fit(train, train["klike_score"])
    rmse = np.sqrt(((model.predict(test) - test["klike_score"]) ** 2).mean())
    assert rmse < 0.65 * test["klike_score"].std()


def test_model_handles_missing_values(data):
    model = make_pipeline("creative").fit(data, data["klike_score"])
    row = data.iloc[[0]].copy()
    row[["has_subtitle", "video_duration_s", "music_voice_ratio"]] = [pd.NA, np.nan, np.nan]
    assert np.isfinite(model.predict(row)).all()


# --- recommendations

def test_at_least_three_recommendations_ranked_by_gain(engine, data):
    for _, campaign in data.sample(25, random_state=0).iterrows():
        recs = engine.recommend(campaign).recommendations
        assert len(recs) >= 3
        gains = [r.score_gain for r in recs if r.confidence in ("high", "medium") and r.lever != "video_duration_s"]
        assert gains == sorted(gains, reverse=True)


def test_never_recommends_something_harmful(engine, data):
    for _, campaign in data.sample(60, random_state=1).iterrows():
        for rec in engine.recommend(campaign).recommendations:
            if rec.confidence == "keep":
                continue
            assert rec.score_ci[1] > 0
            estimate, _, high = rec.lifts[OBJECTIVE_METRIC[campaign["objective"]]]
            assert high > 0 and estimate > -0.10
            if any(h < 0 for _, _, h in rec.lifts.values()):   # other trade-offs must be shown
                assert rec.tradeoffs and rec.confidence == "medium"


def test_format_advice_depends_on_the_platform(engine):
    linkedin = engine.recommend({**GOOD_CREATIVE, "objective": "conversions", "platform": "LinkedIn", "format": "vertical"})
    tiktok = engine.recommend({**GOOD_CREATIVE, "objective": "engagement", "platform": "TikTok", "format": "quadrado"})
    assert any(r.lever == "format" and r.proposed == "horizontal" for r in linkedin.recommendations)
    assert any(r.lever == "format" and r.proposed == "vertical" for r in tiktok.recommendations)
    # There is no horizontal video on TikTok in the data, so it's never suggested there.
    assert not any(r.lever == "format" and r.proposed == "horizontal" for r in tiktok.recommendations)


def test_respects_the_campaign_objective(engine):
    # On TikTok, square -> vertical helps engagement but is estimated to cost conversions.
    report = engine.recommend({**GOOD_CREATIVE, "objective": "conversions", "platform": "TikTok", "format": "quadrado"})
    assert not any(r.lever == "format" and r.confidence != "keep" for r in report.recommendations)


def test_never_recommends_retargeting(engine, data):
    for _, campaign in data.sample(25, random_state=2).iterrows():
        assert all(r.lever != "is_retargeting" for r in engine.recommend(campaign).recommendations)
