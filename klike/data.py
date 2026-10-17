"""Load the Excel file and clean it.

These are the same steps as 01_eda.ipynb, written as functions so the model and the
recommendations engine don't depend on running the notebook first.
"""

import datetime as dt
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "klike_challenge_dataset.xlsx"
CLEAN_PATH = ROOT / "data" / "klike_clean.csv"

NUMERIC = [
    "video_duration_s", "music_voice_ratio", "impressions", "clicks", "ctr", "cpc",
    "spend", "conversions", "revenue", "roas", "avg_watch_time_s", "engagement_rate",
    "klike_score",
]
BOOLEAN = ["is_retargeting", "has_subtitle", "has_cta", "has_hook", "has_face"]

# Excel turned numbers like "3.9" into dates (3 September), but kept a display format that
# still shows the original digits. Rendering the date through that format gives them back.
CORRUPTION_FORMATS = {
    "d.m":     lambda d: f"{d.day}.{d.month}",
    "dd.mm":   lambda d: f"{d.day:02d}.{d.month:02d}",
    "yyyy.m":  lambda d: f"{d.year}.{d.month}",
    "yyyy.mm": lambda d: f"{d.year}.{d.month:02d}",
}

# The reported CTR mixes units from row to row.
UNITS = {"fraction": 1.0, "percent": 100.0, "per mille": 1_000.0, "basis points": 10_000.0}


def load_with_recovery(path=RAW_PATH):
    """Read the workbook cell by cell and undo the date conversion.

    Returns the raw frame and a mask of which cells were recovered.
    """
    sheet = openpyxl.load_workbook(path, data_only=True).active
    header = [c.value for c in sheet[1]]
    records, marks = [], []
    for row in sheet.iter_rows(min_row=2):
        record, mark = {}, {}
        for column, cell in zip(header, row):
            value, mark[column] = cell.value, False
            render = CORRUPTION_FORMATS.get(cell.number_format)
            if isinstance(value, dt.datetime) and render:
                value, mark[column] = float(render(value)), True
            record[column] = value
        records.append(record)
        marks.append(mark)
    return pd.DataFrame(records, columns=header), pd.DataFrame(marks, columns=header).astype(bool)


def _ctr_unit(reported, derived, tolerance):
    """Which unit each row's CTR was written in ('unresolved' if none fits)."""
    labels = pd.Series("unresolved", index=reported.index, dtype=object)
    best = pd.Series(np.inf, index=reported.index)
    for name, factor in UNITS.items():
        expected = derived * factor
        error = (reported - expected).abs() / expected.replace(0, np.nan)
        better = (error <= tolerance) & (error < best)
        labels, best = labels.mask(better, name), best.mask(better, error)
    return labels


def _matches(reported, expected):
    return (reported - expected).abs() <= 0.005 + 0.005 * expected.abs()  # values have 2 decimals


def _repair_counts(df):
    """Find broken cells with ctr = clicks/impressions, cpc = spend/clicks, roas = revenue/spend.

    Each column (except impressions and revenue) appears in two of these identities, so the
    pattern of which ones break tells us which cell is wrong.
    """
    derived_ctr = df["clicks"] / df["impressions"].replace(0, np.nan)
    tolerance = (1.5 / df["clicks"].where(df["clicks"] > 0)).clip(lower=0.005).fillna(0.005)
    unit = _ctr_unit(df["ctr"], derived_ctr, tolerance)
    ctr_broken = unit.eq("unresolved")

    expected_cpc = df["spend"] / df["clicks"].replace(0, np.nan)
    has_cpc = df["cpc"].notna() & expected_cpc.notna()
    cpc_broken = has_cpc & ~_matches(df["cpc"], expected_cpc)

    expected_roas = df["revenue"] / df["spend"].replace(0, np.nan)
    has_roas = df["roas"].notna() & expected_roas.notna()
    roas_broken = has_roas & ~_matches(df["roas"], expected_roas)

    # Only ctr breaks -> impressions is wrong (clicks is confirmed by cpc).
    bad_impressions = ctr_broken & has_cpc & ~cpc_broken
    factor = UNITS[unit[~ctr_broken].value_counts().idxmax()]
    df.loc[bad_impressions, "impressions"] = (
        df.loc[bad_impressions, "clicks"] / (df.loc[bad_impressions, "ctr"] / factor)
    ).round()

    # cpc and roas both break -> spend is wrong (it's the only column they share).
    bad_spend = cpc_broken & roas_broken
    df.loc[bad_spend, "spend"] = (df.loc[bad_spend, "cpc"] * df.loc[bad_spend, "clicks"]).round(2)
    return df


def clean(raw):
    df = raw.copy()
    for column in NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in BOOLEAN:
        df[column] = df[column].astype("boolean")   # keeps "unknown" separate from False

    df = _repair_counts(df)

    # Recompute every rate from the counts, so they all share one unit.
    df["ctr"] = df["clicks"] / df["impressions"].replace(0, np.nan)
    df["cpc"] = df["spend"] / df["clicks"].replace(0, np.nan)
    df["roas"] = df["revenue"] / df["spend"].replace(0, np.nan)
    df["engagement_rate_frac"] = df["engagement_rate"] / 10_000  # assumed to be basis points

    for column in NUMERIC + BOOLEAN:
        if df[column].isna().any():
            df[f"{column}_missing"] = df[column].isna()

    # Zero revenue on awareness/engagement campaigns means "not tracked", not "sold nothing".
    untracked = df["objective"].isin(["awareness", "engagement"]) & df["revenue"].eq(0)
    df["revenue_tracked"] = (~untracked).astype("boolean").mask(df["revenue"].isna())
    df["roas_valid"] = df["roas"].where(df["revenue_tracked"].fillna(False))
    df["completion_rate"] = (df["avg_watch_time_s"] / df["video_duration_s"].replace(0, np.nan)).clip(upper=1.0)
    df["cpa"] = df["spend"] / df["conversions"].replace(0, np.nan)
    return df


def build():
    """Excel -> clean CSV in data/."""
    df = clean(load_with_recovery()[0])
    CLEAN_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    return load_clean()


def load_clean():
    """The clean dataset (built from the Excel file the first time)."""
    if not CLEAN_PATH.exists():
        return build()
    df = pd.read_csv(CLEAN_PATH, parse_dates=["date"])
    for column in BOOLEAN + ["revenue_tracked"]:
        df[column] = df[column].astype("boolean")
    return df
