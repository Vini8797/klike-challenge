"""Get creative recommendations for a campaign from the command line.

    python recommend.py KLK-0116              # a campaign from the dataset
    python recommend.py KLK-0116 KLK-0003     # several
    python recommend.py --random 3            # random campaigns
    python recommend.py --file campaign.json  # a new campaign (same columns as the dataset)
    python recommend.py KLK-0116 --json       # JSON output
    python recommend.py --portfolio           # biggest opportunities across all campaigns (~1.5 min)
"""

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

from klike.data import load_clean
from klike.model import LABELS
from klike.recommender import RecommendationEngine


def portfolio_summary(engine):
    portfolio = engine.portfolio()
    confident = portfolio[portfolio["confidence"].isin(["high", "medium"])]
    duration = confident[confident["lever"] == "video_duration_s"]
    confident = confident[confident["lever"] != "video_duration_s"]   # duration is about CTR, not score
    summary = (confident.groupby("lever")
               .agg(campaigns=("campaign_id", "nunique"), mean_gain=("score_gain", "mean"), total_gain=("score_gain", "sum"))
               .sort_values("total_gain", ascending=False)
               .rename(index=LABELS))
    per_campaign = portfolio.drop_duplicates("campaign_id")
    return "\n".join([
        f"{len(per_campaign)} campaigns · mean Klike Score {per_campaign['actual_score'].mean():.1f} "
        f"→ {per_campaign['potential_score'].mean():.1f} if every recommended change were made (upper bound)",
        "",
        "Where the points are (high/medium confidence recommendations):",
        summary.round(1).to_string(),
        "",
        f"Duration changes (for CTR, not score): {duration['campaign_id'].nunique()} campaigns, "
        f"mean CTR lift {duration['ctr_lift'].mean():+.0%}.",
    ])


def to_json(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return None if pd.isna(value) else str(value)


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # Windows terminals default to cp1252
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Klike creative recommendations")
    parser.add_argument("campaign_ids", nargs="*", help="e.g. KLK-0116")
    parser.add_argument("--file", help="JSON file with a campaign (or a list of campaigns)")
    parser.add_argument("--random", type=int, metavar="N", help="N random campaigns")
    parser.add_argument("--portfolio", action="store_true", help="summary across all campaigns")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    data = load_clean()
    engine = RecommendationEngine().fit(data)   # takes a couple of seconds

    if args.portfolio:
        print(portfolio_summary(engine))
        return

    campaigns = []
    for campaign_id in args.campaign_ids:
        match = data[data["campaign_id"] == campaign_id]
        if match.empty:
            parser.error(f"unknown campaign id: {campaign_id}")
        campaigns.append(match)
    if args.random:
        sample = data.sample(args.random)
        campaigns += [sample.iloc[[i]] for i in range(len(sample))]
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            loaded = json.load(f)
        campaigns += [pd.DataFrame([c]) for c in (loaded if isinstance(loaded, list) else [loaded])]
    if not campaigns:
        parser.print_help()
        return

    reports = [engine.recommend(c) for c in campaigns]
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2, ensure_ascii=False, default=to_json))
    else:
        print(("\n\n" + "─" * 100 + "\n\n").join(str(r) for r in reports))


if __name__ == "__main__":
    main()
