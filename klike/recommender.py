"""Recommendations engine: one campaign in, a ranked list of creative changes out.

For each change the campaign could make (add a hook, cut text, re-cut the format...), the engine
estimates:
- how much the Klike Score changes, using the creative model from Part 2;
- how much CTR, engagement and conversions change on that campaign's platform.
Both come with 95% intervals from bootstrapping. Then it keeps the changes that are likely to
help, drops anything that would hurt what the campaign is paying for, and ranks by score gain.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV

from klike.data import NUMERIC
from klike.model import BINARY_LEVERS, PLATFORMS, engineer, make_pipeline

METRICS = {"ctr": "CTR", "engagement": "engagement rate", "conversions": "conversions at the same reach"}

MIN_SUPPORT = 15        # a format needs this many campaigns on the platform to be suggested
MIN_PEERS = 50          # campaigns needed to compare against platform + category peers
MIN_SCORE_GAIN = 0.5    # points
MIN_MUSIC_STEP = 0.10
MIN_DURATION_STEP = 5   # seconds

# The metric each objective is paying for. A change expected to cost 10% or more of it is never
# recommended, even if it raises the score. This is stricter than the bar for gains on purpose:
# being wrong about the metric a campaign pays for costs more than missing a few score points.
OBJECTIVE_METRIC = {"conversions": "conversions", "app_install": "conversions", "traffic": "ctr",
                    "engagement": "engagement", "awareness": "engagement"}
OBJECTIVE_TOLERANCE = -0.10

ACTIONS = {"has_hook": "Add a hook in the first 3 seconds", "has_face": "Show a human face on screen",
           "has_cta": "Add an explicit call to action", "has_subtitle": "Add subtitles"}
STRENGTHS = {"has_hook": "the hook in the first 3 seconds", "has_face": "the human face on screen",
             "has_cta": "the call to action", "has_subtitle": "the subtitles"}
PEER_NOUNS = {"has_hook": "open with a hook", "has_face": "show a face", "has_cta": "carry a CTA",
              "has_subtitle": "are subtitled"}
FORMAT_NAMES = {"vertical": "vertical (9:16)", "quadrado": "square (1:1)", "horizontal": "horizontal (16:9)"}

TEXT_LEVELS = ["low", "medium", "high"]
FORMATS = ["vertical", "quadrado", "horizontal"]
CONTROLS = {
    "category": ["E-commerce", "SaaS", "Lead Gen", "App Install", "Branding"],
    "objective": ["conversions", "traffic", "engagement", "awareness", "app_install"],
    "target_audience_age": ["18-24", "25-34", "35-44", "45+"],
}


# ---------------------------------------------------------------- effect models

def metric_target(df, metric):
    """Log of the metric, so effects come out as relative lifts."""
    if metric == "ctr":
        return np.log(df["ctr"].where(df["ctr"] > 0))
    if metric == "engagement":
        # 13 values below 1 basis point (eleven of them exactly 0.1) look like placeholders.
        return np.log(df["engagement_rate_frac"].where(df["engagement_rate"] >= 1))
    if metric == "conversions":
        return np.log((df["conversions"] + 0.5) / df["impressions"])
    raise KeyError(metric)


class LiftDesign:
    """Turns campaigns into the columns of the metric models.

    Each creative lever appears once for all platforms and once per platform. Ridge shrinks the
    per-platform copies toward the shared effect, so a platform only gets its own answer when
    its data clearly says so (useful with only 83 LinkedIn campaigns). Category, objective, age
    and retargeting are included as controls, so a lever isn't credited for what they did.
    """

    def fit(self, X):
        X = engineer(X)
        self.binary_means = {c: X[c].mean() for c in BINARY_LEVERS + ["is_retargeting"]}
        self.music_median = X["music_voice_ratio"].median()
        self.duration_median = X["video_duration_s"].median()
        return self

    @staticmethod
    def _onehot(series, levels):
        values = series.astype(str).to_numpy()
        return np.column_stack([values == level for level in levels]).astype(float)

    def _levers(self, X):
        binary = np.column_stack([X[c].fillna(self.binary_means[c]).to_numpy(float) for c in BINARY_LEVERS])
        music = X["music_voice_ratio"].fillna(self.music_median).to_numpy(float)
        # log2 duration with a squared term, so a "sweet spot" length is possible.
        duration = np.log2(X["video_duration_s"].fillna(self.duration_median).clip(5, 90).to_numpy(float) / 16)
        return np.column_stack([binary, self._onehot(X["text_density"], TEXT_LEVELS),
                                self._onehot(X["format"], FORMATS), music, duration, duration ** 2])

    def transform(self, X):
        X = engineer(X)
        levers = self._levers(X)
        platforms = self._onehot(X["platform"], PLATFORMS)
        parts = [levers]
        for j in range(len(PLATFORMS)):
            parts += [levers * platforms[:, [j]], platforms[:, [j]]]
        parts += [self._onehot(X[column], levels) for column, levels in CONTROLS.items()]
        parts.append(X["is_retargeting"].fillna(self.binary_means["is_retargeting"]).to_numpy(float)[:, None])
        parts.append(X[["has_subtitle", "video_duration_s", "music_voice_ratio"]].isna().to_numpy(float))
        return np.hstack(parts)


class LinearEffect:
    """A ridge model plus bootstrap copies of its coefficients.

    Because the model is linear, the effect of a change is just (new row - old row) · coefficients,
    so all bootstrap intervals come from one matrix product.
    """

    def __init__(self, transformer, X, y, alpha, n_bootstrap, rng, log_scale):
        self.transformer, self.log_scale = transformer, log_scale
        self.coef = Ridge(alpha=alpha).fit(X, y).coef_
        self.draws = np.empty((n_bootstrap, X.shape[1]))
        for b in range(n_bootstrap):
            idx = rng.integers(0, len(y), len(y))
            self.draws[b] = Ridge(alpha=alpha).fit(X[idx], y[idx]).coef_

    def effect(self, current, changed):
        """(estimate, low, high) for each changed row vs the current row.

        In points for the score; as a relative lift (0.25 = +25%) for the metrics.
        """
        X = self.transformer.transform(pd.concat([current, changed], ignore_index=True))
        diff = X[1:] - X[0]
        point = diff @ self.coef
        low, high = np.percentile(diff @ self.draws.T, [2.5, 97.5], axis=1)
        if self.log_scale:
            point, low, high = np.expm1(point), np.expm1(low), np.expm1(high)
        return np.column_stack([point, low, high])


# ---------------------------------------------------------------- output

@dataclass
class Recommendation:
    lever: str
    action: str
    current: object
    proposed: object
    score_gain: float
    score_ci: tuple
    lifts: dict              # metric -> (estimate, low, high)
    confidence: str          # high | medium | test | keep
    evidence: str = ""
    tradeoffs: list = field(default_factory=list)

    @property
    def message(self):
        if self.confidence == "keep":
            return f"{self.action}. {self.evidence}".strip()
        ctr, ctr_low, ctr_high = self.lifts["ctr"]
        if self.lever == "video_duration_s":
            parts = [f"{self.action}: CTR {ctr:+.0%} (95% CI {ctr_low:+.0%} to {ctr_high:+.0%}) on this platform."]
        else:
            parts = [f"{self.action}: Klike Score {self.score_gain:+.1f} pts "
                     f"(95% CI {self.score_ci[0]:+.1f} to {self.score_ci[1]:+.1f})."]
        gains = [f"{est:+.0%} {METRICS[m]}" for m, (est, low, _) in self.lifts.items()
                 if low > 0 and not (m == "ctr" and self.lever == "video_duration_s")]
        if gains:
            parts.append("On this platform it is associated with " + ", ".join(gains) + " (model-adjusted).")
        if self.tradeoffs:
            parts.append("Trade-off: " + "; ".join(self.tradeoffs) + ".")
        if self.evidence:
            parts.append(self.evidence)
        return " ".join(parts)

    def to_dict(self):
        record = asdict(self)
        record["current"] = None if pd.isna(self.current) else self.current
        record["message"] = self.message
        return record


@dataclass
class Report:
    campaign_id: str
    context: dict
    actual_score: float
    expected_score: float    # what the creative model expects from the creative alone
    potential_score: float   # actual score + estimated gain from all the changes (capped at 100)
    recommendations: list
    scored: bool = True      # False for a new campaign, where the score is only a prediction

    def to_dict(self):
        record = asdict(self)
        record["recommendations"] = [r.to_dict() for r in self.recommendations]
        return record

    def __str__(self):
        c = self.context
        tags = {"test": "  [unproven — A/B test it]", "keep": "  [strength to keep]"}
        if self.scored:
            headline = (f"Klike Score {self.actual_score:.1f}  (creative model expects {self.expected_score:.1f}; "
                        f"potential with the changes below ≈ {self.potential_score:.1f})")
        else:
            headline = (f"Predicted Klike Score {self.expected_score:.1f}  "
                        f"(potential with the changes below ≈ {self.potential_score:.1f})")
        lines = [f"Campaign {self.campaign_id} · {c['platform']} · {c['category']} · "
                 f"{c['objective']} · age {c['target_audience_age']}", headline, ""]
        for rank, rec in enumerate(self.recommendations, 1):
            lines += [f"{rank}. {rec.message}{tags.get(rec.confidence, '')}", ""]
        return "\n".join(lines).rstrip()


# ---------------------------------------------------------------- engine

class RecommendationEngine:
    def __init__(self, n_bootstrap=300, seed=0):
        self.n_bootstrap, self.seed = n_bootstrap, seed

    def fit(self, df):
        self.data = df.reset_index(drop=True)

        # Score effects: the Part 2 creative model, bootstrapped.
        self.score_model = make_pipeline("creative").fit(df, df["klike_score"])
        pre = self.score_model[:-1]
        self.score_effect = LinearEffect(pre, pre.transform(df), df["klike_score"].to_numpy(),
                                         self.score_model.named_steps["model"].alpha_,
                                         self.n_bootstrap, np.random.default_rng(self.seed + 1), log_scale=False)

        # Metric effects: one model per metric.
        rng = np.random.default_rng(self.seed)
        self.metric_effects = {}
        for metric in METRICS:
            y = metric_target(df, metric)
            rows, y = df[y.notna()], y[y.notna()].to_numpy()
            design = LiftDesign().fit(rows)
            X = design.transform(rows)
            alpha = RidgeCV(alphas=np.logspace(-2, 3, 41)).fit(X, y).alpha_
            self.metric_effects[metric] = LinearEffect(design, X, y, alpha, self.n_bootstrap, rng, log_scale=True)

        # Simple per-platform facts used to build and explain the candidates.
        self.format_support = df.groupby(["platform", "format"], observed=True).size().to_dict()
        self.music_target = df.groupby("platform", observed=True)["music_voice_ratio"].quantile(0.25).round(2).to_dict()
        self.duration_range = {p: tuple(g["video_duration_s"].dropna().quantile([0.10, 0.90]).round())
                               for p, g in df.groupby("platform", observed=True)}
        base = self.data.head(1)
        self.music_slope = float(np.diff(self.score_model.predict(pd.concat(
            [base.assign(music_voice_ratio=0.0), base.assign(music_voice_ratio=1.0)])))[0])
        self.peer_stats = self._peer_stats()
        return self

    def _peer_stats(self):
        """Share of top- and bottom-quartile creatives using each option, per platform (+ category)."""
        groups = {p: g for p, g in self.data.groupby("platform", observed=True)}
        for key, g in self.data.groupby(["platform", "category"], observed=True):
            if len(g) >= MIN_PEERS:
                groups[key] = g
        stats = {}
        for key, g in groups.items():
            top = g[g["klike_score"] >= g["klike_score"].quantile(0.75)]
            bottom = g[g["klike_score"] <= g["klike_score"].quantile(0.25)]
            shares = {(lever, True): (top[lever].dropna().astype(bool).mean(), bottom[lever].dropna().astype(bool).mean())
                      for lever in BINARY_LEVERS}
            for lever in ["text_density", "format"]:
                for value in g[lever].dropna().unique():
                    shares[(lever, value)] = ((top[lever] == value).mean(), (bottom[lever] == value).mean())
            stats[key] = {"n": len(g), "shares": shares}
        return stats

    # ---- candidates
    def _candidates(self, row):
        """Every change this campaign could make: [(lever, {column: new value}, action text)]."""
        r = row.iloc[0]
        platform = r["platform"]
        out = []
        for lever in BINARY_LEVERS:
            if r[lever] is True:
                continue
            action = ACTIONS[lever]
            if pd.isna(r[lever]):
                action += " (not recorded for this campaign — check the video)"
            out.append((lever, {lever: True}, action))

        for level in ["medium", "low", "high"]:
            if level != r["text_density"]:
                out.append(("text_density", {"text_density": level},
                            f"Change on-screen text from {r['text_density']} to {level}"))

        for fmt in FORMAT_NAMES:   # only formats the platform actually runs
            if fmt != r["format"] and self.format_support.get((platform, fmt), 0) >= MIN_SUPPORT:
                out.append(("format", {"format": fmt},
                            f"Re-cut the video from {FORMAT_NAMES.get(r['format'], r['format'])} to {FORMAT_NAMES[fmt]}"))

        target = self.music_target[platform]
        if pd.notna(r["music_voice_ratio"]) and r["music_voice_ratio"] - target >= MIN_MUSIC_STEP:
            out.append(("music_voice_ratio", {"music_voice_ratio": target},
                        f"Make the audio voice-led: music/voice ratio {r['music_voice_ratio']:.2f} → ~{target:.2f}"))

        if pd.notna(r["video_duration_s"]):
            low, high = self.duration_range[platform]
            for seconds in np.arange(low, high + 1, 1.0):
                if abs(seconds - r["video_duration_s"]) >= MIN_DURATION_STEP:
                    verb = "Shorten" if seconds < r["video_duration_s"] else "Lengthen"
                    out.append(("video_duration_s", {"video_duration_s": float(seconds)},
                                f"{verb} the video from {r['video_duration_s']:.0f}s to ~{seconds:.0f}s"))
        return out

    def _evidence(self, lever, proposed, row):
        r = row.iloc[0]
        if lever == "video_duration_s":
            low, high = self.duration_range[r["platform"]]
            return (f"Duration does not move the Klike Score; this is the CTR optimum for {r['platform']} "
                    f"within its typical range ({low:.0f}-{high:.0f}s).")
        if lever == "music_voice_ratio":
            return (f"A small but consistent effect: each 0.1 shift toward voice is worth "
                    f"≈ {-0.1 * self.music_slope:+.1f} pts across all {len(self.data)} campaigns — a polish step.")
        key = (r["platform"], r["category"])
        label = f"{r['platform']} {r['category']}" if key in self.peer_stats else r["platform"]
        stats = self.peer_stats.get(key) or self.peer_stats[r["platform"]]
        top, bottom = stats["shares"].get((lever, proposed), (np.nan, np.nan))
        noun = PEER_NOUNS.get(lever) or (
            f"use {proposed} on-screen text" if lever == "text_density" else f"are {FORMAT_NAMES.get(proposed, proposed)}")
        return f"{top:.0%} of top-quartile {label} creatives {noun}, vs {bottom:.0%} of the bottom quartile (n={stats['n']})."

    # ---- main entry point
    def recommend(self, campaign, min_recommendations=3, max_recommendations=6):
        """Ranked recommendations for one campaign (a dict, a Series or a 1-row DataFrame)."""
        row = self._as_frame(campaign)
        candidates = self._candidates(row)
        changed = pd.concat([self._apply(row, change) for _, change, _ in candidates], ignore_index=True)
        score = self.score_effect.effect(row, changed)
        lifts = {m: effect.effect(row, changed) for m, effect in self.metric_effects.items()}
        scored = [{"lever": lever, "change": change, "action": action,
                   "score": tuple(float(v) for v in score[i]),
                   "lifts": {m: tuple(float(v) for v in lifts[m][i]) for m in METRICS}}
                  for i, (lever, change, action) in enumerate(candidates)]

        # Keep the best option per lever. Duration doesn't move the score, so for duration
        # "best" means best CTR.
        best = {}
        for c in scored:
            key = (c["lifts"]["ctr"][0],) if c["lever"] == "video_duration_s" else (c["score"][0], c["lifts"]["ctr"][0])
            if c["lever"] not in best or key > best[c["lever"]][0]:
                best[c["lever"]] = (key, c)
        ranked = sorted((c for _, c in best.values()), key=lambda c: (c["score"][0], c["lifts"]["ctr"][0]), reverse=True)

        goal = OBJECTIVE_METRIC.get(row.iloc[0]["objective"])

        def harmless(c):
            return c["score"][2] > 0 and all(high > 0 for _, _, high in c["lifts"].values())

        def verdict(c):
            goal_estimate, _, goal_high = c["lifts"][goal] if goal else (0.0, 0.0, 1.0)
            if goal_high < 0 or goal_estimate <= OBJECTIVE_TOLERANCE:
                return None                                  # hurts what the campaign pays for
            score_sure = c["score"][0] >= MIN_SCORE_GAIN and c["score"][1] > 0
            metric_sure = any(low > 0 for _, low, _ in c["lifts"].values())
            tradeoff = any(high < 0 for _, _, high in c["lifts"].values())
            if c["lever"] == "video_duration_s":             # only worth it for CTR
                if c["lifts"]["ctr"][1] > 0 and harmless(c):
                    return "medium"
                return "test" if c["lifts"]["ctr"][0] > 0.02 and harmless(c) else None
            if score_sure:
                return "high" if metric_sure and not tradeoff else "medium"
            if metric_sure and c["score"][0] > -MIN_SCORE_GAIN and harmless(c):
                return "medium"
            if c["score"][0] > 0 and harmless(c):
                return "test"
            return None

        verdicts = [(c, verdict(c)) for c in ranked]
        picked = [self._build(c, row, v) for c, v in verdicts if v in ("high", "medium")][:max_recommendations]
        # A strong creative may have fewer than 3 real improvements. Fill up with untested ideas
        # first, then with strengths to keep — never with something expected to hurt.
        for c, v in verdicts:
            if len(picked) >= min_recommendations:
                break
            if v == "test":
                picked.append(self._build(c, row, "test"))
        if len(picked) < min_recommendations:
            picked += self._strengths(row)[:min_recommendations - len(picked)]
        return self._report(row, picked)

    def _strengths(self, row):
        """What the creative already does right, ordered by how much losing it would cost."""
        r = row.iloc[0]
        losses = [(lever, {lever: False}, f"Keep {STRENGTHS[lever]} in every new variant")
                  for lever in BINARY_LEVERS if r[lever] is True]
        if r["text_density"] == "medium":
            losses.append(("text_density", {"text_density": "high"}, "Keep on-screen text at medium density"))
        if not losses:
            return []
        changed = pd.concat([self._apply(row, change) for _, change, _ in losses], ignore_index=True)
        items = []
        for (lever, _, action), (point, low, high) in zip(losses, self.score_effect.effect(row, changed)):
            peer = self._evidence(lever, True if lever in BINARY_LEVERS else "medium", row)
            items.append((point, Recommendation(
                lever=lever, action=action, current=r[lever], proposed=r[lever], score_gain=0.0,
                score_ci=(0.0, 0.0), lifts={m: (0.0, 0.0, 0.0) for m in METRICS}, confidence="keep",
                evidence=f"Losing it would cost ≈ {-point:.1f} pts (95% CI {-high:.1f} to {-low:.1f}). {peer}")))
        return [rec for _, rec in sorted(items, key=lambda item: item[0])]

    # ---- helpers
    @staticmethod
    def _as_frame(campaign):
        if isinstance(campaign, pd.DataFrame):
            row = campaign.iloc[[0]].copy()
        elif isinstance(campaign, pd.Series):
            row = campaign.to_frame().T
        else:
            row = pd.DataFrame([campaign])
        row = row.reset_index(drop=True)
        for column in NUMERIC:
            if column in row:
                row[column] = pd.to_numeric(row[column], errors="coerce")
        for column in BINARY_LEVERS + ["is_retargeting"]:
            row[column] = row[column].map(lambda v: np.nan if pd.isna(v) else bool(v)).astype(object)
        if row.iloc[0]["platform"] not in PLATFORMS:
            raise ValueError(f"platform must be one of {PLATFORMS}")
        return row

    @staticmethod
    def _apply(row, change):
        changed = row.copy()
        for column, value in change.items():
            changed[column] = [value]
        return changed

    def _build(self, c, row, confidence):
        lever, proposed = c["lever"], next(iter(c["change"].values()))
        tradeoffs = [f"{METRICS[m]} {est:+.0%}" for m, (est, _, high) in c["lifts"].items() if high < 0 and est <= -0.03]
        if c["score"][2] < 0:
            tradeoffs.append(f"Klike Score {c['score'][0]:+.1f} pts")
        return Recommendation(lever=lever, action=c["action"], current=row.iloc[0][lever], proposed=proposed,
                              score_gain=c["score"][0], score_ci=(c["score"][1], c["score"][2]), lifts=c["lifts"],
                              confidence=confidence, evidence=self._evidence(lever, proposed, row), tradeoffs=tradeoffs)

    def _report(self, row, recommendations):
        r = row.iloc[0]
        expected = float(self.score_model.predict(row)[0])
        everything = row.copy()
        for rec in recommendations:
            if rec.confidence != "keep":
                everything = self._apply(everything, {rec.lever: rec.proposed})
        gain = float(self.score_model.predict(everything)[0]) - expected
        has_actual = "klike_score" in row and pd.notna(r["klike_score"])
        actual = float(r["klike_score"]) if has_actual else expected
        return Report(
            campaign_id=str(r["campaign_id"]) if "campaign_id" in row else "new campaign",
            context={k: r[k] for k in ["platform", "category", "objective", "target_audience_age"]},
            actual_score=actual, expected_score=expected,
            potential_score=float(np.clip(actual + max(gain, 0), 0, 100)),
            recommendations=recommendations, scored=bool(has_actual))

    def portfolio(self):
        """Run the engine on every campaign: one row per recommendation."""
        rows = []
        for _, campaign in self.data.iterrows():
            report = self.recommend(campaign)
            for rank, rec in enumerate(report.recommendations, 1):
                rows.append({"campaign_id": report.campaign_id, "platform": report.context["platform"],
                             "category": report.context["category"], "rank": rank, "lever": rec.lever,
                             "proposed": rec.proposed, "confidence": rec.confidence,
                             "score_gain": rec.score_gain, "ctr_lift": rec.lifts["ctr"][0],
                             "actual_score": report.actual_score, "potential_score": report.potential_score})
        return pd.DataFrame(rows)
