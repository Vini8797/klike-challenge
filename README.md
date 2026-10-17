# Klike Data Science Challenge

An exploratory analysis of 500 video ad campaigns, a model for the Klike Score, and a
recommendations engine that tells an advertiser what to change in a creative and what the
change is worth.

**The dataset arrives damaged, and not at random.** Excel had converted 558 numeric cells
into dates — including 16 Klike Scores, which turn out to be 16 of the 17 lowest scores in
the data. Dropping the rows that will not parse is the obvious move and the wrong one: it
deletes the failing creatives, which is precisely what a product like Klike exists to
identify. Recovering them instead is the decision the rest of the project rests on.

| | |
|---|---|
| **Score model** | Ridge on pre-launch features only · MAE 7.1, R² 0.69 on held-out data |
| **With post-launch metrics** | MAE 5.5, R² 0.80 — useful as a benchmark, useless as advice |
| **Biggest creative lever** | A hook in the first 3 seconds: **+17 score points**, +68% CTR |
| **Platform effect** | Only format reverses: vertical is +9 pts on TikTok, −12 on LinkedIn |
| **Engine** | 1,732 recommendations across 500 campaigns in 93s, each quantified with a 95% interval |

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python recommend.py KLK-0116       # recommendations for one campaign
python recommend.py --random 3     # 3 random campaigns
python recommend.py --file new_campaign.json   # a campaign not in the dataset
python recommend.py --portfolio    # summary across all campaigns (~1.5 min)

python -m pytest                   # tests
jupyter lab                        # notebooks, in order: 01, 02, 03
```

A new campaign needs the same columns as the dataset:

```json
{"platform": "TikTok", "category": "E-commerce", "objective": "conversions", "target_audience_age": "18-24",
 "is_retargeting": false, "video_duration_s": 45, "format": "horizontal", "has_subtitle": false,
 "has_cta": true, "has_hook": false, "has_face": false, "text_density": "high", "music_voice_ratio": 0.8}
```

## Structure

```
01_eda.ipynb              Part 1 — cleaning and EDA
02_modeling.ipynb         Part 2 — modeling
03_recommendations.ipynb  Part 3 — recommendations engine
recommend.py              command line for the engine
klike/
  data.py                 reads and cleans the Excel file
  model.py                features and the score model
  recommender.py          the engine
tests/                    cleaning, model and engine
data/klike_clean.csv      cleaned dataset
figures/                  charts used below
```

The notebooks carry the reasoning; `klike/` does the same work as a library, so the command
line runs without them.

---

## Part 1 — Data and EDA

[01_eda.ipynb](01_eda.ipynb)

### Six things were wrong

| Problem | Scale | What I did |
|---|---|---|
| Numbers stored as dates by Excel | 558 cells, 6 columns | Each cell keeps its display format (`d.m`, `yyyy.mm`…), which decodes the original digits exactly |
| `ctr` mixing fractions, per-mille and basis points | 3 conventions, row by row | Recomputed CTR, CPC and ROAS from clicks, impressions, spend and revenue |
| Wrong values in `impressions` and `spend` | 4 + 2 cells | Found by identity, not by eye (below) |
| Missing values | 5–9% across 7 columns | Left as NaN; imputation happens inside the model's CV folds, not before the split |
| Heavy skew in spend, revenue, impressions, clicks | log-normal | Outliers judged on a log scale, flagged but kept |
| Revenue of exactly zero | 72 campaigns | Mostly awareness and engagement — revenue not tracked, so excluded from ROAS rather than scored as failures |

**No rows dropped.**

### Finding the broken cell, not the odd row

CTR, CPC and ROAS are computed from the same four columns, so the three identities overlap.
When one breaks, the *pattern* of which ones break names the guilty column: if only the CTR
identity fails, `clicks` is vouched for by CPC, so `impressions` is wrong.

The two `spend` repairs confirm themselves. Rebuilding spend from `cpc × clicks` makes the
ROAS identity — which was never used in the repair — match the ROAS that shipped in the
file, to both decimals.

### What the naive cleaning would have cost

Not a hypothetical. I trained the same model twice, once on the recovered data and once with
the 16 corrupted targets dropped, and scored both out-of-fold:

| | bias on worst 10% | RMSE on worst 10% | RMSE overall |
|---|---|---|---|
| Recovered (this project) | 9.22 | 12.42 | 8.31 |
| Dropped (naive cleaning) | 10.82 | 13.47 | 8.40 |

Overall RMSE barely moves — which is exactly why the damage would pass unnoticed. On the
worst creatives the naive model gets **more optimistic** and less accurate. For a product
whose job is to tell someone their creative is weak, that is the wrong direction to be wrong
in.

![Recovered scores](figures/01_recovered_targets.png)

### Which attributes move which metric

Difference in the median when the attribute is present:

| Attribute | Klike Score | CTR | Conversions | Engagement |
|---|---|---|---|---|
| Hook in first 3s | **+17 pts** | +68% | +33% | +42% |
| Human face | +12 pts | +37% | +45% | +38% |
| Call to action | +8 pts | +14% | +28% | +10% |
| Subtitles | +7 pts | +15% | +9% | +12% |
| Retargeting | +3 *(ns)* | +71% | **+237%** | +19% |

**Text density is not monotonic.** Medium scores best (65), then low (59), then high (50) —
so the advice for a text-heavy creative is "reduce to medium", not "reduce text".

**Retargeting is the cleanest result here.** It drives conversions up 237% and leaves the
score flat: the Klike Score measures the video, not the audience. So the engine never
recommends it, and never claims credit for what targeting would have delivered anyway. ROAS
responds to no creative attribute at all — it is driven by price and offer, neither of which
is in this dataset.

![Creative levers](figures/01_creative_levers.png)
![Text density](figures/01_text_density.png)

### Platforms differ in one way only

Hook, face, CTA and subtitles have the same effect on TikTok, Meta and LinkedIn. **Format is
the exception**: vertical scores 9 points higher on TikTok and 12 lower on LinkedIn — though
LinkedIn has only 11 vertical videos, so that side is directional. Duration has no effect on
the score anywhere, and only a modest TikTok-only effect on CTR, peaking near 21 seconds.

LinkedIn's median ROAS is 0.6 against TikTok's 3.0.

![Format by platform](figures/01_format_by_platform.png)

---

## Part 2 — Modeling

[02_modeling.ipynb](02_modeling.ipynb)

### Two models, because the engine needs a creative one

- **Creative model** — only what exists before the ad runs: video attributes, platform,
  category, objective, audience. This is what the engine uses.
- **Full model** — adds CTR, completion rate and conversion rate, which exist only after
  launch. A benchmark, not a source of advice: "get more clicks" is not a recommendation.

### Feature engineering

Each candidate was kept only if it improved cross-validated R².

| Feature | Δ R² | Verdict |
|---|---|---|
| platform × format | **+0.019** | kept — vertical has opposite signs on TikTok and LinkedIn, so a lone `format` column averages them to nothing |
| duration² | 0.000 | dropped |
| hook × face | −0.001 | dropped |
| platform × text density | −0.002 | dropped |
| month of launch | −0.004 | dropped |

For the full model, CTR in log scale beat raw CTR decisively (+0.109 vs +0.066 R²).

Deliberately excluded: spend, impressions and revenue (they measure budget, not creative
quality) and average watch time (it tracks video length; completion rate is used instead).

### Results

5-fold cross-validation, 5 repeats:

| Model | MAE | R² |
|---|---|---|
| Baseline (mean) | 12.6 | 0.00 |
| **Ridge · creative** | **6.6** | **0.71** |
| Gradient boosting · creative | 6.9 | 0.68 |
| Random forest · creative | 7.7 | 0.62 |
| Ridge · full | 4.9 | 0.84 |

Ridge wins, and the tree models find nothing extra — so the relationship is close to
additive. That is convenient twice over: the accurate model is also the interpretable one,
and each coefficient *is* the number of points a change is worth, which is exactly the shape
the engine needs.

### Where it fails

On a 100-campaign test set: MAE 7.1, R² 0.69 (full model: 5.5 and 0.80). Seven points
separates weak from strong creatives but cannot promise an exact score.

It overestimates the worst creatives by ~10 points and underestimates the best — so part of
what makes a video excellent or terrible is not in these columns.

![Predicted vs actual](figures/02_predicted_vs_actual.png)
![Effect sizes](figures/02_effect_sizes.png)

The ranking matches creative practice: the first seconds decide whether people keep
watching, some text helps when the video plays muted, faces and subtitles help. Context —
category, audience age — matters far less than the creative choices.

**One finding for the Klike team.** Holding the creative fixed, retargeting adds about 4
points to the score, and that effect vanishes once CTR enters the model. The score looks
like it is absorbing part of the audience effect through CTR.

---

## Part 3 — Recommendations engine

[03_recommendations.ipynb](03_recommendations.ipynb)

### How it works

1. List the changes available to this campaign — add a hook, face, CTA or subtitles; change
   text density; move to a format that is common on the platform; shift the music/voice
   balance; change duration.
2. Estimate each one: score effect from the creative model, and CTR, engagement and
   conversion effects from separate per-platform models. 95% intervals come from bootstrap.
3. Drop what is unlikely to help, and anything that would hurt the metric tied to the
   campaign's objective — a change that lifts engagement but costs conversions is not
   offered to a conversions campaign.
4. Rank what survives by score gain.
5. Attach a comparison against similar campaigns.

Retargeting is never recommended. It is a targeting decision, not a creative one.

### Example

```
$ python recommend.py KLK-0116

Campaign KLK-0116 · LinkedIn · Lead Gen · engagement · age 35-44
Klike Score 10.4  (creative model expects 15.6; potential with the changes below ≈ 77.5)

1. Add a hook in the first 3 seconds: Klike Score +17.1 pts (95% CI +15.7 to +18.4). On this platform
   it is associated with +45% CTR, +7% engagement rate, +71% conversions at the same reach
   (model-adjusted). 86% of top-quartile LinkedIn creatives open with a hook, vs 19% of the bottom
   quartile (n=83).
2. Change on-screen text from high to medium: Klike Score +13.5 pts (95% CI +11.4 to +15.4). ...
3. Show a human face on screen: Klike Score +11.8 pts (95% CI +10.6 to +13.1). ...
4. Add subtitles: Klike Score +8.3 pts ...
5. Re-cut the video from vertical (9:16) to horizontal (16:9): Klike Score +8.3 pts ...
6. Add an explicit call to action: Klike Score +8.1 pts ...
```

KLK-0116 is the worst creative in the dataset. Same attributes on TikTok and recommendation
5 flips to vertical.

### Context changes the answer

![Lifts by platform](figures/03_lifts_by_platform.png)

The hook lifts CTR and conversions on every platform. A face pays off most on TikTok.
Cutting heavy text and shortening the video only move CTR on Meta.

Objective matters too. On TikTok, square → vertical raises the score and engagement but is
estimated at **−27% conversions**, so the engine offers it to engagement campaigns and
withholds it from conversion campaigns.

### Campaigns that are already good

The challenge asks for 3 recommendations, but the best creative scores 99.1 and has nothing
left to fix. Every campaign still gets 3 items: where real changes run out, the engine fills
in with A/B tests worth running, or with what to protect — *"keep the hook; removing it
would cost about 17 points."*

320 campaigns (64%) get 3 or more changes backed by confident evidence.

### Across the portfolio

![Portfolio](figures/03_portfolio_opportunity.png)

Mean score 60.2, rising to 86.6 if every recommendation were applied — an upper bound, since
it assumes effects add. The biggest single opportunity is the hook: 244 campaigns lack one.

---

## Part 4 — Product questions

### 1. What I would extract from the video itself

The dataset records *whether* a video has a hook, a face or text. With the file I would
measure how good each one is:

- **First 3 seconds** — cuts, motion, time to first face, product or spoken word.
- **Pacing** — shot count and average shot length. Probably matters more than total
  duration, which predicts nothing here.
- **Faces** — screen time, eye contact, studio vs UGC.
- **Text (OCR)** — screen coverage and whether it is on screen long enough to read.
- **Audio (speech-to-text)** — words per minute, when the brand is named, how the CTA is
  delivered, music tempo, trending sounds on TikTok.
- **Branding** — when the logo appears, how long the product is on screen.
- **Embeddings** — to retrieve high-scoring similar creatives as concrete examples.

With per-second retention curves, the engine could point at the exact moment viewers leave.

### 2. Production

Two flows. **Pre-launch**: upload a video, get a score and recommendations. **Post-launch**:
a daily job refreshes scores with real metrics and pushes the biggest opportunities.

Videos land in object storage; queue workers extract features. That is the heavy part —
GPUs, autoscaling, a per-video cache. Features go to a feature store and an API serves the
app. The models are linear, so inference is cheap and bootstrap coefficients precompute.

Retraining is scheduled, and a new version ships only if it is at least as accurate *and*
its effect estimates are stable. Peer comparisons enforce a minimum group size, so one
client's data is never readable through another's benchmark.

**Validating the advice matters more than any of that.** These numbers are correlations from
historical data. When a client applies a recommendation it should run as an A/B test — Meta
and TikTok both support it — and the result should feed back to correct the estimate.

### 3. With more time

- Causal estimates — matching or double ML — instead of adjusted correlations.
- A hierarchical model by platform and category, to handle thin groups like LinkedIn.
- Rank by *effort-adjusted* gain: adding subtitles is not the same job as reshooting. And
  convert CTR gains into expected revenue.
- Model combinations of changes rather than assuming they add.
- Chase the retargeting-through-CTR effect; confirm the `engagement_rate` unit with the team.
- A small UI for the engine.

---

## Assumptions and limitations

- Effects are measured from historical data, not experiments. The models adjust for
  platform, category, objective, audience and retargeting, but confounders remain possible.
- `engagement_rate` is assumed to be in basis points — there is no column in the dataset to
  verify it against. 13 values below 1 look like placeholders and are excluded from the
  engagement model.
- LinkedIn has 83 campaigns and 11 vertical videos. Its estimates are directional.
- The "potential score" assumes recommended changes add up, so treat it as a ceiling.