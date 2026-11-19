# Method, calibration, assumptions and sources

Entry: `main.py` for make.fun `gta6-trophy-rarity-forecast-01`.
Task: forecast each GTA VI PS5 trophy's earn rate on Exophase at
2026-11-26T00:00:00Z (a week after launch) from every read the house made
before the deadline, as five quantiles per trophy, scored by mean pinball
loss on the logit scale.

## 1. What is being forecast

Exophase prints, for each of the game's 55 trophies (1 platinum, 2 gold,
6 silver, 46 bronze), the percent of the players it tracks who have earned
it, to 2 decimals, plus the players-tracked count. The house reads the page
every 30 minutes on launch night, then every 10 minutes. The answer is the
first valid read in the hour from the target. The program runs offline in
a sandbox, so it forecasts from the case file alone.

## 2. The model

### 2.1 Saturating curves, fitted per trophy

A trophy's earn rate is cumulative progression: slow at first, steep while
the bulk of players reach it, flat once almost everyone who will earn it
has. The program fits, per trophy,

```
r(h) = S / (1 + exp(-(h - h50) / w)),   w = h50 / 3,
```

`h` in hours since 2026-11-19T00:00:00Z. The midpoint `h50` is chosen by
grid search over {4, 6, 9, 13, 19, 28, 40, 58, 84, 120, 175, 250, 360}
hours; for each candidate the asymptote `S` is the median of the asymptotes
each observed point implies, and the score is the mean squared logit error
plus `0.15 * (ln(h50 / h50_prior))^2`, a pull toward the archetype prior.
The fit uses one point per distinct displayed rate (at the hour it first
appeared), because Exophase is flat between refreshes and repeated reads
add no signal.

Archetype midpoints (priors) and prior asymptotes, from the trophy
descriptions:

| Archetype | Examples | h50 prior | S prior |
|---|---|---|---|
| story Ch.1 / 2 / 3 / 4 / final | Complete Chapter 1..4, the final mission | 20 / 32 / 48 / 64 / 95 h | 0.50 |
| side mission | Complete Breakfast Run, The Cartel Heist, … | 70 h | 0.16 |
| activity | Complete an arcade game, Try each activity once | 45 h | 0.10 |
| skill | Gold medal in all Shooting Range challenges, Win a drone race | 90 h | 0.07 |
| grind / collectible | Complete 250 Stunts, 3 Card sets, 100 vehicles, 5 Bank Vaults, $250k laundering | 130 h | 0.03 |
| 100% completion | Attain 100% Completion (gold) | 170 h | 0.015 |
| platinum | Vice Got a Price! | 220 h | 0.002 |

Archetypes are classified from the description by keyword rules in
`classify()` (chapter numbers, "Complete the final mission", count-based
patterns like "Complete 250" / "100 locations" / "Earn $", "gold medal" /
"Win a" / "Beat ", everything else an activity). The priors matter most
when a trophy's history is short or flat; with two days of refreshes the
fitted midpoint dominates.

### 2.2 Damped trend blend

A second forecast is the logit rate linear in `ln(1 + h)` (OLS over the
same distinct points), slope halved toward 0, re-anchored at the last
observed rate, and capped at the fitted asymptote `S` (a rate cannot pass
its own ceiling). The final median is the logit blend
`0.75 * curve + 0.25 * trend`: the curve is the primary belief, the trend
is insurance against a wrong curve shape.

### 2.3 Structural guards (on the medians)

- Platinum requires every other trophy, so its median is capped at the
  smallest other median.
- 100% completion requires every non-platinum trophy, so its median is
  capped at the smallest non-platinum, non-completion median.
- The main story chain (Chapter 1, 2, 3, 4, final mission) is played in
  order, so its medians are forced non-increasing.

### 2.4 The intro trophy

"Just Like Paradise" (Complete Gone Fishing) sat at 100% among the first
638 tracked players: (almost) everyone who plays finishes the first
mission. Its median is `100 - min(0.6, days * 0.13)` percent (99.99% over
90 minutes, 99.41% over the 4.5-day final horizon), with a spread that
widens with the horizon: the share of tracked owners who never play grows
as the base broadens.

### 2.5 Spreads (logit sigma, combined in quadrature)

- base 0.26
- 0.70 x the curve fit's residual std (capped at 0.8)
- 0.30 x sqrt(horizon in days)
- 0.30 x log10(forecast growth in tracked players) — dilution uncertainty
- archetype extra: grind 0.10, platinum / 100% 0.15
- 0.50 x |logit(curve) - logit(trend)| (capped at 1.2) — when the two
  models disagree, the band widens

Quantiles use the standard normal quantiles at the case's levels
(Acklam's inverse CDF), rounded to 4 decimals half-up like the starter.

### 2.6 Tracked players (for the dilution term only)

`ln P` is fitted against `ln(1 + h)` by OLS and shrunk toward a
launch-phase prior exponent of 1.0, with weight `span / (span + 12 h)`.
The prior exponent comes from the disclosed first refresh:
`638 -> 3,868` players between release and ~4.5 h in, i.e.
`(1 + 4.5)^b = 6.06`, `b ~= 1.06`; we use 1.0.

## 3. The first-refresh prior (the practice lesson)

Exophase refreshes in jumps, hours apart. The rules disclose the launch
night sequence: the page was unchanged (638 players, 943 trophies earned)
from 2026-11-18T23:48Z through the 04:00Z read, then refreshed by 04:30Z
to 3,868 players and 4,276 trophies earned, with every rate but the one at
100% lower.

Reconstruction. Trophy 2 sits at 100% (all 638 earned it). Non-intro
trophies earned: 943 - 638 = 305. After the refresh, trophy 2 still at
100% accounts for 3,868 of the 4,276 trophies, so non-intro trophies
earned are 4,276 - 3,868 = 408. In aggregate, non-intro rates fell by a
factor of (408 / 3,868) / (305 / 638) = **0.2206**. In cohort terms, the
new 3,230 players earned non-intro trophies at a rate `k` of the old
cohort's, where `1 + (6.06 - 1) * k = 408 / 305`, so `k = 0.067`: the new
cohort had earned almost nothing beyond the intro mission, which is what
launch night looks like — the first tracked players are the most hardcore,
everyone after them is hours behind.

When a case's history contains **no refresh at all** and the horizon is
at most 12 hours (exactly the practice shape), the program does not
extrapolate the flat lines. It applies this prior: players multiply by
`F = 6.06`, each non-intro trophy's median is its last rate times
`(1 + (F - 1) * k) / F`, with `k` by archetype (story/side 0.10, activity/
skill 0.08, grind 0.03, platinum/100% 0.01; the rate-weighted mean is
0.067, matching the aggregate derivation), and a wide spread of 0.45
logit units (plus the archetype extra) because the per-trophy fall is
heterogeneous. Once the history contains refreshes, the data takes over
and this prior is not used.

## 4. External base rates (context for the priors, not fitted inputs)

- **GTA V trophy rarity (PSN-wide, long term):** the platinum "Los
  Santos Legend" is ~0.1% of all players (PSN profile screenshots, 2026),
  ~1.5% on PSNProfiles (self-selected trophy hunters); the first story
  trophy "Welcome to Los Santos" is ~38.6% of players; mid-story trophies
  run 15.6% / 9.9% / 7.0%; 70 gold medals is 0.2%. GTA V's platinum
  takes 150+ hours, of which 100+ are GTA Online — GTA VI launches as
  single-player only, so its platinum (100% completion, no online) is
  more attainable than GTA V's, but still a multi-day grind nobody
  finishes in meaningful numbers in week one.
  Sources: PSNProfiles/PocketPSN trophy pages for GTA V; fullsweepmap.com
  GTA 6 trophy-list explainer (Sep 2026).
- **GTA VI launch scale:** Sensor Tower counted 7.71M digital preorders
  worldwide by Nov 3 and ~12.1M / $1.18B by Nov 13; Compound's base case
  is 14M copies in week one (10M low, 18M high); DFC Intelligence put
  day one at ~25M units; Newzoo's week-one range is 37–51M; PS5 took ~74%
  of preorders. So the PS5 player base at Nov 26 is of order 10–25M
  people, while Exophase tracks only the small self-selected fraction who
  sync profiles — hence tracked players grow fast all week and dilution
  is a first-order effect, not a footnote.
  Sources: Compound (compound.ie, Nov 18 2026), Reuters via Sensor
  Tower, Axis Intelligence, GamesRadar+/DFC.

## 5. Backtests (local, run before the deadline)

### 5.1 Practice cases (real house data, reconstructed answers)

The kit ships no practice answers. The answers are reconstructed from
the aggregates the rules disclose, as in section 3: every non-intro
trophy's rate at the target is its last rate times 0.2206, trophy 2
stays 100%. That reconstruction reproduces the house baseline's stated
practice score (our scorer gives the starter loss 1.402, typical miss
0.677 pp, 90% coverage 0.0; the rules say "about 1.51 / about 0.68 /
about 0"), so it is consistent with the true answers up to per-trophy
heterogeneity.

| Forecast | Loss | Typical miss | 90% coverage |
|---|---|---|---|
| This entry (p00 and p01, identical) | **0.116** | 0.020 pp | 0.98 |
| House baseline (starter) | 1.402 | 0.677 pp | 0.0 |

Sensitivity: if the true per-trophy falls are heterogeneous rather than
uniform — lognormal factors around 0.2206, rescaled to the same aggregate
— the entry's practice loss is 0.26 on average at sigma 0.5 and 0.63 at
sigma 1.0 (5 draws each), still well below the baseline's ~1.40. The
practice pays nothing; its value is that it proves the refresh prior
works on real house data.

### 5.2 Synthetic final (pipeline + shape test, known truth)

A synthetic final-like case (119 reads, Nov 18 23:50Z – Nov 21 11:00Z,
refreshes every 4 h, players 638 -> 24,464 in history -> 63,349 at the
target; per-trophy logistic progression with archetype saturations;
truth = the last refresh before the target). This tests the full program
on final-shaped data, including the story-chain guard and the platinum.

| Forecast | Loss | Typical miss | 90% coverage |
|---|---|---|---|
| This entry | **0.297** | 1.66 pp | 0.98 |
| House baseline (starter) | 0.630 | 4.24 pp | 0.98 |

Median spot checks (entry vs truth): Ch.1 68.9% vs 68.0%, Ch.2 50.5%
vs 55.0%, final mission 15.2% vs 20.3%, platinum 0.046% vs 0.060%,
100% completion 0.21% vs 0.63%, 250 Stunts 0.29% vs 0.38%.

### 5.3 Determinism and limits

Two runs on the same case write byte-identical outputs. Runtime ~0.3 s
per case (limit 8 s). Output ~2.8 KB per case (limit 65,536 bytes).
Python standard library only; no network; no randomness.

## 6. Limitations, stated plainly

- The practice answers are a reconstruction, not the house's answers.
  The uniform-fall version is the neutral assumption; the heterogeneous
  versions (loss 0.26–0.63) are the honest range.
- The synthetic final was built with the same logistic family the model
  uses, so its 0.297 is optimistic. The real final's shape errors will
  be larger; the spreads are sized for that, and the practice range is
  the better guide to the real loss.
- The archetype priors (midpoints, asymptotes, first-refresh `k`) are
  judgement calls from GTA V analogues and the one disclosed refresh.
  They matter most where history is thin (the platinum, the grinds);
  the story trophies are driven by their own two days of data.
- The real final case (f00, cut at the deadline) will have ~2 days of
  reads at 10-minute cadence; this program was tested on the practice
  shape (7–8 reads, flat) and the synthetic final shape (119 reads,
  refresh steps), not on f00 itself, which does not exist yet.
- The house model forecast (an AI model at maximum reasoning effort,
  called at the deadline minus 2 h) is the unknown bar alongside the two
  baselines; beating it is not guaranteed by beating the baselines.
