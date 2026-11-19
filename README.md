# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/georgeonweb3/makefun-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/georgeonweb3/makefun-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
contest **gta6-trophy-rarity-forecast-01**: forecast, for every Grand Theft
Auto VI PlayStation 5 trophy, its earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z — one week after launch — as five quantiles per
trophy (5%, 25%, 50%, 75%, 95%). Scored by mean pinball loss on the logit
scale; a prize needs a loss strictly below the best house entry.

## What it is

`main.py` (Python 3.13, standard library only, no network, deterministic,
~0.3 s per case) reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (or
`/input`) and writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (or
`/output`).

## Method in one paragraph

A trophy's earn rate follows a saturating curve: slow at first, steep while
the bulk of players reach it, flat once almost everyone who will earn it
has. Each trophy is forecast with a logistic curve in hours since release,
`r(h) = S / (1 + exp(-(h - h50) / w))`, with the midpoint `h50` fitted to
the trophy's own history by grid search (pulled toward its archetype's
midpoint: story chapters peak early, grinds and the platinum late) and the
asymptote `S` solved from that history, so the level is the trophy's own.
A damped log-time trend of the logit rate is blended in at 25% as a hedge
against a wrong curve shape. Structural guards enforce what must be true:
the platinum cannot exceed any other trophy, 100% completion cannot exceed
any non-platinum trophy, and the story chain (Chapter 1 .. final mission)
is non-increasing. The one special behaviour is the Exophase refresh: the
tracker updates its counts in jumps hours apart, and on launch night the
first refresh (638 -> 3,868 players tracked) crashed every rate but the
100% one. When a case's history contains no refresh at all, the program
applies a first-refresh prior calibrated on that disclosed refresh instead
of extrapolating flat lines. Full method, calibration, assumptions and
sources: [METHOD.md](METHOD.md).

## Backtest (local, before the deadline)

| Test | This entry | House baseline (starter) |
|---|---|---|
| Practice p00/p01, reconstructed answers (uniform 0.2206x fall) | loss **0.116**, typical miss 0.020 pp, 90% coverage 0.98 | loss 1.402, miss 0.677 pp, coverage 0.0 |
| Practice, heterogeneous falls (lognormal, sigma 0.5 / 1.0) | loss 0.26 / 0.63 (mean of 5 draws) | ~1.40 |
| Synthetic final (2 days of history, 4.5-day horizon, known truth) | loss **0.297**, miss 1.66 pp, coverage 0.98 | loss 0.630, miss 4.24 pp, coverage 0.98 |

The practice answers are reconstructed from the aggregates the rules
disclose (943 -> 4,276 trophies earned, 638 -> 3,868 players, every rate
but the 100% one down); the uniform-fall version is the neutral
assumption and the heterogeneous versions show the range. The synthetic
final shares the logistic family with the model, so its 0.297 is
optimistic; the practice range (0.12–0.63) is the honest calibration.
The real final is scored on 2026-11-26 rates nobody has yet.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Files

- `main.py` — the forecaster (the whole entry; nothing else runs)
- `METHOD.md` — method, calibration, assumptions, sources, backtests
- `LICENSE` — MIT
- `.makefun/owner.json` — claim file naming the entrant's wallet
