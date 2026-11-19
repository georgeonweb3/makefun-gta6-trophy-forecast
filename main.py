"""Forecast GTA VI PS5 trophy earn rates on Exophase, one week after launch.

Entry for make.fun gta6-trophy-rarity-forecast-01 (program/v1, Python 3.13,
standard library only, no network). Reads every ``*.in`` case in the input
directory (``$MAKEFUN_INPUT_DIR`` or ``/input``) and writes one
``<id>.out`` per case to the output directory (``$MAKEFUN_OUTPUT_DIR`` or
``/output``): five quantiles (5/25/50/75/95%) of each trophy's earn rate, in
percent, at the case's target.

Method (full writeup in METHOD.md). A trophy's earn rate is the share of
tracked players who have earned it, and that share follows a saturating
curve: slow at first, steep while the bulk of players reach it, flat once
almost everyone who will earn it has. So each trophy is forecast with a
logistic curve in hours since release, ``r(h) = S / (1 + exp(-(h - h50) /
w))``. The curve's midpoint ``h50`` is fitted to the trophy's own history
by grid search, pulled toward its archetype's midpoint (story chapters
peak early, grinds and the platinum late); its asymptote ``S`` is solved
from the history, so the level is the trophy's own, not the archetype's.
A damped log-time trend of the logit rate is blended in at 25% as a hedge
against a wrong curve shape. Structural guards then enforce what must be
true: the platinum cannot exceed any other trophy, 100% completion cannot
exceed any non-platinum trophy, and the main story chain (Chapter 1 ..
the final mission) is non-increasing. Spreads on the logit scale combine
the fit residual, a horizon term, a dilution term that grows with the
forecast growth in tracked players, and the disagreement between the
curve and the trend, so the bands are honest about a 4.5-day horizon.

One behaviour is special, and it is the whole lesson of the practice
cases: Exophase refreshes its counts in jumps, hours apart, not
continuously. On launch night the page sat unchanged at 638 players
tracked for hours, then refreshed to 3,868 players, and every rate but the
one at 100% fell hard (943 trophies earned became 4,276: the new cohort
had earned almost nothing beyond the intro mission). When a case's history
contains no refresh at all, this program therefore does not extrapolate
the flat lines: it applies a first-refresh prior, calibrated on that
disclosed refresh (players x6.06, new-cohort earn rate ~7% of the old
cohort's for non-intro trophies, archetype by archetype). Once the history
contains refreshes, the data takes over and the prior is only shrinkage.

Deterministic: no randomness, no wall-clock reads, no dict-order
dependence; two runs on the same case write the same bytes.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# --- calibration constants (sources and derivations in METHOD.md) --------
# Launch-phase power-law exponent for tracked players, ln P ~ b * ln(1+h):
# the disclosed first refresh took P from 638 to 3,868 between release and
# ~4.5 h in, i.e. (1+4.5)^b = 6.06, b ~= 1.06. We use 1.0, and shrink the
# fitted exponent toward it when the history is short.
BETA_P_PRIOR = 1.0
# Hours of history span that earn the fitted P exponent equal weight with
# the prior: span / (span + PRIOR_HALF_LIFE_H).
PRIOR_HALF_LIFE_H = 12.0
# First-refresh prior: players multiply by this at the first refresh
# (3,868 / 638 = 6.06, disclosed in the rules).
FIRST_REFRESH_F = 6.06
# New-cohort earn rate, as a fraction of the old cohort's rate, by
# archetype. Aggregate derivation: non-intro trophies earned went 305 ->
# 408 while players went 638 -> 3,868, so 1 + (F-1)*k = 408/305, k = 0.067
# on average; story trophies keep more of it (new players do reach the
# early chapters), grinds keep almost none.
K_FIRST = {
    "story": 0.10,
    "side": 0.10,
    "activity": 0.08,
    "skill": 0.08,
    "grind": 0.03,
    "platinum": 0.01,
    "completion": 0.01,
}
# Archetype curve midpoints (hours since release) and prior asymptotes
# (fraction of players): chapters peak early and high, side content next,
# activities mid, grinds and the platinum late and low. The midpoint is a
# prior the grid search is pulled toward; the asymptote is only used when
# a trophy has no usable history at all.
H50_PRIOR = {
    "story": 40.0,
    "side": 70.0,
    "activity": 45.0,
    "skill": 90.0,
    "grind": 130.0,
    "platinum": 220.0,
    "completion": 170.0,
}
H50_CHAPTER = {1: 20.0, 2: 32.0, 3: 48.0, 4: 64.0, 99: 95.0}
S_PRIOR = {
    "story": 0.50,
    "side": 0.16,
    "activity": 0.10,
    "skill": 0.07,
    "grind": 0.03,
    "platinum": 0.002,
    "completion": 0.015,
}
# Grid of curve midpoints the fit tries, in hours.
H50_GRID = [4.0, 6.0, 9.0, 13.0, 19.0, 28.0, 40.0, 58.0, 84.0, 120.0,
            175.0, 250.0, 360.0]
# Weight of the midpoint prior in the grid-search score (logit^2 units).
H50_PRIOR_WEIGHT = 0.15
# Blend weight of the S-curve against the damped trend, in logit space.
W_CURVE = 0.75
# Spread pieces, in logit units (combined in quadrature).
SPREAD_BASE = 0.26          # irreducible trophy-level uncertainty
SPREAD_RESID = 0.70         # weight on the fit's residual std (capped)
SPREAD_HORIZON = 0.30       # per sqrt(day) of horizon
SPREAD_DILUTION = 0.30      # per decade of forecast player growth
SPREAD_DISAGREE = 0.50      # weight on curve-vs-trend disagreement (capped)
SPREAD_FIRST_REFRESH = 0.45  # spread while the first refresh is a prior
SPREAD_ARCH_EXTRA = {"grind": 0.10, "platinum": 0.15, "completion": 0.15}


# --- small numeric helpers ------------------------------------------------

def clamp_pct(x):
    """Clamp a percent into the range the logit can take."""
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    """Round to 4 decimals, half up, as JavaScript Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation, |err| < 1.2e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def parse_time(s):
    """Parse an ISO 8601 UTC timestamp (or a plain date) to a datetime."""
    if s is None:
        return None
    s = str(s).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (slope, intercept, resid_std, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if n == 1:
        return 0.0, ys[0], 0.0, 1
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return b, a, resid, n


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


# --- trophy archetypes -----------------------------------------------------

_COUNT_RE = re.compile(
    r"Complete \d+|\d+ locations|\d+ different|\d+ Bank Vaults|every spot"
    r"|all of|all garage|Discover \d+|Register \d+|Earn \$|Spend \$"
)
_CHAPTER_RE = re.compile(r"Chapter (\d+)")


def classify(trophy):
    """Archetype of a trophy from its tier and description.

    Returns (archetype, chapter): archetype in story/side/activity/skill/
    grind/platinum/completion; chapter is the story chapter number for
    story trophies (99 for the final mission), else None.
    """
    tier = (trophy.get("tier") or "").lower()
    desc = (trophy.get("description") or "").strip()
    if tier == "platinum":
        return "platinum", None
    if "100% Completion" in desc or "100 percent" in desc.lower():
        return "completion", None
    m = _CHAPTER_RE.search(desc)
    if m:
        return "story", int(m.group(1))
    if desc == "Complete the final mission.":
        return "story", 99
    if _COUNT_RE.search(desc):
        return "grind", None
    if desc.startswith("Complete "):
        return "side", None
    low = desc.lower()
    if "gold medal" in low or desc.startswith("Win ") or desc.startswith("Beat "):
        return "skill", None
    return "activity", None


def h50_prior(arch, chapter):
    if arch == "story" and chapter is not None:
        return H50_CHAPTER.get(chapter, H50_PRIOR["story"])
    return H50_PRIOR.get(arch, 60.0)


# --- the forecast ----------------------------------------------------------

def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0

    # Hours since release-day start (game.release_date, else first read).
    t0 = parse_time((case.get("game") or {}).get("release_date"))
    if t0 is None and history:
        t0 = parse_time(history[0].get("at"))
    if t0 is None:
        t0 = parse_time(case.get("at"))

    def hours_of(dt):
        return (dt - t0).total_seconds() / 3600.0 if dt is not None and t0 is not None else 0.0

    at_dt = parse_time(case.get("at"))
    target_dt = parse_time(case.get("target"))
    if target_dt is None and at_dt is not None:
        target_dt = at_dt
    h_target = hours_of(target_dt)

    # Per-read series: hours, players, rates.
    reads = []
    for read in history:
        dt = parse_time(read.get("at")) or parse_time(read.get("slot"))
        p = read.get("players_tracked")
        rates = read.get("rates") or []
        if p is None or p <= 0:
            continue
        reads.append((hours_of(dt), p, rates))
    if not reads:
        return fallback_forecast(case)

    # How many refreshes the history shows (a read whose players or any
    # rate changed from the previous read). Exophase refreshes in jumps;
    # a history with none is launch night before the first refresh.
    n_refresh = 0
    for j in range(1, len(reads)):
        if reads[j][1] != reads[j - 1][1] or reads[j][2] != reads[j - 1][2]:
            n_refresh += 1

    xs = [math.log1p(max(h, 0.0)) for h, _, _ in reads]
    h_last = reads[-1][0]
    h_first = reads[0][0]
    span_h = max(h_last - h_first, 0.0)
    p_last = reads[-1][1]
    x_last = math.log1p(max(h_last, 0.0))
    x_target = math.log1p(max(h_target, 0.0))
    growth_x = max(x_target - x_last, 0.0)

    # --- tracked players at the target (for the dilution spread term) -----
    ys_p = [math.log(p) for _, p, _ in reads]
    b_p_fit, _, _, _ = ols(xs, ys_p)
    w_data = span_h / (span_h + PRIOR_HALF_LIFE_H) if span_h > 0 else 0.0
    b_p = w_data * b_p_fit + (1.0 - w_data) * BETA_P_PRIOR
    p_target = p_last * math.exp(b_p * growth_x)
    p_target = min(max(p_target, p_last), p_last * 50.0)
    dilution_log10 = math.log10(max(p_target / p_last, 1.0))

    # Last non-null rate per trophy, for anchoring.
    last_rate = []
    for i in range(len(trophies)):
        rate = None
        for _, _, rates in reversed(reads):
            if i < len(rates) and rates[i] is not None:
                rate = float(rates[i])
                break
        last_rate.append(rate if rate is not None else 0.0)

    archetypes = [classify(t) for t in trophies]
    first_refresh = n_refresh == 0 and horizon_hours <= 12.0

    medians = []
    spreads = []
    for i, trophy in enumerate(trophies):
        arch, chapter = archetypes[i]
        r_last = last_rate[i]

        # The intro trophy: (almost) everyone who plays finishes the first
        # mission, so it sits near 100% and only drifts down as tracked
        # players who never played accumulate.
        if r_last >= 99.5:
            med = 100.0 - min(0.6, horizon_days * 0.13)
            spr = 0.10 + 0.15 * horizon_days
            medians.append(clamp_pct(med))
            spreads.append(spr)
            continue

        if first_refresh:
            # No refresh seen yet: the next refresh is the first one, and
            # the disclosed launch-night refresh is the calibration. The
            # median rate is the old rate times the dilution factor
            # (1 + (F-1)*k) / F; the spread is wide because the per-trophy
            # fall is heterogeneous.
            k = K_FIRST.get(arch, 0.08)
            med = r_last * (1.0 + (FIRST_REFRESH_F - 1.0) * k) / FIRST_REFRESH_F
            spr = SPREAD_FIRST_REFRESH + SPREAD_ARCH_EXTRA.get(arch, 0.0)
            medians.append(clamp_pct(med))
            spreads.append(spr)
            continue

        # --- main model: saturating curve + damped trend ------------------
        # Distinct displayed points: one per rate value, at the hour it
        # first appeared (a read that repeats the previous rate adds no
        # signal: Exophase is flat between refreshes).
        points = []
        prev = None
        for h, _, rates in reads:
            if i < len(rates) and rates[i] is not None:
                r = float(rates[i])
                if prev is None or r != prev:
                    points.append((h, r))
                    prev = r
        if points and points[-1][1] != r_last:
            points.append((h_last, r_last))
        pts = [(h, max(r, 0.01)) for h, r in points if r is not None]

        hp_prior = h50_prior(arch, chapter)
        s_prior = S_PRIOR.get(arch, 0.05)

        # Grid-search the curve midpoint. For each candidate, the asymptote
        # is the median of the asymptotes each point implies, and the score
        # is the mean squared logit error plus a pull toward the prior.
        best = None
        if len(pts) >= 2:
            for h50 in H50_GRID:
                w = h50 / 3.0
                cand_s = []
                for h, r in pts:
                    z = (h - h50) / w
                    z = max(min(z, 30.0), -30.0)
                    cand_s.append(min(r / 100.0 * (1.0 + math.exp(-z)), 1.0))
                s_fit = median(cand_s)
                if s_fit <= 0:
                    continue
                err = 0.0
                for h, r in pts:
                    z = (h - h50) / w
                    z = max(min(z, 30.0), -30.0)
                    pred = s_fit / (1.0 + math.exp(-z)) * 100.0
                    err += (logit(pred) - logit(r)) ** 2
                err /= len(pts)
                score = err + H50_PRIOR_WEIGHT * (math.log(h50 / hp_prior) ** 2)
                if best is None or score < best[0]:
                    best = (score, h50, s_fit, err)
        if best is not None:
            _, h50, s_fit, resid_ms = best
        else:
            # Too little history to fit a shape: the prior midpoint, and an
            # asymptote solved from the last rate alone.
            h50 = hp_prior
            w = h50 / 3.0
            z = max(min((h_last - h50) / w, 30.0), -30.0)
            s_fit = min(max(r_last, 0.01) / 100.0 * (1.0 + math.exp(-z)), 1.0)
            if s_fit <= 0:
                s_fit = s_prior
            resid_ms = 0.25
        w = h50 / 3.0
        z_t = max(min((h_target - h50) / w, 30.0), -30.0)
        med_curve = clamp_pct(s_fit / (1.0 + math.exp(-z_t)) * 100.0)

        # Damped trend: logit(rate) linear in ln(1+h), slope halved toward
        # 0, capped at the fitted asymptote (a rate cannot pass its own
        # ceiling). It is the hedge against a wrong curve shape.
        xs_p = [math.log1p(max(h, 0.0)) for h, _ in pts]
        ys_p = [logit(r) for _, r in pts]
        b_tr, a_tr, _, _ = ols(xs_p, ys_p)
        b_tr = 0.5 * b_tr
        if len(pts) >= 2:
            lg_tr = a_tr + b_tr * x_target
            # Re-anchor the trend at the last point so a level shift in the
            # fit cannot move the forecast off the last observed rate.
            lg_last_fit = a_tr + b_tr * x_last
            lg_tr += logit(r_last) - lg_last_fit
            med_trend = clamp_pct(min(inv_logit(lg_tr), s_fit * 100.0))
        else:
            med_trend = clamp_pct(r_last)

        # Logit blend, then the spread.
        lg = W_CURVE * logit(med_curve) + (1.0 - W_CURVE) * logit(med_trend)
        med = clamp_pct(inv_logit(lg))
        disagree = abs(logit(med_curve) - logit(med_trend))
        spr = math.sqrt(
            SPREAD_BASE ** 2
            + (SPREAD_RESID * min(math.sqrt(resid_ms), 0.8)) ** 2
            + (SPREAD_HORIZON * math.sqrt(horizon_days)) ** 2
            + (SPREAD_DILUTION * dilution_log10) ** 2
            + SPREAD_ARCH_EXTRA.get(arch, 0.0) ** 2
            + (SPREAD_DISAGREE * min(disagree, 1.2)) ** 2
        )
        medians.append(med)
        spreads.append(spr)

    # --- structural guards on the medians ----------------------------------
    # Platinum requires every other trophy: it cannot exceed the smallest.
    # 100% completion requires every non-platinum trophy.
    # The main story chain is played in order: non-increasing.
    idx_plat = [i for i, (a, _) in enumerate(archetypes) if a == "platinum"]
    idx_comp = [i for i, (a, _) in enumerate(archetypes) if a == "completion"]
    others = [medians[i] for i in range(len(trophies)) if i not in idx_plat]
    if idx_plat and others:
        cap = min(others)
        for i in idx_plat:
            medians[i] = min(medians[i], cap)
    others_np = [medians[i] for i in range(len(trophies))
                 if i not in idx_plat and i not in idx_comp]
    if idx_comp and others_np:
        cap = min(others_np)
        for i in idx_comp:
            medians[i] = min(medians[i], cap)
    chain = sorted(
        ((ch, i) for i, (a, ch) in enumerate(archetypes) if a == "story" and ch is not None),
        key=lambda t: t[0],
    )
    running = None
    for _, i in chain:
        if running is None or medians[i] < running:
            running = medians[i]
        else:
            medians[i] = running

    # --- quantiles -----------------------------------------------------------
    z = [norm_ppf(level) for level in case.get("levels") or LEVELS]
    rows = []
    for i, trophy in enumerate(trophies):
        lg = logit(medians[i])
        s = spreads[i]
        q = [round4(clamp_pct(inv_logit(lg + zj * s))) for zj in z]
        # Rounding can only tie neighbours; enforce non-decreasing anyway.
        for j in range(1, len(q)):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def fallback_forecast(case):
    """No usable history at all: archetype priors, very wide bands."""
    z = [norm_ppf(level) for level in case.get("levels") or LEVELS]
    prior_med = {
        "story": 30.0, "side": 12.0, "activity": 8.0, "skill": 5.0,
        "grind": 2.0, "platinum": 0.05, "completion": 0.5,
    }
    rows = []
    for trophy in case["trophies"]:
        arch, _ = classify(trophy)
        med = prior_med.get(arch, 5.0)
        lg = logit(med)
        q = [round4(clamp_pct(inv_logit(lg + zj * 1.2))) for zj in z]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
