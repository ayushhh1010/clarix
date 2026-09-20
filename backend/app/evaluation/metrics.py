"""
Retrieval metrics with confidence intervals and paired significance tests.

Why the statistics are here rather than a bare mean
---------------------------------------------------
A retrieval eval on a few hundred queries produces noisy point estimates.
"Recall@10 went from 0.71 to 0.74" is not a result -- with n=200 that
difference is well inside the noise, and shipping on it means shipping on a
coin flip. Every number this module reports comes with an interval, and every
comparison between two systems is paired and corrected for multiple testing.

Three choices worth stating:

  Paired, not unpaired. Both systems are run over the *same* queries, so the
  per-query difference removes query difficulty as a variance source. An
  unpaired test on the same data is strictly less powerful and would call
  real wins insignificant.

  Bootstrap, not a t-test. Recall@k is a bounded, discrete, heavily
  non-normal statistic -- at k=10 it takes eleven possible values per query.
  A t-test's normality assumption is not met; the bootstrap makes no such
  assumption.

  Holm, not Bonferroni. When ablating several arms we run several
  comparisons at once, and the family-wise error rate has to be controlled
  or one of them will look significant by chance. Holm is uniformly more
  powerful than Bonferroni and just as valid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Resamples for bootstrap intervals. 10,000 is the conventional floor for a
# stable 95% interval; below ~2,000 the endpoints themselves become noisy.
BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_SEED = 20260919


# --------------------------------------------------------------------------
# Per-query metrics
#
# Each takes the ranked list of retrieved ids and the set of relevant ids,
# and returns a scalar for that one query. Aggregation and intervals are
# handled separately, so the same per-query vector feeds both.
# --------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items appearing in the top k."""
    if not relevant:
        return math.nan
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return math.nan
    return len(set(retrieved[:k]) & relevant) / k


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1 if any relevant item is in the top k. The 'did it work at all' metric."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1/rank of the first relevant item, or 0 if none was retrieved."""
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def dcg_at_k(gains: list[float], k: int) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains[:k], start=1))


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int,
              graded: dict[str, float] | None = None) -> float:
    """
    Normalised discounted cumulative gain.

    `graded` allows non-binary relevance; without it every relevant item has
    gain 1. Unlike recall, nDCG rewards ranking a relevant item first rather
    than eighth, which is what actually matters when only the top few chunks
    fit the context budget.
    """
    gains = [(graded or {}).get(r, 1.0) if r in relevant else 0.0 for r in retrieved]
    ideal = sorted(((graded or {}).get(r, 1.0) for r in relevant), reverse=True)
    best = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / best if best > 0 else 0.0


# --------------------------------------------------------------------------
# Aggregation with intervals
# --------------------------------------------------------------------------

@dataclass
class Estimate:
    """A metric with a bootstrap confidence interval."""

    name: str
    value: float
    ci_low: float
    ci_high: float
    n: int
    confidence: float = 0.95

    def __str__(self) -> str:
        return (f"{self.name}={self.value:.4f} "
                f"[{self.ci_low:.4f}, {self.ci_high:.4f}] n={self.n}")

    @property
    def margin(self) -> float:
        return (self.ci_high - self.ci_low) / 2

    def to_dict(self) -> dict:
        return {
            "name": self.name, "value": round(self.value, 6),
            "ci_low": round(self.ci_low, 6), "ci_high": round(self.ci_high, 6),
            "n": self.n, "confidence": self.confidence,
        }


def bootstrap_mean(
    values: list[float] | np.ndarray,
    name: str = "metric",
    confidence: float = 0.95,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Estimate:
    """
    Mean with a percentile bootstrap interval.

    NaNs are dropped (a query with no known relevant item cannot score), and
    the reported `n` is the count that actually contributed -- reporting the
    nominal query count would overstate the evidence.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return Estimate(name, math.nan, math.nan, math.nan, 0, confidence)
    if n == 1:
        return Estimate(name, float(arr[0]), float(arr[0]), float(arr[0]), 1, confidence)

    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(resamples, n), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return Estimate(
        name=name,
        value=float(arr.mean()),
        ci_low=float(np.quantile(means, alpha)),
        ci_high=float(np.quantile(means, 1 - alpha)),
        n=n,
        confidence=confidence,
    )


# --------------------------------------------------------------------------
# Comparing two systems
# --------------------------------------------------------------------------

@dataclass
class Comparison:
    """A paired comparison between two systems on the same queries."""

    metric: str
    baseline: str
    candidate: str
    baseline_mean: float
    candidate_mean: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    p_adjusted: float | None = None
    significant: bool | None = None

    def __str__(self) -> str:
        star = ""
        if self.significant is not None:
            star = " *" if self.significant else " (ns)"
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        return (f"{self.metric}: {self.baseline_mean:.4f} -> {self.candidate_mean:.4f} "
                f"(d={self.delta:+.4f} [{self.ci_low:+.4f}, {self.ci_high:+.4f}], "
                f"p={p:.4f}, n={self.n}){star}")

    def to_dict(self) -> dict:
        return {
            "metric": self.metric, "baseline": self.baseline,
            "candidate": self.candidate,
            "baseline_mean": round(self.baseline_mean, 6),
            "candidate_mean": round(self.candidate_mean, 6),
            "delta": round(self.delta, 6),
            "ci_low": round(self.ci_low, 6), "ci_high": round(self.ci_high, 6),
            "p_value": round(self.p_value, 6),
            "p_adjusted": round(self.p_adjusted, 6) if self.p_adjusted is not None else None,
            "n": self.n, "significant": self.significant,
        }


def paired_bootstrap(
    baseline: list[float],
    candidate: list[float],
    metric: str = "metric",
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
    confidence: float = 0.95,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> Comparison:
    """
    Paired bootstrap on per-query differences.

    The p-value is two-sided and computed by permutation on the *signs* of
    the paired differences, which is the exact null for "the two systems are
    interchangeable on each query". It costs nothing extra over the bootstrap
    and makes no distributional assumption.
    """
    b = np.asarray(baseline, dtype=float)
    c = np.asarray(candidate, dtype=float)
    if len(b) != len(c):
        raise ValueError(f"paired test needs equal lengths, got {len(b)} and {len(c)}")

    keep = ~(np.isnan(b) | np.isnan(c))
    b, c = b[keep], c[keep]
    n = len(b)
    if n == 0:
        return Comparison(metric, baseline_name, candidate_name,
                          math.nan, math.nan, math.nan, math.nan, math.nan, 1.0, 0)

    diff = c - b
    observed = float(diff.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(resamples, n))
    boot = diff[idx].mean(axis=1)
    alpha = (1 - confidence) / 2

    # Sign-flip permutation: under the null, flipping the sign of any
    # query's difference is equally likely.
    flips = rng.choice([-1.0, 1.0], size=(resamples, n))
    null = (diff * flips).mean(axis=1)
    p = float((np.abs(null) >= abs(observed)).mean())

    return Comparison(
        metric=metric,
        baseline=baseline_name,
        candidate=candidate_name,
        baseline_mean=float(b.mean()),
        candidate_mean=float(c.mean()),
        delta=observed,
        ci_low=float(np.quantile(boot, alpha)),
        ci_high=float(np.quantile(boot, 1 - alpha)),
        p_value=p,
        n=n,
    )


def holm_bonferroni(comparisons: list[Comparison], alpha: float = 0.05) -> list[Comparison]:
    """
    Control the family-wise error rate across several comparisons.

    Running k tests at alpha=0.05 gives roughly a 1-(1-0.05)^k chance that at
    least one is spuriously significant -- 23% at k=5. Holm's step-down
    procedure fixes that and is uniformly more powerful than Bonferroni.

    Mutates and returns the input, setting `p_adjusted` and `significant`.
    """
    ordered = sorted(comparisons, key=lambda c: c.p_value)
    k = len(ordered)
    running = 0.0
    for i, comp in enumerate(ordered):
        adjusted = min(1.0, (k - i) * comp.p_value)
        # Enforce monotonicity: adjusted p-values must not decrease.
        running = max(running, adjusted)
        comp.p_adjusted = running
        comp.significant = running < alpha
    return comparisons


def required_n(effect: float, sd: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """
    Sample size needed to detect `effect` with the given power (paired).

    Reported alongside a null result so "no significant difference" can be
    distinguished from "we had no chance of detecting one". A null result at
    n=50 when 400 were needed is not evidence of no effect.

    Uses the normal approximation, which is adequate for planning.
    """
    if effect == 0 or sd == 0:
        return 0
    z_alpha = 1.959963985 if alpha == 0.05 else abs(_z(1 - alpha / 2))
    z_beta = 0.841621234 if power == 0.80 else abs(_z(power))
    return int(math.ceil(((z_alpha + z_beta) * sd / abs(effect)) ** 2))


def _z(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# --------------------------------------------------------------------------
# Full run summary
# --------------------------------------------------------------------------

@dataclass
class RunResult:
    """Per-query metric vectors for one system configuration."""

    system: str
    query_ids: list[str] = field(default_factory=list)
    per_query: dict[str, list[float]] = field(default_factory=dict)
    latency_ms: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def record(self, query_id: str, metrics: dict[str, float], latency_ms: float) -> None:
        self.query_ids.append(query_id)
        for name, value in metrics.items():
            self.per_query.setdefault(name, []).append(value)
        self.latency_ms.append(latency_ms)

    def summarise(self, confidence: float = 0.95) -> dict[str, Estimate]:
        out = {
            name: bootstrap_mean(values, name=name, confidence=confidence)
            for name, values in self.per_query.items()
        }
        if self.latency_ms:
            lat = np.asarray(self.latency_ms)
            out["latency_p50_ms"] = Estimate(
                "latency_p50_ms", float(np.percentile(lat, 50)),
                math.nan, math.nan, len(lat), confidence)
            out["latency_p95_ms"] = Estimate(
                "latency_p95_ms", float(np.percentile(lat, 95)),
                math.nan, math.nan, len(lat), confidence)
        return out


def standard_metrics(retrieved: list[str], relevant: set[str],
                     ks: tuple[int, ...] = (1, 5, 10, 20)) -> dict[str, float]:
    """The per-query metric bundle used by every retrieval eval."""
    out: dict[str, float] = {"mrr": reciprocal_rank(retrieved, relevant)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        out[f"hit@{k}"] = hit_at_k(retrieved, relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    return out
