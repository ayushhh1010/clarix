"""
Tests for the retrieval metrics and their statistics.

Two layers. The first pins metric definitions against hand-computed values.
The second checks the *statistics themselves* are calibrated: a confidence
interval that does not cover at its nominal rate, or a test whose false
positive rate exceeds alpha, produces confident wrong conclusions -- which is
worse than having no statistics at all. These are slow-ish (thousands of
simulated experiments) but they are the tests that make the reported
intervals trustworthy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.evaluation.metrics import (
    Comparison,
    bootstrap_mean,
    hit_at_k,
    holm_bonferroni,
    ndcg_at_k,
    paired_bootstrap,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    required_n,
    standard_metrics,
)

# --- metric definitions ----------------------------------------------------

def test_recall_at_k_counts_relevant_in_prefix():
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, {"a", "d"}, 2) == 0.5
    assert recall_at_k(retrieved, {"a", "d"}, 4) == 1.0
    assert recall_at_k(retrieved, {"z"}, 4) == 0.0


def test_recall_is_nan_when_nothing_is_relevant():
    """A query with no known gold item cannot score; it must not count as 0."""
    assert math.isnan(recall_at_k(["a"], set(), 5))


def test_precision_and_hit():
    assert precision_at_k(["a", "b", "c", "d"], {"a", "d"}, 2) == 0.5
    assert hit_at_k(["a", "b"], {"b"}, 2) == 1.0
    assert hit_at_k(["a", "b"], {"z"}, 2) == 0.0


@pytest.mark.parametrize("retrieved,expected", [
    (["gold", "x", "y"], 1.0),
    (["x", "gold", "y"], 0.5),
    (["x", "y", "gold"], 1 / 3),
    (["x", "y", "z"], 0.0),
])
def test_reciprocal_rank(retrieved, expected):
    assert reciprocal_rank(retrieved, {"gold"}) == pytest.approx(expected)


def test_ndcg_rewards_ranking_the_gold_item_earlier():
    first = ndcg_at_k(["gold", "x", "y"], {"gold"}, 3)
    third = ndcg_at_k(["x", "y", "gold"], {"gold"}, 3)
    assert first == pytest.approx(1.0)
    assert third == pytest.approx(1 / math.log2(4))
    assert first > third


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)


def test_ndcg_handles_graded_relevance():
    graded = {"a": 3.0, "b": 1.0}
    good = ndcg_at_k(["a", "b"], {"a", "b"}, 2, graded)
    worse = ndcg_at_k(["b", "a"], {"a", "b"}, 2, graded)
    assert good == pytest.approx(1.0)
    assert worse < good


def test_standard_metrics_bundle_shape():
    m = standard_metrics(["a", "b"], {"b"}, ks=(1, 5))
    assert set(m) == {"mrr", "recall@1", "hit@1", "ndcg@1", "recall@5", "hit@5", "ndcg@5"}
    assert m["recall@1"] == 0.0
    assert m["recall@5"] == 1.0


# --- bootstrap intervals ---------------------------------------------------

def test_bootstrap_interval_brackets_the_sample_mean():
    values = [0.0] * 30 + [1.0] * 70
    est = bootstrap_mean(values, name="recall@10")
    assert est.value == pytest.approx(0.70)
    assert est.ci_low < 0.70 < est.ci_high
    assert est.n == 100


def test_bootstrap_drops_nans_and_reports_the_contributing_n():
    est = bootstrap_mean([1.0, 0.0, math.nan, 1.0], name="m")
    assert est.n == 3
    assert est.value == pytest.approx(2 / 3)


def test_bootstrap_handles_degenerate_inputs():
    empty = bootstrap_mean([], name="m")
    assert empty.n == 0 and math.isnan(empty.value)
    single = bootstrap_mean([0.5], name="m")
    assert single.n == 1 and single.value == 0.5


def test_bootstrap_interval_narrows_as_n_grows():
    rng = np.random.default_rng(0)
    wide = bootstrap_mean(rng.binomial(1, 0.6, 50).tolist(), name="m")
    narrow = bootstrap_mean(rng.binomial(1, 0.6, 2000).tolist(), name="m")
    assert narrow.margin < wide.margin


@pytest.mark.slow
def test_bootstrap_ci_is_calibrated():
    """
    A nominal 95% interval must cover the true mean about 95% of the time.

    This is the test that makes every reported interval meaningful. Run over
    400 simulated experiments; allowing 89-99% coverage keeps it from being
    flaky while still catching a genuinely miscalibrated interval.
    """
    rng = np.random.default_rng(7)
    true_p = 0.62
    covered = 0
    trials = 400
    for i in range(trials):
        sample = rng.binomial(1, true_p, 120).astype(float)
        est = bootstrap_mean(sample, name="m", resamples=1200, seed=1000 + i)
        covered += est.ci_low <= true_p <= est.ci_high
    rate = covered / trials
    assert 0.89 <= rate <= 0.99, f"coverage {rate:.1%}, expected ~95%"


# --- paired comparison -----------------------------------------------------

def test_paired_bootstrap_detects_a_real_improvement():
    # Candidate wins on 25 queries, ties on 75. A genuine effect.
    baseline = [0.0] * 25 + [1.0] * 75
    candidate = [1.0] * 25 + [1.0] * 75
    comp = paired_bootstrap(baseline, candidate, metric="recall@10")
    assert comp.delta == pytest.approx(0.25)
    assert comp.p_value < 0.01
    assert comp.ci_low > 0


def test_paired_bootstrap_reports_no_effect_when_there_is_none():
    values = [0.0, 1.0] * 60
    comp = paired_bootstrap(values, list(values), metric="recall@10")
    assert comp.delta == pytest.approx(0.0)
    assert comp.p_value > 0.05
    assert comp.ci_low <= 0 <= comp.ci_high


def test_paired_bootstrap_requires_equal_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        paired_bootstrap([1.0, 0.0], [1.0])


def test_paired_bootstrap_drops_pairs_with_a_nan_on_either_side():
    comp = paired_bootstrap([1.0, math.nan, 0.0], [1.0, 1.0, 1.0])
    assert comp.n == 2


@pytest.mark.slow
def test_paired_test_false_positive_rate_respects_alpha():
    """
    Under the null, p < 0.05 should occur about 5% of the time.

    A test that fires more often than that manufactures significant results
    from noise -- the exact failure that makes an eval harness worse than
    useless. 600 null experiments; the binomial 99.9% upper bound around
    5% at n=600 is ~8%.
    """
    rng = np.random.default_rng(3)
    false_positives = 0
    trials = 600
    for i in range(trials):
        a = rng.binomial(1, 0.5, 80).astype(float)
        b = rng.binomial(1, 0.5, 80).astype(float)
        comp = paired_bootstrap(a, b, resamples=800, seed=5000 + i)
        false_positives += comp.p_value < 0.05
    rate = false_positives / trials
    assert rate <= 0.09, f"false positive rate {rate:.1%}, expected ~5%"


# --- multiple comparisons --------------------------------------------------

def _comp(p: float, name: str) -> Comparison:
    return Comparison(metric=name, baseline="b", candidate="c",
                      baseline_mean=0.5, candidate_mean=0.6, delta=0.1,
                      ci_low=0.0, ci_high=0.2, p_value=p, n=100)


def test_holm_adjusts_upward_and_stays_monotone():
    comps = holm_bonferroni([_comp(0.001, "a"), _comp(0.02, "b"), _comp(0.04, "c")])
    by_name = {c.metric: c for c in comps}
    assert by_name["a"].p_adjusted == pytest.approx(0.003)
    assert by_name["b"].p_adjusted == pytest.approx(0.04)
    assert by_name["c"].p_adjusted == pytest.approx(0.04)
    ordered = sorted(comps, key=lambda c: c.p_value)
    adj = [c.p_adjusted for c in ordered]
    assert adj == sorted(adj), "adjusted p-values must not decrease"


def test_holm_is_less_conservative_than_bonferroni():
    """The largest p-value is multiplied by 1 under Holm, by k under Bonferroni."""
    comps = holm_bonferroni([_comp(0.01, "a"), _comp(0.04, "b")])
    largest = max(comps, key=lambda c: c.p_value)
    assert largest.p_adjusted == pytest.approx(0.04)  # Bonferroni would give 0.08


def test_holm_marks_borderline_results_not_significant():
    comps = holm_bonferroni([_comp(0.03, "a"), _comp(0.04, "b"), _comp(0.045, "c")])
    assert not any(c.significant for c in comps), (
        "three borderline p-values should not all survive correction"
    )


def test_holm_caps_adjusted_p_at_one():
    comps = holm_bonferroni([_comp(0.6, "a"), _comp(0.7, "b"), _comp(0.8, "c")])
    assert all(c.p_adjusted <= 1.0 for c in comps)


# --- power -----------------------------------------------------------------

def test_required_n_grows_as_the_effect_shrinks():
    big = required_n(effect=0.10, sd=0.4)
    small = required_n(effect=0.02, sd=0.4)
    assert small > big > 0


def test_required_n_matches_the_textbook_value():
    # Paired, alpha=.05, power=.80, effect/sd = 0.25 -> ~126 (Cohen).
    assert 120 <= required_n(effect=0.25, sd=1.0) <= 130


def test_required_n_is_zero_for_a_degenerate_request():
    assert required_n(effect=0.0, sd=0.4) == 0
    assert required_n(effect=0.1, sd=0.0) == 0
