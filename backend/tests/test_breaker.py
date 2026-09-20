"""
Circuit breaker tests.

Without a breaker, one dead provider makes every query pay its full timeout
before failing over -- so a single outage degrades everything rather than
nothing. These pin the state machine that prevents that.
"""

from __future__ import annotations

from app.llm.breaker import Breaker, BreakerRegistry, State


def test_starts_closed_and_allows():
    b = Breaker("groq")
    assert b.state() is State.CLOSED
    assert b.allows()


def test_opens_after_the_threshold():
    b = Breaker("groq", threshold=3)
    for _ in range(2):
        b.record_failure("500")
    assert b.state() is State.CLOSED, "must not open early"
    b.record_failure("500")
    assert b.state(now=0.0) is State.OPEN
    assert not b.allows(now=0.0)


def test_success_resets_the_failure_count():
    b = Breaker("groq", threshold=3)
    b.record_failure("a")
    b.record_failure("b")
    b.record_success()
    b.record_failure("c")
    assert b.state() is State.CLOSED, "counter should have reset"


def test_half_opens_after_the_cooldown():
    b = Breaker("groq", threshold=1, cooldown=60.0)
    b.record_failure("down", now=100.0)
    assert b.state(now=130.0) is State.OPEN
    assert b.state(now=161.0) is State.HALF_OPEN


def test_half_open_admits_exactly_one_probe():
    """
    Releasing the whole backlog on recovery is how a struggling provider
    gets knocked over again the moment it returns.
    """
    b = Breaker("groq", threshold=1, cooldown=10.0)
    b.record_failure("down", now=0.0)
    assert b.allows(now=20.0) is True
    assert b.allows(now=20.0) is False
    assert b.allows(now=20.0) is False


def test_a_successful_probe_closes_the_circuit():
    b = Breaker("groq", threshold=1, cooldown=10.0)
    b.record_failure("down", now=0.0)
    assert b.allows(now=20.0)
    b.record_success()
    assert b.state() is State.CLOSED
    assert b.allows()


def test_a_failed_probe_reopens_immediately():
    """It already had its second chance; do not make it earn another."""
    b = Breaker("groq", threshold=5, cooldown=10.0)
    b.record_failure("x", now=0.0)
    for _ in range(4):
        b.record_failure("x", now=0.0)
    assert b.state(now=0.0) is State.OPEN

    assert b.allows(now=20.0)
    b.record_failure("still down", now=20.0)
    assert b.state(now=21.0) is State.OPEN
    assert not b.allows(now=21.0)
    assert b.state(now=35.0) is State.HALF_OPEN, "cooldown restarts from the probe"


def test_trips_and_last_error_are_recorded_for_observability():
    b = Breaker("groq", threshold=1)
    b.record_failure("upstream 503 from groq", now=0.0)
    assert b.total_trips == 1
    assert "503" in b.last_error


def test_registry_creates_one_breaker_per_name():
    reg = BreakerRegistry(threshold=2)
    assert reg.get("groq") is reg.get("groq")
    assert reg.get("groq") is not reg.get("cerebras")
    assert reg.get("groq").threshold == 2


def test_registry_snapshot_reports_state():
    reg = BreakerRegistry(threshold=1)
    reg.get("groq").record_failure("boom", now=0.0)
    reg.get("gemini")
    snap = reg.snapshot()
    assert snap["groq"]["state"] in {"open", "half_open"}
    assert snap["groq"]["trips"] == 1
    assert snap["gemini"]["state"] == "closed"


def test_registry_reset_all():
    reg = BreakerRegistry(threshold=1)
    reg.get("groq").record_failure("boom", now=0.0)
    reg.reset_all()
    assert reg.get("groq").state() is State.CLOSED
