"""
Circuit breaker, one per provider.

Quota tracking (`budget.py`) handles refusals we can predict. This handles
the ones we cannot: a provider that is down, timing out, or 500ing. Without
it every request pays that provider's full timeout before failing over, so
one dead provider degrades every query rather than none of them.

In-process and per-worker on purpose. A shared breaker would need a round
trip to decide whether to make a round trip, and with a handful of
processes each learning independently the worst case is a few wasted
requests per process per recovery window. Quota state is shared -- that one
has to be, or two workers would each spend the whole allowance.

Standard three states:

    closed     normal; failures counted
    open       failing fast; no requests attempted until the cooldown ends
    half-open  one probe allowed; success closes, failure re-opens
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)

# Consecutive failures before opening. Low, because the alternative to a
# tripped provider is simply the next one in the chain -- being wrong costs
# a slightly worse answer, while being slow costs every concurrent request.
DEFAULT_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 60.0


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class Breaker:
    name: str
    threshold: int = DEFAULT_THRESHOLD
    cooldown: float = DEFAULT_COOLDOWN_SECONDS

    failures: int = 0
    opened_at: float | None = None
    _half_open_in_flight: bool = False
    # Retained for observability; the router logs these on failover.
    total_trips: int = 0
    last_error: str = ""

    def state(self, now: float | None = None) -> State:
        if self.opened_at is None:
            return State.CLOSED
        now = now if now is not None else time.monotonic()
        if now - self.opened_at >= self.cooldown:
            return State.HALF_OPEN
        return State.OPEN

    def allows(self, now: float | None = None) -> bool:
        """Whether to attempt a request right now."""
        state = self.state(now)
        if state is State.CLOSED:
            return True
        if state is State.OPEN:
            return False
        # Half-open: exactly one probe at a time. Letting the whole backlog
        # through on recovery is how a struggling provider gets knocked over
        # again the moment it comes back.
        if self._half_open_in_flight:
            return False
        self._half_open_in_flight = True
        return True

    def record_success(self) -> None:
        if self.opened_at is not None:
            logger.info("circuit for %s closed after recovery", self.name)
        self.failures = 0
        self.opened_at = None
        self._half_open_in_flight = False

    def record_failure(self, error: str = "", now: float | None = None) -> None:
        now = now if now is not None else time.monotonic()
        self.last_error = error[:200]
        self._half_open_in_flight = False

        # A failed probe re-opens immediately rather than needing another
        # `threshold` failures: it already had its second chance.
        if self.opened_at is not None:
            self.opened_at = now
            logger.warning("circuit for %s re-opened: %s", self.name, self.last_error)
            return

        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = now
            self.total_trips += 1
            logger.warning(
                "circuit for %s opened after %d failures: %s",
                self.name, self.failures, self.last_error,
            )

    def reset(self) -> None:
        self.failures = 0
        self.opened_at = None
        self._half_open_in_flight = False


@dataclass
class BreakerRegistry:
    """One breaker per provider name, created on demand."""

    threshold: int = DEFAULT_THRESHOLD
    cooldown: float = DEFAULT_COOLDOWN_SECONDS
    _breakers: dict[str, Breaker] = field(default_factory=dict)

    def get(self, name: str) -> Breaker:
        breaker = self._breakers.get(name)
        if breaker is None:
            breaker = Breaker(name=name, threshold=self.threshold, cooldown=self.cooldown)
            self._breakers[name] = breaker
        return breaker

    def snapshot(self) -> dict[str, dict]:
        """Current state of every breaker, for health endpoints."""
        return {
            name: {
                "state": b.state().value,
                "failures": b.failures,
                "trips": b.total_trips,
                "last_error": b.last_error,
            }
            for name, b in self._breakers.items()
        }

    def reset_all(self) -> None:
        for breaker in self._breakers.values():
            breaker.reset()
