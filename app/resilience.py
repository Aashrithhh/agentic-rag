"""Retry, circuit breaker, and fallback utilities for LLM and database calls.

Uses tenacity for retries with exponential backoff, and implements
per-upstream circuit breakers to prevent cascade failures.

Circuit breaker states:
  - CLOSED  (0): Normal operation, requests pass through
  - HALF_OPEN (1): Testing if upstream recovered (limited requests)
  - OPEN (2): Upstream considered down, requests fail fast

Fallback model strategy:
  - Primary model fails → try fallback model with quality downgrade tag
  - All models fail → return fallback response or raise
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Literal, TypeVar

from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ── Circuit Breaker ──────────────────────────────────────────────────


class CircuitState(IntEnum):
    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    failure_threshold: int = 5          # failures before opening
    recovery_timeout: float = 60.0     # seconds before trying half-open
    half_open_max_calls: int = 2       # test calls in half-open state
    success_threshold: int = 2         # successes to close from half-open


class CircuitBreaker:
    """Per-upstream circuit breaker with thread-safe state management."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has elapsed
                if time.monotonic() - self._last_failure_time >= self.config.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._success_count = 0
                    logger.info("Circuit breaker '%s': OPEN → HALF_OPEN", self.name)
            return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info("Circuit breaker '%s': HALF_OPEN → CLOSED", self.name)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker '%s': HALF_OPEN → OPEN", self.name)
            elif (self._state == CircuitState.CLOSED and
                  self._failure_count >= self.config.failure_threshold):
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker '%s': CLOSED → OPEN (failures=%d)",
                    self.name, self._failure_count,
                )

    def reset(self) -> None:
        """Force reset to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0

    def status(self) -> dict[str, Any]:
        """Return current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.name,
            "state_value": int(self.state),
            "failure_count": self._failure_count,
            "last_failure": self._last_failure_time,
        }


# ── Global circuit breakers (one per upstream service) ───────────────

_circuit_breakers: dict[str, CircuitBreaker] = {}
_cb_lock = threading.Lock()


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """Get or create a circuit breaker for an upstream service."""
    if name not in _circuit_breakers:
        with _cb_lock:
            if name not in _circuit_breakers:
                _circuit_breakers[name] = CircuitBreaker(name)
    return _circuit_breakers[name]


def all_circuit_breaker_statuses() -> list[dict[str, Any]]:
    """Return status of all circuit breakers."""
    return [cb.status() for cb in _circuit_breakers.values()]


# Pre-register circuit breakers for known upstreams
UPSTREAM_OPENAI = "openai"
UPSTREAM_COHERE = "cohere"
UPSTREAM_VALIDATION_API = "validation_api"
UPSTREAM_DATABASE = "database"

for _name in [UPSTREAM_OPENAI, UPSTREAM_COHERE, UPSTREAM_VALIDATION_API, UPSTREAM_DATABASE]:
    get_circuit_breaker(_name)


# ── Fallback model strategy ─────────────────────────────────────────


@dataclass
class FallbackModelConfig:
    """Configuration for fallback model chain."""
    primary_model: str = ""
    fallback_models: list[str] = field(default_factory=list)
    quality_tags: dict[str, str] = field(default_factory=dict)


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and request is rejected."""
    pass


def _get_fallback_chain() -> list[dict[str, str]]:
    """Return the ordered list of models to try with quality tags."""
    from app.config import settings
    return [
        {"model": settings.openai_model, "quality": "primary", "provider": "openai"},
        {"model": settings.fallback_model, "quality": "degraded", "provider": "openai"},
    ]


def invoke_with_retry(
    chain: Any,
    inputs: dict,
    *,
    node_name: str = "unknown",
    max_attempts: int = 3,
    fallback: Any | None = None,
    upstream: str = UPSTREAM_OPENAI,
) -> Any:
    """Call ``chain.invoke(inputs)`` with retry, circuit breaker, and fallback model.

    1. Check circuit breaker — fail fast if open
    2. Retry with exponential backoff
    3. On persistent failure, try fallback model
    4. If all fail, return fallback value or raise
    """
    from app.metrics import metrics

    cb = get_circuit_breaker(upstream)

    # Circuit breaker check
    if not cb.allow_request():
        metrics.inc(f"circuit_breaker_rejected.{upstream}")
        logger.warning(
            "%s node: Circuit breaker OPEN for '%s' — trying fallback model",
            node_name, upstream,
        )
        # Try fallback model directly
        result = _try_fallback_model(inputs, node_name=node_name)
        if result is not None:
            return result
        if fallback is not None:
            return fallback
        raise CircuitOpenError(
            f"Circuit breaker open for {upstream} and no fallback available"
        )

    _retry_invoke = retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(chain.invoke)

    try:
        result = _retry_invoke(inputs)
        cb.record_success()
        metrics.inc(f"llm_call_success.{node_name}")
        return result
    except Exception as exc:
        cb.record_failure()
        logger.error(
            "%s node: LLM call failed after %d attempts: %s",
            node_name, max_attempts, exc,
        )

        # Try fallback model
        fallback_result = _try_fallback_model(inputs, node_name=node_name)
        if fallback_result is not None:
            metrics.inc(f"llm_call_fallback_model.{node_name}")
            return fallback_result

        if fallback is not None:
            logger.warning("%s node: returning static fallback response", node_name)
            metrics.inc(f"llm_call_fallback.{node_name}")
            return fallback
        metrics.inc(f"llm_call_failure.{node_name}")
        raise


def _try_fallback_model(
    inputs: dict,
    *,
    node_name: str = "unknown",
) -> Any | None:
    """Attempt to use a fallback model when the primary fails.

    Returns the result with a quality downgrade tag, or None if
    the fallback also fails.
    """
    from app.config import settings
    from app.metrics import metrics

    if not settings.fallback_model:
        return None

    try:
        from langchain_openai import ChatOpenAI

        fallback_llm = ChatOpenAI(
            model=settings.fallback_model,
            api_key=settings.openai_api_key,
            temperature=0,
            max_tokens=4096,
        )

        # Simple invoke without structured output
        # The caller may need to handle the raw response
        prompt_text = str(inputs)
        response = fallback_llm.invoke(prompt_text)

        # Tag the response with quality downgrade
        if hasattr(response, "content"):
            response.content = (
                f"[QUALITY: DEGRADED — generated by fallback model "
                f"{settings.fallback_model}]\n\n{response.content}"
            )

        metrics.inc(f"fallback_model_success.{node_name}")
        logger.warning(
            "%s node: Used fallback model '%s' (quality: degraded)",
            node_name, settings.fallback_model,
        )
        return response

    except Exception as exc:
        logger.error(
            "%s node: Fallback model also failed: %s", node_name, exc,
        )
        metrics.inc(f"fallback_model_failure.{node_name}")
        return None


def db_invoke_with_retry(
    func: Callable[..., T],
    *args: Any,
    node_name: str = "unknown",
    max_attempts: int = 3,
    **kwargs: Any,
) -> T:
    """Wrap a database call with retry and circuit breaker."""
    from app.metrics import metrics

    cb = get_circuit_breaker(UPSTREAM_DATABASE)

    if not cb.allow_request():
        metrics.inc(f"circuit_breaker_rejected.database")
        raise CircuitOpenError("Database circuit breaker is open")

    _retry_func = retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)

    try:
        result = _retry_func(*args, **kwargs)
        cb.record_success()
        metrics.inc(f"db_call_success.{node_name}")
        return result
    except Exception as exc:
        cb.record_failure()
        logger.error(
            "%s node: DB call failed after %d attempts: %s",
            node_name, max_attempts, exc,
        )
        metrics.inc(f"db_call_failure.{node_name}")
        raise
