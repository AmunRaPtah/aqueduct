"""Resilient HTTP for connectors.

Every outbound API call routes through here so the whole pipeline gets one
consistent failure policy instead of eleven hand-rolled retry loops:

- **Structured errors** — `TransientError` (worth retrying: timeouts, 5xx, 429,
  connection resets) vs `PermanentError` (a 4xx that won't fix itself). Callers
  can catch the distinction; `RateLimitError` and `CircuitOpenError` are transient
  subtypes carrying extra context.
- **Rate limiting** — a per-host minimum interval (token-bucket-ish), and it
  *honors the server*: a `Retry-After` header or a `429` parks that host until the
  cooldown elapses, so we back off the moment we're told to.
- **Backoff** — transient failures retry with exponential backoff + full jitter,
  capped, preferring any server-supplied `Retry-After`.
- **Circuit breaker** — per host: after enough consecutive failures the circuit
  opens and calls fail fast (`CircuitOpenError`) for a cooldown, instead of
  hammering a dead endpoint; one trial request half-opens it.

Tests monkeypatch `_open` (the urlopen seam) and `_sleep`/`_monotonic`, so the
policy is exercised offline with no real network and no real waiting.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request

from . import obs

USER_AGENT = "aqueduct/0.1 (data pipeline)"

# --- tunables (module-level so callers/tests can override) -----------------
DEFAULT_RETRIES = 3
BACKOFF_BASE = 1.0          # seconds; delay ~ BACKOFF_BASE * 2**attempt + jitter
BACKOFF_CAP = 30.0          # never sleep longer than this between tries
BREAKER_THRESHOLD = 5       # consecutive failures before a host's circuit opens
BREAKER_COOLDOWN = 30.0     # seconds the circuit stays open before a trial request
DEFAULT_MIN_INTERVAL = 0.0  # per-host floor between requests (0 = no throttle)
MAX_PARK = 60.0             # never park a host longer than this, even if the server
                            # asks for more via Retry-After. An uncapped park would
                            # block the whole single-threaded harvest in one sleep and
                            # defeat the circuit breaker; past this bound we'd rather
                            # fail fast and let the breaker open than wedge the run.


# --- seams (patched in tests) ----------------------------------------------
def _open(req, timeout):  # pragma: no cover - thin wrapper over the stdlib
    return urllib.request.urlopen(req, timeout=timeout)


_sleep = time.sleep
_monotonic = time.monotonic


# --- structured errors -----------------------------------------------------
class NetworkError(Exception):
    """Base for all network failures; carries the url and (optional) HTTP status."""

    def __init__(self, message: str, *, url: str | None = None, status: int | None = None):
        super().__init__(message)
        self.url = url
        self.status = status


class TransientError(NetworkError):
    """A failure worth retrying (timeout, connection reset, 5xx, 408)."""


class PermanentError(NetworkError):
    """A failure that won't fix itself on retry (most 4xx: 400/401/403/404...)."""


class RateLimitError(TransientError):
    """HTTP 429 (or explicit Retry-After); `retry_after` is seconds to wait."""

    def __init__(self, message: str, *, url=None, status=429, retry_after: float | None = None):
        super().__init__(message, url=url, status=status)
        self.retry_after = retry_after


class CircuitOpenError(TransientError):
    """The per-host circuit is open; the call failed fast without hitting the network."""


# --- per-host rate limiter + circuit breaker -------------------------------
class _HostState:
    __slots__ = ("next_allowed", "fails", "open_until")

    def __init__(self):
        self.next_allowed = 0.0   # monotonic time the host may be called again
        self.fails = 0            # consecutive failures (for the breaker)
        self.open_until = 0.0     # monotonic time the open circuit may retry


class RateLimiter:
    """Per-host pacing + circuit breaking, shared across all connectors."""

    def __init__(self, min_interval: float = DEFAULT_MIN_INTERVAL,
                 threshold: int = BREAKER_THRESHOLD, cooldown: float = BREAKER_COOLDOWN):
        self.min_interval = min_interval
        self.threshold = threshold
        self.cooldown = cooldown
        self._hosts: dict[str, _HostState] = {}
        self.intervals: dict[str, float] = {}  # optional per-host overrides

    def _state(self, host: str) -> _HostState:
        return self._hosts.setdefault(host, _HostState())

    def before(self, host: str) -> None:
        """Block until `host` may be called; raise CircuitOpenError if its circuit is open."""
        st = self._state(host)
        now = _monotonic()
        if st.open_until > now:
            raise CircuitOpenError(
                f"circuit open for {host} ({st.open_until - now:.1f}s left)", url=host)
        wait = st.next_allowed - now
        if wait > 0:
            _sleep(min(wait, MAX_PARK))  # defensive: never block longer than MAX_PARK

    def _interval(self, host: str) -> float:
        return self.intervals.get(host, self.min_interval)

    def on_success(self, host: str) -> None:
        st = self._state(host)
        st.fails = 0
        st.open_until = 0.0
        st.next_allowed = _monotonic() + self._interval(host)

    def on_failure(self, host: str, *, retry_after: float | None = None) -> None:
        st = self._state(host)
        st.fails += 1
        # honor an explicit cooldown (Retry-After / 429), else just the host interval,
        # but never park longer than MAX_PARK — an uncapped server value would block
        # the harvest in a single sleep and outlast the watchdog (see MAX_PARK).
        park = retry_after if retry_after is not None else self._interval(host)
        st.next_allowed = _monotonic() + min(max(park, 0.0), MAX_PARK)
        if st.fails >= self.threshold:
            st.open_until = _monotonic() + self.cooldown
            obs.log("net.circuit_open", host=host, fails=st.fails, cooldown=self.cooldown)

    def is_open(self, host: str) -> bool:
        return self._state(host).open_until > _monotonic()


# the shared limiter every connector funnels through
LIMITER = RateLimiter()


def _host(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc or url
    except Exception:  # noqa: BLE001
        return url


def _retry_after(headers) -> float | None:
    """Parse a Retry-After header (seconds form only) into a float, if present/sane."""
    if headers is None:
        return None
    val = headers.get("Retry-After")
    if not val:
        return None
    try:
        return max(0.0, float(val))
    except (TypeError, ValueError):
        return None  # HTTP-date form is rare here; treat as absent


def _backoff(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(retry_after, BACKOFF_CAP)
    base = min(BACKOFF_BASE * (2 ** attempt), BACKOFF_CAP)
    return base / 2 + random.uniform(0, base / 2)  # full jitter around the base


def request(url: str, *, data: bytes | None = None, headers: dict | None = None,
            timeout: int = 30, retries: int = DEFAULT_RETRIES,
            limiter: RateLimiter | None = None) -> bytes:
    """GET/POST `url`, returning the response body, with the full resilience policy.

    Raises `PermanentError` immediately on a non-retryable status, or the last
    `TransientError` after exhausting `retries`.
    """
    limiter = limiter or LIMITER
    host = _host(url)
    hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
    last: TransientError | None = None
    for attempt in range(retries):
        limiter.before(host)  # paces + fails fast if the circuit is open
        try:
            req = urllib.request.Request(url, data=data, headers=hdrs)
            with _open(req, timeout) as resp:
                body = resp.read()
            limiter.on_success(host)
            return body
        except urllib.error.HTTPError as e:
            ra = _retry_after(getattr(e, "headers", None))
            if e.code == 429:
                err: TransientError = RateLimitError(
                    f"429 rate limited: {url}", url=url, retry_after=ra)
            elif e.code == 408 or 500 <= e.code < 600:
                err = TransientError(f"HTTP {e.code}: {url}", url=url, status=e.code)
            else:  # 4xx that won't fix itself — fail fast, don't trip the breaker
                limiter.on_failure(host, retry_after=ra)
                raise PermanentError(f"HTTP {e.code}: {url}", url=url, status=e.code) from e
            limiter.on_failure(host, retry_after=ra)
            last = err
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            limiter.on_failure(host)
            last = TransientError(f"{type(e).__name__}: {url} ({e})", url=url)
        # transient: back off (unless that was the final attempt) and retry
        if attempt < retries - 1:
            delay = _backoff(attempt, getattr(last, "retry_after", None))
            obs.log("net.retry", host=host, attempt=attempt + 1, retries=retries,
                    delay=round(delay, 2), error=str(last))
            _sleep(delay)
    raise last or TransientError(f"request failed: {url}", url=url)


def get_bytes(url: str, *, timeout: int = 30, retries: int = DEFAULT_RETRIES,
              headers: dict | None = None) -> bytes:
    """Fetch raw bytes (XML, PDF, ...)."""
    return request(url, timeout=timeout, retries=retries, headers=headers)


def get_json(url: str, *, timeout: int = 30, retries: int = DEFAULT_RETRIES,
             headers: dict | None = None) -> dict:
    """Fetch and parse a JSON body."""
    return json.loads(request(url, timeout=timeout, retries=retries, headers=headers))
