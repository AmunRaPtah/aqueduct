"""Resilience layer: error classification, backoff/retry, rate limiting, breaker."""

from __future__ import annotations

import io
import urllib.error

import pytest

from aqueduct import net


class _Resp(io.BytesIO):
    """Context-manager byte stream that mimics an http response object."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


def _http_error(code, headers=None):
    return urllib.error.HTTPError("http://x", code, f"e{code}", headers or {}, None)


@pytest.fixture
def nosleep(monkeypatch):
    """Make sleeps instant and the clock advance by each slept amount (no real waiting)."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(net, "_monotonic", lambda: clock["t"])
    monkeypatch.setattr(net, "_sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    # isolate limiter state per test
    monkeypatch.setattr(net, "LIMITER", net.RateLimiter())
    return clock


def test_success_returns_body(monkeypatch, nosleep):
    monkeypatch.setattr(net, "_open", lambda req, timeout: _Resp(b"ok"))
    assert net.request("http://h/a") == b"ok"


def test_permanent_4xx_fails_fast_without_retry(monkeypatch, nosleep):
    calls = []
    def boom(req, timeout):
        calls.append(1)
        raise _http_error(404)
    monkeypatch.setattr(net, "_open", boom)
    with pytest.raises(net.PermanentError) as e:
        net.request("http://h/a", retries=3)
    assert e.value.status == 404
    assert len(calls) == 1  # 404 is not retried


def test_transient_5xx_retries_then_raises(monkeypatch, nosleep):
    calls = []
    def boom(req, timeout):
        calls.append(1)
        raise _http_error(503)
    monkeypatch.setattr(net, "_open", boom)
    with pytest.raises(net.TransientError):
        net.request("http://h/a", retries=3)
    assert len(calls) == 3  # all attempts used


def test_transient_then_success(monkeypatch, nosleep):
    seq = [lambda: (_ for _ in ()).throw(_http_error(500)), lambda: _Resp(b"done")]
    def opener(req, timeout):
        return seq.pop(0)()
    monkeypatch.setattr(net, "_open", opener)
    assert net.request("http://h/a", retries=3) == b"done"


def test_429_is_rate_limit_and_honors_retry_after(monkeypatch, nosleep):
    slept = []
    monkeypatch.setattr(net, "_sleep", lambda s: slept.append(s))
    monkeypatch.setattr(net, "_open",
                        lambda req, timeout: (_ for _ in ()).throw(
                            _http_error(429, {"Retry-After": "7"})))
    with pytest.raises(net.RateLimitError):
        net.request("http://h/a", retries=2)
    assert slept and slept[0] == 7.0  # waited exactly as told, not jittered backoff


def test_circuit_opens_after_threshold(monkeypatch, nosleep):
    lim = net.RateLimiter(threshold=3, cooldown=30)
    monkeypatch.setattr(net, "_open",
                        lambda req, timeout: (_ for _ in ()).throw(_http_error(500)))
    # 3 consecutive failures (one request, 3 attempts) trips the breaker
    with pytest.raises(net.TransientError):
        net.request("http://h/a", retries=3, limiter=lim)
    assert lim.is_open("h")
    # next call fails fast as CircuitOpenError without calling _open
    monkeypatch.setattr(net, "_open",
                        lambda req, timeout: pytest.fail("circuit should be open"))
    with pytest.raises(net.CircuitOpenError):
        net.request("http://h/a", retries=2, limiter=lim)


def test_circuit_resets_on_success(monkeypatch, nosleep):
    lim = net.RateLimiter(threshold=5)
    monkeypatch.setattr(net, "_open", lambda req, timeout: _Resp(b"ok"))
    net.request("http://h/a", limiter=lim)
    lim.on_failure("h"); lim.on_failure("h")
    net.request("http://h/a", limiter=lim)  # success clears the failure streak
    assert lim._state("h").fails == 0


def test_get_json_parses(monkeypatch, nosleep):
    monkeypatch.setattr(net, "_open", lambda req, timeout: _Resp(b'{"x": 1}'))
    assert net.get_json("http://h/a") == {"x": 1}


def test_connectors_route_through_net(monkeypatch, nosleep):
    """A connector's _get is the shared client: patching net._open feeds them all."""
    from aqueduct.sources import chembl, pdb
    monkeypatch.setattr(net, "_open", lambda req, timeout: _Resp(b'{"ok": true}'))
    assert chembl._get("http://chembl/x") == {"ok": True}           # raise-style
    # None-style connectors swallow network errors and return None
    monkeypatch.setattr(net, "_open",
                        lambda req, timeout: (_ for _ in ()).throw(_http_error(404)))
    assert pdb._get("http://pdb/missing") is None
