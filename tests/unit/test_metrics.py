"""Tests for cozyvoice.metrics — verify metric objects exist and are usable."""

from __future__ import annotations

from prometheus_client import Counter, Histogram


def test_tts_metrics_exist():
    from cozyvoice.metrics import tts_requests_total, tts_duration_seconds, tts_fallback_total

    assert isinstance(tts_requests_total, Counter)
    assert isinstance(tts_duration_seconds, Histogram)
    assert isinstance(tts_fallback_total, Counter)


def test_stt_metrics_exist():
    from cozyvoice.metrics import stt_requests_total, stt_duration_seconds

    assert isinstance(stt_requests_total, Counter)
    assert isinstance(stt_duration_seconds, Histogram)


def test_brain_metrics_exist():
    from cozyvoice.metrics import brain_requests_total, brain_duration_seconds

    assert isinstance(brain_requests_total, Counter)
    assert isinstance(brain_duration_seconds, Histogram)


def test_tts_requests_total_can_be_incremented():
    from cozyvoice.metrics import tts_requests_total

    before = tts_requests_total.labels(provider="test_provider", status="ok")._value.get()
    tts_requests_total.labels(provider="test_provider", status="ok").inc()
    after = tts_requests_total.labels(provider="test_provider", status="ok")._value.get()
    assert after == before + 1.0


def test_tts_duration_seconds_can_be_observed():
    from cozyvoice.metrics import tts_duration_seconds

    # Should not raise
    tts_duration_seconds.labels(provider="test_provider").observe(0.123)


def test_tts_fallback_total_can_be_incremented():
    from cozyvoice.metrics import tts_fallback_total

    before = tts_fallback_total.labels(
        failed_provider="provA", next_provider="provB"
    )._value.get()
    tts_fallback_total.labels(failed_provider="provA", next_provider="provB").inc()
    after = tts_fallback_total.labels(
        failed_provider="provA", next_provider="provB"
    )._value.get()
    assert after == before + 1.0


def test_stt_requests_total_can_be_incremented():
    from cozyvoice.metrics import stt_requests_total

    before = stt_requests_total.labels(status="ok")._value.get()
    stt_requests_total.labels(status="ok").inc()
    after = stt_requests_total.labels(status="ok")._value.get()
    assert after == before + 1.0


def test_brain_requests_total_can_be_incremented():
    from cozyvoice.metrics import brain_requests_total

    before = brain_requests_total.labels(endpoint="/chat", status="ok")._value.get()
    brain_requests_total.labels(endpoint="/chat", status="ok").inc()
    after = brain_requests_total.labels(endpoint="/chat", status="ok")._value.get()
    assert after == before + 1.0
