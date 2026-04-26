"""Prometheus metrics for CozyVoice."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

# TTS metrics
tts_requests_total = Counter(
    "cozyvoice_tts_requests_total",
    "TTS requests",
    ["provider", "status"],
)
tts_duration_seconds = Histogram(
    "cozyvoice_tts_duration_seconds",
    "TTS latency",
    ["provider"],
)
tts_fallback_total = Counter(
    "cozyvoice_tts_fallback_total",
    "TTS fallback triggers",
    ["failed_provider", "next_provider"],
)

# STT metrics
stt_requests_total = Counter(
    "cozyvoice_stt_requests_total",
    "STT requests",
    ["status"],
)
stt_duration_seconds = Histogram(
    "cozyvoice_stt_duration_seconds",
    "STT latency",
)

# Brain integration metrics
brain_requests_total = Counter(
    "cozyvoice_brain_requests_total",
    "Brain API calls",
    ["endpoint", "status"],
)
brain_duration_seconds = Histogram(
    "cozyvoice_brain_duration_seconds",
    "Brain call latency",
    ["endpoint"],
)
