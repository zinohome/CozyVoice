"""Simple in-memory IP rate limiter for unauthenticated endpoints."""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import HTTPException, Request


class RateLimiter:
    def __init__(self, requests_per_minute: int = 30) -> None:
        self._rpm = requests_per_minute
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def check(self, request: Request) -> None:
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        # Clean old entries outside the 60-second window
        self._requests[ip] = [t for t in self._requests[ip] if now - t < 60]
        if len(self._requests[ip]) >= self._rpm:
            raise HTTPException(
                status_code=429,
                detail={"error": {"code": "RATE_LIMITED", "message": "Too many requests"}},
            )
        self._requests[ip].append(now)
