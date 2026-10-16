# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 high availability: cross-region failover guarded by a
circuit breaker, layered on top of the SDK's built-in retry.

Resilience happens at two levels:

1. WITHIN a region the SDK already retries transient 429 / 5xx with
   Retry-After-aware exponential backoff -- see ``Client(max_retries=)``
   and ``dashscope/v2/transport/_retry.py``. You do not reimplement it.
2. ACROSS regions this recipe adds a per-region circuit breaker and
   fails over to the next healthy region when one keeps erroring, so a
   whole-region outage does not take your application down.

The demo runs fully offline and deterministically: a mock transport
makes the primary region return 503 and the secondary return 200. Drop
``http_client_factory`` to hit the real gateways -- note that API keys
are region-specific (see ``legacy_1x/set_region_example.py``).
"""

import time

import httpx

import dashscope
from dashscope.v2 import errors

# Fail over across these regions, in order. Both have a dedicated
# shared gateway, so no workspace is required for the demo.
REGIONS = ["cn-beijing", "ap-southeast-1"]

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


class CircuitBreaker:
    """Minimal per-region circuit breaker.

    CLOSED    -> requests flow; consecutive failures are counted.
    OPEN      -> requests are refused fast until ``reset_timeout``
                 elapses, then the breaker moves to HALF_OPEN.
    HALF_OPEN -> one probe is allowed; success closes the breaker,
                 failure re-opens it.
    """

    def __init__(self, failure_threshold=2, reset_timeout=30.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = CLOSED
        self.failure_count = 0
        self._opened_at = 0.0

    def allow(self):
        """Whether a request may be attempted right now."""
        if self.state == OPEN:
            if time.monotonic() - self._opened_at >= self.reset_timeout:
                self.state = HALF_OPEN
                return True
            return False
        return True

    def record_success(self):
        self.failure_count = 0
        self.state = CLOSED

    def record_failure(self):
        self.failure_count += 1
        if (
            self.state == HALF_OPEN
            or self.failure_count >= self.failure_threshold
        ):
            self.state = OPEN
            self._opened_at = time.monotonic()


def _is_retryable(exc):
    """Connection errors and 5xx / 429 are worth failing over on."""
    if isinstance(exc, errors.APIConnectionError):
        return True
    status = getattr(exc, "status_code", None)
    return status is not None and (status >= 500 or status == 429)


class MultiRegionClient:
    """Try each region in order, guarded by a circuit breaker."""

    def __init__(
        self,
        regions,
        http_client_factory=None,
        max_retries=0,
        failure_threshold=2,
        reset_timeout=30.0,
    ):
        self._regions = list(regions)
        self._factory = http_client_factory
        self._max_retries = max_retries
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout
        self._breakers = {}
        self._clients = {}
        for region in self._regions:
            self._breakers[region] = CircuitBreaker(
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
            )

    def _client(self, region):
        if region not in self._clients:
            kwargs = {
                "region": region,
                "max_retries": self._max_retries,
            }
            if self._factory is not None:
                # Offline demo: injected mock transport + dummy creds.
                # The dummy workspace also silences the legacy-gateway
                # warning; real regional use needs a region-specific
                # workspace AND API key (see legacy_1x/set_region_example).
                kwargs["api_key"] = "sk-mock"
                kwargs["workspace"] = "ws-demo"
                kwargs["http_client"] = self._factory(region)
            self._clients[region] = dashscope.Client(**kwargs)
        return self._clients[region]

    def chat(self, prompt, model="qwen3.7-plus"):
        """Return ``(region, response)`` from the first healthy region."""
        last_exc = None
        for region in self._regions:
            breaker = self._breakers[region]
            if not breaker.allow():
                print(f"  [{region}] circuit OPEN -> skip")
                continue
            try:
                resp = self._client(region).chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                breaker.record_success()
                print(f"  [{region}] OK (circuit {breaker.state})")
                return region, resp
            except errors.DashScopeError as exc:
                last_exc = exc
                retryable = _is_retryable(exc)
                if retryable:
                    breaker.record_failure()
                print(
                    f"  [{region}] {type(exc).__name__} "
                    f"status={exc.status_code} retryable={retryable} "
                    f"-> circuit {breaker.state}",
                )
                if not retryable:
                    raise  # a 4xx is a real bug; do not mask it
        raise errors.APIConnectionError(
            f"all regions failed: {last_exc}",
        ) from last_exc


def _mock_http_client(region):
    """Offline transport: the primary region 503s, every other 200s.

    The handler closes over ``region`` instead of sniffing the request
    host, so it stays correct when a workspace changes the gateway.
    """
    is_primary = region == REGIONS[0]

    def handler(_request):
        if is_primary:
            return httpx.Response(
                503,
                json={
                    "code": "ServiceUnavailable",
                    "message": "mocked regional outage",
                },
            )
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": f"来自 {region} 的应答",
                            },
                        },
                    ],
                },
                "usage": {"total_tokens": 10},
                "request_id": f"mock-{region}",
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def main():
    """Drive three calls to show the breaker opening and failing over."""
    mc = MultiRegionClient(
        REGIONS,
        http_client_factory=_mock_http_client,
        max_retries=0,  # keep the demo instant; SDK retry is separate
        failure_threshold=2,
    )

    # Three calls show the breaker progressing CLOSED -> OPEN: the
    # first two probe the dead primary and fail over; the third sees
    # the primary OPEN and goes straight to the healthy region.
    for i in range(1, 4):
        print(f"----- call {i} -----")
        served_by, resp = mc.chat("你好")
        print(f"  served by: {served_by}")
        print(f"  answer   : {resp.output_text}")
        print(f"  request_id: {resp.request_id}")


if __name__ == "__main__":
    main()
