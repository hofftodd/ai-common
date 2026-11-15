"""ai-swap leases against a fake ai-swap (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest

from ai_common import swap


class FakeSwap:
    """Just enough of ai-swap's lease API: POST /api/leases takes or renews, DELETE
    releases. ``lease_api=False`` answers like an ai-swap that predates it."""

    def __init__(self, lease_api: bool = True, api_key: str = ""):
        self.lease_api = lease_api
        self.api_key = api_key
        self.leases: dict[str, dict] = {}
        self.requests: list[tuple[str, str, dict]] = []
        self.n = 0
        self.lapse_next_renewal = False

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        self.requests.append((req.method, req.url.path, body))
        if self.api_key and req.headers.get("authorization") != f"Bearer {self.api_key}":
            return httpx.Response(401, json={"error": "missing or wrong API key"})
        if not self.lease_api:
            return httpx.Response(404, text="404 page not found\n")
        if req.method == "POST" and req.url.path == "/api/leases":
            if body.get("model") == "nope":
                return httpx.Response(404, json={"error": "unknown model nope"})
            if lid := body.get("lease_id"):
                if self.lapse_next_renewal:
                    self.lapse_next_renewal = False
                    self.leases.pop(lid, None)
                if lid not in self.leases:
                    return httpx.Response(404, json={"error": "no such lease (it may have lapsed)"})
            else:
                self.n += 1
                lid = f"L{self.n}"
                self.leases[lid] = {"model": body["model"], "owner": body["owner"]}
            exp = f"2026-11-15T12:00:{self.n:02d}Z"
            return httpx.Response(
                200, json={"lease_id": lid, "lease": {"expires": exp}, "node": "studio"}
            )
        if req.method == "DELETE" and req.url.path.startswith("/api/leases/"):
            lid = req.url.path.rsplit("/", 1)[1]
            return httpx.Response(200, json={"released": self.leases.pop(lid, None) is not None})
        return httpx.Response(404, text="404 page not found\n")


@pytest.fixture(autouse=True)
def fresh_log_once():
    swap._unsupported.clear()


async def test_lease_is_taken_and_released():
    fake = FakeSwap()
    t = httpx.MockTransport(fake)
    async with swap.lease("http://swap.test/", "qwen", "ai-factory run 1", 90, transport=t) as held:
        assert held.held and held.id == "L1" and held.expires == "2026-11-15T12:00:01Z"
        assert fake.leases == {"L1": {"model": "qwen", "owner": "ai-factory run 1"}}
    assert fake.leases == {}
    assert fake.requests[0] == (
        "POST", "/api/leases", {"model": "qwen", "owner": "ai-factory run 1", "ttl_seconds": 90},
    )  # fmt: skip
    assert fake.requests[-1][:2] == ("DELETE", "/api/leases/L1")


async def test_lease_is_renewed_at_a_third_of_its_ttl():
    fake = FakeSwap()
    t = httpx.MockTransport(fake)
    async with swap.lease(
        "http://swap.test", "qwen", "me", ttl_s=90, renew_every_s=0.03, transport=t
    ) as held:
        await asyncio.sleep(0.2)
        assert held.renewals >= 3
    renewals = [b for m, _, b in fake.requests if m == "POST" and "lease_id" in b]
    assert renewals[0] == {"model": "qwen", "lease_id": "L1", "ttl_seconds": 90}
    assert fake.leases == {}


async def test_a_lapsed_lease_is_taken_afresh():
    fake = FakeSwap()
    fake.lapse_next_renewal = True
    t = httpx.MockTransport(fake)
    async with swap.lease(
        "http://swap.test", "qwen", "me", ttl_s=90, renew_every_s=0.01, transport=t
    ) as held:
        await asyncio.sleep(0.05)
        assert held.id == "L2"
    assert fake.leases == {}


async def test_release_on_error_in_the_body():
    fake = FakeSwap()
    with pytest.raises(RuntimeError):
        async with swap.lease(
            "http://swap.test", "qwen", "me", transport=httpx.MockTransport(fake)
        ):
            raise RuntimeError("step failed")
    assert fake.leases == {} and fake.requests[-1][0] == "DELETE"


async def test_older_ai_swap_makes_lease_a_no_op_logged_once(caplog):
    fake = FakeSwap(lease_api=False)
    t = httpx.MockTransport(fake)
    with caplog.at_level(logging.INFO, logger="ai_common.swap"):
        for _ in range(2):
            async with swap.lease("http://old.test", "qwen", "me", transport=t) as held:
                assert not held.held
    assert sum("no lease API" in r.message for r in caplog.records) == 1
    assert [m for m, _, _ in fake.requests] == ["POST", "POST"]  # no renewals, no release


async def test_unknown_model_is_an_error_not_an_old_server():
    t = httpx.MockTransport(FakeSwap())
    with pytest.raises(swap.SwapError, match="unknown model nope"):
        async with swap.lease("http://swap.test", "nope", "me", transport=t):
            pass


async def test_api_key_is_sent_as_bearer():
    fake = FakeSwap(api_key="k1")
    t = httpx.MockTransport(fake)
    with pytest.raises(swap.SwapError, match="HTTP 401"):
        async with swap.lease("http://swap.test", "qwen", "me", transport=t):
            pass
    async with swap.lease("http://swap.test", "qwen", "me", api_key="k1", transport=t) as held:
        assert held.held
    assert fake.leases == {}


def test_auth_headers_and_client():
    assert swap.auth_headers() == {}
    assert swap.auth_headers("k") == {"Authorization": "Bearer k"}
    c = swap.client("http://swap.test/", "k")
    assert c.headers["authorization"] == "Bearer k" and str(c.base_url) == "http://swap.test"
