"""A small async ai-swap client: API key headers, and model leases.

A lease keeps one model loaded for a client across many requests (an agent step that pauses
between turns while its tools run): while it lasts, loads that need the model's memory wait
instead of evicting it. It lapses unless renewed within its TTL.

    async with lease("http://studio.local:8080", "qwen3-coder", owner="ai-factory run 12"):
        ...  # the model stays loaded; the lease is renewed every ttl/3 and released on exit

An ai-swap without the lease API (it answers 404 with no JSON error) makes ``lease`` a no-op,
logged once per server.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger("ai_common.swap")

DEFAULT_TTL_S = 600


class SwapError(Exception):
    pass


def auth_headers(api_key: str = "") -> dict[str, str]:
    """Headers for an ai-swap request: ``Authorization: Bearer <key>`` when a key is set."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def client(
    base_url: str,
    api_key: str = "",
    timeout: float | httpx.Timeout = 30,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """An httpx client for one ai-swap, with its API key on every request."""
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        headers=auth_headers(api_key),
        timeout=timeout,
        transport=transport,
    )


@dataclass
class Lease:
    model: str
    owner: str
    id: str = ""  # "" = no lease held (the server has no lease API)
    expires: str = ""  # RFC 3339, as the server reports it
    renewals: int = 0

    @property
    def held(self) -> bool:
        return bool(self.id)


_unsupported: set[str] = set()  # servers already logged as having no lease API


def _error(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        return ""
    return str(body.get("error", "")) if isinstance(body, dict) else ""


def _expires(body: dict[str, Any]) -> str:
    inner = body.get("lease")
    if isinstance(inner, dict) and inner.get("expires"):
        return str(inner["expires"])
    return str(body.get("expires", ""))


async def _post(c: httpx.AsyncClient, body: dict[str, Any]) -> httpx.Response:
    return await c.post("/api/leases", json=body)


@contextlib.asynccontextmanager
async def lease(
    base_url: str,
    model: str,
    owner: str,
    ttl_s: int = DEFAULT_TTL_S,
    *,
    reason: str = "",
    renew_every_s: float | None = None,
    api_key: str = "",
    timeout: float = 600,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncIterator[Lease]:
    """Hold a lease on ``model`` for the body of the ``async with``.

    Taking the lease may load the model, so ``timeout`` is generous. The lease is renewed in
    the background every ``ttl_s / 3`` seconds (``renew_every_s`` overrides), taken afresh if
    it lapsed, and released on exit. Raises SwapError if the server refuses the lease
    (unknown model, no capacity)."""
    held = Lease(model, owner)
    base = base_url.rstrip("/")
    async with client(base, api_key, timeout, transport) as c:
        take = {"model": model, "owner": owner, "ttl_seconds": ttl_s}
        if reason:
            take["reason"] = reason
        r = await _post(c, take)
        if r.status_code == 404 and not _error(r):
            if base not in _unsupported:
                _unsupported.add(base)
                log.info("%s has no lease API (older ai-swap); not leasing", base)
            yield held
            return
        if r.status_code != 200:
            raise SwapError(
                f"lease on {model} refused: HTTP {r.status_code} {_error(r) or r.text[:200]}"
            )
        body = r.json()
        held.id, held.expires = str(body["lease_id"]), _expires(body)

        every = renew_every_s if renew_every_s is not None else ttl_s / 3

        async def renew() -> None:
            while True:
                await asyncio.sleep(every)
                try:
                    r = await _post(c, {"model": model, "lease_id": held.id, "ttl_seconds": ttl_s})
                    if r.status_code == 404:
                        log.warning("lease on %s lapsed; taking a new one", model)
                        r = await _post(c, take)
                    if r.status_code != 200:
                        log.warning("renewing the lease on %s: HTTP %s %s", model,
                                    r.status_code, _error(r) or r.text[:200])  # fmt: skip
                        continue
                    body = r.json()
                    held.id, held.expires = str(body["lease_id"]), _expires(body)
                    held.renewals += 1
                except (httpx.HTTPError, ValueError, KeyError) as e:
                    log.warning("renewing the lease on %s: %s", model, e)

        task = asyncio.create_task(renew())
        try:
            yield held
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            try:
                await c.delete(f"/api/leases/{held.id}", params={"model": model})
            except httpx.HTTPError as e:
                log.warning("releasing the lease on %s: %s (it lapses in %ss)", model, e, ttl_s)
