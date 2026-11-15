"""The web, for local models: search through ai-search (or any SearXNG), and fetch pages only
from public addresses.

The fetch guard resolves every host before each request, including each redirect hop, and
refuses any that resolves to a non-public address, so a page can't steer a fetch into the LAN,
the host or a cloud metadata service.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

USER_AGENT = "ai-common/1 (+local models; public pages only)"
MAX_BYTES = 3_000_000
MAX_REDIRECTS = 3
PAGE_CHARS = 15_000
TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")


class WebError(Exception):
    pass


# -- search ---------------------------------------------------------------------------------


@dataclass
class Hit:
    title: str
    url: str
    snippet: str


async def search(
    urls: list[str] | str,
    query: str,
    count: int = 8,
    token: str = "",
    timeout: float = 30,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[Hit]:
    """Top ``count`` results for one query from the first search URL that answers.

    ``urls`` are SearXNG-compatible base URLs, tried in order (e.g. a LAN name, then a
    Tailscale name). A 5xx, 429 or connection error moves on to the next; any other answer is
    the service speaking, so another route won't help. ``token`` is sent as
    ``Authorization: Bearer`` (ai-search accepts it)."""
    bases = [urls] if isinstance(urls, str) else urls
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    last = "no search URL"
    for base in bases:
        try:
            async with httpx.AsyncClient(timeout=timeout, transport=transport) as c:
                r = await c.get(
                    f"{base.rstrip('/')}/search",
                    params={"q": query, "format": "json", "count": str(count)},
                    headers=headers,
                )
            if r.status_code != 200:
                last = f"{base}: HTTP {r.status_code} {r.text[:200]}"
                if r.status_code < 500 and r.status_code != 429:
                    break  # the service answered: another route won't help
                continue
            rows = (r.json() or {}).get("results") or []
            return [
                Hit(str(x.get("title", "")).strip(), str(x.get("url", "")),
                    str(x.get("content", "")).strip())
                for x in rows[:count]
                if str(x.get("url", "")).startswith(("http://", "https://"))
            ]  # fmt: skip
        except (httpx.HTTPError, ValueError) as e:
            last = f"{base}: {e}"
    raise WebError(f"search failed ({last})")


# -- fetch ----------------------------------------------------------------------------------


def is_public(host: str) -> bool:
    """Every address the host resolves to is public."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise WebError(f"cannot resolve {host}") from e
    return all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


async def check_url(url: str) -> None:
    """Raise WebError unless ``url`` is http(s) on a public host."""
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise WebError("only http(s) URLs")
    if not await asyncio.to_thread(is_public, p.hostname):
        raise WebError(f"{p.hostname} is not a public address")


@dataclass
class Page:
    url: str  # as requested
    title: str = ""
    text: str = ""
    error: str = ""
    final: str = ""  # after redirects, if different

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.error


async def fetch(
    url: str,
    timeout: float = 20.0,
    page_chars: int = PAGE_CHARS,
    max_bytes: int = MAX_BYTES,
    max_redirects: int = MAX_REDIRECTS,
    user_agent: str = USER_AGENT,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Page:
    """One public page as text. Never raises: a failure is the page's ``error``.

    Redirects are followed by hand (at most ``max_redirects``), each target checked like
    the first; only text types are read, at most ``max_bytes`` of them; HTML becomes text."""
    try:
        target = url
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, transport=transport,
            headers={"User-Agent": user_agent, "Accept": "text/html,text/plain;q=0.9"},
        ) as c:  # fmt: skip
            for _ in range(max_redirects + 1):
                await check_url(target)
                async with c.stream("GET", target) as r:
                    if r.is_redirect:
                        target = urljoin(target, r.headers.get("location", ""))
                        continue
                    if r.status_code != 200:
                        return Page(url, error=f"HTTP {r.status_code}")
                    ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                    if ctype and ctype not in TEXT_TYPES:
                        return Page(url, error=f"not a text page ({ctype})")
                    body = b""
                    async for chunk in r.aiter_bytes():
                        body += chunk
                        if len(body) > max_bytes:
                            break
                    raw = body.decode(r.encoding or "utf-8", errors="replace")
                    title, text = to_text(raw) if "html" in (ctype or "html") else ("", raw)
                    text = re.sub(r"\s+", " ", text).strip()
                    final = target if target != url else ""
                    return Page(url, title, text[:page_chars], final=final)
            return Page(url, error="too many redirects")
    except WebError as e:
        return Page(url, error=str(e))
    except (httpx.HTTPError, UnicodeError, ValueError) as e:
        return Page(url, error=f"{type(e).__name__}: {e}"[:200])


_DROP = re.compile(
    r"(?is)<(script|style|noscript|svg|nav|header|footer|aside|form|template)\b[^>]*>.*?</\1>"
)


def to_text(raw: str) -> tuple[str, str]:
    """(title, readable text) of an HTML page, standard library only."""
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
    title = html.unescape(re.sub(r"\s+", " ", m.group(1))).strip() if m else ""
    body = _DROP.sub(" ", raw)
    body = re.sub(r"(?is)<!--.*?-->", " ", body)
    body = re.sub(r"(?i)<(br|/p|/div|/li|/h[1-6]|/tr)\b[^>]*>", "\n", body)
    body = html.unescape(re.sub(r"(?s)<[^>]+>", " ", body))
    return title, body
