"""The fetch guard, HTML to text, and search failover, against a fake web (ported from
ai-delegate's test_web and ai-factory's test_research)."""

from __future__ import annotations

from collections import Counter

import httpx
import pytest

from ai_common import web

PAGE_A = """<html><head><title>Widget Spec 2.0</title><script>alert('x')</script></head>
<body><nav>menu menu</nav><p>The widget protocol uses port 7070 by default.</p>
<p>Version 2.0 was released in March 2026 by the Widget Group.</p></body></html>"""
PAGE_B = "<html><title>Widget news</title><p>Critics say widgets are slow.</p></html>"

SEARCH = {
    "widget release": ["https://spec.example.org/a", "https://news.example.com/b"],
}


async def chunks():
    for _ in range(web.MAX_BYTES // 65536 + 10):  # well past the cap, 64 KiB at a time
        yield b"z" * 65536


def web_handler(hits: Counter, seen: list[httpx.Request] | None = None):
    def h(req: httpx.Request) -> httpx.Response:
        url = str(req.url).split("?")[0]
        hits[url] += 1
        if seen is not None:
            seen.append(req)
        if req.url.host == "search.test":
            q = req.url.params["q"]
            rows = [{"title": u, "url": u, "content": "..."} for u in SEARCH.get(q, [])]
            rows.append({"title": "not a web page", "url": "ftp://x.example/", "content": ""})
            return httpx.Response(200, json={"results": rows})
        if url == "https://evil.example/redirect":
            return httpx.Response(302, headers={"location": "http://10.0.0.5/secret"})
        if url == "https://hop.example/1":
            return httpx.Response(301, headers={"location": "/2"})
        if url == "https://hop.example/2":
            return httpx.Response(302, headers={"location": "https://spec.example.org/a"})
        if url == "https://loop.example/":
            return httpx.Response(302, headers={"location": "https://loop.example/"})
        if url == "https://big.example/":
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=chunks())
        if url == "https://files.example.org/x.pdf":
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")
        body = {"https://spec.example.org/a": PAGE_A, "https://news.example.com/b": PAGE_B}.get(url)
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=body)

    return h


@pytest.fixture
def fake_dns(monkeypatch):
    """No real DNS: 10.x and *.internal are private, everything else public."""
    monkeypatch.setattr(
        web, "is_public", lambda host: not (host.startswith("10.") or host.endswith(".internal"))
    )


def test_html_to_text_drops_chrome():
    title, text = web.to_text(PAGE_A)
    assert title == "Widget Spec 2.0"
    assert "port 7070" in text and "alert" not in text and "menu" not in text


async def test_fetch_guard(fake_dns):
    hits: Counter = Counter()
    t = httpx.MockTransport(web_handler(hits))
    ok = await web.fetch("https://spec.example.org/a", transport=t)
    assert ok.ok and ok.title == "Widget Spec 2.0" and ok.final == ""
    bad = await web.fetch("https://evil.example/redirect", transport=t)
    assert not bad.ok and "not a public address" in bad.error
    assert hits["http://10.0.0.5/secret"] == 0  # refused before any request went out
    assert (
        "not a text page" in (await web.fetch("https://files.example.org/x.pdf", transport=t)).error
    )
    assert "only http" in (await web.fetch("file:///etc/passwd", transport=t)).error
    assert "not a public" in (await web.fetch("http://db.internal/", transport=t)).error
    assert (await web.fetch("https://nowhere.example/", transport=t)).error == "HTTP 404"


async def test_fetch_follows_checked_redirects_and_caps(fake_dns):
    t = httpx.MockTransport(web_handler(Counter()))
    hop = await web.fetch("https://hop.example/1", transport=t)
    assert hop.ok and hop.final == "https://spec.example.org/a"
    assert (await web.fetch("https://loop.example/", transport=t)).error == "too many redirects"
    big = await web.fetch("https://big.example/", transport=t, page_chars=10**9)
    assert web.MAX_BYTES < len(big.text) <= web.MAX_BYTES + 64 * 1024
    short = await web.fetch("https://spec.example.org/a", transport=t, page_chars=20)
    assert len(short.text) == 20


async def test_fetch_sends_the_user_agent(fake_dns):
    seen: list[httpx.Request] = []
    t = httpx.MockTransport(web_handler(Counter(), seen))
    await web.fetch("https://spec.example.org/a", transport=t, user_agent="ai-test/1")
    assert seen[-1].headers["user-agent"] == "ai-test/1"


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8700/api/nodes",
        "http://localhost/",
        "http://10.0.0.5/x",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "file:///etc/passwd",
    ],
)
async def test_fetch_refuses_private_addresses(url):
    # Real resolution: these are all literals or localhost, which resolve offline.
    def refuse(req: httpx.Request) -> httpx.Response:
        raise AssertionError(f"request went out to {req.url}")

    page = await web.fetch(url, transport=httpx.MockTransport(refuse))
    assert not page.ok and page.error


def test_is_public():
    assert not web.is_public("127.0.0.1")
    assert not web.is_public("192.168.1.10")
    assert web.is_public("1.1.1.1")
    with pytest.raises(web.WebError, match="cannot resolve"):
        web.is_public("no-such-host.invalid")


async def test_search_fails_over_then_reports():
    hits: Counter = Counter()

    def h(req: httpx.Request) -> httpx.Response:
        if req.url.host == "down.internal":
            raise httpx.ConnectError("no route")
        if req.url.host == "busy.test":
            return httpx.Response(503, text="overloaded")
        return web_handler(hits)(req)

    t = httpx.MockTransport(h)
    got = await web.search(
        ["http://down.internal:1", "http://busy.test", "http://search.test"], "widget release",
        transport=t,
    )  # fmt: skip
    assert [x.url for x in got] == SEARCH["widget release"]  # the ftp:// row is dropped
    assert got[0].snippet == "..."
    assert len(await web.search("http://search.test/", "widget release", count=1, transport=t)) == 1
    with pytest.raises(web.WebError, match="search failed"):
        await web.search(["http://down.internal:1"], "x", transport=t)


async def test_search_stops_when_the_service_answers_and_sends_the_token():
    seen: list[httpx.Request] = []

    def h(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        if req.url.host == "auth.test":
            ok = req.headers.get("authorization") == "Bearer s3cret"
            return httpx.Response(200, json={"results": []}) if ok else httpx.Response(401)
        return httpx.Response(200, json={"results": []})

    t = httpx.MockTransport(h)
    with pytest.raises(web.WebError, match="HTTP 401"):
        await web.search(["http://auth.test", "http://other.test"], "q", transport=t)
    assert [r.url.host for r in seen] == ["auth.test"]  # a 401 isn't a route problem
    assert await web.search(["http://auth.test"], "q", token="s3cret", transport=t) == []
    params = seen[-1].url.params
    assert params["q"] == "q" and params["format"] == "json" and params["count"] == "8"
    await web.search(["http://other.test"], "q", transport=t)
    assert "authorization" not in seen[-1].headers
