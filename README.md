# ai-common

Python code shared by [ai-factory](https://github.com/hofftodd/ai-factory) and
[ai-delegate](https://github.com/hofftodd/ai-delegate), which used to copy it between them.
Part of the [ai-tools](https://github.com/hofftodd/ai-tools) suite.

| Module | What it holds |
|---|---|
| `ai_common.sandbox` | Hardened `docker run` arguments (`--rm --init`, all capabilities dropped, no-new-privileges, pids limit, the clone at `/workspace`, a read-only control mount, `HOME=/tmp/home`); the isolated `--internal` network and the socat relay on port 8701 that is its only way out; reading a process's output line by line, however long the lines |
| `ai_common.web` | `is_public(host)`; `fetch(url)`, which checks every redirect hop against public addresses, caps the size, reads only text types and turns HTML into text; `search(urls, query)`, a SearXNG-shaped client that fails over between URLs and can send a Bearer token (ai-search accepts one) |
| `ai_common.swap` | An ai-swap client: `auth_headers(api_key)`, `client(base_url, api_key)`, and `lease(base_url, model, owner, ttl_s)`, an async context manager that holds a model lease, renews it every `ttl/3` and releases it on exit (a no-op, logged once, against an ai-swap without the lease API) |

What differs between the callers (container and network names, labels, the control mount, the
results directory, the per-line output cap) is a parameter, not a setting read here: this
package reads no configuration of its own.

## Using it

Add it as a git dependency:

```toml
dependencies = ["ai-common @ git+ssh://git@github.com/hofftodd/ai-common.git"]
```

or, while working on both inside an ai-tools checkout, as a path source:

```toml
[tool.uv.sources]
ai-common = { path = "../ai-common", editable = true }
```

## Developing

```bash
uv sync
uv run pytest && uv run ruff check && uv run mypy src
```

Tests marked `docker` need a Docker engine and an image with `curl` and `socat`
(`AI_COMMON_TEST_IMAGE`, default `ai-factory/executor-base:latest`); they skip without one.
