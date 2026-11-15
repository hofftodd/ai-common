"""Code shared by ai-factory and ai-delegate.

- ``ai_common.sandbox``: hardened ``docker run`` arguments, the isolated executor network and
  the socat relay that is its only way out, and reading a process's output line by line.
- ``ai_common.web``: the public-address guard, a safe page fetch, and a SearXNG-shaped search
  client with URL failover.
- ``ai_common.swap``: a small ai-swap client: API key headers and model leases.
"""

__version__ = "0.1.0"
