"""The sandbox: docker run arguments, the relay, and reading output line by line.

The output tests are ai-factory's test_proc_lines; the egress test is ai-factory's
test_isolated_network_reaches_only_the_gateway, run only where Docker and the image are."""

from __future__ import annotations

import asyncio
import http.server
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from ai_common import sandbox
from ai_common.sandbox import Proc, Relay, RunSpec

# -- docker run arguments -------------------------------------------------------------------


def spec(tmp_path: Path, **kw: object) -> RunSpec:
    base: dict[str, object] = dict(
        name="step-1", workspace=tmp_path / "ws", ctl_dir=tmp_path / "ctl",
        ctl_mount="/ai-factory", image="img:1", command=["sh", "-c", "true"],
    )  # fmt: skip
    base.update(kw)
    return RunSpec(**base)  # type: ignore[arg-type]


def pairs(args: list[str], flag: str) -> list[str]:
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


def test_docker_args_are_hardened(tmp_path):
    s = spec(tmp_path, env={"A": "1"}, labels={"ai-factory": "1"}, cpus="2", memory="4g")
    args = sandbox.docker_args(s)
    assert args[:5] == ["docker", "run", "--rm", "--init", "--name"]
    assert args[-4:] == ["img:1", "sh", "-c", "true"]
    assert pairs(args, "--user") == [f"{os.getuid()}:{os.getgid()}"]
    assert pairs(args, "--cap-drop") == ["ALL"]
    assert pairs(args, "--security-opt") == ["no-new-privileges"]
    assert pairs(args, "--pids-limit") == ["1024"]
    assert pairs(args, "--cpus") == ["2"] and pairs(args, "--memory") == ["4g"]
    assert pairs(args, "-v") == [f"{tmp_path}/ws:/workspace", f"{tmp_path}/ctl:/ai-factory:ro"]
    assert pairs(args, "-w") == ["/workspace"]
    assert pairs(args, "-e") == ["HOME=/tmp/home", "A=1"]
    assert pairs(args, "--label") == ["ai-factory=1"]
    assert "--network" not in args and "--add-host" not in args


def test_network_isolated_or_open(tmp_path):
    iso = sandbox.docker_args(spec(tmp_path, network="exec-net", add_host="host-gateway"))
    assert pairs(iso, "--network") == ["exec-net"] and "--add-host" not in iso
    open_ = sandbox.docker_args(spec(tmp_path, add_host="192.168.5.2"))
    assert pairs(open_, "--add-host") == ["host.docker.internal:192.168.5.2"]
    assert "--network" not in open_


def test_role_mounts_and_results(tmp_path):
    s = spec(
        tmp_path, user="1000:1000", workspace_readonly=True, readonly_dirs=["tests"],
        writable_dirs=["out"], results_dir=tmp_path / "res", results_mount="/ai-factory-results",
    )  # fmt: skip
    args = sandbox.docker_args(s)
    assert pairs(args, "--user") == ["1000:1000"]
    assert pairs(args, "-v") == [
        f"{tmp_path}/ws:/workspace:ro",
        f"{tmp_path}/ctl:/ai-factory:ro",
        f"{tmp_path}/ws/tests:/workspace/tests:ro",
        f"{tmp_path}/ws/out:/workspace/out:rw",
        f"{tmp_path}/res:/ai-factory-results",
    ]
    with pytest.raises(sandbox.SandboxError):
        sandbox.docker_args(spec(tmp_path, results_dir=tmp_path / "res"))


# -- the relay, against a fake docker -------------------------------------------------------


class FakeDocker:
    def __init__(self, answers: dict[str, tuple[int, str, str]] | None = None):
        self.calls: list[list[str]] = []
        self.answers = answers or {}

    def __call__(self, *args: str, timeout: float | None = None) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        key = " ".join(args[:2])
        code, out, err = self.answers.get(key, (0, "", ""))
        return subprocess.CompletedProcess(["docker", *args], code, out, err)


RELAY = Relay(
    name="ai-factory-gw-x", alias="ai-factory-gw", network="ai-factory-exec-x", image="base",
    labels={"ai-factory": "1"}, user="1000:1000", port_label="ai-factory.gateway_port",
)  # fmt: skip


def test_relay_is_created_with_network_and_attached():
    d = FakeDocker({"network inspect": (1, "", "no such"), "inspect -f": (1, "", "no such")})
    assert sandbox.ensure_relay(RELAY, 9000, d) == "http://ai-factory-gw:8701"
    verbs = [" ".join(c[:2]) for c in d.calls]
    assert verbs == ["network inspect", "network create", "inspect -f", "run -d", "network connect"]
    assert d.calls[1] == ["network", "create", "--internal", "--label", "ai-factory=1",
                          "ai-factory-exec-x"]  # fmt: skip
    run = d.calls[3]
    assert run[-4:] == ["base", "socat", "TCP-LISTEN:8701,fork,reuseaddr",
                        "TCP:host.docker.internal:9000"]  # fmt: skip
    for flag in ("--read-only", "--cap-drop", "--security-opt", "--pids-limit", "--memory"):
        assert flag in run
    assert pairs(run, "--user") == ["1000:1000"]
    assert pairs(run, "--label") == ["ai-factory=1", "ai-factory.gateway_port=9000"]
    assert d.calls[4] == ["network", "connect", "--alias", "ai-factory-gw", "ai-factory-exec-x",
                          "ai-factory-gw-x"]  # fmt: skip


def test_running_relay_on_the_right_port_is_kept_and_wrong_port_replaced():
    d = FakeDocker({"inspect -f": (0, "true 9000\n", "")})
    sandbox.ensure_relay(RELAY, 9000, d)
    assert [c[0] for c in d.calls] == ["network", "inspect"]
    d = FakeDocker({"inspect -f": (0, "true 8000\n", "")})
    sandbox.ensure_relay(RELAY, 9000, d)
    assert ["rm", "-f", "ai-factory-gw-x"] in d.calls and any(
        c[:2] == ["run", "-d"] for c in d.calls
    )


def test_relay_without_port_label_checks_only_that_it_runs():
    r = Relay(name="ai-delegate-gw-9000", alias="gw-9000", network="ai-delegate-exec", image="b")
    d = FakeDocker({"inspect -f": (0, "true\n", "")})
    assert sandbox.ensure_relay(r, 9000, d) == "http://gw-9000:8701"
    assert d.calls[-1] == ["inspect", "-f", "{{.State.Running}}", "ai-delegate-gw-9000"]
    d = FakeDocker({"inspect -f": (1, "", ""), "run -d": (0, "", "")})
    sandbox.ensure_relay(r, 9000, d)
    run = next(c for c in d.calls if c[:2] == ["run", "-d"])
    assert "--user" not in run and "--label" not in run


def test_relay_failures_raise_and_clean_up():
    d = FakeDocker({"inspect -f": (1, "", ""), "network connect": (1, "", "boom")})
    with pytest.raises(sandbox.SandboxError, match="cannot attach the relay: boom"):
        sandbox.ensure_relay(RELAY, 9000, d)
    assert d.calls[-1] == ["rm", "-f", "ai-factory-gw-x"]
    d = FakeDocker({"network inspect": (1, "", ""), "network create": (1, "", "denied")})
    with pytest.raises(sandbox.SandboxError, match="cannot create network"):
        sandbox.ensure_relay(RELAY, 9000, d)
    d = FakeDocker({"network inspect": (1, "", ""), "network create": (1, "", "already exists"),
                    "inspect -f": (0, "true 9000", "")})  # fmt: skip
    sandbox.ensure_relay(RELAY, 9000, d)  # lost a race to create it: fine


def test_teardown():
    d = FakeDocker()
    sandbox.teardown("gw", "net", d)
    assert d.calls == [["rm", "-f", "gw"], ["network", "rm", "net"]]


# -- output line by line (ai-factory's test_proc_lines) -------------------------------------


async def _lines(script: str, **kw: object) -> list[str]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )  # fmt: skip
    p = Proc(proc, **kw)  # type: ignore[arg-type]
    out = [line async for line in p.lines()]
    assert await p.wait() == 0
    return out


async def test_a_megabyte_line_comes_through_whole():
    # A coding agent's JSON event carrying a whole file: far over readline's 64 KiB.
    out = await _lines(
        "import sys; sys.stdout.write('before\\n' + 'x' * 1_000_000 + '\\nafter\\n')"
    )
    assert out == ["before", "x" * 1_000_000, "after"]


async def test_lines_split_across_reads_and_a_last_line_without_newline():
    out = await _lines(
        "import sys, time\n"
        "sys.stdout.write('one\\ntw'); sys.stdout.flush(); time.sleep(0.05)\n"
        "sys.stdout.write('o\\n\\nthree'); sys.stdout.flush()"
    )
    assert out == ["one", "two", "", "three"]


async def test_a_line_over_the_cap_is_cut_with_a_note(monkeypatch):
    monkeypatch.setattr(sandbox, "READ_CHUNK", 300)
    out = await _lines(
        "import sys; sys.stdout.write('y' * 5000 + '\\nnext\\n')",
        max_line_bytes=1000, cut_label="ai-factory",
    )  # fmt: skip
    assert out == ["y" * 1000 + sandbox.cut_note(4000, "ai-factory"), "next"]
    assert sandbox.cut_note(4000, "ai-factory") == " ...[ai-factory: line cut, 4000 more bytes]"


async def test_no_cap_keeps_every_byte(monkeypatch):
    monkeypatch.setattr(sandbox, "READ_CHUNK", 300)
    out = await _lines("import sys; sys.stdout.write('y' * 5000)", max_line_bytes=None)
    assert out == ["y" * 5000]


async def test_multibyte_characters_split_between_reads(monkeypatch):
    monkeypatch.setattr(sandbox, "READ_CHUNK", 7)
    out = await _lines("import sys; sys.stdout.write('héllo wörld ✓\\n')")
    assert out == ["héllo wörld ✓"]


async def test_kill_a_local_process_group():
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(60)", start_new_session=True,
    )  # fmt: skip
    p = Proc(proc)
    assert p.handle == f"local:{proc.pid}"
    await p.kill()
    assert proc.returncode is not None


# -- egress lockdown in real containers -----------------------------------------------------

IMAGE = os.environ.get("AI_COMMON_TEST_IMAGE", "ai-factory/executor-base:latest")


def _docker_ready() -> bool:
    try:
        return sandbox.docker_ok() == "" and sandbox.image_present(IMAGE)
    except (OSError, subprocess.SubprocessError):
        return False


class _Hello(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = b"ai-common-probe"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a: object) -> None:
        pass


@pytest.mark.docker
@pytest.mark.skipif(not _docker_ready(), reason=f"docker or image {IMAGE} not available")
def test_isolated_network_reaches_only_the_gateway(tmp_path):
    """On the isolated network a container reaches the gateway through the relay, and
    nothing else: not the host, the LAN or the internet."""
    bind = "127.0.0.1" if sys.platform == "darwin" else "0.0.0.0"
    server = http.server.ThreadingHTTPServer((bind, 0), _Hello)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    tag = f"{os.getpid()}-{time.time_ns() % 10**6}"
    relay = Relay(name=f"ai-common-test-gw-{tag}", alias="ai-common-gw",
                  network=f"ai-common-test-{tag}", image=IMAGE,
                  labels={"ai-common-test": "1"}, port_label="ai-common.gateway_port")  # fmt: skip
    ws, ctl = tmp_path / "ws", tmp_path / "ctl"
    ws.mkdir()
    ctl.mkdir()

    def run(script: str, network: str) -> subprocess.CompletedProcess[str]:
        s = RunSpec(
            name=f"ai-common-egress-{time.time_ns() % 10**9}", workspace=ws, ctl_dir=ctl,
            ctl_mount="/ctl", image=IMAGE, command=["sh", "-c", script],
            network=network, add_host="host-gateway",
        )  # fmt: skip
        return subprocess.run(sandbox.docker_args(s), capture_output=True, text=True, timeout=120)

    try:
        url = sandbox.ensure_relay(relay, port)
        through = run(f"curl -fsS --max-time 10 {url}/", relay.network)
        assert through.returncode == 0 and "ai-common-probe" in through.stdout, through.stderr
        for target in (f"http://host.docker.internal:{port}/", "http://1.1.1.1/",
                       "https://example.com/"):  # fmt: skip
            r = run(f"curl -fsS --max-time 5 {target}", relay.network)
            assert r.returncode != 0, f"{target} was reachable from the isolated network"
        assert (
            run(f"curl -fsS --max-time 10 http://host.docker.internal:{port}/", "").returncode == 0
        )
    finally:
        sandbox.teardown(relay.name, relay.network)
        server.shutdown()
