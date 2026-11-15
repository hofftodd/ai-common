"""Docker sandboxes for coding agents, as ai-factory and ai-delegate run them.

One container per step: ``--rm --init``, every capability dropped, no-new-privileges, a pids
limit, the step's clone mounted at /workspace, its control directory mounted read-only, and
HOME=/tmp/home. Nothing else from the host.

Containers sit on one of two networks:

- isolated (``RunSpec.network`` set): an internal Docker network with no route anywhere. The
  only other thing on it is a relay container that forwards ``RELAY_PORT`` to the caller's
  model gateway on the host, so an agent reaches its model through the gateway's budgets and
  nothing else: not the LAN, the model servers or the internet.
- open (``RunSpec.network`` empty): the default bridge plus ``host.docker.internal``
  (``RunSpec.add_host``), for projects whose tests need the network.

What differs between the callers (container and network names, labels, the control mount,
the results directory, the per-line cap on output) is a parameter here.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import signal
import subprocess
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

WORKSPACE = "/workspace"
HOST_ALIAS = "host.docker.internal"
RELAY_PORT = 8701
CONTAINER_UID = 1000

# Runs one docker CLI command: (args, timeout) -> completed process with text output.
Docker = Callable[..., subprocess.CompletedProcess[str]]


class SandboxError(RuntimeError):
    pass


def docker(
    *args: str, timeout: float | None = 120, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a docker CLI command and capture its output. ``env`` (default: this process's)
    is where a caller resolves DOCKER_HOST."""
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=None if env is None else dict(env),
    )


def docker_ok(run: Docker = docker) -> str:
    """'' if Docker answers, else why not."""
    if not shutil.which("docker"):
        return "docker is not installed"
    try:
        r = run("info", "--format", "{{.ServerVersion}}", timeout=15)
    except subprocess.TimeoutExpired:
        return "docker info timed out"
    return "" if r.returncode == 0 else (r.stderr.strip() or "docker info failed")[:300]


def image_present(image: str, run: Docker = docker) -> bool:
    return run("image", "inspect", image, timeout=30).returncode == 0


def label_args(labels: Mapping[str, str]) -> list[str]:
    out: list[str] = []
    for k, v in labels.items():
        out += ["--label", f"{k}={v}"]
    return out


# -- one step's container ---------------------------------------------------------------------


@dataclass
class RunSpec:
    name: str  # unique; the container's name
    workspace: Path  # host path of the clone, mounted at /workspace
    ctl_dir: Path  # host path of the control dir, mounted read-only at ``ctl_mount``
    ctl_mount: str  # e.g. /ai-factory, /ai-delegate
    image: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    user: str = ""  # "uid:gid"; "" = this process's uid:gid
    cpus: str = ""  # "" = no limit
    memory: str = ""
    pids_limit: int = 1024
    # The internal network to join (isolated). "" = the default bridge (open), with
    # ``add_host`` mapped as host.docker.internal when set (e.g. "host-gateway").
    network: str = ""
    add_host: str = ""
    # Role sandboxing: the workspace mounted read-only, and workspace-relative directories
    # mounted over it read-only or writable.
    workspace_readonly: bool = False
    readonly_dirs: list[str] = field(default_factory=list)
    writable_dirs: list[str] = field(default_factory=list)
    # Host directory mounted writable at ``results_mount`` (test results).
    results_dir: Path | None = None
    results_mount: str = ""


def mounts(spec: RunSpec) -> list[str]:
    ws = f"{spec.workspace}:{WORKSPACE}" + (":ro" if spec.workspace_readonly else "")
    out = ["-v", ws, "-v", f"{spec.ctl_dir}:{spec.ctl_mount}:ro"]
    for rel, mode in [
        *((d, "ro") for d in spec.readonly_dirs),
        *((d, "rw") for d in spec.writable_dirs),
    ]:
        out += ["-v", f"{spec.workspace / rel}:{WORKSPACE}/{rel}:{mode}"]
    if spec.results_dir:
        if not spec.results_mount:
            raise SandboxError("results_dir needs results_mount")
        out += ["-v", f"{spec.results_dir}:{spec.results_mount}"]
    return out


def docker_args(spec: RunSpec) -> list[str]:
    """The full ``docker run`` command line for one step. The caller creates any mount
    points (``readonly_dirs``/``writable_dirs``) and chowns the mounts if it runs as root."""
    env = {"HOME": "/tmp/home", **spec.env}
    args = [
        "docker", "run", "--rm", "--init", "--name", spec.name,
        "--user", spec.user or f"{os.getuid()}:{os.getgid()}",
    ]  # fmt: skip
    if spec.cpus:
        args += ["--cpus", spec.cpus]
    if spec.memory:
        args += ["--memory", spec.memory]
    args += [
        "--pids-limit", str(spec.pids_limit),
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        *mounts(spec),
        "-w", WORKSPACE,
        *label_args(spec.labels),
    ]  # fmt: skip
    if spec.network:
        args += ["--network", spec.network]
    elif spec.add_host:
        args += ["--add-host", f"{HOST_ALIAS}:{spec.add_host}"]
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    return [*args, spec.image, *spec.command]


async def start(
    spec: RunSpec,
    env: Mapping[str, str] | None = None,
    max_line_bytes: int | None = None,
    cut_label: str = "ai-common",
) -> Proc:
    """Start the step's container. ``env`` is the docker CLI's own environment (where a
    caller resolves DOCKER_HOST); the container gets only ``spec.env``."""
    proc = await asyncio.create_subprocess_exec(
        *docker_args(spec),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
        env=None if env is None else dict(env),
    )
    return Proc(
        proc, f"docker:{spec.name}", container=spec.name,
        max_line_bytes=max_line_bytes, cut_label=cut_label, docker_env=env,
    )  # fmt: skip


# -- output ---------------------------------------------------------------------------------

# Output is read in chunks, so a line of any length is fine: a coding agent's JSON events can
# carry whole files on one line (StreamReader.readline stops at 64 KiB and raises). Past
# ``max_line_bytes`` a line is cut, keeping memory bounded.
READ_CHUNK = 64 * 1024
MAX_LINE_BYTES = 4 * 1024 * 1024


def cut_note(dropped: int, label: str = "ai-common") -> str:
    return f" ...[{label}: line cut, {dropped} more bytes]"


class Proc:
    """A started step: its output line by line, its exit status, and a way to kill it.

    ``handle`` names it for a supervisor (``docker:<name>`` or ``local:<pgid>``)."""

    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        handle: str = "",
        container: str | None = None,
        max_line_bytes: int | None = MAX_LINE_BYTES,
        cut_label: str = "ai-common",
        docker_env: Mapping[str, str] | None = None,
    ):
        self.proc = proc
        self.handle = handle or (f"docker:{container}" if container else f"local:{proc.pid}")
        self.container = container
        self.max_line_bytes = max_line_bytes
        self.cut_label = cut_label
        self._docker_env = docker_env

    async def lines(self) -> AsyncIterator[str]:
        """Each line of the process's output, without its newline, however long (cut at
        ``max_line_bytes`` with a note; None = never cut)."""
        assert self.proc.stdout is not None
        cap = self.max_line_bytes
        buf = bytearray()
        dropped = 0  # bytes of the current line past the cap

        def take(part: bytes) -> None:
            nonlocal dropped
            if cap is None:
                buf.extend(part)
                return
            room = cap - len(buf)
            if room > 0:
                buf.extend(part[:room])
            dropped += max(0, len(part) - max(room, 0))

        def line() -> str:
            nonlocal dropped
            text = bytes(buf).decode(errors="replace")
            if dropped:
                text += cut_note(dropped, self.cut_label)
            buf.clear()
            dropped = 0
            return text

        while True:
            chunk = await self.proc.stdout.read(READ_CHUNK)
            if not chunk:
                if buf or dropped:
                    yield line()
                return
            start = 0
            while (end := chunk.find(b"\n", start)) != -1:
                take(chunk[start:end])
                yield line()
                start = end + 1
            take(chunk[start:])

    async def wait(self) -> int:
        return await self.proc.wait()

    async def kill(self) -> None:
        """Remove the container, or SIGTERM then SIGKILL the process group."""
        if self.proc.returncode is not None:
            return
        if self.container:
            await asyncio.to_thread(docker, "rm", "-f", self.container, env=self._docker_env)
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.proc.wait(), 10)


# -- the isolated network and its relay -----------------------------------------------------


def ensure_network(name: str, labels: Mapping[str, str], run: Docker = docker) -> None:
    """Create the internal (no route anywhere) network ``name`` if it doesn't exist."""
    if run("network", "inspect", name).returncode == 0:
        return
    r = run("network", "create", "--internal", *label_args(labels), name)
    if r.returncode != 0 and "already exists" not in r.stderr:
        raise SandboxError(f"cannot create network {name}: {r.stderr.strip()}")


@dataclass
class Relay:
    """The relay container that is an isolated network's only way out: it sits on the
    default bridge (so it can reach the host) and on ``network`` as ``alias``, and forwards
    ``RELAY_PORT`` to ``host.docker.internal:<gateway_port>``."""

    name: str  # container name
    alias: str  # its name on the network
    network: str
    image: str  # must have socat
    labels: dict[str, str] = field(default_factory=dict)
    user: str = ""  # "" = the image's user
    host_ip: str = "host-gateway"  # where host.docker.internal points
    # A label recording the gateway port, so a relay left forwarding to another port is
    # replaced. "" = only check that it runs (fine when ``name`` contains the port).
    port_label: str = ""
    timeout: float = 120

    @property
    def url(self) -> str:
        return f"http://{self.alias}:{RELAY_PORT}"


def ensure_relay(relay: Relay, gateway_port: int, run: Docker = docker) -> str:
    """Make sure the internal network and the relay exist and forward to ``gateway_port``;
    returns the base URL containers use to reach the gateway."""
    ensure_network(relay.network, relay.labels, run)
    fmt = "{{.State.Running}}"
    want = ["true"]
    if relay.port_label:
        fmt += f' {{{{index .Config.Labels "{relay.port_label}"}}}}'
        want.append(str(gateway_port))
    r = run("inspect", "-f", fmt, relay.name)
    if r.returncode == 0:
        if r.stdout.split() == want:
            return relay.url
        run("rm", "-f", relay.name)
    labels = dict(relay.labels)
    if relay.port_label:
        labels[relay.port_label] = str(gateway_port)
    r = run(
        "run", "-d", "--name", relay.name,
        *(["--user", relay.user] if relay.user else []),
        "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--memory", "64m", "--pids-limit", "64",
        "--add-host", f"{HOST_ALIAS}:{relay.host_ip}",
        *label_args(labels),
        relay.image,
        "socat", f"TCP-LISTEN:{RELAY_PORT},fork,reuseaddr", f"TCP:{HOST_ALIAS}:{gateway_port}",
        timeout=relay.timeout,
    )  # fmt: skip
    if r.returncode != 0:
        raise SandboxError(f"cannot start the gateway relay: {r.stderr.strip()[:300]}")
    r = run("network", "connect", "--alias", relay.alias, relay.network, relay.name)
    if r.returncode != 0 and "already exists" not in r.stderr:
        run("rm", "-f", relay.name)
        raise SandboxError(f"cannot attach the relay: {r.stderr.strip()[:300]}")
    return relay.url


def teardown(relay_name: str, network: str = "", run: Docker = docker) -> None:
    """Remove the relay and (when given) the network, best effort. The network stays while
    containers are still attached to it."""
    run("rm", "-f", relay_name)
    if network:
        run("network", "rm", network)
