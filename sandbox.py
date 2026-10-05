"""Docker sandbox lifecycle for the Verifier pipeline.

One container per problem, reused across all LLM calls for that problem
via `docker exec`. Matches the dominant benchmark pattern (SWE-bench,
Terminal-Bench, AgentBench all use per-task containers). Containers are
destroyed at problem end.

Authentication
--------------
Claude subscription auth lives in the macOS Keychain as "Claude Code-
credentials". The container can't reach the Keychain, so we extract the
OAuth JSON to a temp file (mode 0644 so the non-root `node` container user
can read it; /tmp is host-only) and bind-mount it read-only at
/home/node/.claude/.credentials.json (claude's plaintext fallback path).

Codex subscription auth is already on disk at ~/.codex/auth.json, so a
plain bind-mount of ~/.codex suffices.

Hardening
---------
Container runs with --cap-drop=ALL, --security-opt=no-new-privileges,
--pids-limit, memory/cpu caps, and credential mounts as :ro. This is
reasonable for research-grade isolation without going to gVisor.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

DEFAULT_IMAGE = "reasoning-eval-sandbox:latest"

# Minimum set of files codex needs in its state dir. Everything else
# (sessions/, archived_sessions/, history.jsonl, cache/) is runtime-
# generated and unnecessary for a fresh call.
_CODEX_STATE_FILES = (
    "auth.json",
    "config.toml",
    "installation_id",
    "internal_storage.json",
    ".codex-global-state.json",
)
_CODEX_RESUME_DIRS = ("sessions", "archived_sessions")


def _staging_dir() -> str:
    """Directory for files/dirs that become bind-mount sources.

    These must live somewhere the Docker VM can see. On macOS VM-based
    engines with default mount config (colima, lima, podman machine),
    only $HOME is shared — a bind mount whose source is under
    tempfile's default /var/folders/... silently appears in the
    container as an empty root-owned dir. Symptoms: codex dies with
    "could not update PATH: Permission denied (os error 13)" (empty,
    unwritable $CODEX_HOME) and claude with "stream ended without
    result event" (empty creds mount). 0700 keeps the OAuth tokens
    staged here host-user-only; the Docker daemon resolves mount
    sources as root, so the restrictive parent doesn't affect the
    container. Override with OWRE_STAGING_DIR if $HOME itself
    isn't shared into the VM.
    """
    root = os.environ.get("OWRE_STAGING_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "reasoning-eval", "staging")
    os.makedirs(root, mode=0o700, exist_ok=True)
    return root


def refresh_claude_credentials() -> str | None:
    """Extract the Claude OAuth token from the macOS Keychain to a
    temp file. Returns the path (mode 0644, so the non-root container user
    can read it; /tmp is host-only) or None if we can't get it (not on
    macOS, not logged in, security CLI missing)."""
    try:
        r = subprocess.run(
            ["security", "find-generic-password",
             "-s", "Claude Code-credentials", "-w"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, OSError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None

    fd, path = tempfile.mkstemp(prefix="reasoning-eval-claude-creds-",
                                suffix=".json", dir=_staging_dir())
    try:
        with os.fdopen(fd, "w") as f:
            f.write(r.stdout)
        # 0644 (world-readable) so the non-root container user (UID 1000)
        # can read the file. The file contains an OAuth access token, so
        # we rely on the 0700 staging dir rather than mode bits for
        # secrecy. On shared hosts, switch to a UID-matching scheme.
        os.chmod(path, 0o644)
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def prepare_claude_config(retries: int = 5, retry_delay: float = 0.2) -> str | None:
    """Snapshot ~/.claude.json to a temp file the container can mount
    without racing the host.

    The host may be actively running Claude Code (including the one
    running this pipeline), which writes ~/.claude.json periodically
    via atomic rename. If the container mounts the host file directly
    :ro, a concurrent read while the file is mid-rewrite yields a
    corrupted-looking JSON (we've observed "Unterminated string" errors
    at this layer). Taking a snapshot + validating + retrying avoids
    that race entirely.

    Returns the tempfile path (mode 0644) or None if we can't get a
    readable + parseable snapshot after `retries` attempts.
    """
    src = Path(os.path.expanduser("~/.claude.json"))
    if not src.exists():
        return None
    fd, path = tempfile.mkstemp(prefix="reasoning-eval-claude-config-",
                                suffix=".json", dir=_staging_dir())
    os.close(fd)
    last_err: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            shutil.copy2(str(src), path)
            # Container user (UID 1000) needs to read this.
            os.chmod(path, 0o644)
            with open(path) as f:
                json.load(f)
            return path
        except (json.JSONDecodeError, OSError) as e:
            last_err = e
            time.sleep(retry_delay)
    # Give up; clean up the tempfile so we don't leak.
    try:
        os.unlink(path)
    except OSError:
        pass
    if last_err:
        print(f"WARNING: could not snapshot ~/.claude.json: {last_err}")
    return None


def cleanup_claude_config(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def prepare_codex_state_dir(resume_state_dir: str | None = None) -> str | None:
    """Copy the minimum set of codex subscription state files into a
    fresh tempdir so the container can bind-mount it writable without
    risking host state pollution.

    Codex writes to ~/.codex during normal operation (trusted-project
    state in config.toml, new session rollouts, etc.). Mounting the
    host ~/.codex :ro causes codex exec to fail. Mounting it :rw
    leaks per-problem state back to the host. This gives each problem
    a fresh, isolated writable copy that's destroyed at cleanup time.

    Returns a tempdir path, or None if ~/.codex doesn't exist.
    """
    src = Path(os.path.expanduser("~/.codex"))
    if not src.exists():
        return None
    dst = tempfile.mkdtemp(prefix="reasoning-eval-codex-state-", dir=_staging_dir())
    for name in _CODEX_STATE_FILES:
        f = src / name
        if f.exists():
            shutil.copy2(f, Path(dst) / name)
    if resume_state_dir:
        resume_root = Path(resume_state_dir)
        for name in _CODEX_RESUME_DIRS:
            source_dir = resume_root / name
            if source_dir.is_dir():
                shutil.copytree(source_dir, Path(dst) / name)
    # Container user (UID 1000) needs to write here. Some bind-mount
    # backends (Docker Desktop/VirtioFS) present the mount as owned by
    # the container UID, but others (Linux Docker Engine, non-VirtioFS
    # macOS engines) keep the host's real UID with this mode as the only
    # write bit — 0755 then makes the dir unwritable to UID 1000 and
    # codex dies on its first $CODEX_HOME write. This is a throwaway
    # per-problem tempdir destroyed at cleanup and the container is the
    # isolation boundary, so world-writable is fine here.
    try:
        os.chmod(dst, 0o777)
        for f in Path(dst).rglob("*"):
            try:
                # Same non-owner-write property as the dir above: on
                # engines that keep the host UID, 0644/0755 would leave
                # the copied state read-only to UID 1000.
                os.chmod(f, 0o777 if f.is_dir() else 0o666)
            except OSError:
                pass
    except OSError:
        pass
    return dst


def cleanup_credentials(path: str | None) -> None:
    """Remove a single credential file (extracted claude creds)."""
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def cleanup_codex_state_dir(path: str | None) -> None:
    """Remove a per-problem codex state tempdir."""
    if not path:
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def snapshot_codex_resume_state(state_dir: str, destination: str) -> None:
    """Persist only non-secret Codex rollout directories for later resume."""
    source_root = Path(state_dir)
    destination_root = Path(destination)
    temporary = destination_root.with_name(destination_root.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=True)
    for name in _CODEX_RESUME_DIRS:
        source_dir = source_root / name
        if source_dir.is_dir():
            shutil.copytree(source_dir, temporary / name)
    shutil.rmtree(destination_root, ignore_errors=True)
    os.replace(temporary, destination_root)


def image_digest(image: str = DEFAULT_IMAGE) -> str | None:
    """Return the local image digest (sha256:...) so call metadata can
    record which build was actually run. None if the image isn't local.

    Uses `docker images -q` rather than `docker image inspect` because
    the latter fails on containerd-backed Docker daemons for locally-
    built tags that weren't pushed to a registry.
    """
    try:
        r = subprocess.run(
            ["docker", "images", "-q", "--no-trunc", image],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        # TimeoutExpired derives from SubprocessError, NOT OSError, so a hung
        # Docker daemon used to escape this handler and traceback. That was
        # survivable while this ran mid-run; now that `_preflight` makes it the
        # first code every command executes, it would traceback at the front
        # door instead of printing the "or Docker isn't running" message --
        # which is precisely the right diagnosis for a hung daemon.
        return None
    if r.returncode != 0:
        return None
    digest = r.stdout.strip()
    return digest or None


def start_sandbox(
    pid: str,
    run_id: str,
    *,
    image: str = DEFAULT_IMAGE,
    claude_creds_path: str | None = None,
    claude_config_path: str | None = None,
    codex_state_dir: str | None = None,
    workspace_host_path: str | None = None,
    memory: str = "10g",
    cpus: str = "2",
    allow_network: bool = True,
) -> str:
    """Start a detached sandbox container for one problem. Returns the
    container ID (long form). The container runs `sleep infinity` and is
    intended to be used via `docker exec` until stop_sandbox() kills it.

    Auto-removes on stop thanks to --rm. Pass allow_network=False to run the
    container with --network=none (closed egress; see the hardening block).
    """
    home = os.path.expanduser("~")

    # We run as the `node` user inside the container (UID 1000). Claude
    # refuses --dangerously-skip-permissions under root, and Anthropic's
    # own devcontainer uses the same non-root pattern.
    #
    # IMPORTANT: mount only the specific auth files, NOT the ~/.claude
    # directory as a whole. Claude's Bash tool writes session state
    # under .claude/session-env/..., which would fail on a :ro mount.
    # Letting the container create its own .claude/ in the ephemeral FS
    # avoids that class of failure entirely.
    mounts: list[str] = []
    # Codex needs to write config.toml (trusted-project state) during
    # exec, so we give it a per-problem writable tempdir copy of the
    # host's minimal subscription state rather than mounting ~/.codex
    # directly. Falls back to the host dir :ro if no tempdir was
    # prepared (smoke tests, dev).
    if codex_state_dir:
        mounts += ["-v", f"{codex_state_dir}:/home/node/.codex:rw"]
    else:
        mounts += ["-v", f"{home}/.codex:/home/node/.codex:ro"]

    if claude_creds_path:
        # Claude's plaintext OAuth fallback. Extracted from Keychain by
        # refresh_claude_credentials() and chmod'd 0644 so the non-root
        # container user can read it.
        mounts += [
            "-v", f"{claude_creds_path}:/home/node/.claude/.credentials.json:ro",
        ]

    # ~/.claude.json holds feature flags (cachedGrowthBookFeatures) that
    # gate behavior including --json-schema honoring. Without it claude
    # silently returns unstructured text even when a schema is passed.
    #
    # We mount a SNAPSHOT of the host file, not the host file directly.
    # Claude Code's writes to .claude.json are not always atomic (see
    # upstream issues #29051, #29217, #29250); a concurrent container
    # read observing a mid-write state gets "Unterminated string in
    # JSON" and refuses to proceed. prepare_claude_config() does a
    # validated snapshot + retry at run start to sidestep this.
    if claude_config_path:
        mounts += ["-v", f"{claude_config_path}:/home/node/.claude.json:ro"]
    else:
        # Fallback: mount the host file directly (smoke tests, dev).
        claude_config = Path(home) / ".claude.json"
        if claude_config.exists():
            mounts += ["-v", f"{claude_config}:/home/node/.claude.json:ro"]
    if workspace_host_path:
        os.makedirs(workspace_host_path, exist_ok=True)
        mounts += ["-v", f"{workspace_host_path}:/workspace:rw"]

    hardening = [
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=8192",
        f"--memory={memory}",
        f"--memory-swap={memory}",
        f"--cpus={cpus}",
    ]
    if not allow_network:
        # Close network egress so the sandbox shell can't fetch a benchmark
        # mirror directly (bypassing the web_search blocklist). Valid only when
        # the model runs host-side (reasoning_agent); an in-container CLI
        # (claude/codex) would lose API access -- harness._prepare_sandbox_run
        # guards that. Also disables runtime `pip install`.
        hardening.append("--network=none")

    # Container name helps operators see what's running with `docker ps`.
    # Docker name regex is [a-zA-Z0-9][a-zA-Z0-9_.-]+ so we sanitize the
    # pid in case it has characters Docker rejects.
    safe_pid = "".join(c if c.isalnum() or c in "_.-" else "_" for c in pid)[:64]
    name = f"reasoning-eval-{run_id}-{safe_pid}"[:128]

    cmd = [
        "docker", "run", "-d", "--rm",
        "--name", name,
        *mounts,
        *hardening,
        "-e", "CLAUDE_CODE_MAX_OUTPUT_TOKENS=120000",
        "--workdir", "/workspace",
        image,
        "sleep", "infinity",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(
            f"docker run failed (returncode={r.returncode}): "
            f"stderr={r.stderr.strip()!r}"
        )
    container_id = r.stdout.strip()
    if not container_id:
        raise RuntimeError("docker run returned empty container id")
    return container_id


def stop_sandbox(container_id: str) -> None:
    """Stop the sandbox (auto-removes thanks to --rm). Idempotent."""
    try:
        subprocess.run(
            ["docker", "stop", "-t", "2", container_id],
            capture_output=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass


def exec_prefix(container_id: str) -> list[str]:
    """The argv prefix that turns a CLI invocation into a `docker exec`
    in the named container. Caller appends the actual CLI args."""
    return ["docker", "exec", container_id]


def container_tool_versions(image: str = DEFAULT_IMAGE) -> dict:
    """Run a throwaway container to capture the versions of the tools
    actually installed inside the image. These may drift from the host
    versions; the paper methodology should cite these, not the host.

    Returns a dict with one string per tool (or None on failure).
    """
    def _one(argv: list[str]) -> str | None:
        try:
            r = subprocess.run(
                ["docker", "run", "--rm", "--entrypoint", argv[0], image, *argv[1:]],
                capture_output=True, text=True, timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None
        if r.returncode != 0:
            return None
        return (r.stdout or r.stderr).strip()

    return {
        "claude": _one(["claude", "--version"]),
        "codex": _one(["codex", "--version"]),
        "python3": _one(["python3", "--version"]),
        "pari_gp": _one(["gp", "--version-short"]),
        "node": _one(["node", "--version"]),
    }


def pip_freeze_in_container(container_id: str) -> str | None:
    """Capture `pip freeze` inside a running sandbox, returning the
    output as a string. Records the exact Python env state (including
    any packages an agent `pip install`-ed at runtime)."""
    try:
        r = subprocess.run(
            ["docker", "exec", container_id, "pip", "freeze"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def copy_from_container(container_id: str, src: str, dst: str) -> bool:
    """`docker cp <container>:<src> <dst>`. Returns True on success."""
    try:
        r = subprocess.run(
            ["docker", "cp", f"{container_id}:{src}", dst],
            capture_output=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return r.returncode == 0


def copy_to_container(container_id: str, src: str, dst: str) -> bool:
    """Restore a host directory as UID 1000 inside a hardened container."""
    source = Path(src)
    if not source.is_dir():
        return False
    try:
        mkdir = subprocess.run(
            [
                "docker", "exec", "-u", "1000:1000", container_id,
                "mkdir", "-p", dst,
            ],
            capture_output=True,
            timeout=30,
        )
        if mkdir.returncode != 0:
            return False
        # `docker cp` creates root-owned files. A container started with
        # --cap-drop=ALL cannot chown them afterward, even via `docker exec -u
        # 0`. Stream a tar archive into a process running as the runtime user
        # instead, so restored files are writable without adding CAP_CHOWN.
        with tempfile.TemporaryFile() as archive:
            packed = subprocess.run(
                ["tar", "-C", str(source), "-cf", "-", "."],
                stdout=archive,
                stderr=subprocess.PIPE,
                timeout=60,
            )
            if packed.returncode != 0:
                return False
            archive.seek(0)
            restored = subprocess.run(
                [
                    "docker", "exec", "-i", "-u", "1000:1000",
                    container_id, "tar", "-C", dst, "-xf", "-",
                ],
                stdin=archive,
                capture_output=True,
                timeout=60,
            )
        return restored.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def snapshot_container_directory(container_id: str, src: str, dst: str) -> bool:
    """Atomically replace a host snapshot with one container directory."""
    destination = Path(dst)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True, exist_ok=True)
    if not copy_from_container(container_id, src.rstrip("/") + "/.", str(temporary)):
        shutil.rmtree(temporary, ignore_errors=True)
        return False
    shutil.rmtree(destination, ignore_errors=True)
    os.replace(temporary, destination)
    return True


def inspect_container(container_id: str) -> dict | None:
    """Run `docker inspect <container>` and return the parsed JSON.

    Captures the authoritative exit state — actual returncode, start/
    stop timestamps, OOMKilled flag, actual resource limits applied,
    restart count, and any docker-daemon-level error. If our pipeline
    sees a mysterious nonzero call returncode, this is where we'd find
    out whether the container died of OOM vs something else.

    Returns None on any failure — docker inspect is purely diagnostic,
    the run should continue without it.
    """
    try:
        r = subprocess.run(
            ["docker", "inspect", container_id],
            capture_output=True, text=True, timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    # `docker inspect` returns a list. For a single ID we want the
    # first (only) element.
    return data[0] if isinstance(data, list) and data else None


def dpkg_list_in_container(container_id: str) -> str | None:
    """Capture the installed apt packages inside the container (the
    system-level equivalent of `pip freeze`). Complements
    pip_freeze_in_container — one run of both tells you the complete
    package state for reproducibility."""
    try:
        r = subprocess.run(
            ["docker", "exec", container_id, "dpkg", "-l"],
            capture_output=True, text=True, timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout
