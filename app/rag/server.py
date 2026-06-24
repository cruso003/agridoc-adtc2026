"""llama-server lifecycle for the bench runner.

Two launchers, selectable on the CLI:

  docker  (default) — reuses the existing `adtc-profiler:latest` image whose
                       entrypoint is llama-server, mounting the model dir. This
                       matches run_smollm2_sweep.sh so no new infra is needed.
  exec              — runs a native `llama-server` binary directly (for hosts
                       that have llama.cpp installed).

A model KEY (e.g. "llama-3.2-1b-q4") resolves to submissions/<key>/model/*.gguf.
Use as a context manager: the server is started, waited-on, and always stopped.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from . import config, llm


def resolve_gguf(model_key: str, submissions: str | Path = config.DEFAULT_SUBMISSIONS) -> Path:
    """Find the single GGUF for a model key under submissions/<key>/model/."""
    model_dir = Path(submissions) / model_key / "model"
    ggufs = sorted(model_dir.glob("*.gguf"))
    if not ggufs:
        raise FileNotFoundError(
            f"no .gguf for model '{model_key}' under {model_dir} "
            f"(expected submissions/<key>/model/<file>.gguf)"
        )
    if len(ggufs) > 1:
        print(f"[server] {model_key}: multiple GGUFs, using {ggufs[0].name}")
    return ggufs[0]


class LlamaServer:
    """Start/stop one llama-server for one GGUF; health-gate before use."""

    def __init__(
        self,
        model_key: str,
        launcher: str = "docker",
        port: int = 8080,
        submissions: str | Path = config.DEFAULT_SUBMISSIONS,
        image: str = "adtc-profiler:latest",
        server_bin: str = "llama-server",
        ctx_size: int = config.SERVER_CTX_SIZE,
        container_name: str = "rag-bench-server",
    ):
        self.model_key = model_key
        self.launcher = launcher
        self.port = port
        self.gguf = resolve_gguf(model_key, submissions)
        self.image = image
        self.server_bin = server_bin
        self.ctx_size = ctx_size
        self.container_name = container_name
        self.base_url = f"http://127.0.0.1:{port}"
        self._proc: subprocess.Popen | None = None

    # ── launch ────────────────────────────────────────────────────────────────
    def _start_docker(self) -> None:
        # Clean any stale container with our name first.
        subprocess.run(["docker", "rm", "-f", self.container_name],
                       capture_output=True, text=True)
        # Docker Desktop accepts a Windows absolute path with forward slashes.
        host_dir = str(self.gguf.parent).replace("\\", "/")
        cmd = [
            "docker", "run", "--rm", "-d",
            "--name", self.container_name,
            "-p", f"{self.port}:8080",
            "-v", f"{host_dir}:/model:ro",
            "--entrypoint", "llama-server",
            self.image,
            "-m", f"/model/{self.gguf.name}",
            "--host", "0.0.0.0", "--port", "8080",
            "--log-disable", "--ctx-size", str(self.ctx_size),
        ]
        print(f"[server] docker start {self.model_key} ({self.gguf.name}) on :{self.port}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"docker run failed: {res.stderr.strip()}")

    def _start_exec(self) -> None:
        cmd = [
            self.server_bin,
            "-m", str(self.gguf),
            "--host", "127.0.0.1", "--port", str(self.port),
            "--log-disable", "--ctx-size", str(self.ctx_size),
        ]
        print(f"[server] exec start {self.model_key} ({self.gguf.name}) on :{self.port}")
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)

    def start(self, ready_timeout: float = 240.0, poll: float = 3.0) -> None:
        if self.launcher == "docker":
            self._start_docker()
        elif self.launcher == "exec":
            self._start_exec()
        else:
            raise ValueError(f"unknown launcher: {self.launcher!r}")

        deadline = time.monotonic() + ready_timeout
        while time.monotonic() < deadline:
            if llm.health(self.base_url):
                print(f"[server] {self.model_key} healthy")
                return
            if self.launcher == "exec" and self._proc and self._proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early (code {self._proc.returncode})")
            time.sleep(poll)
        self.stop()
        raise TimeoutError(f"{self.model_key}: server not healthy within {ready_timeout}s")

    # ── teardown ──────────────────────────────────────────────────────────────
    def stop(self) -> None:
        if self.launcher == "docker":
            subprocess.run(["docker", "stop", self.container_name],
                           capture_output=True, text=True)
        elif self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        print(f"[server] {self.model_key} stopped")

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
