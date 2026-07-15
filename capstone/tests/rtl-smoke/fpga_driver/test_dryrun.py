#!/usr/bin/env python3
"""Offline dry-run: drive the full scaffold against the mock Socket.IO server.

No board, no network beyond localhost. Proves the transport, the five actions,
the UART cross-chunk marker matching, the end-to-end sweep, and that the
harvested RESULT lines parse into the paper breakdown. Run under a Python that
has python-socketio + aiohttp (see README.md "Offline test"):

    python test_dryrun.py

Exits 0 on success. Also exposes test_* functions for pytest, if available.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `import fpga_driver` works

from fpga_driver.fpga_console import FpgaConsole  # noqa: E402
from fpga_driver import run_rtl_smoke  # noqa: E402
from fpga_driver import config as C  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port(port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"mock server did not come up on :{port}")


class _MockServer:
    def __init__(self) -> None:
        self.port = _free_port()
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "_MockServer":
        self.proc = subprocess.Popen(
            [sys.executable, str(HERE / "mock_server.py"),
             "--host", "127.0.0.1", "--port", str(self.port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _wait_port(self.port)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def run() -> None:
    tmp_img = HERE / ".dryrun_fw_payload.bin"
    tmp_img.write_bytes(b"\x7fELF" + b"\x00" * 4096)  # tiny fake image
    capture_path = HERE / ".dryrun_capture.txt"
    try:
        with _MockServer() as srv:
            console = FpgaConsole(srv.url, allow_unverified=True,
                                  logger=lambda m: print(f"[drv] {m}"))
            console.connect(timeout=10)
            frame = ""
            users = None
            try:
                # good-citizen helpers: user count + the auto-shutdown Lock
                users = console.user_count()
                console.lock()
                console.power(True)

                # -- the five actions, exercised individually --
                # 1: upload (multipart) + load (load_state loading->done)
                console.upload_boot_image("fw.bin", str(tmp_img))
                console.load_boot_image("fw.bin")
                # 2: reset + wait for the shell prompt (cross-chunk marker)
                console.reset(wait_prompt=True, prompt_timeout=15)
                # 3: read UART until a marker -- done inside run_command below
                capture = run_rtl_smoke.run_smoke(
                    console, tmp_img, "fw.bin", do_upload=False)
                # 4: toggle a virtual switch (verify via switch_state)
                console.set_switch(1, True)
                # 5: trace dump (POST trace-start -> trace_result text)
                frame = console.trace_dump(timeout=10)
            finally:
                console.unlock()
                console.close()

        # -- assertions --
        assert "buildroot login" in console.uart_text or "# " in capture, \
            "no shell prompt seen"
        for needle in ("mode=bump  alloc_free=7",
                       "mode=norevoke  alloc_free=60",
                       "mode=revoke  alloc_free=65",
                       "borrow-cost-fpga: RESULT vs-raw"):
            assert needle in capture, f"missing RESULT fragment: {needle!r}"
        assert frame and "end of dump" in frame, \
            f"trace_dump did not return the dump text: {frame!r}"
        assert users == 1, f"expected user_count 1, got {users!r}"

        capture_path.write_text(capture)
        rc = run_rtl_smoke.parse_uart(capture_path)
        assert rc == 0, f"--parse-uart exited {rc}"
        print("\nOK: dry-run passed (5 actions + end-to-end sweep + parse).")
    finally:
        tmp_img.unlink(missing_ok=True)
        capture_path.unlink(missing_ok=True)


def test_dryrun() -> None:  # pytest entry point
    run()


if __name__ == "__main__":
    run()
