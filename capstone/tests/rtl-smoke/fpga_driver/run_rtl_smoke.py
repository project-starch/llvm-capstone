#!/usr/bin/env python3
"""End-to-end agent-driven rtl-smoke run over the FPGA web console.

Replaces the human clicking ~5 buttons per run (README.md "Run (human-driven)")
with a headless sequence:

    upload fw_payload.bin  ->  load (JTAG 0x80000000)  ->  reset, await shell
    ->  run borrow_cost + the three revoke_cost .user/.dom pairs over UART
    ->  harvest the RESULT lines  ->  run-revoke-cost-fpga-qemu.sh --parse-uart

The board actions go through FpgaConsole (config.py holds the protocol). The
guest commands + RESULT/marker strings are the same ones the human run and the
QEMU validation use, so the harvested capture drops straight into the existing
parser.

Because config.PROTOCOL_SOURCE is still "placeholder", this refuses to hit a real
board unless --allow-unverified is given (use it only against the mock server, or
after wiring config.py from the real client JS). It never fetches a URL or seeks
credentials on its own -- the token'd URL is a required CLI argument.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
# Allow running as a script (python run_rtl_smoke.py) or as a module.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fpga_driver import config as C  # type: ignore
    from fpga_driver.fpga_console import FpgaConsole, ActionTimeout  # type: ignore
else:  # pragma: no cover
    from . import config as C
    from .fpga_console import FpgaConsole, ActionTimeout

REMOTE_DIR = "/root/rtl-smoke"
# The Capstone kernel module. The .doms open /dev/capstone (via libcapstone's
# capstone_init); that node is created by this module, and the fpga image does
# NOT auto-load it at boot -- so a module image must insmod it first (else
# "failed to initialize Capstone").
#
# HARDWARE NOTE (real FPGA): `insmod` of ANY module *hangs this board* -- the
# kernel module loader's icache/text-patch path does not complete on this CVA6
# (same class as the RFENCE boot gap). Build capstone BUILT-IN (CONFIG in-tree,
# obj-y) so /dev/capstone exists at boot with no runtime insmod. See the
# fpga-gdb-boot-run history note.
CAPSTONE_MODULE = "/capstone.ko"
# Load address our OpenSBI fw_payload runs from (M-mode entry), and the address
# the (bypassed) bootrom normally places the board DTB -- OpenSBI passes it as a1.
BOOT_LOAD_ADDR = 0x80000000
DTB_ADDR = 0x82200000

# The runs the human sequence performs (README.md steps 3-4). Each entry:
#   (label, controller.user, payload.dom, done_marker_key)
RUNS = [
    ("borrow", "borrow_cost_fpga.user", "borrow_cost_fpga.dom", "borrow_complete"),
    ("bump", "revoke_cost_fpga.user", "revoke_cost_fpga_bump.dom", "measurement_complete"),
    ("norevoke", "revoke_cost_fpga.user", "revoke_cost_fpga_norevoke.dom", "measurement_complete"),
    ("revoke", "revoke_cost_fpga.user", "revoke_cost_fpga_revoke.dom", "measurement_complete"),
]


def load_capstone_module(console: FpgaConsole, module: str = CAPSTONE_MODULE) -> str:
    """insmod the Capstone module and confirm /dev/capstone appeared. Raises if
    the module fails to load (e.g. a vermagic mismatch -- a UP image must ship a
    capstone.ko built against the same SMP setting)."""
    cmd = (f"insmod {module} && ls /dev/capstone && echo CAPSTONE_MOD_OK "
           f"|| echo CAPSTONE_MOD_FAIL")
    out = console.run_command(cmd, C.UART["module_loaded"], timeout=30.0)
    if "CAPSTONE_MOD_FAIL" in out or "CAPSTONE_MOD_OK" not in out:
        raise RuntimeError(f"insmod {module} failed (no /dev/capstone):\n{out}")
    return out


def throttled_send(console: FpgaConsole, text: str, delay: float = 0.05) -> None:
    """Type `text` one char at a time. The board's UART RX FIFO overruns on a
    bulk write and silently drops characters (corrupting commands, e.g.
    `/capstone.ko` -> `/capstoneko`), so long commands must be throttled. (Even
    throttled the RX is occasionally lossy -- callers should verify/retry.)"""
    for ch in text:
        console._emit("uart_send", text=ch)
        time.sleep(delay)


def login_root(console: FpgaConsole, tries: int = 4) -> bool:
    """After the login prompt, log in as root (no password on this image) and
    CONFIRM a live shell with an echo probe whose marker can't appear in the
    command echo (quote-split RDY''OK)."""
    console.wait_uart(r"login:", timeout=180.0)
    for _ in range(tries):
        console._emit("uart_send", text="root\r")
        time.sleep(1.5)
        start = len(console.uart_text)
        console._emit("uart_send", text="echo RDY''OK\r")
        try:
            console.wait_uart(r"RDYOK", timeout=8.0, search_from=start)
            return True
        except ActionTimeout:
            console._emit("uart_send", text="\r")
            time.sleep(1.5)
    return False


def boot_via_gdb(console: FpgaConsole, image_name: str,
                 *, host_image: str, host_dtb: str,
                 load_addr: int = BOOT_LOAD_ADDR, dtb_addr: int = DTB_ADDR) -> None:
    """Boot our image via the GDB session instead of reset-board (VERIFIED live).

    reset-board makes the bootrom reload the SPI-resident firmware, clobbering a
    JTAG-loaded image. Instead: `monitor reset halt` halts the hart at the reset
    vector (clean M-mode) WITHOUT running the bootrom, then OpenOCD (which runs on
    the host) loads our image + the board DTB from host files straight into DRAM
    while halted, we set the entry registers, and `continue`. Non-persistent.

    `host_image` / `host_dtb` are paths on the OpenOCD host (relative to its CWD,
    the console app dir -- uploads land in `images/`). Because the bootrom is
    bypassed, the DTB it normally places at `dtb_addr` must be loaded here and
    passed in a1, else Linux has no device tree (no console, no boot)."""
    prompt = C.GDB_PROMPT
    console.gdb_start()
    console.gdb_cmd("monitor reset halt", prompt, timeout=60.0)
    # OpenOCD reads the host files and writes DRAM while the core is halted
    # (~2 min for the 15 MB image). No console load-image (that poisons the TAP
    # for the gdb attach) and no clobber (halted -> bootrom never runs).
    console.gdb_cmd(f"monitor load_image {host_image} {load_addr:#x} bin",
                    prompt, timeout=240.0)
    console.gdb_cmd(f"monitor load_image {host_dtb} {dtb_addr:#x} bin",
                    prompt, timeout=40.0)
    console.gdb_cmd(f"set $pc = {load_addr:#x}", prompt)
    console.gdb_cmd("set $a0 = 0", prompt)          # hartid
    console.gdb_cmd(f"set $a1 = {dtb_addr:#x}", prompt)  # DTB pointer
    # Resume; `continue` doesn't return a prompt, so fire it and watch the UART.
    console._emit("gdb_input", text="continue\n")
    if not login_root(console):
        raise RuntimeError("gdb-boot reached login but could not confirm a shell")
    # Detach: the .doms do Capstone domain switches that desync an attached debug
    # session ("packet queue is empty, aborting"). gdb_stop leaves Linux running.
    console.gdb_stop()
    time.sleep(4.0)


def boot_board(console: FpgaConsole, image: Path, image_name: str,
               *, boot_method: str = "reset", do_upload: bool = True,
               dtb: Optional[Path] = None, host_dir: str = "images") -> None:
    """Get our image to a Linux shell. `reset` = upload+load+reset-board (works
    only if the board boots the JTAG-loaded image); `gdb` = the self-serve
    GDB-driven boot that bypasses the SPI-reload reset (needs the board `dtb`,
    since the bypassed bootrom no longer supplies one)."""
    if do_upload:
        console.upload_boot_image(image_name, str(image))
    if boot_method == "gdb":
        if dtb is None:
            raise ValueError("boot_method='gdb' needs the board dtb= (the "
                             "bypassed bootrom no longer places it)")
        dtb_name = dtb.name
        if do_upload:
            console.upload_boot_image(dtb_name, str(dtb))
        boot_via_gdb(console, image_name,
                     host_image=f"{host_dir}/{image_name}",
                     host_dtb=f"{host_dir}/{dtb_name}")
    elif boot_method == "reset":
        console.load_boot_image(image_name)
        console.reset(wait_prompt=True)
    else:
        raise ValueError(f"unknown boot_method: {boot_method!r}")


def run_smoke(
    console: FpgaConsole,
    image: Path,
    image_name: str,
    *,
    do_upload: bool = True,
    remote_dir: str = REMOTE_DIR,
    boot_method: str = "reset",
    dtb: Optional[Path] = None,
    load_module: bool = True,
) -> str:
    """Drive one full sweep; return the concatenated UART capture (RESULT lines
    included). `console` must already be connected. `load_module=False` for a
    built-in-capstone image (/dev/capstone at boot; insmod hangs this board)."""
    boot_board(console, image, image_name,
               boot_method=boot_method, do_upload=do_upload, dtb=dtb)
    capture: List[str] = []
    if load_module:
        # A module image needs /dev/capstone loaded before any pair.
        capture.append(f"# === insmod {CAPSTONE_MODULE} ===\n"
                       f"{load_capstone_module(console)}\n")
    for label, user, dom, marker_key in RUNS:
        cmd = f"{remote_dir}/{user} {remote_dir}/{dom}"
        marker = C.UART[marker_key]
        out = console.run_command(cmd, marker, timeout=240.0)
        capture.append(f"# === {label}: {cmd} ===\n{out}\n")
    return "".join(capture)


def parse_uart(capture_path: Path) -> int:
    """Hand the harvested capture to the existing bundled parser."""
    script = Path(__file__).resolve().parent.parent / "run-revoke-cost-fpga-qemu.sh"
    return subprocess.call(["bash", str(script), "--parse-uart", str(capture_path)])


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url",
                    help="full token'd console URL, e.g. https://fpga.corank.info/<token>/ "
                         "(kept out of the repo; never logged). Required unless --parse-only.")
    ap.add_argument("--token", default=os.environ.get("FPGA_TOKEN"),
                    help="access token, if the protocol echoes it in the handshake "
                         "(config.CONNECT.auth_key); defaults to $FPGA_TOKEN")
    ap.add_argument("--image", type=Path,
                    help="fw_payload.bin to upload+load. Required unless --parse-only.")
    ap.add_argument("--image-name", default=None,
                    help="name to store the image under (default: the file's basename)")
    ap.add_argument("--no-upload", action="store_true",
                    help="skip upload; the named image is already on the console")
    ap.add_argument("--boot-method", choices=("reset", "gdb"), default="reset",
                    help="how to start the image: 'reset' (upload+load+reset-board) "
                         "or 'gdb' (self-serve GDB boot that bypasses the SPI-reload "
                         "reset). Use 'gdb' when the board boots resident SPI firmware "
                         "on reset instead of the JTAG-loaded image.")
    ap.add_argument("--remote-dir", default=REMOTE_DIR,
                    help=f"where the .user/.dom artifacts live in the guest (default {REMOTE_DIR})")
    ap.add_argument("--capture-out", type=Path, default=None,
                    help="write the harvested UART capture here (default: a temp file)")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="run even though config.PROTOCOL_SOURCE is 'placeholder' "
                         "(mock-server / bring-up only -- see PROTOCOL.md)")
    ap.add_argument("--no-lock", action="store_true",
                    help="do not take the auto-shutdown Lock while running "
                         "(default: take it, release it when done)")
    ap.add_argument("--ignore-users", action="store_true",
                    help="run even if other clients are connected to the shared "
                         "board (default: back off if >1 user is present)")
    ap.add_argument("--parse-only", type=Path, default=None,
                    help="skip the board; just run --parse-uart on this capture file")
    args = ap.parse_args(argv)

    if args.parse_only:
        return parse_uart(args.parse_only)

    if not args.url:
        ap.error("--url is required (unless --parse-only)")
    if not args.image:
        ap.error("--image is required (unless --parse-only)")
    if not args.image.is_file():
        ap.error(f"image not found: {args.image}")
    image_name = args.image_name or args.image.name

    console = FpgaConsole(
        args.url, token=args.token,
        allow_unverified=args.allow_unverified,
        logger=lambda m: print(f"[fpga] {m}", file=sys.stderr),
    )
    console.connect()
    locked = False
    try:
        # Good-citizen check on the shared board: back off if someone else is on.
        users = console.user_count()
        if users is not None and users > 1 and not args.ignore_users:
            print(f"[fpga] {users} clients connected -- the board is shared and "
                  f"in use; backing off. Re-run with --ignore-users to override.",
                  file=sys.stderr)
            return 3
        if not args.no_lock:
            console.lock()
            locked = True
            print("[fpga] took the auto-shutdown Lock", file=sys.stderr)
        console.power(True)
        capture = run_smoke(
            console, args.image, image_name,
            do_upload=not args.no_upload, remote_dir=args.remote_dir,
            boot_method=args.boot_method,
        )
    finally:
        if locked:
            console.unlock()
            print("[fpga] released the Lock", file=sys.stderr)
        console.close()

    if args.capture_out:
        out_path = args.capture_out
        out_path.write_text(capture)
    else:
        fd, tmp = tempfile.mkstemp(prefix="fpga-uart-", suffix=".txt")
        os.close(fd)
        out_path = Path(tmp)
        out_path.write_text(capture)
    print(f"[fpga] UART capture -> {out_path}", file=sys.stderr)

    return parse_uart(out_path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
