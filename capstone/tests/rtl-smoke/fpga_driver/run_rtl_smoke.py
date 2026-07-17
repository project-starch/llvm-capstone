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
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
# Allow running as a script (python run_rtl_smoke.py) or as a module.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fpga_driver import config as C  # type: ignore
    from fpga_driver.fpga_console import FpgaConsole  # type: ignore
else:  # pragma: no cover
    from . import config as C
    from .fpga_console import FpgaConsole

REMOTE_DIR = "/root/rtl-smoke"
# The Capstone kernel module. The .doms open /dev/capstone (via libcapstone's
# capstone_init); that node is created by this module, and the fpga image does
# NOT auto-load it at boot -- so the sweep must insmod it first, exactly like the
# QEMU domain-smoke does. Missing this yields "failed to initialize Capstone".
CAPSTONE_MODULE = "/capstone.ko"
# Load address our OpenSBI fw_payload runs from (M-mode entry).
BOOT_LOAD_ADDR = 0x80000000

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


def boot_via_gdb(console: FpgaConsole, image_name: str,
                 *, load_addr: int = BOOT_LOAD_ADDR) -> None:
    """Boot our image using the console's GDB session instead of reset-board.

    reset-board makes the bootrom reload the SPI-resident firmware, clobbering
    our JTAG-loaded DRAM image. The debug module's `monitor reset halt` instead
    halts the hart at the reset vector (clean M-mode) WITHOUT running the bootrom;
    we then point the PC at our already-loaded image and `continue`. Non-persistent
    (flashes nothing).

    The exact OpenOCD ordering (load-image vs gdb_start attach, restore vs the
    console JTAG load) is worked out against the live session -- see the
    fpga-gdb-boot history note. This is the recipe the driver drives; adjust the
    command list here if the live session needs a different order."""
    prompt = C.GDB_PROMPT
    # 1. JTAG-load our image into DRAM (no reset -> bootrom does not reload SPI).
    console.load_boot_image(image_name)
    # 2. Attach OpenOCD + gdb-multiarch.
    console.gdb_start()
    # 3. Halt at the reset vector via the debug module (not the reset-board button).
    console.gdb_cmd("monitor reset halt", prompt, timeout=60.0)
    # 4. Enter at our OpenSBI (a0=hartid; the fpga OpenSBI embeds its own DTB).
    console.gdb_cmd(f"set $pc = {load_addr:#x}", prompt)
    console.gdb_cmd("set $a0 = 0", prompt)
    # 5. Resume. `continue` does not return a (gdb) prompt (target runs), so fire
    #    it and watch the UART for OpenSBI -> Linux -> the shell login.
    console._emit("gdb_input", text="continue\n")
    console.wait_uart(C.UART["login_prompt"], timeout=180.0)


def boot_board(console: FpgaConsole, image: Path, image_name: str,
               *, boot_method: str = "reset", do_upload: bool = True) -> None:
    """Get our image to a Linux shell. `reset` = upload+load+reset-board (works
    only if the board boots the JTAG-loaded image); `gdb` = the self-serve
    GDB-driven boot that bypasses the SPI-reload reset."""
    if do_upload:
        console.upload_boot_image(image_name, str(image))
    if boot_method == "gdb":
        boot_via_gdb(console, image_name)
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
) -> str:
    """Drive one full sweep; return the concatenated UART capture (RESULT lines
    included). `console` must already be connected."""
    boot_board(console, image, image_name,
               boot_method=boot_method, do_upload=do_upload)
    capture: List[str] = []
    # The .doms need /dev/capstone -- load the module before running any pair.
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
