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
import base64
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

# The runs the human sequence performs (README.md steps 3-4). Each entry:
#   (label, controller.user, payload.dom, done_marker_key)
RUNS = [
    ("borrow", "borrow_cost_fpga.user", "borrow_cost_fpga.dom", "borrow_complete"),
    ("bump", "revoke_cost_fpga.user", "revoke_cost_fpga_bump.dom", "measurement_complete"),
    ("norevoke", "revoke_cost_fpga.user", "revoke_cost_fpga_norevoke.dom", "measurement_complete"),
    ("revoke", "revoke_cost_fpga.user", "revoke_cost_fpga_revoke.dom", "measurement_complete"),
]


def run_smoke(
    console: FpgaConsole,
    image: Path,
    image_name: str,
    *,
    do_upload: bool = True,
    remote_dir: str = REMOTE_DIR,
) -> str:
    """Drive one full sweep; return the concatenated UART capture (RESULT lines
    included). `console` must already be connected."""
    if do_upload:
        data_b64 = base64.b64encode(image.read_bytes()).decode("ascii")
        console.upload_boot_image(image_name, data_b64, image.stat().st_size)
    console.load_boot_image(image_name)
    console.reset(wait_prompt=True)

    capture: List[str] = []
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
    ap.add_argument("--remote-dir", default=REMOTE_DIR,
                    help=f"where the .user/.dom artifacts live in the guest (default {REMOTE_DIR})")
    ap.add_argument("--capture-out", type=Path, default=None,
                    help="write the harvested UART capture here (default: a temp file)")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="run even though config.PROTOCOL_SOURCE is 'placeholder' "
                         "(mock-server / bring-up only -- see PROTOCOL.md)")
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
    try:
        console.power(True)
        capture = run_smoke(
            console, args.image, image_name,
            do_upload=not args.no_upload, remote_dir=args.remote_dir,
        )
    finally:
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
