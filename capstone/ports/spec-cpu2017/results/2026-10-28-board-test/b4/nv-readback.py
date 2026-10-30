#!/usr/bin/env python3
"""nv-readback.py <name> <sha256>: read the console's resident-bitstream record (flash_state.nv_bitstream_name and, when
the console exposes it, nv_bitstream_sha256) on a fresh session, read-only: no power, lock or flash call.
Exit 0: the name matches AND the sha256 matches; 3: the name matches and the console exposes no sha256; 1: a mismatch;
2: no data. The console URL comes from CAPSTONE_FPGA_URL_FILE and is never printed."""
import os, sys, time, pathlib
sys.path.insert(0, "/mnt/hdd/" + os.environ["USER"] + "/capstone/wt-spec/capstone/tests/rtl-smoke")
from fpga_driver.fpga_console import FpgaConsole  # noqa: E402

if len(sys.argv) != 3:
    sys.exit(__doc__)
name, sha = sys.argv[1], sys.argv[2].lower()
url = pathlib.Path(os.environ["CAPSTONE_FPGA_URL_FILE"]).read_text().strip()
fs = {}
try:
    c = FpgaConsole(url); c.connect()
    try:
        end = time.time() + 30
        while time.time() < end:
            with c._cond:
                fs = dict(c._state.get("flash_state") or {})
            if fs.get("nv_bitstream_name") is None:
                try:
                    r = c._http.get(f"{c._api_base}/state", timeout=30)
                    if r.ok:
                        fs = dict((r.json() or {}).get("flash_state") or {})
                except Exception:
                    pass
            if fs.get("nv_bitstream_name") is not None:
                break
            time.sleep(2)
    finally:
        c.close()
except Exception as e:
    print(f"NVRB ERROR {type(e).__name__}"); sys.exit(2)
rn, rs = fs.get("nv_bitstream_name"), (fs.get("nv_bitstream_sha256") or "").lower()
print(f"NVRB name {rn!r} sha256 {rs or 'NOT EXPOSED'}")
if rn is None:
    sys.exit(2)
if rn != name or (rs and rs != sha):
    print("NVRB MISMATCH"); sys.exit(1)
sys.exit(0 if rs else 3)
