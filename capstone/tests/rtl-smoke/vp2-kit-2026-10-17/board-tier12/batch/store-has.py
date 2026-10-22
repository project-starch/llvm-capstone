#!/usr/bin/env python3
"""store-has.py <name> [<name> ...]: exit 0 if the console's bitstream store lists every name, 1 if any is missing,
2 on any failure. Read-only (GET /api/bitstreams); the console URL comes from CAPSTONE_FPGA_URL_FILE or the default
secrets file and is never printed. Run only when no board session is live on that console."""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "rtl-smoke"))
try:
    from fpga_driver.fpga_console import FpgaConsole
    url = pathlib.Path(os.environ.get("CAPSTONE_FPGA_URL_FILE") or (pathlib.Path.home() / ".claude-kisp/secrets/fpga-console-url")).read_text().strip()
    c = FpgaConsole(url); c.connect()
    try:
        r = c._http.get(f"{c._api_base}/bitstreams", timeout=30); r.raise_for_status(); j = r.json()
        items = j if isinstance(j, list) else (j.get("bitstreams") or j.get("files") or [])
        names = [x if isinstance(x, str) else (x.get("name") or x.get("filename")) for x in items]
    finally:
        c.close()
except Exception as e:
    print(f"STORE ERROR {type(e).__name__}"); sys.exit(2)
if not names:
    print("STORE NO DATA: the store listing is empty"); sys.exit(2)
miss = [n for n in sys.argv[1:] if n not in names]
print(f"STORE {len(names)} names; wanted {sys.argv[1:]}; missing {miss}")
sys.exit(1 if miss else 0)
