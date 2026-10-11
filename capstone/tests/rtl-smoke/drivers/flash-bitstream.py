#!/usr/bin/env python3
"""Flash a bitstream onto the FPGA board. A FLASH IS ASK-FIRST: run this only with the lead's confirmation.

    python3 flash-bitstream.py <server-side name> <local .bit> <expected sha256>

ONE FpgaConsole throughout: check the local hash, upload via POST /api/bitstreams/upload if the name is
not registered, flash, power-cycle (mandatory: the flash writes SPI only), then read nv_bitstream_name
back on the SAME session. Exit 0 only if the readback names the flashed name. Used for R-42 (2026-09-25)
and R-43 v2 (2026-09-29). The console URL is read from the secrets file and never printed."""
import os, sys, time, pathlib, hashlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fpga_driver.fpga_console import FpgaConsole  # noqa: E402

if len(sys.argv) != 4:
    sys.exit("usage: flash-bitstream.py <server-side name> <local .bit> <expected sha256>")
NAME, SRC, SHA = sys.argv[1], sys.argv[2], sys.argv[3].lower()
URL = pathlib.Path(os.environ.get("CAPSTONE_FPGA_URL_FILE") or (pathlib.Path.home() / ".claude-kisp/secrets/fpga-console-url")).read_text().strip()

def log(m): print(f"[flash {time.strftime('%H:%M:%S')}] {m}", flush=True)

def nvbit(c, poll=20.0):
    end = time.time() + poll
    while True:
        with c._cond:
            v = (c._state.get("flash_state") or {}).get("nv_bitstream_name")
        if v is None:
            st = c.get_state() if hasattr(c, "get_state") else {}
            v = ((st or {}).get("flash_state") or {}).get("nv_bitstream_name")
        if v is not None or time.time() >= end:
            return v
        time.sleep(1.0)

def stored(c):
    r = c._http.get(f"{c._api_base}/bitstreams", timeout=30)
    r.raise_for_status(); j = r.json()
    items = j if isinstance(j, list) else (j.get("bitstreams") or j.get("files") or [])
    return [x if isinstance(x, str) else (x.get("name") or x.get("filename")) for x in items]

def main():
    h = hashlib.sha256(open(SRC, "rb").read()).hexdigest()
    if h != SHA:
        log(f"LOCAL HASH MISMATCH {h} -- abort"); return 1
    log("local sha256 matches the synth lane's sealed hash")
    c = FpgaConsole(URL); c.connect()
    locked = False
    try:
        log(f"resident before: {nvbit(c, 10)!r}")
        names = stored(c)
        if NAME not in names:
            log("not in the server store -- uploading via API")
            with open(SRC, "rb") as fh:
                r = c._http.post(f"{c._api_base}/bitstreams/upload", data={"name": NAME},
                                 files={"file": (NAME, fh)}, timeout=600)
            log(f"upload HTTP {r.status_code}")
            if not r.ok: return 1
            names = stored(c)
        if NAME not in names:
            log("NOT in the store after upload -- abort, no SPI write"); return 1
        log("present in the server store")
        c.power(True); time.sleep(15.0)
        c.lock(); locked = True
        log(f"flashing {NAME} (~90 s)")
        c.flash_bitstream(NAME)
        log("MANDATORY power cycle: off 8 s, on 15 s")
        c.power(False); time.sleep(8.0); c.power(True); time.sleep(15.0)
        rb = nvbit(c, 30)
        log(f"resident AFTER power cycle: {rb!r}")
        return 0 if rb == NAME else 1
    finally:
        try: c.power(False); log("powered off")
        except Exception as e: log(f"power off failed: {e}")
        if locked:
            try: c.unlock(); log("unlocked")
            except Exception as e: log(f"unlock failed: {e}")
        try: c.close()
        except Exception: pass

if __name__ == "__main__":
    rc = main(); print(f"rc={rc}", flush=True); os._exit(rc)
