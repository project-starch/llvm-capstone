"""allruns.py IMAGE [CHECKER]: run a granule checker (default granule-strict.py) over EVERY load-carrying scalar run of
IMAGE, not only the runs the libcall twin removes. Needs rescan-nosplit.py's environment (C81_SCANNER_DIR, C81_OBJDUMP, C81_NM)."""
import sys
import os
HERE = os.path.dirname(os.path.abspath(__file__))
img = sys.argv[1]; checker = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "granule-strict.py")
src = open(os.path.join(HERE, "rescan-nosplit.py")).read().split("thr = int(sys.argv[3])")[0]
ns = {}; exec(src, ns)
g = {}; exec(open(checker).read().split('if __name__ == "__main__":')[0], g)
runs = ns["runs"](img, 16); hits = 0
if not runs: sys.exit(f"NO DATA: no runs in {img}")
for k, off, t in runs:
    f = g["flag"](t)
    if f: hits += 1; print("FLAG", f"{k[0]}+{off:#x}", f"{k[1]}B unit={k[2]}", "--", f, "\n     ", t[:240])
print(f"{img}: {len(runs)} load-carrying runs, {hits} flagged ({checker.split('/')[-1]})")
