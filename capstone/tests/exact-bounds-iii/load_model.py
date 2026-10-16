#!/usr/bin/env python3
"""Load the project's bounds encoder oracle from capstone-ariane at a PINNED commit, rather than
keeping a copy here that would silently drift from the RTL it models."""
import importlib.util, pathlib, subprocess, sys, tempfile

REV = "d93a8d66b"          # the bitstream's RTL: one scheme, R-54 renormalisation
PATH = "verif/tests/custom/capstone/bounds-model.py"

def load():
    import os
    env = os.environ.get("CAPSTONE_ARIANE")
    sub = None
    if env:
        if not pathlib.Path(env).is_dir():
            sys.exit("ERROR: CAPSTONE_ARIANE=%s is not a directory" % env)
        sub = pathlib.Path(env)
    else:
        root = pathlib.Path(__file__).resolve()
        for p in root.parents:
            if (p / "capstone" / "capstone-ariane").is_dir():
                sub = p / "capstone" / "capstone-ariane"; break
    if sub is None:
        sys.exit("ERROR: capstone-ariane not found above %s and CAPSTONE_ARIANE unset -- "
                 "this is an error, not a zero" % pathlib.Path(__file__).resolve())
    try:
        src = subprocess.run(["git", "-C", str(sub), "show", "%s:%s" % (REV, PATH)],
                             capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        sys.exit("ERROR: cannot read %s:%s in %s (%s)" % (REV, PATH, sub, e.stderr.strip()))
    if "def compress" not in src:
        sys.exit("ERROR: %s:%s does not define compress() -- wrong file or revision" % (REV, PATH))
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False); f.write(src); f.close()
    spec = importlib.util.spec_from_file_location("bounds_model", f.name)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

if __name__ == "__main__":
    m = load()
    print("loaded bounds-model.py from capstone-ariane %s" % REV)
    print("exact small:", m.decompress(*m.compress(0x10000000, 0x10000040)[:3], 0x10000000))
