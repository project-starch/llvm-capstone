#!/usr/bin/env python3
"""Load the project's bounds encoder oracle from capstone-ariane at a PINNED commit, rather than
keeping a copy here that would silently drift from the RTL it models.

A WORKTREE HAS NO SUBMODULE CHECKOUT: `capstone/capstone-ariane` exists there as an EMPTY
directory, so "the path is a directory" is not enough -- the revision has to resolve. This tries
every candidate, verifies the revision in each, and falls back to the main clone via
`git rev-parse --git-common-dir`. It exits non-zero and names every candidate it tried rather than
returning an empty result, because a missing oracle must read as an error and not as a zero.
"""
import importlib.util, os, pathlib, subprocess, sys, tempfile

REV = "d93a8d66b"          # the bitstream's RTL: one scheme, R-54 renormalisation
PATH = "verif/tests/custom/capstone/bounds-model.py"


def _candidates():
    seen = []
    env = os.environ.get("CAPSTONE_ARIANE")
    if env:
        seen.append(pathlib.Path(env))
    here = pathlib.Path(__file__).resolve()
    for p in here.parents:
        c = p / "capstone" / "capstone-ariane"
        if c not in seen:
            seen.append(c)
    # From a worktree the submodule is unchecked out; the main clone has it.
    try:
        common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                cwd=str(here.parent), capture_output=True, text=True, check=True
                                ).stdout.strip()
        if common:
            main = pathlib.Path(common).parent / "capstone" / "capstone-ariane"
            if main not in seen:
                seen.append(main)
    except (subprocess.CalledProcessError, OSError):
        pass
    return seen


def _read(sub):
    """The file's text from `sub` at REV, or None if that repo cannot serve it."""
    if not sub.is_dir():
        return None
    try:
        subprocess.run(["git", "-C", str(sub), "cat-file", "-e", REV + "^{commit}"],
                       capture_output=True, check=True)
        out = subprocess.run(["git", "-C", str(sub), "show", "%s:%s" % (REV, PATH)],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return None
    return out if "def compress" in out else None


def load():
    tried = []
    for sub in _candidates():
        tried.append(str(sub))
        src = _read(sub)
        if src is None:
            continue
        f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        f.write(src); f.close()
        spec = importlib.util.spec_from_file_location("bounds_model", f.name)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        m.__oracle_source__ = "%s at %s:%s" % (sub, REV, PATH)
        return m
    sys.exit("ERROR: no capstone-ariane could serve %s:%s. Tried:\n  %s\n"
             "Set CAPSTONE_ARIANE to a checkout that has that revision. "
             "(A git worktree's submodule directory is EMPTY -- use the main clone.)"
             % (REV, PATH, "\n  ".join(tried)))


if __name__ == "__main__":
    m = load()
    print("loaded %s" % m.__oracle_source__)
    iE, B, T, E = m.compress(0x10000000, 0x10000040)
    print("self-check, exact small region: %s" % (m.decompress(iE, B, T, 0x10000000),))
