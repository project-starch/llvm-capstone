"""Running one application-domain image on a persistent Capstone VM, for corpus runners.

The sql-repros and c-repros runners each carried their own copy of these helpers; they live here
once. Nothing in this module judges anything -- see verdicts.py for that.
"""
import fcntl
import json
import re
import subprocess
import sys
from pathlib import Path

from verdicts import sha256

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "capstone/ports/common/application/run.py"


def vm(state, *argv, timeout=300):
    """`python -m capstone_vm --state STATE ARGV...`, output captured."""
    return subprocess.run([sys.executable, "-m", "capstone_vm", "--state", str(state), *argv],
                          cwd=str(REPO / "capstone/runtime/host"), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout)


def require_up(state):
    if "running" not in vm(state, "status").stdout:
        sys.exit("the VM is not up; bring it up first, this runner will not boot it")


def run_app(state, image, app_args, out_base, run_args=(), timeout=600):
    """Run IMAGE on the VM; returns (console text, result dict). The run's own output goes to
    OUT_BASE.out and the launcher's result to OUT_BASE.json. A timeout is written into the text
    as `[runner] TIMEOUT`, so a caller can never mistake it for a quiet run."""
    out_base = Path(out_base)
    result_path, log_path = out_base.with_suffix(".json"), out_base.with_suffix(".out")
    command = [sys.executable, str(RUN), "--state", str(state), "--cwd", "/tmp",
               "--result", str(result_path), *run_args, str(image), "--", *app_args]
    with log_path.open("w") as log:
        try:
            subprocess.run(command, cwd=str(RUN.parent), stdout=log, stderr=subprocess.STDOUT,
                           timeout=timeout)
        except subprocess.TimeoutExpired:
            log.write("\n[runner] TIMEOUT\n")
    text = log_path.read_text(errors="replace")
    result = {}
    if result_path.is_file() and result_path.stat().st_size:
        result = json.loads(result_path.read_text())
    return text, result


def share_lock(share, name):
    """One run at a time per share: two would stage into the same directories."""
    lock = open(Path(share) / f".{name}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another run holds the share's lock")
    return lock


def platform(state, compiler, runner):
    """The platform record every bundle carries: the VM's own identity hashes (capstone_vm records
    each file it boots), the compiler and the runner. Missing entries are said, not omitted."""
    config = json.loads((Path(state) / "config.json").read_text())
    files = {k: f.get("sha256") for k, f in config.get("identity", {}).get("files", {}).items()}
    record = {k: files.get(k, "unrecorded") for k in ("qemu", "firmware", "kernel", "rootfs")}
    record.update({k: v for k, v in files.items() if k not in record})
    record["compiler"] = sha256(compiler) if compiler and Path(compiler).is_file() else "unrecorded"
    record["runner"] = sha256(runner)
    record["judge"] = sha256(Path(__file__).with_name("verdicts.py"))
    identity = config.get("identity", {})
    record["profile"] = identity.get("profile", "physical")
    record["exact_bounds"] = bool(identity.get("exact_bounds", False))
    return record


def profile(state):
    """'virtual' or 'physical', as the VM itself was started (capstone_vm records it)."""
    identity = json.loads((Path(state) / "config.json").read_text()).get("identity", {})
    return identity.get("profile", "physical")


def sdk_identity(sdk):
    """(heap, profile) an application SDK was built with, read from the SDK itself, never from a
    label. On the virtual profile malloc is the local mallocng whatever the heap option says, so
    the profile is reported as the heap there."""
    cache = (Path(sdk) / "CMakeCache.txt").read_text()
    heap = re.search(r"^CAPSTONE_APPLICATION_HEAP:STRING=(\S+)", cache, re.M)
    virtual = re.search(r"^CAPSTONE_APPLICATION_VIRTUAL:BOOL=(\S+)", cache, re.M)
    if virtual and virtual.group(1).upper() in ("ON", "TRUE", "1"):
        return "virtual-mallocng", "virtual"
    return (heap.group(1) if heap else None), "physical"
