"""Running one application-domain image on a persistent Capstone VM, for corpus runners.

The sql-repros and c-repros runners each carried their own copy of these helpers; they live here
once. Nothing in this module judges anything -- see verdicts.py for that.

A STATE may also be a batch session made by tools/board/appvm-session.py instead of a running VM: its config.json
carries a "session" entry. Then run_app does not run anything. In the collect phase it records each call's image,
arguments, cwd and environment; one boot (the vp2 board, or a QEMU kit as the session's own control) runs them all;
in the read phase run_app hands each call the result that boot recorded for it. A runner runs twice, collect then
read, unchanged, as long as its staged files are unique per call and it reads nothing back from the share.
"""
import fcntl
import json
import re
import shutil
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


def session(state):
    """The batch session a state directory stands for, or None for a real VM."""
    return json.loads((Path(state) / "config.json").read_text()).get("session")


def on_board(state):
    """True when the run's results come from the vp2 board: the bundle then records arm vp2-<arm>."""
    s = session(state)
    return bool(s and s.get("kind") == "board")


def require_up(state):
    if session(state):
        return
    if "running" not in vm(state, "status").stdout:
        sys.exit("the VM is not up; bring it up first, this runner will not boot it")


def run_app(state, image, app_args, out_base, run_args=(), timeout=600):
    """Run IMAGE on the VM; returns (console text, result dict). The run's own output goes to
    OUT_BASE.out and the launcher's result to OUT_BASE.json. A timeout is written into the text
    as `[runner] TIMEOUT`, so a caller can never mistake it for a quiet run."""
    out_base = Path(out_base)
    result_path, log_path = out_base.with_suffix(".json"), out_base.with_suffix(".out")
    if session(state):
        return _session_run(state, image, app_args, out_base, run_args, timeout)
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


def _session_call(image, app_args, out_base, run_args, timeout):
    """One run_app call as a session records it. Only --cwd and -e are supported; anything else is refused, since a
    board boot could not honour it."""
    cwd, env, rest = "/tmp", [], list(run_args)
    while rest:
        flag = rest.pop(0)
        if flag == "--cwd" and rest:
            cwd = rest.pop(0)
        elif flag in ("-e", "--env") and rest:
            env.append(rest.pop(0))
        else:
            raise ValueError(f"a batch session cannot honour run_app argument {flag!r}")
    return {"name": Path(out_base).name, "sha256": sha256(image), "args": [str(x) for x in app_args],
            "cwd": cwd, "env": env, "timeout": timeout}


def _session_run(state, image, app_args, out_base, run_args, timeout):
    s, root = session(state), Path(state).parent
    call = _session_call(image, app_args, out_base, run_args, timeout)
    log_path = Path(out_base).with_suffix(".out")
    if s["phase"] == "collect":
        images = root / "collect" / "images"
        images.mkdir(parents=True, exist_ok=True)
        staged = images / f"{Path(image).stem}-{call['sha256'][:16]}.dom"
        if not staged.exists():
            shutil.copy2(image, staged)
        call["image"] = staged.name
        with (root / "plan.jsonl").open("a") as plan:
            plan.write(json.dumps(call, sort_keys=True) + "\n")
        log_path.write_text("[runner] SESSION COLLECT: nothing ran\n")
        return "[runner] SESSION COLLECT: nothing ran\n", {}
    results = json.loads((root / "results.json").read_text())
    if call["name"] not in results:
        raise KeyError(f"the session's boot recorded no step {call['name']!r}")
    recorded, text, result = results[call["name"]]
    if {k: recorded[k] for k in ("sha256", "args", "cwd", "env")} != {k: call[k] for k in ("sha256", "args", "cwd", "env")}:
        raise ValueError(f"step {call['name']!r} differs from what was collected: {recorded} vs {call}")
    result = dict(result, version=1, image_sha256=call["sha256"]) if result.get("kind") in ("exit", "signal") else {}
    log_path.write_text(text)
    Path(out_base).with_suffix(".json").write_text(json.dumps(result) + "\n" if result else "")
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
    if config.get("session"):
        recorded = Path(state).parent / "platform.json"
        # In the collect phase nothing has booted; the runner's bundle from that pass is discarded, and says so.
        record = (json.loads(recorded.read_text()) if recorded.is_file()
                  else {k: "unrecorded (session collect: nothing ran)" for k in ("qemu", "firmware", "kernel", "rootfs")})
        record.update(compiler=sha256(compiler) if compiler and Path(compiler).is_file() else "unrecorded",
                      runner=sha256(runner), judge=sha256(Path(__file__).with_name("verdicts.py")))
        return record
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
    label: musl mallocng on the virtual profile, level0 on the physical one."""
    cache = (Path(sdk) / "CMakeCache.txt").read_text()
    virtual = re.search(r"^CAPSTONE_APPLICATION_VIRTUAL:BOOL=(\S+)", cache, re.M)
    if virtual and virtual.group(1).upper() in ("ON", "TRUE", "1"):
        return "virtual-mallocng", "virtual"
    return "level0", "physical"
