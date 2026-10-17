"""Running one application-domain image on a persistent Capstone VM, for corpus runners.

The sql-repros and c-repros runners each carried their own copy of these helpers; they live here
once. Nothing in this module judges anything -- see verdicts.py for that.

THE BOARD BACKEND (virtualvm.py's, the same CAPSTONE_VIRTUAL_BOARD stage:/judge: switch): run_app() becomes one
board step per call, keyed by OUT_BASE's name. The step reproduces what the VM does for a run: the image at
/mnt/host/applications/<stem>-<sha256>.dom (run.py's path, which the fault record names), `capstone-exec -- image
args` in the run's cwd with the fault record in a file, and the host's `capstone-vm:` lines after the output. Stage
mode records the call and returns the QEMU prediction's own result for it (CAPSTONE_VIRTUAL_BOARD_PREDICTION, the
prediction run's --raw directory), so the runner takes the prediction's branches and stages every call it made;
judge mode returns the board's. Unset, nothing changes.
"""
import atexit
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import virtualvm
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
    if virtualvm.board():
        return board_run_app(*virtualvm.board(), image, app_args, out_base, run_args)
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


APPS = "host/applications"
FAULTREC = "@@T3-FAULTREC "
_board = {"started": False, "judged": set(), "runs": None}


def board_command(image_name, app_args, run_args):
    """The board step for one run_app call: what the VM's run_application() does, in a shell."""
    cwd, env, rest = "/tmp", [], list(run_args)
    while rest:
        flag = rest.pop(0)
        if flag == "--cwd" and rest:
            cwd = rest.pop(0)
        elif flag in ("-e", "--env") and rest:
            env.append(rest.pop(0))
        else:
            sys.exit(f"BOARD: run_app option {flag!r} has no board form; refused")
    words = [f"/mnt/host/applications/{image_name}", *map(str, app_args)]
    return (f"unset CAPSTONE_EXEC_DIAGNOSTICS; cd {shlex.quote(cwd)} || exit 125; rm -f /tmp/t3fault; "
            + shlex.join(["env", "CAPSTONE_FAULT_RECORD=/tmp/t3fault", *env, "/mnt/vm/capstone-vexec", "--", *words])
            + f'; rc=$?; if [ -s /tmp/t3fault ]; then echo "{FAULTREC}$(cat /tmp/t3fault)"; '
            + 'echo "capstone-vm: $(cat /tmp/t3fault)"; fi; '
            + '[ $rc -gt 128 ] && echo "capstone-vm: application terminated by signal $((rc-128))"; exit $rc')


def board_run_app(mode, where, bundle, image, app_args, out_base, run_args):
    out_base, home = Path(out_base), where / bundle
    key, digest = out_base.name, sha256(image)
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", key):
        sys.exit(f"BOARD: run key {key!r} is not shell-safe")
    name = f"{Path(image).stem}-{digest}.dom"
    command = board_command(name, app_args, run_args)
    steps_p = home / "steps.json"
    if mode == "stage":
        if not _board["started"]:
            if home.exists():
                sys.exit(f"{home} exists; a bundle is staged once")
            _board["started"] = True
        pred = Path(os.environ.get("CAPSTONE_VIRTUAL_BOARD_PREDICTION", "/nonexistent"))
        out_p, json_p = pred / f"{key}.out", pred / f"{key}.json"
        if not out_p.is_file():
            sys.exit(f"BOARD STAGE REFUSED {bundle}: no prediction record {out_p}")
        text = out_p.read_text(errors="replace")
        result = json.loads(json_p.read_text()) if json_p.is_file() and json_p.stat().st_size else {}
        if result.get("image_sha256", digest) != digest:
            sys.exit(f"BOARD STAGE REFUSED {bundle}: {key}'s prediction ran image {result['image_sha256'][:16]}, "
                     f"this run stages {digest[:16]}")
        steps = json.loads(steps_p.read_text()) if steps_p.exists() else []
        if any(st["name"] == key for st in steps):
            sys.exit(f"BOARD STAGE REFUSED {bundle}: run key {key} repeats")
        (home / "stage" / APPS).mkdir(parents=True, exist_ok=True)
        if not (home / "stage" / APPS / name).exists():
            shutil.copy2(image, home / "stage" / APPS / name)
        steps.append({"name": key, "command": command})
        steps_p.write_text(json.dumps(steps, indent=1) + "\n")
        (home / "stage.SHA256SUMS").write_text(
            "".join(f"{h}  {p}\n" for p, h in virtualvm.stage_sums(home / "stage").items()))
        print(f"BOARD-STAGED {bundle} {key}: {name[:40]}", flush=True)
    else:
        step = next((st for st in json.loads(steps_p.read_text()) if st["name"] == key), None)
        if step is None or step["command"] != command:
            sys.exit(f"BOARD JUDGE REFUSED {bundle}: run {key} is not the staged step")
        sums = dict(l.split("  ", 1)[::-1] for l in (home / "stage.SHA256SUMS").read_text().splitlines())
        if sums.get(f"{APPS}/{name}") != digest:
            sys.exit(f"BOARD JUDGE REFUSED {bundle}: {key}'s image {digest[:16]} is not what the boot staged")
        if _board["runs"] is None:
            serial = home / "serial.log"
            text = virtualvm.BOARD_HEARTBEAT.sub("", serial.read_text(errors="replace").replace("\r\n", "\n"))
            _board["runs"] = virtualvm.split(text)
            atexit.register(lambda: [print(f"BOARD JUDGE: staged run {st['name']} was never judged", file=sys.stderr)
                                     for st in json.loads(steps_p.read_text()) if st["name"] not in _board["judged"]])
        _board["judged"].add(key)
        if key not in _board["runs"]:
            text, result = "\n[runner] TIMEOUT\n", {}       # the boot ended before this step began
        else:
            text, res = _board["runs"][key]
            record = next((l[len(FAULTREC):] for l in text.splitlines() if l.startswith(FAULTREC)), None)
            text = "".join(l for l in text.splitlines(keepends=True) if not l.startswith(FAULTREC))
            marker = f"{virtualvm.BOARD_TIMEOUT}{key}\n"
            if marker in text or res.get("kind") == "none":
                text, result = text.replace(marker, "") + "\n[runner] TIMEOUT\n", {}
            else:
                result = {"version": 1, "kind": res["kind"], "value": res["value"]}
                if record is not None:
                    result["fault"] = record.strip()
                result["image_sha256"] = digest
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_base.with_suffix(".out").write_text(text)
    if result:
        out_base.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
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
    b = virtualvm.board()
    if b and b[0] == "judge":
        return virtualvm.platform(None, compiler, runner)
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
    label: musl mallocng on the virtual profile, level0 on the physical one."""
    cache = (Path(sdk) / "CMakeCache.txt").read_text()
    virtual = re.search(r"^CAPSTONE_APPLICATION_VIRTUAL:BOOL=(\S+)", cache, re.M)
    if virtual and virtual.group(1).upper() in ("ON", "TRUE", "1"):
        return "virtual-mallocng", "virtual"
    return "level0", "physical"
