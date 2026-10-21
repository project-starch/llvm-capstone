#!/usr/bin/env python3
"""appvm-session.py: run an appvm.run_app corpus runner's calls in ONE boot, on the vp2 board or on a QEMU kit.

    appvm-session.py init SESSION (--board-kit K | --virtual-kit K)
    <runner> --state SESSION/state ...      # phase collect: run_app records each call, runs nothing
    appvm-session.py boot SESSION [--timeout S]
    <runner> --state SESSION/state ...      # phase read: run_app hands each call its recorded result

The runner is unchanged. Its staged share files land in SESSION/collect/share, which the boot links at /mnt/host.
Its run_app calls become steps of one boardvm.BoardBatch (or virtualvm.Batch) boot; each runs
`cd <cwd> && env <-e...> capstone-vexec <image> <args>`. The QEMU kit is the session mechanism's own control: the
same session on QEMU must give the runner's direct appvm reading.

A runner fits only if its staged files are unique per call (or identical where they repeat) and it reads nothing
back from the share. A step that never ran reads kind none, which no observe() takes as a run.
"""
import argparse
import json
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
import boardvm  # noqa: E402
import virtualvm  # noqa: E402


def config_path(session):
    return Path(session) / "state" / "config.json"


def init(a):
    s = Path(a.session)
    if s.exists():
        sys.exit(f"{s} exists; a session starts fresh")
    (s / "state").mkdir(parents=True)
    (s / "collect" / "share").mkdir(parents=True)
    (s / "collect" / "images").mkdir(parents=True)
    kind, kit = ("board", a.board_kit) if a.board_kit else ("virtual", a.virtual_kit)
    config_path(s).write_text(json.dumps({
        "share": str(s / "collect" / "share"), "identity": {"profile": "virtual"},
        "session": {"phase": "collect", "kind": kind, "kit": str(Path(kit).resolve())}}, indent=2) + "\n")
    print(f"session {s}: {kind} kit {kit}; phase collect")
    return 0


def plan(session):
    """The collected calls, one per name; a name collected twice must be the same call."""
    calls = {}
    for line in (Path(session) / "plan.jsonl").read_text().splitlines():
        c = json.loads(line)
        if c["name"] in calls and calls[c["name"]] != c:
            sys.exit(f"step {c['name']!r} was collected twice with different calls")
        calls[c["name"]] = c
    if not calls:
        sys.exit("NO DATA: nothing was collected")
    return calls


def step_name(name, taken):
    safe = re.sub(r"[^A-Za-z0-9_.+-]", "_", name)
    out, n = safe, 1
    while out in taken:
        n += 1
        out = f"{safe}.{n}"
    taken.add(out)
    return out


def boot(a):
    s = Path(a.session)
    cfg = json.loads(config_path(s).read_text())
    sess = cfg["session"]
    if sess["phase"] != "collect":
        sys.exit(f"the session is in phase {sess['phase']}, not collect")
    calls = plan(s)
    board = sess["kind"] == "board"
    Batch, kit = boardvm.select(None if board else sess["kit"], sess["kit"] if board else None)
    batch = Batch(s / "stage")
    root = str(Path(batch.put("images", s / "collect" / "images")).parent)
    (s / "collect" / "share" / ".session-share").write_text(f"{s}\n")
    batch.put("share", s / "collect" / "share")
    # /mnt/host: a symlink where nothing is, a bind mount over a directory the image already has (the QEMU kit's
    # rootfs carries one). The OK line comes only from the marker read THROUGH /mnt/host, not from the commands.
    batch.step("session-share", f"if [ -L /mnt/host ] || [ ! -e /mnt/host ]; then mkdir -p /mnt && ln -sfn {root}/share "
                                f"/mnt/host; else mount -o bind {root}/share /mnt/host; fi; "
                                f"[ -f /mnt/host/.session-share ] && echo SESSION-SHARE-OK || "
                                f"echo 'SESSION: /mnt/host does not show the stage'")
    names, taken = {}, {"session-share"}
    for name, c in calls.items():
        step = names[name] = step_name(name, taken)
        env = " ".join(shlex.quote(e) for e in c["env"])
        batch.step(step, f"cd {shlex.quote(c['cwd'])} && exec env {env} {root}/capstone-vexec "
                         f"{root}/images/{c['image']} " + " ".join(shlex.quote(x) for x in c["args"]))
        batch.bounds[step] = int(c["timeout"])   # the runner's own limit; a board batch bounds the step at a multiple
    timeout = a.timeout or 300 + sum(min(int(c["timeout"]), 600) for c in calls.values())
    runs, transcript, completed = batch.execute(kit, s / "guest", timeout=timeout)
    share_ok = "SESSION-SHARE-OK" in runs.get("session-share", ("", {}))[0]
    results = {}
    for name, c in calls.items():
        text, result = runs.get(names[name], ("[runner] SESSION: the boot ended before this step ran\n", {"kind": "none"}))
        results[name] = [c, text, result]
    (s / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    record = (boardvm.record_firmware(boardvm.platform(kit, None, HERE), s / "guest") if board
              else virtualvm.platform(kit, None, HERE))
    for k in ("compiler", "runner", "judge"):
        record.pop(k, None)
    (s / "platform.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    sess["phase"] = "read"
    config_path(s).write_text(json.dumps(cfg, indent=2) + "\n")
    ran = sum(1 for n in calls if names[n] in runs)
    print(f"session {s}: {ran}/{len(calls)} steps ran; share {'linked' if share_ok else 'NOT LINKED'}; "
          f"boot {'completed' if completed else 'DID NOT COMPLETE'}; transcript {transcript}; phase read")
    return 0 if completed and share_ok and ran == len(calls) else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("session", type=Path)
    k = i.add_mutually_exclusive_group(required=True)
    k.add_argument("--board-kit", type=Path)
    k.add_argument("--virtual-kit", type=Path)
    b = sub.add_parser("boot")
    b.add_argument("session", type=Path)
    b.add_argument("--timeout", type=int)
    a = p.parse_args()
    return init(a) if a.cmd == "init" else boot(a)


if __name__ == "__main__":
    sys.exit(main())
