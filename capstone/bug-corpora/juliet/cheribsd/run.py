#!/usr/bin/env python3
"""Run a build's images on CheriBSD purecap under CHERI-QEMU, in batches of one boot each, and record what every step
printed. Judges nothing: judge.py reads the observations.

    run.py --build <build dir> --sdk <CheriBSD SDK> --rootfs <purecap rootfs> --image <purecap disk image>
           --revocation default|off|every-free --out <fresh dir> [--pairs 64] [--guests 1] [--limit-s 10]

EACH BATCH is one boot of a disposable guest: a qcow2 overlay over the shared disk image (never opened for writing,
so several guests can boot it at once: --guests), the batch's stage directory exported over virtio-9p, and one shell
script in it that the serial console is told to run. The script copies the images to the guest's /tmp, then runs,
in this order: ok, the ABI probe, the three controls (bounds, revoked, uaf), the two probes (type sizes, malloc
bounds), the seed finder, the batch's images (a case's good image before its bad image), the controls, the ABI probe
and ok again. Every program runs as `timeout -k 5 <limit> ./supervise ./<image> [args]`: supervise
(cpython/pymalloc-repros/observe/supervise.c) traces the program and prints, from outside it, the signal, si_code,
address and pc of the fault that ended it and the mapped object the pc and return address lie in; timeout(1) kills
the whole reaper tree at the deadline (exit 124, 137 after the kill).

THE REVOCATION CONFIGURATION is set by sysctl before the script runs and every knob is read back into the record:
default (runtime_revocation_default=1, every_free=0, the platform as shipped), off (default=0), every-free
(every_free_default=1: free() sweeps before it returns). Only the default run decides a verdict; the other two are
the attribution pair and the sensitivity reading, as in bug-corpora/php/engine-repros' CheriBSD arm.

RECORDED PER BATCH: the planned steps, the sequence of steps the guest actually began (from the transcript), whether
it booted, mounted the share, completed and panicked, the sysctl read-backs, and the wall time.
RECORDED PER STEP: the image, the sha256 of the copy that was staged, the guest seconds it took, the exit,
supervise's fault, object and exit lines, the order in which the Juliet markers appeared, the ABI, probe, seed and malloc-bounds readings, and any @@TWIN
reading. The serial transcripts stay in the run directory and are never committed.
The run takes the host's QEMU lock SHARED (a snapshot-class session: read-only backing image, no network), passing
the turnstile on the way in, so an exclusive caller is neither starved nor excluded wrongly.
Exit 0 when every batch completed, 75 when any boot did not, 1 when a staged image differs from the build record.
"""
import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pexpect

import shared as S

MARK = "JULIET"
DONE = "JULIET_RUNNER_DONE"
MOUNT_TAG = "juliet"
GUEST_SHARE = "/mnt/juliet"


def guest_script(plan, limit_s):
    lines = ["#!/bin/sh", "set -u", "ulimit -c 0",
             f"rm -rf {S.GUEST_IMAGES}; cp -R {GUEST_SHARE}/images {S.GUEST_IMAGES} && chmod +x {S.GUEST_IMAGES}/* "
             f"|| echo {MARK}_STAGE_FAIL",
             f"cd {S.GUEST_IMAGES} || exit 1"]
    for name, image, args, _ in plan:
        lines += [f"echo {MARK}_BEGIN:{name}:$(date +%s)",
                  f"timeout -k 5 {limit_s} ./supervise ./{image} {args}".rstrip() + " 2>&1",
                  f"echo {MARK}_EXIT:{name}:$?"]
    lines += [f"echo {DONE}:$(date +%s)", ""]
    return "\n".join(lines)


def parse(text):
    """{step: exit}, {step: transcript}, [steps begun, in order, duplicates kept], {step: guest seconds it began at}
    (the DONE marker's time closes the last step)"""
    exits, output, begun, began, cur = {}, {}, [], {}, None
    exit_re = re.compile(rf"^{MARK}_EXIT:(\S+):(\d+)$")
    for raw in text.splitlines():
        s = raw.strip("\r\n ")
        if s.startswith(f"{MARK}_BEGIN:"):
            _, cur, t = s.split(":", 2)
            begun.append(cur)
            began[cur] = int(t) if t.isdigit() else None
            output[cur] = []
            continue
        if s.startswith(f"{DONE}:"):
            began[DONE] = int(s.split(":", 1)[1]) if s.split(":", 1)[1].isdigit() else None
            continue
        m = exit_re.match(s)
        if m:
            exits[m.group(1)] = int(m.group(2))
            cur = None
            continue
        if cur is not None:
            output[cur].append(s)
    return exits, {k: "\n".join(v) for k, v in output.items()}, begun, began


def qemu_argv(a, overlay, stage):
    return str(a.sdk / "bin/qemu-system-riscv64cheri"), [
        "-M", "virt", "-m", str(a.memory), "-smp", "1", "-nographic",
        "-bios", str(a.sdk / "share/qemu/bbl-riscv64cheri-virt-fw_jump.bin"),
        "-kernel", str(a.rootfs / "boot/kernel/kernel"),
        "-drive", f"if=none,file={overlay},id=drv,format=qcow2", "-device", "virtio-blk-device,drive=drv",
        "-fsdev", f"local,id=fs0,path={stage},security_model=none",
        "-device", f"virtio-9p-device,fsdev=fs0,mount_tag={MOUNT_TAG}",
        "-append", "vfs.root.mountfrom=ufs:/dev/vtbd0"]


def boot(a, bdir, stage, script_name, n_steps):
    """One guest: boot, set and read back the sysctls, mount the share, run the script, stop. Returns the batch
    record (without the steps) and the transcript."""
    set_knobs, expected, _ = S.REVOCATION[a.revocation]
    overlay = bdir / "overlay.qcow2"
    subprocess.run([str(a.sdk / "bin/qemu-img"), "create", "-f", "qcow2", "-b", str(a.image), "-F", "raw",
                    str(overlay)], check=True, stdout=subprocess.DEVNULL)
    cmd, argv = qemu_argv(a, overlay, stage)
    (bdir / "qemu-command.json").write_text(json.dumps([cmd, *argv], indent=1) + "\n")
    rec = {"booted": False, "mounted": False, "completed": False, "panic": False, "sysctl": {}}
    t0 = time.time()
    log = open(bdir / "serial.log", "w")
    q = pexpect.spawn(cmd, argv, encoding="utf-8", codec_errors="replace", timeout=a.boot_s,
                      env={**os.environ, "TMPDIR": str(bdir)})
    q.logfile = log
    try:
        q.expect(r"login:", timeout=a.boot_s)
        rec["booted"] = True
        q.sendline("root")
        if q.expect([r"Password:", r"# "], timeout=240) == 0:
            q.sendline("")
            q.expect(r"# ", timeout=120)
        q.sendline("stty -echo 2>/dev/null; true")
        q.expect(r"# ", timeout=60)
        for name in expected:   # set what the configuration sets, read back EVERY knob, marked and split as in
            value = set_knobs.get(name)   # ports/php/cheribsd/run-cheribsd-cases.py (the console echoes its input)
            setting = f"sysctl {name}={value} >/dev/null 2>&1; " if value is not None else ""
            q.sendline(f'{setting}V=$(sysctl -n {name}); echo "SYS""CTL:{name}=$V"')
            q.expect(rf"SYSCTL:{re.escape(name)}=(\S*)", timeout=60)
            rec["sysctl"][name] = q.match.group(1)
            q.expect(r"# ", timeout=60)
        q.sendline(f"kldload virtio_p9fs 2>/dev/null; mkdir -p {GUEST_SHARE}; "
                   f"mount -t p9fs -o trans=virtio {MOUNT_TAG} {GUEST_SHARE} && echo MOUNT_OK || echo MOUNT_FAIL")
        q.expect(r"MOUNT_OK|MOUNT_FAIL", timeout=180)
        rec["mounted"] = q.after.strip() == "MOUNT_OK"
        q.expect(r"# ", timeout=60)
        if rec["mounted"]:
            q.sendline(f"sh {GUEST_SHARE}/{script_name}")
            q.expect(DONE, timeout=300 + n_steps * (a.limit_s + 15))
            rec["completed"] = True
            q.expect(r"# ", timeout=120)
    except (pexpect.EOF, pexpect.TIMEOUT) as exc:
        rec["stopped"] = type(exc).__name__
    finally:
        try:
            q.terminate(force=True)
        except Exception:
            pass
        log.flush(); log.close()
    text = (bdir / "serial.log").read_text(errors="replace")
    rec["panic"] = "panic:" in text
    rec["wall_s"] = round(time.time() - t0, 1)
    overlay.unlink(missing_ok=True)
    return rec, text


def qemu_lock():
    """The host's QEMU lock, SHARED, through the turnstile (capstone/tests/capstone-test-env.sh)."""
    lock = Path(os.environ.get("CAPSTONE_QEMU_LOCK", Path.home() / ".capstone-locks/qemu.lock"))
    lock.parent.mkdir(parents=True, exist_ok=True)
    turnstile = open(str(lock) + ".turnstile", "a")
    fcntl.flock(turnstile, fcntl.LOCK_EX)
    held = open(lock, "a")
    fcntl.flock(held, fcntl.LOCK_SH)
    fcntl.flock(turnstile, fcntl.LOCK_UN); turnstile.close()
    return held


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--rootfs", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--revocation", choices=sorted(S.REVOCATION), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=64, help="cases per boot; a batch holds up to 2 x pairs images")
    p.add_argument("--guests", type=int, default=1, help="boots run at once, each on its own overlay")
    p.add_argument("--limit-s", type=int, default=10, help="guest seconds per program")
    p.add_argument("--boot-s", type=int, default=900)
    p.add_argument("--memory", default="2048")
    a = p.parse_args()
    for path in (a.sdk / "bin/qemu-system-riscv64cheri", a.sdk / "bin/qemu-img", a.rootfs / "boot/kernel/kernel",
                 a.image):
        if not path.exists():
            sys.exit(f"missing: {path}")
    if a.out.exists():
        sys.exit(f"{a.out} exists; a run goes into a fresh directory")
    a.out.mkdir(parents=True)
    build = S.load(a.build / "build.json")
    img = a.build / "img"
    rank = {"good": 0, "bad": 1}
    images = sorted((i for i in build["images"] if i["ok"]),
                    key=lambda i: (i["case"], rank[i["arm"]], i["seed_pattern"] or ""))
    per = 2 * a.pairs
    chunks = [images[i:i + per] for i in range(0, len(images), per)]
    t_all = time.time()

    def run_batch(bi):
        chunk = chunks[bi]
        bdir = a.out / f"batch{bi:03d}"
        stage_img = bdir / "stage/images"; stage_img.mkdir(parents=True)
        staged = {}
        for n, h in build["support_images"].items():
            shutil.copy2(img / n, stage_img)
            staged[n] = S.sha256(stage_img / n)
            if staged[n] != h:
                sys.exit(f"{n} differs from the build record")
        for i in chunk:
            shutil.copy2(img / i["image"], stage_img)
            staged[i["image"]] = S.sha256(stage_img / i["image"])
            if staged[i["image"]] != i["sha256"]:
                sys.exit(f"{i['image']} differs from the build record")
        plan = []   # (step, image, args, case record or None)
        for name in S.SUPPORT_HEAD:
            plan.append((name, *S.support_step(name), None))
        for k, i in enumerate(chunk):
            plan.append((f"s{k:03d}", i["image"], "", i))
        for name in S.SUPPORT_TAIL:
            plan.append((name, *S.support_step(name), None))
        script = f"run-{os.getpid()}-{bi:03d}.sh"   # one file per boot, never a shared name
        (bdir / "stage" / script).write_text(guest_script(plan, a.limit_s))
        rec, text = boot(a, bdir, bdir / "stage", script, len(plan))
        exits, output, begun, began = parse(text)
        order = [n for n, _, _, _ in plan] + [DONE]
        steps = []
        for k, (name, image, args, item) in enumerate(plan):
            t0, t1 = began.get(name), began.get(order[k + 1])
            s = {"batch": bdir.name, "step": name, "image": image, "sha256": staged[image],
                 "wall_s": t1 - t0 if t0 is not None and t1 is not None else None,
                 **S.observe(output.get(name, ""), exits.get(name))}
            if item:
                s.update(case=item["case"], cwe=item["cwe"], arm=item["arm"], seed_pattern=item["seed_pattern"],
                         twin_family=item.get("twin"))
            steps.append(s)
        shutil.rmtree(bdir / "stage")   # the copies were verified against the build; the build keeps the images
        batch = {"batch": bdir.name, "plan": [n for n, _, _, _ in plan], "serial_sequence": begun, **rec}
        print(f"{bdir.name}: {len(chunk)} images, booted={rec['booted']} completed={rec['completed']} "
              f"panic={rec['panic']}, {rec['wall_s']:.0f} s", flush=True)
        return batch, steps

    held = qemu_lock()
    try:
        with ThreadPoolExecutor(max(1, a.guests)) as ex:
            results = list(ex.map(run_batch, range(len(chunks))))
    finally:
        held.close()
    batches = [b for b, _ in results]
    steps_out = [s for _, ss in results for s in ss]
    record = {"set": build["set"], "mode": build["mode"], "revocation": a.revocation,
              "sysctl_expected": S.REVOCATION[a.revocation][1], "wall_s": round(time.time() - t_all, 1),
              "limit_s": a.limit_s, "pairs": a.pairs, "guests": a.guests,
              "runner_sha256": S.sha256(Path(__file__)), "shared_sha256": S.sha256(S.HERE / "shared.py"),
              "build_manifest_sha256": build["manifest_sha256"], "toolchain": build["toolchain"],
              "support_images": build["support_images"], "platform": S.vehicle(a.sdk, a.rootfs, a.image),
              "batches": batches, "steps": steps_out}
    (a.out / "observations.json").write_text(json.dumps(record, indent=0))
    print(f"{len(batches)} batches in {record['wall_s']} s -> {a.out / 'observations.json'}")
    sys.exit(0 if all(b["completed"] for b in batches) else 75)


if __name__ == "__main__":
    main()
