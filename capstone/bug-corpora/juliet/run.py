#!/usr/bin/env python3
"""Run a build's images on the virtual profile in QEMU, in serial batches of one boot each, and record what every step
printed. Judges nothing: judge.py reads the observations.

    run.py --build <build dir> (--kit <virtual kit> | --board-kit <board kit>) --helper <juliet-timeout> --out <fresh dir> [--pairs 32]

EACH BATCH (tools/virtualvm.Batch, one boot, the shared QEMU lock; with --board-kit, tools/boardvm.BoardBatch, one cold
board boot), in this order:
  ok, controls (bounds-malloc, uaf-malloc), randcheck (seeded and twin builds),
  the batch's images, a case's good image before its bad image,
  controls again, ok again, the module's --stats.
  Every program runs under the guest helper `juliet-timeout <step> <ms> -- ./capstone-vexec <image>`, which kills the
  whole process group at the deadline, reaps it, and prints `[runner] TIMEOUT ...` (exit 124).

RECORDED PER BATCH: the planned steps and the sequence of steps the guest actually began, read from the serial log.
RECORDED PER STEP: the image, the sha256 of the copy that was staged, the exit (kind, value), the launcher's
`domain fault` line, the last `[CAPSTONE] ...` line QEMU printed (the faulting access), the order in which the Juliet
markers, the fault and the timeout appeared, and any `@@TWIN` reading.
Exit 0 when every batch completed, 75 when any boot did not, 1 when a staged image differs from the build record.
"""
import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import common as C
import boardvm  # noqa: E402  (tools/, put on sys.path by common)
import virtualvm  # noqa: E402

QEMU_LINE = re.compile(r"^\[CAPSTONE\] .*$", re.M)
EVENTS = ("Calling good()", "Finished good()", "Calling bad()", "Finished bad()", "OK-PROGRAM done",
          "CONTROL bounds-malloc mark", "CONTROL uaf-malloc mark", "RETURNED", "domain fault cause=", "[runner] TIMEOUT")


def platform(a, runner):
    """The run's platform. A board kit's record is the board's identity (bitstream, monitor, module, launcher) WITHOUT a
    firmware hash: every board boot bakes its own payload, so its firmware hash differs from batch to batch (each batch's
    is kept in its own record) while judge.py requires one platform across the census, seeded, twin and probe runs."""
    return boardvm.platform(a.board_kit, None, runner) if a.board_kit else virtualvm.platform(a.kit, None, runner)


def observe(text, result):
    fault = next((l.strip() for l in text.splitlines() if "domain fault cause=" in l), None)
    q = QEMU_LINE.findall(text)
    pos = sorted((text.find(e), e) for e in EVENTS if e in text)
    return {"exit": [result.get("kind"), result.get("value")], "fault_line": fault, "qemu_line": q[-1] if q else None,
            "events": [e for _, e in pos], "timeout": "[runner] TIMEOUT" in text,
            "twin": dict(re.findall(r"^@@TWIN (\w+)=(\S+)$", text, re.M)),
            "randcheck": dict(re.findall(r"RAND seed=(\d) first2=(\d\d)", text))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build", type=Path, required=True)
    k = p.add_mutually_exclusive_group(required=True)
    k.add_argument("--kit", type=Path, help="a virtual (QEMU) kit")
    k.add_argument("--board-kit", type=Path, help="a board kit (tools/boardvm.py): each batch is one cold board boot")
    p.add_argument("--helper", type=Path, required=True, help="juliet-timeout, built static for the guest")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=32, help="cases per boot; a batch holds up to 2 x pairs images")
    p.add_argument("--limit-ms", type=int, default=10000)
    a = p.parse_args()
    Batch, kit = boardvm.select(a.kit, a.board_kit)
    if a.out.exists():
        sys.exit(f"{a.out} exists; a run goes into a fresh directory")
    a.out.mkdir(parents=True)
    build = C.load(a.build / "build.json")
    img = a.build / "img"
    rank = {"good": 0, "bad": 1}
    images = sorted((i for i in build["images"] if i["ok"]), key=lambda i: (i["case"], rank[i["arm"]], i["seed_pattern"] or ""))
    per = 2 * a.pairs
    batches, steps_out, t_all = [], [], time.time()
    for bi in range(0, len(images), per):
        chunk = images[bi:bi + per]
        bdir = a.out / f"batch{bi // per:03d}"
        b = Batch(bdir / "stage")
        stage_img = bdir / "images"; stage_img.mkdir(parents=True)
        staged = {}
        for n in build["support_images"]:
            shutil.copy2(img / f"{n}.dom", stage_img)
            staged[n] = C.sha256(stage_img / f"{n}.dom")
            if staged[n] != build["support_images"][n]:
                sys.exit(f"{n}.dom differs from the build record")
        for i in chunk:
            shutil.copy2(img / f"{i['image']}.dom", stage_img)
            staged[i["image"]] = C.sha256(stage_img / f"{i['image']}.dom")
            if staged[i["image"]] != i["sha256"]:
                sys.exit(f"{i['image']}.dom differs from the build record")
        b.put("images", stage_img)
        b.put("juliet-timeout", a.helper)
        plan = []

        def step(name, image, args="", item=None):
            b.step(name, f"./juliet-timeout {name} {a.limit_ms} -- ./capstone-vexec ./images/{image}.dom {args}")
            plan.append((name, image, item))
        step("ok-0", "ok")
        step("ctl-bounds-0", "controls", "bounds-malloc"); step("ctl-uaf-0", "controls", "uaf-malloc")
        if "randcheck" in build["support_images"]:
            step("randcheck", "randcheck")
        for k, i in enumerate(chunk):
            step(f"s{k:03d}", i["image"], item=i)
        step("ctl-bounds-1", "controls", "bounds-malloc"); step("ctl-uaf-1", "controls", "uaf-malloc")
        step("ok-1", "ok")
        b.step("stats", "./capstone-vexec --stats")
        t0 = time.time()
        steps, serial, completed = b.execute(kit, bdir / "work", 300 + 15 * len(b.steps))
        # the order the guest actually began steps in, duplicates kept: split() returns a dict and would hide them
        log = Path(serial).read_text(errors="replace") if Path(serial).exists() else ""
        serial_sequence = re.findall(rf"^{re.escape(virtualvm.BEGIN)}(\S+)", log, re.M)
        for name, image, item in plan:
            text, result = steps.get(name, ("", {"kind": "none"}))
            rec = {"batch": bdir.name, "step": name, "image": image, "sha256": staged[image], **observe(text, result)}
            if item:
                rec.update(case=item["case"], cwe=item["cwe"], arm=item["arm"], seed_pattern=item["seed_pattern"],
                           twin_family=item.get("twin"))
            steps_out.append(rec)
        stats = re.search(r"\{.*\}", steps.get("stats", ("", {}))[0], re.S)
        batches.append({"batch": bdir.name, "plan": [n for n, _, _ in plan], "serial_sequence": serial_sequence,
                        "completed": completed, **({"firmware": boardvm.record_firmware({}, bdir / "work")} if a.board_kit else {}),
                        "wall_s": round(time.time() - t0, 1), "stats": json.loads(stats.group(0)) if stats else None})
        print(f"{bdir.name}: {len(chunk)} images, completed={completed}, {time.time() - t0:.0f} s", flush=True)
    record = {"set": build["set"], "mode": build["mode"], "wall_s": round(time.time() - t_all, 1),
              "limit_ms": a.limit_ms, "helper_sha256": C.sha256(a.helper), "runner_sha256": C.sha256(Path(__file__)),
              "cc_sha256": build["cc_sha256"], "platform": platform(a, Path(__file__)),
              "batches": batches, "steps": steps_out}
    (a.out / "observations.json").write_text(json.dumps(record, indent=0))
    print(f"{len(batches)} batches in {record['wall_s']} s -> {a.out / 'observations.json'}")
    sys.exit(0 if all(x["completed"] for x in batches) else 75)


if __name__ == "__main__":
    main()
