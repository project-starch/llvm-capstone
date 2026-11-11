#!/usr/bin/env python3
"""Run a configured quick gate within one budget; this is not full SPEC proof.

The plan is a nonempty JSON list of {"name": str, "argv": [str, ...]}, with
optional "cwd". Commands execute directly, without implicit shell parsing.
Builds and lock acquisition must be steps in this same plan to share its budget.
"""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

SCOPE = "Configured quick gate only; not full SPEC qualification"


def validate_plan(value):
    if not isinstance(value, list) or not value:
        raise ValueError("plan must be a nonempty JSON list")
    names = set()
    for step in value:
        if not isinstance(step, dict) or set(step) - {"name", "argv", "cwd"}:
            raise ValueError("each step must contain name, argv and optional cwd only")
        name, argv = step.get("name"), step.get("argv")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("step names must be nonempty and unique")
        names.add(name)
        if not isinstance(argv, list) or not argv or any(
            not isinstance(arg, str) or "\0" in arg for arg in argv
        ) or not argv[0]:
            raise ValueError("argv must be a nonempty list of strings with a nonempty executable and no NUL")
        if "cwd" in step and (
            not isinstance(step["cwd"], str) or not step["cwd"] or "\0" in step["cwd"]
        ):
            raise ValueError("cwd must be a nonempty string without NUL")
    return value


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def cleanup(process, deadline):
    """Signal the owned group even if its leader already exited."""
    result = {"term_sent": False, "kill_sent": False}
    if group_exists(process.pid):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            result["term_sent"] = True
        except ProcessLookupError:
            pass
        grace = min(0.5, max(0.0, deadline - time.monotonic()) / 2)
        until = time.monotonic() + grace
        while time.monotonic() < until and group_exists(process.pid):
            process.poll()
            time.sleep(min(0.01, max(0.0, until - time.monotonic())))
        if group_exists(process.pid):
            try:
                os.killpg(process.pid, signal.SIGKILL)
                result["kill_sent"] = True
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
        result["leader_reaped"] = True
    except subprocess.TimeoutExpired:
        result["leader_reaped"] = False
    return result


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}")


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="fresh evidence directory")
    parser.add_argument("--budget-seconds", type=float, default=300)
    args = parser.parse_args()
    if not math.isfinite(args.budget_seconds) or not 0 < args.budget_seconds <= 300:
        parser.error("budget must be finite, greater than zero and at most 300 seconds")
    try:
        plan = validate_plan(json.loads(args.plan.read_text()))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    try:
        args.out.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(f"output directory must be fresh: {error}")
    write_json(args.out / "plan.json", plan)
    deadline = started + args.budget_seconds
    reserve = min(2.0, args.budget_seconds / 2)
    work_deadline = deadline - reserve
    records, status, reason = [], "PASS", None
    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for index, step in enumerate(plan):
            directory = args.out / f"step-{index:03d}"
            directory.mkdir()
            write_json(directory / "command.json", step)
            row = dict(step, status="NOT_RUN", elapsed_seconds=0.0, returncode=None)
            records.append(row)
            process = None
            began = time.monotonic()
            print(json.dumps({"event": "STEP_START", "index": index,
                              "name": step["name"],
                              "remaining_seconds": max(0.0, work_deadline - began)}), flush=True)
            try:
                remaining = work_deadline - began
                if remaining <= 0:
                    status, reason = "TIMEOUT", "end-to-end deadline reached before step"
                    row["status"] = "TIMEOUT"
                else:
                    with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
                        process = subprocess.Popen(
                            step["argv"], cwd=step.get("cwd"), stdin=subprocess.DEVNULL,
                            stdout=stdout, stderr=stderr, start_new_session=True,
                        )
                        row["pid"] = process.pid
                        row["returncode"] = process.wait(timeout=max(0.0, work_deadline - time.monotonic()))
                    if time.monotonic() >= work_deadline:
                        status, reason = "TIMEOUT", "end-to-end work deadline reached"
                    elif row["returncode"] != 0:
                        status, reason = "FAIL", "command returned nonzero"
                    elif group_exists(process.pid):
                        status, reason = "FAIL", "command left descendants running"
                    row["status"] = status
            except subprocess.TimeoutExpired:
                row["status"], status, reason = "TIMEOUT", "TIMEOUT", "end-to-end work deadline reached"
            except (OSError, KeyboardInterrupt) as error:
                row["status"], status, reason = "FAIL", "FAIL", str(error)
            finally:
                if process is not None:
                    # Cleanup cannot obtain a new budget after a slow build/test.
                    row["cleanup"] = cleanup(process, min(deadline, time.monotonic() + reserve))
                    row["returncode"] = process.returncode
                    if not row["cleanup"]["leader_reaped"]:
                        row["status"], status, reason = "TIMEOUT", "TIMEOUT", "cleanup deadline reached"
                row["elapsed_seconds"] = time.monotonic() - began
                if reason is not None:
                    row["reason"] = reason
                write_json(directory / "result.json", row)
                print(json.dumps({"event": "STEP_END", "index": index,
                                  "name": step["name"], "status": row["status"],
                                  "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
            if status != "PASS":
                break
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    for index in range(len(records), len(plan)):
        step = plan[index]
        row = dict(step, status="NOT_RUN", elapsed_seconds=0.0, returncode=None)
        records.append(row)
        directory = args.out / f"step-{index:03d}"
        directory.mkdir()
        write_json(directory / "command.json", step)
        write_json(directory / "result.json", row)
    if status == "PASS" and time.monotonic() >= deadline:
        status, reason = "TIMEOUT", "end-to-end deadline reached while recording results"
    summary = {
        "scope": SCOPE, "status": status, "reason": reason,
        "budget_seconds": args.budget_seconds, "cleanup_reserve_seconds": reserve,
        "elapsed_seconds": time.monotonic() - started, "steps": records,
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return {"PASS": 0, "FAIL": 1, "TIMEOUT": 124}[status]


if __name__ == "__main__":
    raise SystemExit(main())
