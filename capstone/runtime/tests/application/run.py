#!/usr/bin/env python3
"""Exercise real applications and process recovery in an already running VM.

The shared directory must contain application-contract.dom, application-supervisor,
perl.dom and mruby.dom. This test never boots or restarts the VM.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import selectors
import signal
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=1, help="Mixed lifecycle repetitions after warmup (200 gives 1,000 launches)")
    parser.add_argument("--sublet-image", help="Also test transferred heap, stale reference and node exhaustion")
    parser.add_argument("--report", type=Path, help="Write the checked resource counters and platform hashes")
    args = parser.parse_args()
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "host"))
    cli = [sys.executable, "-m", "capstone_vm", "--state", str(args.state)]

    def call(*words, input=None, timeout=90):
        result = subprocess.run([*cli, *words], input=input, text=True, capture_output=True,
                                timeout=timeout, env=env)
        if result.returncode:
            raise RuntimeError(f"{words[0]} exited {result.returncode}: {result.stderr}\n{result.stdout}")
        return result

    boot_id = call("exec", "cat", "/proc/sys/kernel/random/boot_id").stdout
    result = call("exec", "sh", "-c", "cp /mnt/host/application-supervisor /tmp/application-supervisor && "
                  "chmod +x /tmp/application-supervisor && /tmp/application-supervisor "
                  "/usr/bin/capstone-exec /mnt/host/application-contract.dom")
    if "application sequence: PASS" not in result.stdout:
        raise RuntimeError(result.stdout + result.stderr)
    print(result.stdout, end="")
    result = call("run", "/mnt/host/perl.dom", "-e", 'print join("|", @ARGV), "\\n"',
                  "", "with spaces", "line\nbreak")
    assert result.stdout == "|with spaces|line\nbreak\n", result.stdout
    result = call("run", "/mnt/host/perl.dom", "-e", 'while (<STDIN>) { print uc($_); }',
                  input="from a Linux pipe\n")
    assert result.stdout == "FROM A LINUX PIPE\n", result.stdout
    large = "a Linux pipe with EOF\n" * 50000
    result = call("run", "/mnt/host/perl.dom", "-e", 'while (<STDIN>) { print uc($_); }', input=large)
    assert result.stdout == large.upper()
    result = call("run", "/mnt/host/mruby.dom", "-e", 'puts "mruby: #{6*7}"')
    assert result.stdout == "mruby: 42\n", result.stdout
    result = call("exec", "/mnt/host/application-process-test")
    print(result.stdout, end="")

    # Keep one domain alive while another finishes/faults/reclaims. Cancel the
    # original host CLI and require actual guest cleanup, including blocked I/O.
    for mode, program in (("cpu", "loop"), ("read", "healthy"), ("write", "write-loop")):
        child = subprocess.Popen([*cli, "run", "--cwd", "/tmp", "-e",
                                  "CAPSTONE_CONTRACT=environment with spaces",
                                  "/mnt/host/application-contract.dom", program, "",
                                  "argument with spaces\nand newline"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, env=env)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(child.stdout, selectors.EVENT_READ)
                assert selector.select(30), f"{mode}: no readiness output"
                assert child.stdout.readline() == b"stdout\n"
            assert child.poll() is None, f"{mode}: exited before cancellation"
            assert json.loads(call("exec", "capstone-exec", "--stats").stdout)["live_domains"] == 1
            assert call("run", "/mnt/host/mruby.dom", "-e", 'puts 42').stdout == "42\n"
            if mode == "cpu":
                result = call("exec", "/tmp/application-supervisor", "/usr/bin/capstone-exec",
                              "/mnt/host/application-contract.dom")
                assert "application sequence: PASS" in result.stdout
            # Let the last mode fill the pipe before interrupting the host CLI.
            time.sleep(0.1)
            stop = signal.SIGINT if mode == "cpu" else signal.SIGTERM
            child.send_signal(stop)
            assert child.wait(timeout=20) == 128 + stop
            stats = json.loads(call("exec", "capstone-exec", "--stats").stdout)
            assert stats["live_domains"] == stats["live_regions"] == stats["live_bytes"] == 0, stats
            print(f"overlapping processes / host cancellation during {mode}: PASS")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            child.stdin.close()
            child.stdout.close()
            child.stderr.close()

    with tempfile.TemporaryDirectory() as directory:
        status = Path(directory) / "status.json"
        for mode, kind in (("exit139", "exit"), ("fault-vector", "signal")):
            result = subprocess.run([*cli, "run", "--result", str(status), "--cwd", "/tmp", "-e",
                                     "CAPSTONE_CONTRACT=environment with spaces",
                                     "/mnt/host/application-contract.dom", mode, "",
                                     "argument with spaces\nand newline"], env=env, capture_output=True, timeout=30)
            assert result.returncode == 139
            assert json.loads(status.read_text()) == {"version": 1, "kind": kind,
                                                      "value": 139 if kind == "exit" else 11}
        print("SSH preserves exit 139 versus actual SIGSEGV: PASS")

    if args.sublet_image:
        result = call("exec", "/tmp/application-supervisor", "/usr/bin/capstone-exec", args.sublet_image)
        assert "application sequence: PASS" in result.stdout
        for mode in ("fault-stale", "fault-exhaust", "healthy"):
            result = call("exec", "/tmp/application-supervisor", "/usr/bin/capstone-exec",
                          args.sublet_image, "--mode", mode)
            assert "PASS" in result.stdout
            print(result.stdout, end="")

    before = json.loads(call("exec", "capstone-exec", "--stats").stdout)
    result = call("exec", "/tmp/application-supervisor", "/usr/bin/capstone-exec",
                  "/mnt/host/application-contract.dom", str(args.repeat),
                  timeout=max(90, args.repeat * 20))
    assert "application sequence: PASS" in result.stdout
    after = json.loads(call("exec", "capstone-exec", "--stats").stdout)
    for key in ("live_domains", "live_regions", "live_bytes", "cached_bytes", "poisoned_blocks",
                "nodes_high_water", "nodes_live", "nodes_retired", "tag_pages"):
        assert after[key] == before[key], (key, before, after)
    assert after["nodes_allocated_total"] > before["nodes_allocated_total"]
    print(f"resource reuse after {args.repeat * 5 + 8} mixed starts: PASS")
    assert call("exec", "cat", "/proc/sys/kernel/random/boot_id").stdout == boot_id
    print("real applications, SSH streams, unchanged boot ID: PASS")
    if args.report:
        identity = json.loads((args.state / "config.json").read_text())["identity"]
        report = {"platform": "QEMU supervised CALL, one hart", "boot_id": boot_id.strip(),
                  "mixed_starts": args.repeat * 5 + 8, "sublet": bool(args.sublet_image),
                  "before": before, "after": after,
                  "binaries": {k: v["sha256"] for k, v in identity["files"].items()},
                  "environment": identity["environment"]}
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
