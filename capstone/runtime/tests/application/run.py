#!/usr/bin/env python3
"""Exercise real applications and process recovery in an already running VM.

The shared directory must contain application-contract.dom, application-supervisor,
perl.dom and mruby.dom. This test never boots or restarts the VM.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "host"))
    cli = [sys.executable, "-m", "capstone_vm", "--state", str(args.state)]

    def call(*words, input=None):
        result = subprocess.run([*cli, *words], input=input, text=True, capture_output=True,
                                timeout=90, env=env)
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
    result = call("run", "/mnt/host/mruby.dom", "-e", 'puts "mruby: #{6*7}"')
    assert result.stdout == "mruby: 42\n", result.stdout
    assert call("exec", "cat", "/proc/sys/kernel/random/boot_id").stdout == boot_id
    print("real applications, SSH streams, unchanged boot ID: PASS")


if __name__ == "__main__":
    main()
