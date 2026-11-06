#!/usr/bin/env python3
"""Exercise the same spawn backend on the host before native-task qualification."""
import argparse
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("shell", type=Path)
    p.add_argument("scratch", type=Path)
    args = p.parse_args()
    shell = str(args.shell.resolve())
    args.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hush-tests-", dir=args.scratch) as tmp:
        cwd = Path(tmp)
        child = cwd / "child"
        subprocess.run(["cc", "-O2", "-Wall", "-Wextra", "-Werror", str(HERE / "child.c"),
                        "-o", str(child)], check=True)
        result = subprocess.run([shell, "sh", str(HERE / "semantics.sh"), str(child)],
                                cwd=cwd, text=True, capture_output=True, timeout=60)
        print(result.stdout, end="")
        if result.returncode:
            raise SystemExit(result.stderr + "semantics failed: " + str(result.returncode))
        # Force a writer subprocess and exercise pipe draining past pipe capacity.
        large = "A" * 100000
        script = cwd / "large-heredoc.sh"
        script.write_text('cat <<EOF | "' + str(child) + '" count\n' + large + '\nEOF\n')
        result = subprocess.run([shell, "sh", str(script)], cwd=cwd, text=True,
                                capture_output=True, timeout=30)
        if (result.returncode, result.stdout) != (0, "100001\n"):
            raise SystemExit("large heredoc failed: " + repr(result))
        print("HUSH PASS: large heredoc")
        # Repeated redirection errors must restore stdout and close saved copies.
        command = ('n=0; while test "$n" -lt 100; do '
                   'cat <missing-input 2>errors; n=$((n+1)); done; '
                   '"' + str(child) + '" fds')
        result = subprocess.run([shell, "sh", "-c", command], cwd=cwd, text=True,
                                capture_output=True, timeout=30)
        if (result.returncode, result.stdout) != (0, "0 1 2 \n"):
            raise SystemExit("redirection rollback failed: " + repr(result))
        print("HUSH PASS: redirection rollback")


if __name__ == "__main__":
    main()
