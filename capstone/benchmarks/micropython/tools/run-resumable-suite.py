#!/usr/bin/env python3
"""Run every embedded MicroPython test despite domain-fatal faults and hangs.

Each QEMU round starts at the first test not yet classified.  A cleanly returned test is scored;
when the domain faults or hangs, exactly that next test is recorded and the following round starts
one index later.  Logs and the final TSV stay under the requested output directory.
"""
import argparse
import base64
import pathlib
import re
import subprocess
import sys

LINE = re.compile(rb"Called dom \((\d+)-th time\) retval = (\d+)")
OUTPUT_LINE = re.compile(rb"^MPYOUT (\d+) (\d+) ([0-9a-f]*)$", re.MULTILINE)
OUTPUT_CAPTURE_LIMIT = 4095


def read_expected(path):
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        idx, name, length, word, how = line.split("\t")
        pattern = None
        if how.startswith("regex-exp:"):
            pattern = base64.b64decode(how.removeprefix("regex-exp:"))
            how = "regex-exp"
        rows.append((int(idx), name, None if word == "-" else int(word, 16), how, pattern))
    return rows


def convert_regex_escapes(line):
    """Match MicroPython's test runner conversion for regex .exp lines."""
    converted = []
    escape = False
    for char in line.decode("utf-8"):
        if escape:
            escape = False
            converted.append(char)
        elif char == "\\":
            escape = True
        elif char in "()[]{}.*+^$":
            converted.append("\\" + char)
        else:
            converted.append(char)
    if converted and converted[-1] == "\n":
        converted[-1] = "\r*\n"
    return "".join(converted).encode()


def regex_output_matches(actual, expected):
    """Apply MicroPython's line regex and ######## wildcard normalization."""
    expected_lines = []
    for line in expected.splitlines(keepends=True):
        if line == b"########\n":
            expected_lines.append((line, None))
        else:
            expected_lines.append((line, re.compile(convert_regex_escapes(line))))

    actual_lines = [line + b"\n" for line in actual.split(b"\n")]
    if actual.endswith(b"\n"):
        actual_lines.pop()

    actual_idx = 0
    for expected_idx, (line, pattern) in enumerate(expected_lines):
        if line == b"########\n":
            if expected_idx + 1 >= len(expected_lines):
                del actual_lines[actual_idx:]
                actual_lines.insert(actual_idx, line)
                actual_idx += 1
                continue
            next_pattern = expected_lines[expected_idx + 1][1]
            skip = 0
            while (actual_idx + skip < len(actual_lines)
                   and not next_pattern.match(actual_lines[actual_idx + skip])):
                skip += 1
            if actual_idx + skip >= len(actual_lines):
                return False
            del actual_lines[actual_idx:actual_idx + skip]
            actual_lines.insert(actual_idx, b"########\n")
        else:
            if actual_idx >= len(actual_lines):
                return False
            if pattern.match(actual_lines[actual_idx]):
                actual_lines[actual_idx] = line
        actual_idx += 1

    return b"".join(actual_lines) == expected


def is_target_skip(output):
    """Recognise MicroPython's target-skip conventions."""
    return (output is not None
            and output.startswith((
                b"SKIP\nTraceback (most recent call last):\n",
                b"SKIP-TOO-LARGE\nTraceback (most recent call last):\n",
            ))
            and output.rstrip().endswith(b"SystemExit:"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--expected", required=True)
    ap.add_argument("--guest-runner", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--timeout-multiplier", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--index-base", type=int, default=0,
                    help="add this value to indices in progress output and results.tsv")
    ap.add_argument("--max-infra-retries", type=int, default=3)
    ap.add_argument("--capture-output", action="store_true",
                    help="save up to 4095 output bytes for each returned test")
    args = ap.parse_args()

    repo = pathlib.Path(__file__).resolve().parents[4]
    smoke = repo / "capstone/tests/runtime-qemu/run-domain-smoke.py"
    domain = pathlib.Path(args.domain).resolve()
    guest_runner = pathlib.Path(args.guest_runner).resolve()
    expected = read_expected(args.expected)
    count = len(expected)
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if domain.parent != guest_runner.parent:
        sys.exit("domain and guest runner must share one --share-dir")

    got = {}
    captured = {}
    capture_truncated = set()
    stopped = {}
    start = args.start
    round_no = 0
    infra_retries = 0
    sentinel_label = count + 1

    while start < count:
        log = out_dir / f"round-{round_no:03d}-from-{start:04d}.log"
        remaining = count - start + 1  # includes the past-the-end sentinel
        guest_command = (
            f"/mnt/host/{guest_runner.name} /mnt/host/{domain.name} {start} {remaining}"
        )
        if args.capture_output:
            guest_command += " --dump-output"
        cmd = [
            sys.executable, str(smoke),
            "--share-dir", str(domain.parent),
            "--log-file", str(log),
            "--timeout-multiplier", str(args.timeout_multiplier),
            "--qemu-extra-arg=-icount", "--qemu-extra-arg=shift=0",
            "--guest-command", guest_command,
            "--success-marker", f"Called dom ({sentinel_label}-th time)",
        ]
        print(f"ROUND {round_no}: start={start + args.index_base} "
              f"remaining={remaining - 1}", flush=True)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        data = log.read_bytes() if log.exists() else b""
        returned = [(int(m.group(1)), int(m.group(2))) for m in LINE.finditer(data)]
        for label, word in returned:
            if label <= count:
                got[label - 1] = word & 0xFFFFFFFF
        for match in OUTPUT_LINE.finditer(data):
            idx = int(match.group(1))
            declared_len = int(match.group(2))
            output = bytes.fromhex(match.group(3).decode())
            if len(output) != declared_len:
                sys.exit(f"captured output length mismatch for test {idx}; see {log}")
            if idx < count:
                captured[idx] = output
                if declared_len >= OUTPUT_CAPTURE_LIMIT:
                    capture_truncated.add(idx)
        if any(label == sentinel_label for label, _ in returned):
            start = count
            break

        reached_guest = b"Created domain ID" in data
        if not reached_guest:
            infra_retries += 1
            print(f"  infrastructure did not reach the guest ({infra_retries}/"
                  f"{args.max_infra_retries})", flush=True)
            if infra_retries >= args.max_infra_retries:
                sys.exit(f"infrastructure failed repeatedly; see {log}")
            round_no += 1
            continue
        infra_retries = 0

        next_idx = returned[-1][0] if returned else start
        if next_idx >= count:
            sys.exit(f"run stopped without sentinel after the final test; see {log}")
        kind = "FAULT" if b"domain halted by capability fault" in data else "HANG"
        stopped[next_idx] = kind
        print(f"  {kind} test={next_idx + args.index_base} "
              f"{expected[next_idx][1]}", flush=True)
        start = next_idx + 1
        round_no += 1

    rows = []
    counts = {"PASS": 0, "FAIL": 0, "FAULT": 0, "HANG": 0, "UNSCORED": 0}
    for idx, name, want, how, pattern in expected:
        row_how = how
        if idx in stopped:
            status = stopped[idx]
            got_word = None
        else:
            got_word = got.get(idx)
            if got_word is None:
                status = "HANG"
            elif got_word & 0x80000000 and is_target_skip(captured.get(idx)):
                status = "UNSCORED"
                row_how = "target skip"
            elif pattern is not None:
                actual = captured.get(idx)
                if actual is None or idx in capture_truncated:
                    status = "UNSCORED"
                elif regex_output_matches(actual, pattern):
                    status = "PASS"
                else:
                    status = "FAIL"
            elif want is None:
                status = "UNSCORED"
            elif (got_word & 0x7FFFFFFF) == want:
                status = "PASS"
            else:
                status = "FAIL"
        counts[status] += 1
        rows.append((idx, name, status, got_word, want, row_how))

    report = out_dir / "results.tsv"
    with report.open("w") as f:
        f.write("index\tname\tstatus\tgot\twant\toracle\n")
        for idx, name, status, got_word, want, how in rows:
            got_text = "-" if got_word is None else f"0x{got_word:08x}"
            want_text = "-" if want is None else f"0x{want:08x}"
            f.write(f"{idx + args.index_base}\t{name}\t{status}\t{got_text}\t"
                    f"{want_text}\t{how}\n")

    if args.capture_output:
        output_dir = out_dir / "actual-output"
        output_dir.mkdir(exist_ok=True)
        for idx, output in captured.items():
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", expected[idx][1])
            output_path = output_dir / f"{idx + args.index_base:04d}-{safe_name}.actual"
            output_path.write_bytes(output)
        print(f"OUTPUTS {output_dir}")

    print("COMPLETE " + " ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"REPORT {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
