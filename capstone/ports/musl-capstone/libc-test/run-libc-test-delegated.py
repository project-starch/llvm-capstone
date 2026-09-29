#!/usr/bin/env python3
"""musl's libc-test in the persistent guest, each test a delegated application.

Builds every functional test with the application SDK's compiler driver
(ABI v2), stages the images on the VM's share and runs them through the host
CLI. Verdicts: PASS is exit 0 with no unserved syscall; FAIL carries the status
and the unserved names; FAULT is a domain fault with its record; HUNG is a
test that outlived its timeout. The exclusions are the old runner's, so the
counts compare with docs/../musl-capstone/README.md.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXCLUDE = {
    "dlopen": "dynamic loading", "dlopen_dso": "dynamic loading",
    "tls_align_dlopen": "dynamic loading", "tls_init_dlopen": "dynamic loading",
    "pthread_cancel-points": "threads", "pthread_cancel": "threads",
    "pthread_cond": "threads", "pthread_mutex": "threads", "pthread_mutex_pi": "threads",
    "pthread_robust": "threads", "pthread_tsd": "threads", "sem_init": "threads",
    "sem_open": "threads and shared memory", "vfork": "processes", "spawn": "processes",
    "popen": "processes", "wordexp": "processes", "fcntl": "processes: forks a child",
    "socket": "network", "ipc_msg": "SysV IPC", "ipc_sem": "SysV IPC", "ipc_shm": "SysV IPC",
}
SYSNAME = {25: "fcntl", 29: "ioctl", 34: "mkdirat", 35: "unlinkat", 48: "faccessat", 49: "chdir",
    56: "openat", 57: "close", 59: "pipe2", 61: "getdents64", 62: "lseek", 63: "read", 64: "write",
    65: "readv", 66: "writev", 78: "readlinkat", 79: "fstatat", 80: "fstat", 88: "utimensat",
    93: "exit", 94: "exit_group", 98: "futex", 101: "nanosleep", 113: "clock_gettime",
    115: "clock_nanosleep", 129: "kill", 130: "tkill", 132: "sigaltstack", 134: "rt_sigaction",
    135: "rt_sigprocmask", 160: "uname", 165: "getrusage", 169: "gettimeofday", 172: "getpid",
    173: "getppid", 174: "getuid", 178: "gettid", 198: "socket", 214: "brk", 215: "munmap",
    220: "clone", 221: "execve", 222: "mmap", 226: "mprotect", 233: "madvise", 260: "wait4",
    261: "prlimit64", 278: "getrandom", 291: "statx"}
HOOK = """/* Fold libc-test's t_status into an exit() that reports success. */
extern volatile int t_status;
int __capstone_at_exit(int status) { return status ? status : t_status; }
"""


def names(numbers: str) -> str:
    out = []
    for token in numbers.split():
        match = re.fullmatch(r"(-?\d+)(x\d+)?", token)
        name = SYSNAME.get(int(match.group(1)), match.group(1)) + (match.group(2) or "") if match else token
        if name not in out:
            out.append(name)
    return " ".join(out)


def build(sdk: Path, suite: Path, share: Path, work: Path, only: set[str]) -> dict[str, tuple[str, str]]:
    cc = str(sdk / "capstone-cc")
    common = []
    work.mkdir(parents=True, exist_ok=True)
    flags = ["-std=c99", "-O1", "-Wno-everything", "-D_GNU_SOURCE", f"-I{suite / 'src/common'}"]
    for source in sorted((suite / "src/common").glob("*.c")):
        if source.stem == "runtest":
            continue
        obj = work / f"common-{source.stem}.o"
        subprocess.run([cc, *flags, "-c", str(source), "-o", str(obj)], check=True)
        common.append(str(obj))
    (work / "lt_hook.c").write_text(HOOK)
    hook = work / "lt_hook.o"
    subprocess.run([cc, "-O1", "-Wno-everything", "-c", str(work / "lt_hook.c"), "-o", str(hook)], check=True)
    results = {}
    for source in sorted((suite / "src/functional").glob("*.c")):
        name = source.stem
        if only and name not in only:
            continue
        if name in EXCLUDE:
            results[name] = ("EXCLUDED", EXCLUDE[name])
            continue
        image = share / f"lt-{name}.dom"
        proc = subprocess.run([cc, *flags, str(source), *common, str(hook), "-o", str(image)],
                              capture_output=True, text=True)
        if proc.returncode:
            reason = re.search(r"(undefined symbol: \S+|error: [^\n]{0,90})", proc.stderr)
            results[name] = ("NOBUILD", reason.group(1) if reason else proc.stderr.strip()[-120:])
            image.unlink(missing_ok=True)
            continue
        results[name] = ("BUILT", "")
    return results


def run_one(cli: list[str], name: str, timeout: float, env: dict) -> tuple[str, str]:
    result_file = Path(os.environ.get("TMPDIR", "/tmp")) / f"lt-{name}-{os.getpid()}.json"
    started = time.monotonic()
    try:
        proc = subprocess.run([*cli, "run", "--cwd", "/tmp", "--result", str(result_file),
                               f"/mnt/host/lt-{name}.dom"], capture_output=True, text=True,
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return "HUNG", f"no result after {timeout:.0f}s"
    elapsed = time.monotonic() - started
    record = json.loads(result_file.read_text()) if result_file.exists() else None
    result_file.unlink(missing_ok=True)
    unserved = re.search(r"UNSERVED syscalls: (.*)", proc.stderr)
    noop = re.search(r"NO-OP syscalls: (.*)", proc.stderr)
    extra = f"{elapsed:.1f}s"
    if unserved:
        extra += " UNSERVED " + names(unserved.group(1))
    if noop:
        extra += " NO-OP " + names(noop.group(1))
    if record is None:
        return "NORESULT", f"cli rc={proc.returncode}: {proc.stderr.strip()[-160:]}"
    if record["kind"] == "signal":
        return "FAULT", f"signal {record['value']} {record.get('fault', '')} {extra}"
    status = record["value"]
    if status == 0 and not unserved:
        return "PASS", extra
    return "FAIL", f"status={status} {extra}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="capstone-vm state directory")
    parser.add_argument("--sdk", type=Path, required=True, help="application SDK build directory")
    parser.add_argument("--suite", type=Path, default=Path("/tmp/capstone/libc-test"))
    parser.add_argument("--share", type=Path, required=True, help="the VM's shared directory")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--tests", nargs="*", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    env = dict(os.environ, PYTHONPATH=str(HERE.parents[2] / "runtime/host"))
    cli = [sys.executable, "-m", "capstone_vm", "--state", str(args.state)]
    results = build(args.sdk, args.suite, args.share, args.work, set(args.tests))
    for name in sorted(results):
        if results[name][0] == "BUILT":
            results[name] = run_one(cli, name, args.timeout, env)
            print(f"{results[name][0]:<9} {name:<24} {results[name][1]}", flush=True)
    counts = {}
    for verdict, _ in results.values():
        counts[verdict] = counts.get(verdict, 0) + 1
    summary = " ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f" total={len(results)}"
    print(summary)
    if args.report:
        args.report.write_text(json.dumps({"results": results, "summary": counts,
                                           "sdk": str(args.sdk), "timeout": args.timeout}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
