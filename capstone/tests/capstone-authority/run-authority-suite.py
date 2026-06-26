#!/usr/bin/env python3
"""Run the Capstone authority / provenance negative-test suite.

Boots the Capstone QEMU guest once, loads every domain under domains/ via the
capstone-test.user loader, and classifies each domain's runtime outcome against
the oracle in oracle.tsv:

  tag-fault     -> "Cap mem access requires capability" in QEMU output, no retval
  bounds-fault  -> "Cap mem access OOB" in QEMU output, no retval
  ok            -> "retval = <detail>" present and NO "Cap mem access" fault
  no-trap-today -> domain completes (retval present) with NO "Cap mem access"
                   fault (the broad-bounds granularity gap; flips to
                   bounds-fault once the Step-3 global SHRINK lands)

Exit code 0 iff every domain matches its oracle. Exit code 75 marks an
infrastructure flake (boot/login/mount problems) for retry, matching
run-domain-smoke.py.
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys

import pexpect

TAG_FAULT_MARK = "Cap mem access requires capability"
OOB_FAULT_MARK = "Cap mem access OOB"
ANY_FAULT_RE = re.compile(r"Cap mem access (requires capability|OOB)")
RETVAL_RE = re.compile(r"retval = (\d+)")
INFRA_FLAKE_EXIT_CODE = 75

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent


def env(name, default):
    return os.environ.get(name, str(default))


class InfraFlake(RuntimeError):
    pass


def load_oracle():
    oracle = []
    with (SCRIPT_DIR / "oracle.tsv").open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            domain, expect = parts[0], parts[1]
            detail = parts[2] if len(parts) > 2 else ""
            oracle.append((domain, expect, detail))
    return oracle


def classify(expect, detail, output):
    """Return (passed: bool, observed: str)."""
    fault = ANY_FAULT_RE.search(output)
    retval = RETVAL_RE.search(output)
    observed = []
    if fault:
        observed.append(f"fault[{fault.group(0)}]")
    if retval:
        observed.append(f"retval={retval.group(1)}")
    observed_str = ",".join(observed) or "no-output"

    if expect == "tag-fault":
        return (TAG_FAULT_MARK in output and retval is None), observed_str
    if expect == "bounds-fault":
        return (OOB_FAULT_MARK in output and retval is None), observed_str
    if expect == "ok":
        return (retval is not None and retval.group(1) == detail and fault is None), observed_str
    if expect == "no-trap-today":
        return (fault is None and retval is not None), observed_str
    return False, f"unknown-expect:{expect}"


def expect_prompt(child, timeout, phase):
    try:
        child.expect(r"# ", timeout=timeout)
    except (pexpect.EOF, pexpect.TIMEOUT) as exc:
        raise InfraFlake(f"{phase}: no shell prompt") from exc


def run_cmd(child, command, timeout=30):
    """Send a guest setup command, return its serial output."""
    child.sendline(command)
    expect_prompt(child, timeout=timeout, phase=f"cmd:{command}")
    return child.before.replace("\r\r", "\r")


def run_domain(child, command, timeout):
    """Run a domain-loader command. A trapping domain emits its [CAPSTONE]
    diagnostic and then aborts QEMU (a capability fault inside a domain hits a
    riscv_cpu_do_interrupt assertion), so we accept EOF as a valid terminal
    state and return whatever serial output was captured before it."""
    child.sendline(command)
    idx = child.expect([r"# ", pexpect.EOF], timeout=timeout)
    out = child.before.replace("\r\r", "\r")
    return out, (idx == 1)  # (output, qemu_died)


def main():
    tmp_root = pathlib.Path(env("CAPSTONE_TMP_ROOT", "/tmp/capstone"))
    share_dir = pathlib.Path(env("SHARE_DIR", tmp_root / "capstone-runtime-qemu-share"))
    buildroot_dir = pathlib.Path(env("CAPSTONE_BUILDROOT_DIR", REPO_ROOT / "capstone" / "caplifive-buildroot"))
    qemu_bin = env("CAPSTONE_QEMU_BINARY", REPO_ROOT / "capstone" / "capstone-qemu" / "build" / "qemu-system-riscv64")
    log_file = pathlib.Path(env("AUTHORITY_LOG", tmp_root / "capstone-authority-suite.log"))
    mult = float(env("CAPSTONE_QEMU_TIMEOUT_MULTIPLIER", "2"))
    runtime_dir = SCRIPT_DIR.parent / "runtime-qemu"

    share_dir.mkdir(parents=True, exist_ok=True)

    # Build the loader and the domains (unless told to skip).
    if env("AUTHORITY_NO_BUILD", "") != "1":
        subprocess.run(
            ["bash", str(runtime_dir / "build-capstone-test-user.sh"), str(share_dir / "capstone-test.user")],
            check=True,
        )
        subprocess.run(
            ["bash", str(SCRIPT_DIR / "build-authority-suite.sh"), str(share_dir)],
            check=True,
        )

    oracle = load_oracle()
    only = env("AUTHORITY_ONLY", "")
    if only:
        wanted = set(only.split(","))
        oracle = [row for row in oracle if row[0] in wanted]
        if not oracle:
            raise SystemExit(f"AUTHORITY_ONLY matched no domains: {only}")
    for domain, _, _ in oracle:
        dom = share_dir / f"{domain}.dom"
        if not dom.exists():
            raise SystemExit(f"missing built domain: {dom}")

    image_dir = buildroot_dir / "build" / "images"
    fw_jump, kernel, rootfs = image_dir / "fw_jump.elf", image_dir / "Image", image_dir / "rootfs.ext2"
    for req in (fw_jump, kernel, rootfs):
        if not req.exists():
            raise SystemExit(f"missing buildroot image: {req}")

    qemu_cmd = [
        str(qemu_bin), "-nographic", "-machine", "virt", "-m", "256M", "-smp", "1",
        "-bios", str(fw_jump), "-kernel", str(kernel),
        "-append", "root=/dev/vda rw console=ttyS0 earlycon=sbi",
        "-drive", f"file={rootfs},format=raw,id=hd0",
        "-device", "virtio-blk-device,drive=hd0",
        "-virtfs", f"local,path={share_dir},mount_tag=hostshare,security_model=none,id=hostshare",
        "-chardev", "stdio,mux=on,id=ch0,signal=on",
        "-mon", "chardev=ch0,mode=readline", "-serial", "chardev:ch0",
        "-cpu", "rv64,sstc=false,h=false",
    ]

    def boot_and_run_one(domain, log):
        """Boot the guest, run a single domain, return its serial output.

        Each domain gets its own boot: a trapping domain emits its [CAPSTONE]
        diagnostic and then aborts QEMU, which would otherwise kill the run for
        every subsequent domain."""
        log.write("\n\n===== domain: " + domain + " =====\n")
        log.write(" ".join(qemu_cmd) + "\n\n")
        child = pexpect.spawn(qemu_cmd[0], qemu_cmd[1:], timeout=max(5.0, 5.0 * mult),
                              encoding="utf-8", codec_errors="replace",
                              cwd=str(buildroot_dir), env={"QEMU_AUDIO_DRV": "none", **os.environ})
        child.logfile_read = log
        try:
            try:
                child.expect("buildroot login:", timeout=120 * mult)
            except (pexpect.EOF, pexpect.TIMEOUT) as exc:
                raise InfraFlake(f"{domain}: boot-login") from exc
            child.sendline("root")
            expect_prompt(child, timeout=30 * mult, phase="login")
            run_cmd(child, "dmesg -n 1", timeout=10 * mult)
            run_cmd(child, "stty columns 29999", timeout=10 * mult)
            run_cmd(child, "mkdir -p /mnt/host", timeout=10 * mult)
            run_cmd(child, "mount -t 9p -o trans=virtio,version=9p2000.L hostshare /mnt/host", timeout=20 * mult)
            run_cmd(child, "insmod /capstone.ko", timeout=20 * mult)
            out, _died = run_domain(child, f"/mnt/host/capstone-test.user /mnt/host/{domain}.dom", timeout=30 * mult)
            return out
        finally:
            child.terminate(force=True)

    results = []
    with log_file.open("w", encoding="utf-8") as log:
        for domain, expect, detail in oracle:
            # Retry once on a transient QEMU boot flake (matches the benchmark
            # aggregate runners).
            out = None
            for attempt in (1, 2):
                try:
                    out = boot_and_run_one(domain, log)
                    break
                except InfraFlake as exc:
                    log.write(f"\n[infra-flake attempt {attempt}] {exc}\n")
                    if attempt == 2:
                        print(f"__CAPSTONE_INFRA_FLAKE__ {exc}", file=sys.stderr)
                        return INFRA_FLAKE_EXIT_CODE
            passed, observed = classify(expect, detail, out)
            results.append((domain, expect, observed, passed))

    width = max(len(d) for d, *_ in results)
    print(f"\n{'DOMAIN'.ljust(width)}  {'EXPECT'.ljust(14)}  {'OBSERVED'.ljust(36)}  RESULT")
    print("-" * (width + 64))
    all_pass = True
    for domain, expect, observed, passed in results:
        all_pass &= passed
        print(f"{domain.ljust(width)}  {expect.ljust(14)}  {observed.ljust(36)}  {'PASS' if passed else 'FAIL'}")
    print(f"\nlog: {log_file}")
    if all_pass:
        print("__CAPSTONE_AUTHORITY_SUITE_PASSED__")
        return 0
    print("__CAPSTONE_AUTHORITY_SUITE_FAILED__", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
