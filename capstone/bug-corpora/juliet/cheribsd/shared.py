"""Shared pieces of the CheriBSD Juliet harness: the parent harness's manifests and source rules, the fault codes a
class must raise, the three revocation configurations, the toolchain and vehicle records, and the step-output reader.

The parent harness (../common.py) is loaded by path, not by sys.path: ../select.py would otherwise shadow the standard
library's `select`, which pexpect imports."""
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JULIET = HERE.parent
CAP = JULIET.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("juliet_common", JULIET / "common.py")   # also puts bug-corpora/tools on sys.path
import verdicts  # noqa: E402

SIGPROT = 34
# The si_code a detection must carry, per class (CheriBSD sys/sys/signal.h): PROT_CHERI_TAG for an access through a
# capability whose tag a revocation sweep cleared, PROT_CHERI_BOUNDS for an access outside the capability's bounds.
EXPECTED_SICODE = {415: 2, 416: 2, 121: 1, 122: 1, 124: 1, 126: 1, 127: 1}
SICODE_NAME = {1: "PROT_CHERI_BOUNDS", 2: "PROT_CHERI_TAG", 3: "PROT_CHERI_SEALED", 4: "PROT_CHERI_TYPE",
               5: "PROT_CHERI_PERM", 7: "PROT_CHERI_IMPRECISE", 8: "PROT_CHERI_STORELOCAL", 9: "PROT_CHERI_CINVOKE",
               11: "PROT_CHERI_SYSREG", 12: "PROT_CHERI_UNALIGNED_BASE"}
# Timeouts the guest's timeout(1) reports: 124 at the deadline, 137 when the kill after it was needed.
TIMEOUT_EXITS = (124, 137)

DEFAULT_KNOB = "security.cheri.runtime_revocation_default"
EVERY_FREE_KNOB = "security.cheri.runtime_revocation_every_free_default"
# configuration -> (sysctls set before the run, every knob's expected read-back, abi-probe's runtime_revocation)
REVOCATION = {
    "default": ({}, {DEFAULT_KNOB: "1", EVERY_FREE_KNOB: "0"}, 1),
    "off": ({DEFAULT_KNOB: "0"}, {DEFAULT_KNOB: "0", EVERY_FREE_KNOB: "0"}, 0),
    "every-free": ({EVERY_FREE_KNOB: "1"}, {DEFAULT_KNOB: "1", EVERY_FREE_KNOB: "1"}, 1),
}

CONFIG = "cheribsd-riscv64-purecap.cfg"       # the SDK's own purecap configuration: target, sysroot, -cheri-tgot-tls
FLAGS = ["--config", CONFIG, "-std=gnu11", "-O0", "-fuse-ld=lld"]
SUPERVISE_C = CAP / "bug-corpora/cpython/pymalloc-repros/observe/supervise.c"
ABI_PROBE_C = CAP / "ports/common/host/cheribsd/abi-probe.c"
SICODE_C = CAP / "bug-corpora/tools/cheribsd/sicode.c"
CAPBOUNDS_C = CAP / "bug-corpora/tools/cheribsd/capbounds.c"
MECHANISM_PROBE_C = JULIET / "mechanism-probe.c"

GUEST_IMAGES = "/tmp/images"                   # where the guest script copies a batch's images before running them
SUPPORT_HEAD = ["ok-0", "abi-0", "ctl-bounds-0", "ctl-revoked-0", "ctl-uaf-0", "probe", "capbounds", "randseed"]
SUPPORT_TAIL = ["ctl-bounds-1", "ctl-revoked-1", "ctl-uaf-1", "abi-1", "ok-1"]
# step name -> (support image, args)
SUPPORT_STEPS = {"ok": ("ok", ""), "abi": ("abi-probe", ""), "ctl-bounds": ("abi-probe", "oob"),
                 "ctl-revoked": ("sicode-selftest", "revoked"), "ctl-uaf": ("uaf-control", ""),
                 "probe": ("probe", ""), "capbounds": ("capbounds", ""), "randseed": ("randseed", "")}


def support_step(name):
    return SUPPORT_STEPS[name[:-2] if name[-2:] in ("-0", "-1") else name]


sha256 = C.sha256
load = C.load


def clang(sdk):
    return Path(sdk) / "bin/clang"


def config_text(sdk):
    return (Path(sdk) / "bin" / CONFIG).read_text()


def rootfs_of(sdk):
    """The sysroot the SDK's purecap configuration names, resolved: the rootfs the guest image was made from."""
    m = re.search(r"^--sysroot=<CFGDIR>/(\S+)$", config_text(sdk), re.M)
    if not m:
        sys.exit(f"{CONFIG} names no <CFGDIR>-relative sysroot")
    return (Path(sdk) / "bin" / m.group(1)).resolve()


def _version(exe):
    try:
        out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return f"unreadable: {e}"
    return out.strip().splitlines()[0] if out.strip() else "no output"


def toolchain(sdk):
    """What every image is built with: the compiler, its purecap configuration and the libc it links against. Paths
    are not recorded."""
    sdk = Path(sdk)
    rootfs = rootfs_of(sdk)
    return {"clang": _version(clang(sdk)), "clang_sha256": sha256(clang(sdk).resolve()),
            "config": CONFIG, "config_sha256": sha256(sdk / "bin" / CONFIG),
            "config_lines": [l for l in config_text(sdk).splitlines() if l.strip()],
            "sysroot_libc_sha256": sha256(rootfs / "lib/libc.so.7"), "flags": FLAGS}


def vehicle(sdk, rootfs, image):
    """What a boot runs on: QEMU, firmware, kernel, the rootfs libc, and the disk image's size and mtime (9.5 GB:
    identified, not hashed)."""
    sdk, rootfs, image = Path(sdk), Path(rootfs), Path(image)
    fw = sdk / "share/qemu/bbl-riscv64cheri-virt-fw_jump.bin"
    st = image.stat()
    return {"qemu": _version(sdk / "bin/qemu-system-riscv64cheri"),
            "qemu_sha256": sha256((sdk / "bin/qemu-system-riscv64cheri").resolve()),
            "firmware": {"file": fw.name, "bytes": fw.stat().st_size, "sha256": sha256(fw)},
            "kernel_sha256": sha256(rootfs / "boot/kernel/kernel"),
            "rootfs_libc_sha256": sha256(rootfs / "lib/libc.so.7"),
            "image": {"file": image.name, "bytes": st.st_size,
                      "mtime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))},
            "machine": "virt, 1 hart, the machine's default cpu"}


# ---- reading one step's output ---------------------------------------------------------------------------------

EVENTS = ("Calling good()", "Finished good()", "Calling bad()", "Finished bad()", "OK-PROGRAM done",
          "CONTROL uaf-free mark", "CONTROL uaf-free done", "SELFTEST no fault", "CHERI_BOUNDARY_READY",
          "SUPERVISE fault", "SUPERVISE exit")
FAULT = re.compile(r"^SUPERVISE fault signal=(\d+) code=(\S+) addr=(\S+) pc=0x([0-9a-f]+)\s*$", re.M)
OBJECT = re.compile(r"^SUPERVISE (fault-object|fault-ra) 0x([0-9a-f]+) in (\S+) base=0x([0-9a-f]+)\s*$", re.M)
EXIT = re.compile(r"^SUPERVISE exit (signalled|status)=(\d+)\s*$", re.M)
ABI = re.compile(r"^CHERI_ABI pointer_bytes=(\d+) runtime_revocation=(\d+)\s*$", re.M)


def observe(text, exit_code):
    """What one step's transcript says, as judge.py reads it. `exit_code` is the guest's `$?` after timeout(1) and
    supervise, which passes its child's status through (162 for SIGPROT)."""
    f = FAULT.search(text)
    fault = None
    if f:
        code = int(f.group(2)) if f.group(2).isdigit() else None
        fault = {"signal": int(f.group(1)), "code": code, "addr": f.group(3), "pc": int(f.group(4), 16)}
    objects = {m.group(1): {"addr": int(m.group(2), 16), "path": m.group(3), "base": int(m.group(4), 16)}
               for m in OBJECT.finditer(text)}
    e = EXIT.search(text)
    abi = ABI.search(text)
    pos = sorted((text.find(ev), ev) for ev in EVENTS if ev in text)
    return {"exit": exit_code, "timeout": exit_code in TIMEOUT_EXITS or (e is None and exit_code is not None),
            "fault": fault, "fault_object": objects.get("fault-object"), "fault_ra": objects.get("fault-ra"),
            "supervise_exit": [e.group(1), int(e.group(2))] if e else None,
            "events": [ev for _, ev in pos],
            "abi": {"pointer_bytes": int(abi.group(1)), "runtime_revocation": int(abi.group(2))} if abi else None,
            "twin": dict(re.findall(r"^@@TWIN (\w+)=(\S+)$", text, re.M)),
            "randseed": {p: int(s) for s, p in re.findall(r"^RANDSEED seed=(\d+) first2=(\d\d)$", text, re.M)},
            "probe": {k: dict(re.findall(r"(\S+)=(\S+)", v)) for k, v in re.findall(r"^PROBE (\S+) (.*)$", text, re.M)},
            "capbounds": [l for l in text.splitlines() if re.match(r"^\d+\s+\d+\s+-?\d+\s", l)]}


class Symbols:
    """Function symbols of one ELF (an image or a library of the rootfs) for pc attribution, from llvm-nm; a stripped
    library falls back to its dynamic symbols. `lookup(pc, base)` takes the runtime pc and the object's lowest mapping,
    which supervise prints, and the ELF's lowest LOAD address does the rest."""

    def __init__(self, llvm_bin, path):
        self.funcs = []
        for flag in ([], ["-D"]):
            out = subprocess.run([str(Path(llvm_bin) / "llvm-nm"), "--print-size", "-n", "--defined-only", *flag,
                                  str(path)], capture_output=True, text=True).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) == 4 and parts[2] in "tTwW":
                    self.funcs.append((int(parts[0], 16), int(parts[1], 16), parts[3]))
            if self.funcs:
                break
        hdr = subprocess.run([str(Path(llvm_bin) / "llvm-readelf"), "-l", str(path)], capture_output=True,
                             text=True).stdout
        loads = [int(m.group(1), 16) for m in re.finditer(r"^\s*LOAD\s+\S+\s+(0x[0-9a-f]+)", hdr, re.M)]
        self.link_base = min(loads) if loads else 0

    def lookup(self, pc, base):
        off = pc - base + self.link_base
        for addr, size, name in self.funcs:
            if addr <= off < addr + max(size, 1):
                return name, off - addr
        return None, None
