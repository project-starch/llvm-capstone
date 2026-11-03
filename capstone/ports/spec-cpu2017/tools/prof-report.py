#!/usr/bin/env python3
"""Attribute the capprof windows of one tools/profile.sh run to layers and functions.

usage: prof-report.py --out OUT --platform PLATFORM --kind capstone|plain [--nm llvm-nm]
  -> OUT/profile/window-N.json for every guest invocation's window, OUT/profile.json (the sum),
     and a summary on stdout.

Every counted instruction (vaddr, length, bytes, attempts) goes to exactly one class:
  program   capstone: the domain image's code, located by the launcher's image record (entry and
            code range, OUT/image-records.txt, one record per guest invocation in order) and mapped
            to link addresses by entry - domain_main; plain: the program's executable segments
  launcher  capstone-vexec's executable segments (capstone only)
  profrun   the window helper's segments
  kernel    System.map's text range (PLATFORM/kernel/System.map); module: capstone_vm.ko's sections,
            placed with /sys/module/capstone_vm/sections (OUT/module-sections.txt)
  firmware  OpenSBI, which runs untranslated at 0x80000000
  other     anything else (other processes, unresolved)
An address inside an image whose bytes differ from the image's bytes at that address is not the
image's ("other (bytes differ)"). Functions are the containing symbol of the image's symbol table
(machine-code functions; inlined code counts for the function it is in).
"""
import argparse, bisect, collections, glob, json, os, re, struct, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--platform", required=True)
ap.add_argument("--kind", choices=("capstone", "plain"), required=True)
ap.add_argument("--nm", default="llvm-nm")
A = ap.parse_args()
OUT, PL = A.out, A.platform
NM = A.nm
SYSMAP = os.path.join(PL, "kernel", "System.map")
MODULE = os.path.join(PL, "adapter", "module", "capstone_vm.ko")
LAUNCHER = os.path.join(PL, "adapter", "capstone-vexec")
PROFRUN = os.path.join(PL, "profrun")


def fail(msg):
    print("prof-report:", msg, file=sys.stderr)
    sys.exit(1)


def elf_segments(path):
    """[(vaddr, memsz, file bytes)] of executable PT_LOAD segments, and e_type."""
    data = open(path, "rb").read()
    if data[:4] != b"\x7fELF" or data[4] != 2:
        fail(f"{path} is not a 64-bit ELF")
    e_type, = struct.unpack_from("<H", data, 16)
    phoff, = struct.unpack_from("<Q", data, 32)
    phentsize, phnum = struct.unpack_from("<HH", data, 54)
    segs = []
    for i in range(phnum):
        p_type, p_flags, p_offset, p_vaddr, _, p_filesz, p_memsz, _ = struct.unpack_from(
            "<IIQQQQQQ", data, phoff + i * phentsize)
        if p_type == 1 and p_flags & 1:
            segs.append((p_vaddr, p_memsz, data[p_offset:p_offset + p_filesz]))
    return segs, e_type


def symbols(path, kinds="TtWw"):
    out = subprocess.run([NM, "-n", "--defined-only", path], capture_output=True, text=True,
                         check=True).stdout
    table = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in kinds and not parts[2].startswith((".L", "$")):
            table.append((int(parts[0], 16), parts[2]))
    table.sort()
    return table


class Image:
    def __init__(self, name, path, slide=0):
        self.name, self.path, self.slide = name, path, slide
        self.segs, self.e_type = elf_segments(path)
        self.syms = symbols(path)
        self.addrs = [a for a, _ in self.syms]
        self.sha = subprocess.run(["sha256sum", path], capture_output=True, text=True).stdout[:64]

    def contains(self, va):
        link = va - self.slide
        return any(v <= link < v + m for v, m, _ in self.segs)

    def bytes_match(self, va, length, word):
        link = va - self.slide
        for v, m, b in self.segs:
            if v <= link < v + len(b):
                if link - v + length > len(b):
                    return False
                got = int.from_bytes(b[link - v:link - v + length], "little")
                return got == word & ((1 << (8 * length)) - 1)
        return False

    def function(self, va):
        i = bisect.bisect_right(self.addrs, va - self.slide) - 1
        return self.syms[i][1] if i >= 0 else "??"


def load_window(n):
    path = os.path.join(OUT, "capprof", f"window-{n}.tsv")
    if not os.path.exists(path):
        return None
    recs = []
    for line in open(path):
        if line.startswith("#"):
            continue
        va, ln, b, att, tr = line.split("\t")
        if int(att):
            recs.append((int(va, 16), int(ln), int(b, 16), int(att)))
    return recs


def kernel_table():
    table, text = [], [None, None]
    for line in open(SYSMAP):
        a, t, name = line.split()[:3]
        a = int(a, 16)
        if name == "_stext": text[0] = a
        if name == "_etext": text[1] = a
        if t in "Tt":
            table.append((a, name))
    table.sort()
    return table, text


def module_table():
    secs = {}
    for line in open(os.path.join(OUT, "module-sections.txt")):
        parts = line.split()
        if len(parts) == 2 and parts[1].startswith("0x"):
            secs[parts[0]] = int(parts[1], 16)
    out = subprocess.run([NM, "--defined-only", "--format=sysv", MODULE], capture_output=True,
                         text=True, check=True).stdout
    table = []
    for line in out.splitlines():
        f = [x.strip() for x in line.split("|")]
        if len(f) >= 7 and f[2].upper() in ("T", "W") and f[6] in secs and f[1] and not f[0].startswith(".L"):
            table.append((secs[f[6]] + int(f[1], 16), int(f[4] or "0", 16), f[0]))
    table.sort()
    ranges = [(secs[s], secs[s] + 0x100000) for s in secs if s.startswith(".text")]
    return table, ranges, secs


def insn_class(b, ln):
    if ln == 2:
        return "compressed"
    op = b & 0x7f
    f3 = (b >> 12) & 7
    return {0x5b: f"capstone.f3={f3}" + (f".f7={b >> 25:#x}" if f3 in (1, 7) else ""),
            0x03: "load", 0x23: "store", 0x07: "fp-load", 0x27: "fp-store", 0x63: "branch",
            0x6f: "jal", 0x67: "jalr", 0x53: "fp-op", 0x43: "fp-fma", 0x47: "fp-fma",
            0x4b: "fp-fma", 0x4f: "fp-fma", 0x73: "system", 0x2f: "atomic", 0x0f: "fence",
            0x37: "lui", 0x17: "auipc", 0x13: "alu-imm", 0x1b: "alu-imm", 0x33: "alu", 0x3b: "alu",
            }.get(op, f"op={op:#x}")


# The idle loop and the functions only it calls. Not the context-tracking and irq entry/exit
# functions (ct_kernel_enter/exit, irq_enter_rcu, tick_sched_*): those run on every kernel entry.
IDLE = re.compile(r"^(do_idle|cpu_idle_poll|default_idle_call|arch_cpu_idle|cpuidle_|ct_idle_|"
                  r"tick_nohz_idle_|tick_nohz_next_event|tick_nohz_get_sleep_length|tick_nohz_stop_|tick_nohz_restart|"
                  r"tick_check_broadcast_expired|can_stop_idle_tick|rcu_needs_cpu|timer_clear_idle|"
                  r"sched_idle_|cpu_startup_entry|menu_|nohz_run_idle_balance|hrtimer_next_event_without|"
                  r"get_next_timer_interrupt|rcu_idle_)")

# The guest invocations, in order: each is one window (capstone-run.sh records the index).
invocations = []
for f in sorted(glob.glob(os.path.join(OUT, "invocations", "[0-9][0-9][0-9][0-9].json"))):
    d = json.load(open(f))
    if d.get("route") in ("guest", "guest-plain"):
        invocations.append((os.path.basename(f)[:4], d))
if not invocations:
    fail("no guest invocations under " + OUT)
records = []
rec_path = os.path.join(OUT, "image-records.txt")
if A.kind == "capstone":
    if not os.path.exists(rec_path):
        fail("capstone arm without image-records.txt (launcher built without the profiling hooks?)")
    for line in open(rec_path):
        m = re.search(r"entry=0x([0-9a-f]+) code=0x([0-9a-f]+)-0x([0-9a-f]+)", line)
        if m:
            records.append(tuple(int(m.group(i), 16) for i in (1, 2, 3)) + (line.strip(),))
    if len(records) != len(invocations):
        fail(f"{len(records)} image records for {len(invocations)} guest invocations")

ktab, ktext = kernel_table()
kaddrs = [a for a, _ in ktab]
mtab, mranges, msecs = module_table()
maddrs = [a for a, _, _ in mtab]
# Every static guest binary links at the same low address: the launcher, capstone-job (the job
# helper capstone_vm run puts in front of the launcher) and the shell. Images are tried in order
# and an address is attributed to the first whose bytes match, so the helper is an image of its
# own; the shell (busybox, not a platform file) stays "other".
JOB = os.path.join(PL, "adapter", "capstone-job")
profrun = Image("profrun", PROFRUN)
helpers = [Image("launcher", LAUNCHER), Image("job-helper", JOB)] if A.kind == "capstone" else []

os.makedirs(os.path.join(OUT, "profile"), exist_ok=True)
total_classes, total_funcs, total_mix = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
summary = {"run": OUT, "kind": A.kind, "windows": []}
for k, (inv_id, inv) in enumerate(invocations):
    n = inv.get("detail", {}).get("window")
    if n is None:
        n = k
    recs = load_window(n)
    if recs is None:
        summary["windows"].append({"invocation": inv_id, "window": n, "error": "no window file"})
        print(f"window {n} (invocation {inv_id}): MISSING", file=sys.stderr)
        continue
    exe = inv["exe"]
    sha = subprocess.run(["sha256sum", exe], capture_output=True, text=True).stdout[:64]
    if inv.get("exe_sha256") and sha != inv["exe_sha256"]:
        fail(f"{exe} changed since the run (sha {sha} vs recorded {inv['exe_sha256']})")
    if A.kind == "capstone":
        entry, code_lo, code_hi, rec_line = records[k]
        syms = symbols(exe)
        dm = [a for a, name in syms if name == "domain_main"]
        if not dm:
            fail(f"{exe} has no domain_main symbol")
        prog = Image("program", exe, slide=entry - dm[0])
        images = [prog, *helpers, profrun]
    else:
        prog = Image("program", exe)
        if prog.e_type != 2:
            fail("a plain program must be ET_EXEC (static, no PIE)")
        code_lo = code_hi = None
        images = [prog, profrun]
    classes, funcs = collections.Counter(), collections.Counter()
    mix, mismatch = collections.defaultdict(collections.Counter), collections.Counter()
    for va, ln, b, att in recs:
        cls, fn = None, None
        if A.kind == "capstone" and code_lo <= va < code_hi:
            cls = "program"
            fn = prog.function(va) if prog.bytes_match(va, ln, b) else None
            if fn is None:
                cls, fn = "other (bytes differ)", f"{va:#x}"
        else:
            # Several static images share the low link address: the first whose BYTES match wins;
            # an address inside images that all disagree is unattributed and counted per image.
            containing = [im for im in images if im.contains(va)]
            matching = [im for im in containing if im.bytes_match(va, ln, b)]
            if len(matching) == 1:
                cls, fn = matching[0].name, matching[0].function(va)
            elif matching:   # the same bytes at the same address in two images: no way to tell
                cls, fn = "ambiguous " + "/".join(im.name for im in matching), f"{va:#x}"
            elif containing:
                mismatch["/".join(im.name for im in containing)] += att
        if cls is None:
            if ktext[0] and ktext[0] <= va < ktext[1]:
                i = bisect.bisect_right(kaddrs, va) - 1
                fn = ktab[i][1]
                # The idle loop runs while the guest waits for the host (9p, the SSH session): its
                # attempts vary run to run and are nobody's work, so they are their own class.
                cls = "kernel idle" if IDLE.match(fn) else "kernel"
            elif any(lo <= va < hi for lo, hi in mranges):
                i = bisect.bisect_right(maddrs, va) - 1
                cls, fn = "module", mtab[i][2] if i >= 0 else "??"
            elif va >= 0xffffffff00000000:
                cls, fn = "kernel (outside text)", f"{va:#x}"
            elif 0x80000000 <= va < 0x80200000:
                cls, fn = "firmware", "opensbi"
            else:
                cls, fn = "other", "other"
        classes[cls] += att
        funcs[(cls, fn)] += att
        mix[cls][insn_class(b, ln)] += att
    total = sum(classes.values())
    report = {
        "invocation": inv_id, "window": n, "exe": exe, "exe_sha256": sha, "argv": inv.get("argv"),
        "total_attempts": total, "classes": dict(classes.most_common()),
        "bytes_differ_by_image": dict(mismatch),
        "top_functions": [[c, f, v] for (c, f), v in funcs.most_common(60)],
        "all_functions": [[c, f, v] for (c, f), v in funcs.most_common()],
        "mix": {c: dict(m.most_common()) for c, m in mix.items()},
        "image_record": records[k][3] if A.kind == "capstone" else None,
        "images": {im.name: im.sha for im in images},
    }
    json.dump(report, open(os.path.join(OUT, "profile", f"window-{n}.json"), "w"), indent=1)
    summary["windows"].append({"invocation": inv_id, "window": n, "total_attempts": total,
                               "classes": dict(classes.most_common())})
    total_classes.update(classes); total_funcs.update(funcs)
    for c, m in mix.items():
        total_mix[c].update(m)
    print(f"window {n} (invocation {inv_id}, {os.path.basename(exe)}): {total:,} attempts")
    for c, v in classes.most_common():
        print(f"  {c:24s} {v:>15,}  {100 * v / total:6.2f}%")
grand = sum(total_classes.values())
summary.update({"total_attempts": grand, "classes": dict(total_classes.most_common()),
                "top_functions": [[c, f, v] for (c, f), v in total_funcs.most_common(80)],
                "all_functions": [[c, f, v] for (c, f), v in total_funcs.most_common()],
                "mix": {c: dict(m.most_common()) for c, m in total_mix.items()}})
json.dump(summary, open(os.path.join(OUT, "profile.json"), "w"), indent=1)
busy = grand - total_classes.get("kernel idle", 0)
summary["busy_attempts"] = busy
print(f"ALL WINDOWS: {grand:,} attempts, {busy:,} without the idle loop")
for c, v in total_classes.most_common():
    print(f"  {c:24s} {v:>15,}  {100 * v / grand:6.2f}%")
print("  top functions:")
for (c, f), v in total_funcs.most_common(25):
    print(f"    {100 * v / grand:6.2f}%  {c:10s} {f}")
