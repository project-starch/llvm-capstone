#!/usr/bin/env python3
"""Say, before a domain is ever booted, which of its libc calls need something
the domain cannot provide.

WHY THIS EXISTS. A delegated application's runtime runs the syscalls it knows
and returns -ENOSYS for the rest. musl turns -ENOSYS into an ordinary failed call, and a
caller that does not check gets a plausible-looking wrong answer instead of a
stop. The runtime records what it refused and reports it at exit, but that is
after the fact and only for paths the run happened to take. This is the check
that does not need a run: the image is linked, so every libc source it can
reach is already in it.

HOW. The served set is read out of the delegated runtime's own tables, so it
cannot drift from the implementation: every number the shape table
(runtime/common/delegate.c) puts in CAPSTONE_GROUP_DELEGATED, which the
launcher's task runs as Linux, plus every number the domain's stub
(runtime/delegate.c in this port) answers itself, from its `case SYS_x:`
labels and `n == SYS_x` tests: futex, the signal calls, execve through the
spawn service, the identities and the robust list. The rest of the table
(memory, processes, the signal return) and every number not in it are
refused. For each symbol the image defines, the archive says which musl object
defined it, the object name gives the source file, and the source is scanned
for the SYS_ names it mentions. Names are resolved to numbers through musl's
own generated bits/syscall.h and the compatibility aliases in
src/internal/syscall.h, so SYS_fstatat and SYS_newfstatat compare equal.

THE RULE, and it is the whole design. A source is reported only when EVERY
syscall it names is unserved. A mention is not a need: musl's fstatat tries
statx and falls back to newfstatat, and clock_gettime tries clock_gettime and
falls back to gettimeofday, so both mention something we do not serve and both
work. Requiring that nothing it names is served is what separates "has a path
that needs this" from "has no path at all". The first version of this script
reported mentions and produced ten false lines on an image with no problem,
which is how a gate teaches people to turn it off.

THE BASELINE. Every musl image links some of these no matter what the program
does: mmap, munmap, lite_malloc's brk path, the aio stub. Reporting them on
every image is the same failure as reporting mentions, one level up, so
--baseline takes a reference image and subtracts whatever it also has. What is
left is what THIS program brought. Use any domain built from the same archive;
the smaller the better.

THE RUNTIME'S OWN FUNCTIONS. The SDK's runtime archive replaces some musl
functions outright (mmap and the System V calls from the domain's allocator,
atexit, a few string functions), and the image's copy is then not musl's:
--sdk passes that archive as a port object, and names the libc archive and
the musl tree the SDK was built with.

WHAT IT STILL CANNOT DO. It reads each symbol's own source, not the whole call
graph, and a linked symbol need not be reachable. So a line means "this cannot
work if you call it", not "this will happen". For what a run actually
attempted, the runtime reports the refused syscalls by name at exit and folds
them into the domain's status; that is exact and this is early.
"""
import argparse, os, re, subprocess, sys

def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout

def syscall_numbers(musl):
    num, alias = {}, {}
    gen = os.path.join(musl, "obj/include/bits/syscall.h")
    for line in open(gen):
        m = re.match(r"#define\s+SYS_(\w+)\s+(\d+)\s*$", line)
        if m:
            num[m.group(1)] = int(m.group(2))
    for line in open(os.path.join(musl, "src/internal/syscall.h")):
        m = re.match(r"#define\s+SYS_(\w+)\s+SYS_(\w+)\s*$", line)
        if m:
            alias[m.group(1)] = m.group(2)
    def resolve(name, depth=0):
        if name in num:
            return num[name]
        if name in alias and depth < 8:
            return resolve(alias[name], depth + 1)
        return None
    return resolve

def served_set(runtime, resolve):
    """The numbers the delegated runtime serves: the shape table's DELEGATED
    group, and what the domain's stub answers itself. An empty part is an
    error, never an empty set: a table that moved would otherwise make every
    call look unserved."""
    header = open(os.path.join(runtime, "include/capstone/delegate.h")).read()
    number = {m.group(1): int(m.group(2))
              for m in re.finditer(r"CAPSTONE_SYS_(\w+)\s*=\s*(\d+)", header)}
    table = open(os.path.join(runtime, "common/delegate.c")).read()
    delegated = {number[m.group(1)] for m in re.finditer(
        r"\{\s*CAPSTONE_SYS_(\w+)\s*,\s*CAPSTONE_GROUP_DELEGATED\b", table)}
    stub = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "runtime/delegate.c")).read()
    local = {resolve(name) for name in
             re.findall(r"case\s+SYS_(\w+)\s*:", stub) + re.findall(r"\bn\s*==\s*SYS_(\w+)", stub)}
    local.discard(None)
    if not number or not delegated or not local:
        sys.exit(f"check-domain-support: no served set under {runtime} "
                 f"({len(number)} numbers, {len(delegated)} delegated, {len(local)} in the stub)")
    return delegated | local

def archive_map(nm, archive):
    """symbol -> archive member that defines it."""
    member, out = None, {}
    for line in sh(nm, "--print-armap", archive).splitlines():
        line = line.rstrip()
        if line.endswith(".o:") or line.endswith(".o):"):
            member = line.rstrip(":").split("(")[-1].rstrip(")")
            continue
        m = re.match(r"^([0-9a-f]+)?\s*([A-Za-z])\s+(\S+)$", line)
        if m and m.group(2) in "TtWwDdBbVvRr" and member:
            out.setdefault(m.group(3), member)
    return out

def member_sources(musl, members):
    """src_stdio_fputwc.o -> src/stdio/fputwc.c, by searching the tree rather
    than by guessing where the underscores used to be slashes."""
    index = {}
    for root, _, files in os.walk(os.path.join(musl, "src")):
        for f in files:
            if f.endswith(".c"):
                rel = os.path.relpath(os.path.join(root, f), musl)
                index[rel[:-2].replace("/", "_").replace("-", "_")] = rel
    return {m: index.get(m[:-2]) for m in members}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--sdk", default=None,
                    help="the application SDK the image was linked with: its sdk.json names "
                         "the musl tree, the libc archive and the runtime archive")
    ap.add_argument("--musl", default=None)
    ap.add_argument("--archive", default=None)
    ap.add_argument("--runtime", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "../../runtime"),
                    help="capstone/runtime, whose shape table says what is delegated")
    ap.add_argument("--baseline", default=None,
                    help="a reference image built from the same archive; what it also "
                         "links is the libc's baseline and is subtracted")
    ap.add_argument("--port-objects", nargs="*", default=[],
                    help="the port's own runtime objects; whatever they define is not musl's")
    ap.add_argument("--nm", default=os.environ.get("CAPSTONE_LLVM_BIN", "") + "/llvm-nm" if os.environ.get("CAPSTONE_LLVM_BIN") else "llvm-nm")
    args = ap.parse_args()
    if args.sdk:
        import json
        sdk = json.load(open(os.path.join(args.sdk, "sdk.json")))
        args.musl = args.musl or sdk["musl"]
        args.archive = args.archive or sdk["libc"]
        args.port_objects.append(sdk["runtime"])
    args.musl = args.musl or os.environ.get("MUSL_SRC", "/tmp/capstone/musl-src/musl-1.2.5")
    args.archive = args.archive or os.environ.get("ARCHIVE", "/tmp/capstone/musl-capstone-build/libc-capstone.a")

    resolve = syscall_numbers(args.musl)
    served = served_set(args.runtime, resolve)

    def image_symbols(path):
        out = set()
        for line in sh(args.nm, "--defined-only", path).splitlines():
            m = re.match(r"^[0-9a-f]+\s+([A-Za-z])\s+(\S+)$", line.rstrip())
            if m and not m.group(2).startswith(".L"):
                out.add(m.group(2))
        return out

    defined = image_symbols(args.image)
    baseline = image_symbols(args.baseline) if args.baseline else set()
    defined -= baseline

    # Symbols the port defines itself are not musl's, whatever the archive also
    # holds under that name: level0.c answers to __libc_malloc, and reading
    # lite_malloc.c for it would report a brk that nothing in the image calls.
    ours = set()
    for obj in args.port_objects:
        for line in sh(args.nm, "--defined-only", obj).splitlines():
            m = re.match(r"^[0-9a-f]+\s+([A-Za-z])\s+(\S+)$", line.rstrip())
            if m:
                ours.add(m.group(2))

    amap = archive_map(args.nm, args.archive)
    wanted = {s: amap[s] for s in defined if s in amap and s not in ours}
    sources = member_sources(args.musl, set(wanted.values()))

    cache, findings = {}, []
    for sym in sorted(wanted):
        src = sources.get(wanted[sym])
        if not src:
            continue
        if src not in cache:
            text = open(os.path.join(args.musl, src), errors="replace").read()
            need, has_served = set(), False
            for name in re.findall(r"\bSYS_(\w+)\b", text):
                n = resolve(name)
                if n is None:
                    continue
                if n in served:
                    has_served = True
                else:
                    need.add((n, name))
            cache[src] = set() if has_served else need
        if cache[src]:
            findings.append((sym, src, sorted(cache[src])))

    what = os.path.basename(args.image)
    against = f" beyond {os.path.basename(args.baseline)}" if args.baseline else ""
    if not findings:
        print(f"domain-support: {what} links nothing{against} that needs an unserved syscall")
        return 0
    print(f"domain-support: {what} links libc calls{against} that cannot work in a domain\n")
    for sym, src, ns in findings:
        print(f"  {sym:24s} needs {' '.join(name for _, name in ns):16s} {src}")
    print("\nEach line is a call that will fail with ENOSYS if the program makes it.")
    return 1

sys.exit(main())
