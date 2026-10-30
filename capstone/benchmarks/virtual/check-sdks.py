#!/usr/bin/env python3
"""Integrity check of the two microbenchmark SDKs: list EVERY file that differs between the arms.

Archives are compared member by member, sdk.json key by key. Text files (build.ninja, CMakeCache.txt,
cmake_install.cmake, ...) are compared after replacing each arm's own root with a placeholder, so a
difference that is only the output path is reported as such and not hidden. Exit 0 only when the
differing content is exactly the expected set: the heap.c object in libapplication-runtime.a and the
sdk.json heap switches (plus root-path substitutions, which are listed)."""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

def sha(b): return hashlib.sha256(b).hexdigest()[:16]

def members(path, llvm_ar=None):
    """Parse a GNU/SysV ar archive directly: {member name (#n for repeats): bytes}."""
    data = Path(path).read_bytes()
    assert data[:8] == b'!<arch>\n', path
    pos, names, out, seen = 8, b'', {}, {}
    while pos + 60 <= len(data):
        hdr = data[pos:pos + 60]; size = int(hdr[48:58].decode().strip()); body = data[pos + 60:pos + 60 + size]
        name = hdr[:16].decode().strip()
        pos += 60 + size + (size & 1)
        if name == '/' or name == '/SYM64/':
            continue
        if name == '//':
            names = body; continue
        if name.startswith('/') and name[1:].isdigit():
            off = int(name[1:]); name = names[off:names.index(b'\n', off)].decode()
        name = name.rstrip('/')
        seen[name] = seen.get(name, 0) + 1
        out[f"{name}#{seen[name]}" if seen[name] > 1 else name] = body
    return out

def main():
    r1, r2 = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    llvm_ar = os.path.join(os.environ.get('CAPSTONE_LLVM_BIN', ''), 'llvm-ar')
    files = lambda r: {p.relative_to(r).as_posix() for p in r.rglob('*') if p.is_file()}
    f1, f2 = files(r1), files(r2)
    unexpected, report = [], []
    for only, side in ((f1 - f2, 'T1'), (f2 - f1, 'T2')):
        for f in sorted(only):
            report.append(f"ONLY-IN-{side} {f}"); unexpected.append(f)
    for f in sorted(f1 & f2):
        a, b = (r1 / f).read_bytes(), (r2 / f).read_bytes()
        if a == b:
            continue
        if f.endswith(('.ninja_log', '.ninja_deps', 'CMakeConfigureLog.yaml')):
            report.append(f"BUILD-LOG {f} (timestamps, try-compile scratch names)")
            continue
        if f.endswith('/heap.c.obj'):
            report.append(f"HEAP-OBJECT {f} {sha(a)} {sha(b)}")
            continue
        if f.endswith('.a'):
            m1, m2 = members(r1 / f, llvm_ar), members(r2 / f, llvm_ar)
            diff = sorted(k for k in set(m1) | set(m2) if m1.get(k) != m2.get(k))
            for k in diff:
                report.append(f"ARCHIVE-MEMBER {f}:{k} {sha(m1.get(k, b''))} {sha(m2.get(k, b''))}")
                if not (f.endswith('libapplication-runtime.a') and k == 'heap.c.obj'):
                    unexpected.append(f"{f}:{k}")
            if not diff:
                report.append(f"ARCHIVE-HEADER-ONLY {f}"); unexpected.append(f)
            continue
        if f.endswith('sdk.json'):
            j1, j2 = json.loads(a), json.loads(b)
            for k in sorted(set(j1) | set(j2)):
                if j1.get(k) != j2.get(k):
                    v1, v2 = str(j1.get(k)), str(j2.get(k))
                    pathonly = v1.replace(str(r1), '<ROOT>') == v2.replace(str(r2), '<ROOT>')
                    report.append(f"SDK.JSON {f}:{k} {v1} | {v2}" + (" (output path only)" if pathonly else ""))
                    if k not in ('heap_lifetimes', 'heap_scrub_free', 'heap_object_bounds') and not pathonly:
                        unexpected.append(f"{f}:{k}")
            continue
        try:
            t1 = a.decode().replace(str(r1), '<ROOT>'); t2 = b.decode().replace(str(r2), '<ROOT>')
        except UnicodeDecodeError:
            report.append(f"BINARY {f} {sha(a)} {sha(b)}"); unexpected.append(f); continue
        if t1 == t2:
            report.append(f"PATH-ONLY {f}")
            continue
        import difflib
        d = [l for l in difflib.unified_diff(t1.splitlines(), t2.splitlines(), lineterm='', n=0)
             if l[:1] in '+-' and not l.startswith(('+++', '---'))]
        heaponly = all(('HEAP_LIFETIMES' in l or 'HEAP_SCRUB_FREE' in l or 'HEAP_OBJECT_BOUNDS' in l) for l in d)
        if f.endswith('.ninja_log') or f.endswith('.ninja_deps'):
            report.append(f"BUILD-LOG {f} ({len(d)} lines; timestamps/hashes)")
            continue
        report.append(f"TEXT {f} ({len(d)} changed lines){' heap-switch lines only' if heaponly else ''}")
        for l in d[:6]:
            report.append(f"    {l[:200]}")
        if not heaponly:
            unexpected.append(f)
    print("\n".join(report))
    if unexpected:
        print("UNEXPECTED:", *unexpected, sep="\n  ")
        return 1
    print("INTEGRITY OK: the content differs only in heap.c.obj and the sdk.json heap switches"
          " (every other listed difference is the output path or a build log)")
    return 0

if __name__ == '__main__':
    sys.exit(main())
