#!/usr/bin/env python3
"""Run codegen over every BITCODE object and print the ones that fail, one per line as "<path>\t<first error>".

B0 (docs/plans/b0-silicon-delegated-runtime.md). Under full LTO instruction selection happens at the link, so an
object the backend cannot select for fails the WHOLE link -- and lld keeps every bitcode definition of a runtime
libcall (the fp128 soft-float builtins, the long-double math family) whether the program calls it or not. Running
llc per object with the SAME -mllvm options the link passes as --plugin-opt finds those objects before the link.

Usage: lto-codegen-verify.py <llc> <jobs> "<compile flags containing -mllvm options>" <objects...>
Exit 0 with the failing list on stdout (possibly empty); exit 2 if NO object was bitcode (a broken magic test or
the wrong inputs must never read as "every object codegens")."""
import concurrent.futures as cf, shlex, subprocess, sys

BITCODE_MAGIC = bytes.fromhex("4243c0de")   # hex on purpose: a mangled \x escape once let a verifier pass
                                           # 1377 objects having read none of them


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    llc, jobs, flags, objs = sys.argv[1], int(sys.argv[2]), shlex.split(sys.argv[3]), sys.argv[4:]
    llc_flags = [flags[i + 1] for i, w in enumerate(flags) if w == "-mllvm" and i + 1 < len(flags)]
    seen = [0]

    def check(o):
        with open(o, "rb") as fh:
            if fh.read(4) != BITCODE_MAGIC:
                return None
        seen[0] += 1
        r = subprocess.run([llc, "-mtriple=capstone64-unknown-elf", "-mattr=+m,+a", *llc_flags,
                            "-filetype=obj", o, "-o", "/dev/null"], capture_output=True)
        if r.returncode == 0:
            return None
        err = (r.stderr or b"").decode("utf-8", "replace")
        first = next((l for l in err.splitlines()
                      if "error" in l or "LLVM ERROR" in l or "Cannot select" in l or "Assertion" in l), "")
        return "%s\t%s" % (o, first.strip()[:120])

    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        found = [r for r in ex.map(check, objs) if r]
    if seen[0] == 0:
        print("verification recognised no bitcode among %d objects; refusing to report a clean set" % len(objs),
              file=sys.stderr)
        return 2
    print("VERIFIED %d bitcode objects with llc flags [%s]: %d fail" % (seen[0], " ".join(llc_flags), len(found)),
          file=sys.stderr)
    for line in found:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
