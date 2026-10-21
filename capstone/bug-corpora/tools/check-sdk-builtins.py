#!/usr/bin/env python3
"""A corpus reading must not come from a builtins build (C-81): with malloc/realloc as library builtins LLVM may elide
an allocation whose result is only tested against NULL and assume it succeeded (upstream semantics), so a case that
observes an allocation failure reads nothing. The corpus runners call this before any guest boots.

    check-sdk-builtins.py [--extra WORD]... TARGET...

TARGET forms are sdk_builtins.py's (an SDK, a build directory, a CMake or hosted build, a build.json, or an image
file, whose SDK is found by walking up). Every SDK a target names must pass. --extra adds flags appended after the SDK's
codegen (a runner's --cc-arg / --extra-cflags), since clang lets the LAST of -fbuiltin / -fno-builtin win. A target
passes when its effective setting is -fno-builtin, or when its sdk.json records "builtins_allowed" (the build-sdk.sh
override). Exit 0 all pass; 78 any refused, or a target whose SDK cannot be found (no data is a refusal, not a pass).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdk_builtins import REFUSED, check  # noqa: E402


def main(argv):
    extra, targets, i = [], [], 0
    while i < len(argv):
        if argv[i] == "--extra" and i + 1 < len(argv):
            extra.append(argv[i + 1]); i += 2
        else:
            targets.append(argv[i]); i += 1
    if not targets:
        print("check-sdk-builtins: no target (no data is a refusal)", file=sys.stderr)
        return REFUSED
    rc = 0
    for t in targets:
        ok, why = check(t, extra)
        print(f"SDK-BUILTINS {'OK' if ok else 'REFUSED'} {t}", file=sys.stderr)
        for w in why:
            print(f"    {w}", file=sys.stderr)
        rc = rc if ok else REFUSED
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
