#!/usr/bin/env python3
"""A corpus reading must not come from a builtins build (C-81): with malloc/realloc as library builtins LLVM may elide
an allocation whose result is only tested against NULL and assume it succeeded (upstream semantics), so a case that
observes an allocation failure reads nothing. The corpus runners call this before any guest boots.

    check-sdk-builtins.py [--extra WORD]... TARGET...

A TARGET is an SDK directory (its sdk.json); a port's build directory holding image/sdk/sdk.json or
source/runtime/sdk.json; a CMake build (its CMakeCache's CAPSTONE_SDK, plus CMAKE_C_FLAGS); a hosted corpus build (every
work/*/CMakeCache.txt); or a scripted build recording its SDK in build.json. Every SDK a target names must pass. --extra adds flags appended after the SDK's
codegen (a runner's --cc-arg / --extra-cflags), since clang lets the LAST of -fbuiltin / -fno-builtin win. A target
passes when its effective setting is -fno-builtin, or when its sdk.json records "builtins_allowed" (the build-sdk.sh
override). Exit 0 all pass; 78 any refused, or a target whose SDK cannot be found (no data is a refusal, not a pass).
"""
import json, re, sys
from pathlib import Path

REFUSED = 78


def effective_no_builtin(words):
    last = [w for w in words if w in ("-fbuiltin", "-fno-builtin")]
    return bool(last) and last[-1] == "-fno-builtin"


def _cache_sdk(cache):
    """(sdk.json, the build's own C flags) named by a CMakeCache, or None."""
    text = cache.read_text(errors="replace")
    m = re.search(r"^CAPSTONE_SDK:\w+=(.+)$", text, re.M)
    if not m or not (Path(m.group(1)) / "sdk.json").is_file():
        return None
    flags = re.findall(r"^CMAKE_C_FLAGS(?:_[A-Z]+)?:\w+=(.*)$", text, re.M)
    return Path(m.group(1)) / "sdk.json", " ".join(flags).split()


def find_sdks(target):
    """Every (sdk.json, extra flags) a build directory was made with. A port build is one CMake tree; a hosted corpus
    build is one CMake tree per case under work/NN; a scripted build records its SDK in build.json. All must pass."""
    t = Path(target)
    for cand in (t / "sdk.json", t / "image/sdk/sdk.json", t / "source/runtime/sdk.json"):
        if cand.is_file():
            return [(cand, [])]
    found = []
    for cache in [t / "CMakeCache.txt", *sorted(t.glob("work/*/CMakeCache.txt"))]:
        if cache.is_file():
            found.append(_cache_sdk(cache) or (cache, None))
    bj = t / "build.json"
    if bj.is_file():
        sdk = json.loads(bj.read_text()).get("sdk")
        found.append(((Path(sdk) / "sdk.json"), []) if sdk and (Path(sdk) / "sdk.json").is_file() else (bj, None))
    return found


def check(target, extra=()):
    sdks = find_sdks(target)
    if not sdks:
        return False, [f"{target}: no sdk.json found (looked for sdk.json, image/sdk, source/runtime, CMakeCache, "
                       "work/*/CMakeCache, build.json)"]
    ok, why = True, []
    for j, own in sdks:
        if own is None:
            ok = False; why.append(f"{j}: names no SDK with an sdk.json"); continue
        config = json.loads(j.read_text())
        if config.get("builtins_allowed"):
            why.append(f"{j}: builtins ALLOWED (recorded: {config['builtins_allowed'][:60]})"); continue
        words = list(config.get("codegen", [])) + list(own) + list(extra)
        if effective_no_builtin(words):
            why.append(f"{j}: effective -fno-builtin"); continue
        ok = False
        why.append(f"{j}: effective codegen is NOT -fno-builtin ({[w for w in words if 'builtin' in w] or 'neither flag'})")
    return ok, sorted(set(why))


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
