"""C-81's one reading, shared by every corpus runner: a corpus reading must not come from a builtins build.

With malloc/realloc as library builtins LLVM may elide an allocation whose result is only tested against NULL and
assume it succeeded (upstream semantics), so a case that observes an allocation failure reads nothing. A runner calls
gate([its build inputs]) right after parsing its arguments, before anything is built, staged or booted.

A target is an SDK directory (its sdk.json); a build directory holding image/sdk/sdk.json, source/runtime/sdk.json, runtime/sdk.json or
runtime-virtual/sdk/sdk.json (every one present); a CMake build (its CMakeCache's CAPSTONE_SDK, plus CMAKE_C_FLAGS); a hosted corpus build (every
work/*/CMakeCache.txt); a scripted build recording its SDK in build.json; or an image file, whose SDK is the nearest of
its directory and four ancestors that names one. Every SDK a target names must have an EFFECTIVE -fno-builtin (clang
lets the last of -fbuiltin / -fno-builtin win, over the SDK's codegen, the build's own flags and any extra words) or
record "builtins_allowed" (build-sdk.sh's CAPSTONE_SDK_ALLOW_BUILTINS=1). A target that names none is refused: no data
is a refusal, not a pass. Refusal exits 78.
"""
from pathlib import Path

REFUSED = 78


import json, re, sys


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
    if t.is_file():                      # an image: the build it sits in, found by walking up (find_sdks_up)
        return find_sdks_up(t.parent)
    named = [c for c in (t / "sdk.json", t / "image/sdk/sdk.json", t / "source/runtime/sdk.json", t / "runtime/sdk.json",
                         t / "runtime-virtual/sdk/sdk.json") if c.is_file()]
    if named:                            # a build root can hold more than one (perl: runtime and runtime-virtual/sdk)
        return [(c, []) for c in named]
    found = []
    for cache in [t / "CMakeCache.txt", *sorted(t.glob("work/*/CMakeCache.txt"))]:
        if cache.is_file():
            found.append(_cache_sdk(cache) or (cache, None))
    bj = t / "build.json"
    if bj.is_file():
        sdk = json.loads(bj.read_text()).get("sdk")
        found.append(((Path(sdk) / "sdk.json"), []) if sdk and (Path(sdk) / "sdk.json").is_file() else (bj, None))
    return found


def find_sdks_up(d, levels=4):
    """An image's SDK: the nearest of its directory and up to `levels` ancestors that names one."""
    d = Path(d)
    for _ in range(levels + 1):
        found = find_sdks(d)
        if found:
            return found
        if d.parent == d:
            break
        d = d.parent
    return []


def check(target, extra=()):
    sdks = find_sdks(target)
    if not sdks:
        return False, [f"{target}: no sdk.json found (looked for sdk.json, image/sdk, source/runtime, runtime, runtime-virtual/sdk, CMakeCache, "
                       "work/*/CMakeCache, build.json; for an image, its directory and four ancestors)"]
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


def gate(targets, extra=(), who="runner"):
    """Exit 78 unless every target passes; print the reading either way (stderr)."""
    targets = [t for t in targets if t]
    if not targets:
        print(f"SDK-BUILTINS REFUSED {who}: no build input named (no data is a refusal)", file=sys.stderr)
        sys.exit(REFUSED)
    bad = False
    for t in targets:
        ok, why = check(t, extra)
        print(f"SDK-BUILTINS {'OK' if ok else 'REFUSED'} {t}", file=sys.stderr)
        for w in why:
            print(f"    {w}", file=sys.stderr)
        bad = bad or not ok
    if bad:
        print(f"SDK-BUILTINS {who}: refused before anything starts (exit {REFUSED})", file=sys.stderr)
        sys.exit(REFUSED)
