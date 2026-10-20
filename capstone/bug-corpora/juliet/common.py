"""Shared pieces of the Juliet harness: paths, the seed overlay, the twin instrumentation, and the source facts the
strict checks read."""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TOOLS = REPO / "capstone/bug-corpora/tools"
sys.path.insert(0, str(TOOLS))
CONTROLS_C = REPO / "capstone/bug-corpora/postgres/c-repros/shared/controls.c"

# The cause a detection must carry, per class (docs/design/virtual-capstone/isa.md cause table):
# 25 = a capability whose lifetime ended, 28 = an access outside the capability's bounds.
EXPECTED_CAUSE = {415: 25, 416: 25, 121: 28, 122: 28, 124: 28, 126: 28, 127: 28}
OVER = {121, 122, 126}      # classes whose defective access lies past the end
UNDER = {124, 127}          # classes whose defective access lies below the start

# Juliet's main seeds rand() with time(NULL); a seeded build replaces only that expression. Under musl 1.2.5's
# rand() (src/prng/rand.c: seed = 6364136223846793005*seed + 1, return seed>>33) these seeds give the four patterns
# of the first two rand()%2 results; every seeded batch checks that in the guest before its readings count.
SRAND = "srand( (unsigned)time(NULL) );"
SEEDS = {"11": 4, "10": 3, "01": 2, "00": 1}

# Families whose bad image can complete without the defect happening at run time, and the statement that decides it.
# A twin is the bad image with one print added at that statement (twin_source). judge.py relabels a MISSED case on its
# own twin's reading only where the deciding value is deterministic (wide-snprintf: what swprintf writes); a CWE-170
# twin reads an uninitialised stack word, which another execution need not share, so it is reported, never relabels.
FAMILIES = {
    "wide-snprintf": re.compile(r"wchar_t_(alloca_|declare_)?snprintf_\d\d[a-e]?$"),
    "cwe170": re.compile(r"__CWE170_(char|wchar_t)_(loop|memcpy|strncpy)_\d\d[a-e]?$"),
    "sizeof-ptr": re.compile(r"__sizeof_(double|int64_t|struct)_\d\d[a-e]?$"),
}
TWIN_RULES = {
    # after the copy: how many wide characters swprintf actually wrote into its destination. Both shapes Juliet uses:
    # CWE805 `SNPRINTF(data, 100, L"%s", source);` and CWE806 `SNPRINTF(dest, wcslen(data), L"%s", data);`
    "wide-snprintf": (re.compile(r'^(\s*)SNPRINTF\((\w+), [^;]*?, L"%s", \w+\);\s*$', re.M),
                      lambda m: f'{m.group(1)}printf("@@TWIN wcslen=%u\\n", (unsigned)wcslen({m.group(2)}));', "after"),
    # before the print: whether the unterminated buffer happens to end in a zero
    "cwe170": (re.compile(r"^(\s*)print(W?)Line\(dest\);\s*$", re.M),
               lambda m: f'{m.group(1)}printf("@@TWIN dest99=%u\\n", (unsigned)dest[99]);', "before"),
}


def family(case_name):
    return next((f for f, rx in FAMILIES.items() if rx.search(case_name)), None)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_env():
    """The environment every SDK compile runs in: no CAPSTONE_* (capstone-cc honours CAPSTONE_SDK, which would select
    another SDK than the one recorded) and no CCC_OVERRIDE_OPTIONS."""
    import os
    return {k: v for k, v in os.environ.items() if not k.startswith("CAPSTONE_") and k != "CCC_OVERRIDE_OPTIONS"}


def sdk_fingerprint(sdk_dir):
    """Hashes of everything an image is built from: the wrapper, sdk.json, and each artifact sdk.json names (compiler,
    linker, link script, runtime, libc, builtins), plus the codegen flags. Paths are not recorded."""
    sdk_dir = Path(sdk_dir)
    desc = json.loads((sdk_dir / "sdk.json").read_text())
    fp = {"capstone-cc": sha256(sdk_dir / "capstone-cc"), "sdk.json": sha256(sdk_dir / "sdk.json"),
          "codegen": desc.get("codegen"), "profile": desc.get("profile"), "allocator": desc.get("allocator")}
    for k in ("cc", "ld", "script", "runtime", "libc", "builtins"):
        fp[k] = sha256(desc[k]) if isinstance(desc.get(k), str) and Path(desc[k]).is_file() else None
    return fp


def load(path):
    return json.loads(Path(path).read_text())


def overlay(text):
    """The seeded variant of a case file: main's time seed replaced by JULIET_SEED, nothing else."""
    n = text.count(SRAND)
    if n > 1:
        raise ValueError("more than one srand(time) in one file")
    return text.replace(SRAND, "srand( (unsigned)JULIET_SEED );"), n


def twin_source(text, fam):
    """The twin of one case file: the family's print inserted at its deciding statement, only inside OMITBAD sections.
    Returns (text, number of insertions)."""
    rx, line, where = TWIN_RULES[fam]
    count = 0

    def section(m):
        nonlocal count

        def ins(s):
            nonlocal count
            count += 1
            add = line(s)
            return f"{s.group(0).rstrip()}\n{add}" if where == "after" else f"{add}\n{s.group(0).rstrip()}"
        return rx.sub(ins, m.group(0))
    out = re.sub(r"#ifndef OMITBAD.*?#endif /\* OMITBAD \*/", section, text, flags=re.S)
    return out, count


def bad_sections(juliet, case):
    """The OMITBAD sections of every file of a case (multi-file flows keep the sink in their b..e files)."""
    out = ""
    for f in case["files"]:
        t = (Path(juliet) / "testcases" / f).read_text(errors="replace")
        out += "".join(re.findall(r"#ifndef OMITBAD(.*?)#endif /\* OMITBAD \*/", t, re.S))
    return out


ELEM = {"char": 1, "wchar_t": 4, "int": 4, "short": 2, "long": 8, "int64_t": 8, "double": 8, "float": 4,
        "twoIntsStruct": 8, "size_t": 8, "unsigned": 4}


def declared_sizes(bad, elem=ELEM):
    """({byte size: {"stack"|"heap"}}, number of distinct candidate objects) for the objects a bad section declares or allocates, leaving out the good object
    (`dataGoodBuffer`) and copy sources (`source`, `src`): `T a[N], b[M];` and ALLOCA(...) are stack, malloc/calloc heap.
    Forms: arrays, `N*sizeof(T)` in either order, plain byte counts (CWE-131's `ALLOCA(10)`), single objects
    (`malloc(sizeof(T))`), the return_freed_ptr helper's `strlen("BadSink")+1`, and CWE-135's
    `(strlen((char *)wide)+1)*sizeof(wchar_t)`, where strlen of a wide "A..." string is 1."""
    out, names = {}, set()
    def add(n, kind, name=None):
        out.setdefault(n, set()).add(kind)
        names.add(name or f"<{kind}:{n}>")
    # The defective object is never the good buffer and never a copy's source (read within its bounds in every class)
    skip = lambda name: "good" in name.lower() or "source" in name.lower() or name.lower() == "src"
    for t, decl in re.findall(r"\b(\w+)\s+(\w+\s*\[\d+\][^;]*);", bad):
        if t in elem:
            for name, n in re.findall(r"(\w+)\s*\[(\d+)\]", decl):
                if not skip(name):
                    add(int(n) * elem[t], "stack", name)
    alloc = r"(\w+)\s*=\s*\([^)]*\)\s*(ALLOCA|malloc|calloc)\("
    for name, fn, rest in re.findall(alloc + r"([^;]*)\);", bad):
        if skip(name):
            continue
        kind = "stack" if fn == "ALLOCA" else "heap"
        m = re.fullmatch(r"\s*\(?(\d+)\)?\s*\*\s*sizeof\((\w+)\)\s*", rest) or None
        if m and m.group(2) in elem: add(int(m.group(1)) * elem[m.group(2)], kind, name); continue
        m = re.fullmatch(r"\s*sizeof\((\w+)\)\s*\*\s*\(?(\d+)\)?\s*", rest)
        if m and m.group(1) in elem: add(int(m.group(2)) * elem[m.group(1)], kind, name); continue
        m = re.fullmatch(r"\s*sizeof\((\w+)\)\s*", rest)
        if m and m.group(1) in elem: add(elem[m.group(1)], kind, name); continue
        m = re.fullmatch(r"\s*(\d+)\s*,\s*sizeof\((\w+)\)\s*", rest)
        if m and m.group(2) in elem: add(int(m.group(1)) * elem[m.group(2)], kind, name); continue
        m = re.fullmatch(r"\s*(\d+)\s*", rest)
        if m: add(int(m.group(1)), kind, name)
    m = re.search(r'helperBad\("([^"]*)"\)', bad)
    if m: add(len(m.group(1)) + 1, "heap")
    if "strlen((char *)data)" in bad and "sizeof(wchar_t)" in bad:
        add(2 * elem["wchar_t"], "stack" if "ALLOCA((dataLen+1)" in bad else "heap", "dest")
    return out, len(names)
