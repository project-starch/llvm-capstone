#!/usr/bin/env python3
"""Triage upstream SPATIAL defects (overflow / out-of-bounds) for the three target programs.

The temporal hunt's instruments filtered spatial wording OUT -- wireshark-wmem-defect-triage.md:23
makes "not an overflow" an explicit disqualifier -- so the spatial class was never triaged. This is
the mirror instrument.

The question that decides a candidate's value is NOT "is it an overflow" but **which bound does the
overflow cross**:

  class A  the malloc/g_malloc bound ............ shrink and sublet already fault; CHERI too.
                                                  A tie row: no contribution.
  class B  a sub-allocation bound INSIDE a       . only a ported inner allocator faults
           nested allocator's block                (chunks / slabsublet* / pool*). The prize.
  class C  a sub-object bound inside ONE          . needs sub-object bounds; malloc-granular
           allocation                               bounds are blind. High value.

So filter 2 finds the overflowed buffer's ALLOCATION SITE and reads the allocator off it. It
reports evidence for a human to adjudicate; it never infers a class from the subsystem, which is
the existing triage docs' own rule.

Contract, deliberately: "no data" is an ERROR, not a zero (CLAUDE.md). A population that yields
nothing exits non-zero and says where it looked.

  spatial-triage.py --self-test           prove filter 2 separates a known A from a known B
  spatial-triage.py --program wireshark   triage one program
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

UPSTREAM = Path("/tmp/capstone/upstream")

# Populations mirror the temporal instruments' own choices, so the two hunts are comparable.
PROGRAMS = {
    "wireshark": {
        "repo": "wireshark",
        "pin": "v4.6.8",
        "population": "v4.6.8..origin/master",
        "nested_alloc": r"wmem_alloc|wmem_new|wmem_alloc0|wmem_new0|wmem_realloc|wmem_memdup|wmem_strdup",
        "plain_alloc": r"\bg_malloc|\bg_new|\bmalloc\s*\(|\bg_realloc|tvb_memdup|g_strdup",
        "nested_note": "a wmem scope: the chunk lives inside one big block the system allocator handed out",
    },
    "ffmpeg": {
        "repo": "ffmpeg",
        "pin": "n9.0.1",
        "population": "n9.0.1..origin/master",
        "nested_alloc": r"av_buffer_pool_get|av_refstruct_pool_get|ff_refstruct_pool_get|av_frame_get_buffer|ff_get_buffer|ff_thread_get_buffer",
        "plain_alloc": r"av_malloc|av_mallocz|av_calloc|av_realloc|av_buffer_alloc|\bmalloc\s*\(",
        "nested_note": "a pool buffer or a frame whose planes are carved from ONE AVBuffer",
    },
    "memcached": {
        "repo": "memcached",
        "pin": "1.6.45",
        "population": "1.6.45",          # the pin IS upstream head: 0 commits after it
        "nested_alloc": r"do_item_alloc|item_alloc|slabs_alloc|cache_alloc|do_slabs_alloc|mcp_page_carve",
        "plain_alloc": r"\bmalloc\s*\(|\bcalloc\s*\(|\brealloc\s*\(|\bstrdup\s*\(",
        "nested_note": "a slab item inside its 1 MiB page, or a cache.c object",
    },
}

# filter 1. Spatial wording. Kept here rather than in a shell grep because this must COUNT
# meaningfully, and because an ERE [^\n] does not mean newline (bit this project twice).
SPATIAL = re.compile(
    r"out[- ]of[- ]bounds|\bOOB\b|buffer overflow|overread|over-read|past the end"
    r"|off[- ]by[- ]one|underflow|bounds check|heap-buffer-overflow|overflow of"
    r"|write overflow|read overflow|out of array|overflowing|too small (?:buffer|array)",
    re.I,
)
# Wording that LOOKS spatial and is not: an integer/refcount overflow is a temporal driver here
# (memcached case 03 is exactly this trap), and a leak or DoS is out of class entirely.
NOT_SPATIAL = re.compile(
    r"integer overflow|refcount overflow|reference count overflow|memory leak|\bleak\b"
    r"|memory exhaustion|infinite loop|stack overflow(?! in)|accounting underflow"
    # SPATIAL matches bare `underflow`, so an INTEGER underflow subject passed filter 1 while
    # 'integer overflow' was excluded -- asymmetric, and it let wireshark 830cf562a0 through
    # into a table published as 'verified class A' (retracted 2026-10-05).
    r"|integer underflow|unsigned underflow|signed underflow|size underflow",
    re.I,
)


def git(repo, *args, check=True):
    out = subprocess.run(["git", "-C", str(UPSTREAM / repo), *args],
                         capture_output=True, text=True)
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed rc={out.returncode}: {out.stderr[:200]}")
    return out.stdout


def sha_exists(repo, sha):
    """git cat-file -t FIRST. --is-ancestor on a bad object exits 128, which reads as 'live'.

    A tag is a legitimate ref here (the pins are tags), so accept commit OR tag and let
    rev-parse^{commit} do the peeling -- demanding 'commit' made v4.6.8 read UNRESOLVED.
    """
    out = subprocess.run(["git", "-C", str(UPSTREAM / repo), "cat-file", "-t", sha],
                         capture_output=True, text=True)
    return out.returncode == 0 and out.stdout.strip() in ("commit", "tag")


def in_pin(repo, sha, pin):
    """IN-PIN / NOT-IN-PIN / UNRESOLVED by ANCESTRY. UNRESOLVED is a verdict, never a default.

    NOT SUFFICIENT for liveness on its own -- see fix_present_in_pin.
    """
    if not sha_exists(repo, sha):
        return "UNRESOLVED"
    r = subprocess.run(["git", "-C", str(UPSTREAM / repo), "merge-base", "--is-ancestor", sha, pin],
                       capture_output=True, text=True)
    if r.returncode == 0:
        return "IN-PIN"
    if r.returncode == 1:
        return "NOT-IN-PIN"
    return "UNRESOLVED"


def fix_present_in_pin(repo, sha, pin):
    """Is the FIX's own code already in the pinned source? Read the tree, not the ancestry.

    Ancestry alone is wrong and it cost a false 'live' verdict on wireshark e8ef9df09d: the
    master commit is not an ancestor of v4.6.8, yet v4.6.8 already carries the fix -- it was
    backported under a different hash, which the wireshark triage doc warns about in as many
    words ("master carries the same fixes under different hashes"). Its own rule is the one to
    follow: "Liveness proved by reading the pinned tree, not by the fix date".

    So: take the distinctive lines the fix ADDED and look for them in the pinned file.
      FIX-IN-PIN      the added code is there -> the defect is NOT live at the pin
      DEFECT-LIVE     none of it is there    -> the defect is live (confirm by hand)
      UNRESOLVED      no file, or nothing distinctive enough to test
    """
    files = [f for f in git(repo, "show", "--name-only", "--format=", "-M", sha).split()
             if f.endswith((".c", ".cpp", ".h"))]
    if not files:
        return "UNRESOLVED", []
    diff = git(repo, "show", "--format=", "-U0", sha)
    added = []
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            body = line[1:].strip()
            # Distinctive = long enough to be unique, and not a comment or a lone brace.
            if len(body) >= 25 and not body.startswith(("/*", "*", "//", "#include")):
                added.append(body)
    if not added:
        return "UNRESOLVED", []

    found, probed = 0, 0
    for path in files[:4]:
        try:
            pinned = git(repo, "show", f"{pin}:{path}")
        except RuntimeError:
            continue
        for body in added[:25]:
            probed += 1
            if body in pinned:
                found += 1
    if probed == 0:
        return "UNRESOLVED", []
    # Any distinctive added line present in the pin means the fix (or its backport) is there.
    if found:
        return "FIX-IN-PIN", [f"{found}/{probed} of the fix's added lines are already in {pin}"]
    return "DEFECT-LIVE", [f"none of {probed} distinctive added lines appear in {pin}"]


# ---------------------------------------------------------------- filter 2

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SUBSCRIPT = re.compile(r"(\w+)\s*\[")


# A stack array declaration: `char temp[KEY_MAX_LENGTH + 30];`. memcached 76a6c36 overflows one of
# these, which is neither class A nor B -- no allocator is involved, so the nested story cannot
# apply. Detected explicitly rather than left to fall through as "unresolved".
STACK_DECL = re.compile(
    r"^\s*(?:static\s+)?(?:const\s+)?"
    r"(?:char|u?int\d*_t|unsigned|signed|short|long|uint8_t|uint16_t|uint32_t|float|double|bool)"
    r"[\w\s\*]*\b(\w+)\s*\[[^\]]+\]\s*;")


def buffers_in_diff(diff):
    """Identifiers the fix's hunks subscript, copy into, or ASSIGN -- the overflow candidates.

    Assigned names matter because the overflowed buffer is often an alias: memcached ddee3e2
    reads past `auth_cur`, which is derived from `auth_data = calloc(...)`. Collecting only
    subscripted names loses the allocation site one hop away and the candidate reads as unresolved.
    """
    names, stack_decls = [], []
    for line in diff.splitlines():
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        body = line[1:]
        m = STACK_DECL.match(body)
        if m:
            stack_decls.append(m.group(1))
        for m in SUBSCRIPT.finditer(body):
            names.append(m.group(1))
        for fn in ("memcpy", "memmove", "memset", "strcpy", "strncpy", "snprintf", "sprintf"):
            m = re.search(rf"{fn}\s*\(\s*&?\s*([A-Za-z_]\w*)", body)
            if m:
                names.append(m.group(1))
        # `name = ...` on a changed line: picks up the allocation itself when the fix touched it
        m = re.match(r"\s*(?:[\w\*\s]+?\s+)?\*?\s*([A-Za-z_]\w*)\s*=[^=]", body)
        if m:
            names.append(m.group(1))
    KEYWORDS = {"if", "for", "while", "return", "sizeof", "int", "unsigned", "char", "void",
                "const", "static", "else", "switch", "case", "do", "struct", "typedef"}
    seen, uniq = set(), []
    for n in names:
        if n not in seen and n not in KEYWORDS:
            seen.add(n)
            uniq.append(n)
    return uniq, stack_decls


def classify(program, sha):
    """Locate each candidate buffer's allocation site in the fix's PARENT and read the allocator.

    Returns (proposed_class, evidence_lines). The parent is the vulnerable tree, which is the
    only tree in which the allocation site is the one the defect actually used.
    """
    cfg = PROGRAMS[program]
    repo = cfg["repo"]
    files = [f for f in git(repo, "show", "--name-only", "--format=", "-M", sha).split()
             if f.endswith((".c", ".cpp", ".h"))]
    diff = git(repo, "show", "--format=", sha)
    names, stack_decls = buffers_in_diff(diff)
    nested_re = re.compile(cfg["nested_alloc"])
    plain_re = re.compile(cfg["plain_alloc"])

    evidence, saw_nested, saw_plain = [], False, False
    for path in files[:6]:
        try:
            src = git(repo, "show", f"{sha}^:{path}")
        except RuntimeError:
            continue
        for lineno, line in enumerate(src.splitlines(), 1):
            for name in names[:16]:
                # an assignment to the candidate buffer, on the same line as an allocator call
                if re.search(rf"\b{re.escape(name)}\b\s*=", line) or re.search(rf"\*\s*{re.escape(name)}\b", line):
                    if nested_re.search(line):
                        saw_nested = True
                        evidence.append(f"NESTED  {path}:{lineno}  {line.strip()[:120]}")
                    elif plain_re.search(line):
                        saw_plain = True
                        evidence.append(f"PLAIN   {path}:{lineno}  {line.strip()[:120]}")
    if saw_nested:
        proposed = "B"          # inside a nested allocator's block -- only a ported arm faults
    elif saw_plain:
        proposed = "A"          # crosses the malloc bound -- shrink/sublet and CHERI already fault
    elif stack_decls:
        proposed = "STACK"      # no allocator involved; out of class for the nested story
        evidence.append(f"STACK   declared in the hunk: {', '.join(stack_decls[:4])}")
    else:
        proposed = "?"
    return proposed, evidence, names


def triage(program, limit=None, only_nested=False):
    cfg = PROGRAMS[program]
    repo = cfg["repo"]
    log = git(repo, "log", "--format=%h%x09%s", "--no-merges", cfg["population"]).splitlines()
    if not log:
        print(f"ERROR: population {cfg['population']!r} in {UPSTREAM/repo} produced NO commits. "
              f"Fetch the clone, or the ref is wrong.", file=sys.stderr)
        return 2

    rows = []
    for line in log:
        sha, _, subj = line.partition("\t")
        if not SPATIAL.search(subj) or NOT_SPATIAL.search(subj):
            continue
        rows.append((sha, subj))

    if not rows:
        print(f"ERROR: filter 1 matched 0 of {len(log)} commits for {program}. A zero here is an "
              f"instrument result, not a finding about {program}.", file=sys.stderr)
        return 2

    print(f"## {program}: population {len(log)} commits -> filter 1 kept {len(rows)}")
    print(f"##   nested allocators looked for: {cfg['nested_note']}")
    out = []
    for sha, subj in rows[:limit]:
        cls, evidence, names = classify(program, sha)
        ancestry = in_pin(repo, sha, cfg["pin"])
        source, live_ev = fix_present_in_pin(repo, sha, cfg["pin"])
        # The SOURCE read decides; ancestry is kept only to show where the two disagree, which
        # is the backport case and is the whole reason ancestry cannot be trusted alone.
        disagree = (ancestry == "NOT-IN-PIN" and source == "FIX-IN-PIN")
        out.append({"sha": sha, "subject_redacted": bool(re.search(r"[Ff]ixes from|[Tt]hanks", subj)),
                    "class": cls, "liveness": source, "ancestry": ancestry,
                    "backported_under_another_hash": disagree,
                    "evidence": evidence, "liveness_evidence": live_ev, "buffers": names[:8]})
        flag = "<<<" if cls in ("B", "C") and source == "DEFECT-LIVE" else "   "
        if only_nested and cls not in ("B", "C"):
            continue
        note = "  (BACKPORTED: ancestry said live, the pin says fixed)" if disagree else ""
        print(f"{flag} [{cls:5}] [{source:11}] {sha}{note}")
        for e in evidence[:3]:
            print(f"        {e}")
        for e in live_ev:
            print(f"        {e}")
    print(f"\n## proposed class B: {sum(1 for r in out if r['class']=='B')}, "
          f"class A: {sum(1 for r in out if r['class']=='A')}, "
          f"unresolved: {sum(1 for r in out if r['class']=='?')}")
    Path(f"/tmp/capstone/spatial-triage-{program}.json").write_text(json.dumps(out, indent=2))
    print(f"## written: /tmp/capstone/spatial-triage-{program}.json")
    return 0


def self_test():
    """filter 2 must SEPARATE a known class B from a known class A, and the liveness probe's
    three controls must fire. A classifier that has never disagreed is unproven."""
    ok = True

    # Known class B, adjudicated by hand: wireshark opcua d24613c461. Its overflowed buffer is
    # `plaintext`, allocated at opcua.c:652 of the fix's parent with wmem_alloc(pinfo->pool, ...).
    cls, ev, _ = classify("wireshark", "d24613c461")
    good = cls == "B" and any("wmem_alloc" in e for e in ev)
    print(f"  [{'PASS' if good else 'FAIL'}] known class B (opcua d24613c461) -> {cls}")
    for e in ev[:2]:
        print(f"         {e}")
    ok &= good

    # Known class A: memcached ddee3e2, `auth_data = calloc(1, sb.st_size)` read past by auth_cur.
    # A plain heap allocation, so bounds at malloc granularity already cover it.
    cls_a, ev_a, _ = classify("memcached", "ddee3e2")
    good_a = cls_a == "A"
    print(f"  [{'PASS' if good_a else 'FAIL'}] known class A (memcached ddee3e2) -> {cls_a}")
    for e in ev_a[:2]:
        print(f"         {e}")
    ok &= good_a

    # Known STACK: memcached 76a6c36 overflows `char temp[KEY_MAX_LENGTH + 30]`. No allocator is
    # involved at all, so neither A nor B can be the honest answer -- this control exists because
    # my first run of this self-test used it AS the class-A control and it read '?', which is
    # correct for a stack array and wrong for the control.
    cls_s, ev_s, _ = classify("memcached", "76a6c36")
    good_s = cls_s == "STACK"
    print(f"  [{'PASS' if good_s else 'FAIL'}] known STACK (memcached 76a6c36) -> {cls_s}")
    for e in ev_s[:2]:
        print(f"         {e}")
    ok &= good_s

    # The classifier must DISAGREE across the three. One verdict for all = vacuous.
    sep = len({cls, cls_a, cls_s}) == 3
    print(f"  [{'PASS' if sep else 'FAIL'}] filter 2 separates all three: {cls} / {cls_a} / {cls_s}")
    ok &= sep

    # Liveness controls, the three the wireshark checker uses.
    c1 = in_pin("wireshark", "v4.6.8", "v4.6.8")
    c2 = in_pin("wireshark", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef", "v4.6.8")
    c3 = in_pin("wireshark", "origin/master", "v4.6.8")
    for label, got, want in (("in-pin commit reads IN-PIN", c1, "IN-PIN"),
                             ("nonexistent sha reads UNRESOLVED", c2, "UNRESOLVED"),
                             ("newest commit does not read IN-PIN", c3, "NOT-IN-PIN")):
        good = got == want
        print(f"  [{'PASS' if good else 'FAIL'}] {label}: {got}")
        ok &= good

    # filter 3 must catch the BACKPORT, the defect that made this check necessary. wireshark
    # e8ef9df09d is not an ancestor of v4.6.8, so ancestry calls the defect live; v4.6.8 already
    # carries PFT_RS_K_MAX and the widened allocation, so the source says fixed. The source wins.
    anc = in_pin("wireshark", "e8ef9df09d", "v4.6.8")
    src, src_ev = fix_present_in_pin("wireshark", "e8ef9df09d", "v4.6.8")
    good_b = anc == "NOT-IN-PIN" and src == "FIX-IN-PIN"
    print(f"  [{'PASS' if good_b else 'FAIL'}] backport caught (e8ef9df09d): "
          f"ancestry={anc}, pinned source={src}")
    for e in src_ev:
        print(f"         {e}")
    ok &= good_b

    # ... and it must still be able to say DEFECT-LIVE, or it is a one-sided check. A commit well
    # after the pin on master, whose code cannot be in v4.6.8.
    newest = git("wireshark", "log", "-1", "--format=%h", "origin/master").strip()
    src_live, _ = fix_present_in_pin("wireshark", newest, "v4.6.8")
    good_l = src_live in ("DEFECT-LIVE", "UNRESOLVED")
    print(f"  [{'PASS' if good_l else 'FAIL'}] filter 3 can still say live (master head "
          f"{newest}): {src_live}")
    ok &= good_l

    # filter 1 must reject the trap: an integer/refcount overflow is not spatial.
    trap = SPATIAL.search("refcount overflow frees linked item") and not NOT_SPATIAL.search(
        "refcount overflow frees linked item")
    print(f"  [{'PASS' if not trap else 'FAIL'}] filter 1 rejects 'refcount overflow' as non-spatial")
    ok &= not trap

    # ... and the underflow half of the same trap, which it did NOT reject until 2026-10-05.
    # Two-sided: an integer underflow is out, a buffer underflow (a real spatial shape) stays in.
    for subject, want_spatial in (("fix integer underflow in packet length", False),
                                  ("fix unsigned underflow when computing remaining", False),
                                  ("fix buffer underflow when seeking backwards", True)):
        got = bool(SPATIAL.search(subject)) and not bool(NOT_SPATIAL.search(subject))
        good = got == want_spatial
        print(f"  [{'PASS' if good else 'FAIL'}] filter 1 {'keeps' if want_spatial else 'rejects'}"
              f" {subject!r}: spatial={got}")
        ok &= good

    # KNOWN BLIND SPOT of filter 2, kept as a control so it cannot be forgotten: the allocation is
    # matched anywhere in the fix's file, not near the overflowed object, so a file that happens to
    # contain a heap allocation makes a STACK overflow read as class A. memcached 11b5f9b overflows
    # `char temp[KEY_MAX_LENGTH + 1]` at proxy_lua.c:717 and reads A because of an unrelated
    # realloc in the same file. The control asserts the WRONG answer on purpose: a class-A verdict
    # from this tool is a CANDIDATE, and the allocation site must be opened before it is published.
    cls_blind, _, _ = classify("memcached", "11b5f9b")
    blind = cls_blind == "A"
    print(f"  [{'PASS' if blind else 'CHANGED'}] known blind spot (11b5f9b, a stack array) still "
          f"reads {cls_blind}: an A verdict is a candidate, not a finding")
    if not blind:
        print("         filter 2 changed -- re-read the blind-spot note above and this control")

    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", choices=sorted(PROGRAMS))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--population",
                    help="override the default population, e.g. a program's whole history. "
                         "Fix-reversal cases are acceptable -- every existing memcached and "
                         "FFmpeg corpus case is one -- so the post-pin window is not the only "
                         "useful range.")
    ap.add_argument("--only-nested", action="store_true",
                    help="print only class B/C rows, for a wide population")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.program:
        ap.error("--program or --self-test")
    if a.population:
        PROGRAMS[a.program] = dict(PROGRAMS[a.program], population=a.population)
    return triage(a.program, a.limit, only_nested=a.only_nested)


if __name__ == "__main__":
    sys.exit(main())
