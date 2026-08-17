#!/usr/bin/env python3
"""Write RESULT.txt into each measured case directory.

Kept as a script rather than six hand-written files so the numbers stay in one
place and cannot drift apart between directories.
"""
import os

S = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(S, "cases")

COMMON = """
How to read the domain column. A quiet return is only meaningful because the same
image also faults on other tests and returns the expected value for the sanity
test, so capability checking is live and the interpreter runs. See
../../evidence/domain-round2-2026-08-17.txt.
"""

RESULTS = {
"MPY-T09_bytearray-resize-stale-view": """MPY-T09 / upstream 18168

  stock MicroPython at pin 2e3304a : returns, prints "T09 1 1"
  Capstone domain under QEMU       : returns, prints "T09 1 1", retval 0x0018e966

UNTRAPPED, AND THE CORRUPTION IS VERIFIED.

The retval encodes an FNV-1a of the output. 0x0018e966 was computed from the
stock output BEFORE the domain run, so the match is a prediction that held, not
a number read afterwards and declared to agree.

"T09 1 1" decodes as: the memoryview diverged from the bytearray, so it addresses
storage the bytearray no longer owns, AND the write through it did not reach the
bytearray. proof-dangling.py measures the same thing directly and reports the
view reading recycled heap content (0xf0) where the bytearray reads 'A'.

So a use-after-free WRITE executed inside a pure-capability domain and corrupted
storage, exactly as it does on hardware with no capability support at all. This
is the sharpest row in the corpus: silent on stock, silent on Capstone, and
visible only once the collector is made capability-aware.
""",

"MPY-T10_array-resize-stale-view": """MPY-T10 / upstream 18171

  stock MicroPython at pin 2e3304a : returns, prints "T10 1 1"
  Capstone domain under QEMU       : returns, prints "T10 1 1", retval 0x00289ce0

UNTRAPPED, AND THE CORRUPTION IS VERIFIED. Same shape as MPY-T09 but on
array('I') instead of bytearray, and 0x00289ce0 was likewise computed from the
stock output in advance.

THIS REPRO IS AN ADAPTATION, and the adaptation was checked rather than assumed.
Upstream 18171 uses open() plus readinto() as the vehicle for writing through the
stale view; this domain has MICROPY_VFS=0 and no filesystem, so the write is done
directly instead. The defect under test, resizing an exporter while a memoryview
is live, is unchanged. The unmodified original was run on stock to confirm it too
completes silently there, printing OK, so the substitution did not turn a
crashing case into a quiet one.
""",

"MPY-T11_sort-mutates-under-gc": """MPY-T11 / upstream 17941

  stock MicroPython at pin 2e3304a : SIGSEGV
  Capstone domain under QEMU       : FAULT, cause 24, pc 0x101619b00

FAULTED, BUT NOT FOR TEMPORAL SAFETY, and the corpus records it that way.

Cause 24 is "Cap mem access requires capability": an untagged word used as a
pointer. That is the tag check catching a wild pointer, which is the same defect
the MMU catches on stock, where this segfaults. Capstone did not establish that
an object was dead; it observed that a word was not a capability. Any
pointer-integrity mechanism stops this case.

Counting this as a win would be the easiest way to overstate the result, which is
why domain_behaviour says fault-cause24 and not "caught".
""",

"MPY-T12_dict-eq-reentrant-clear": """MPY-T12 / upstream 18619

  stock MicroPython at pin 2e3304a : SIGSEGV
  Capstone domain under QEMU       : FAULT, cause 24, pc 0x101619b00

FAULTED, BUT NOT FOR TEMPORAL SAFETY. Same reading as MPY-T11, and note the pc is
identical, so both bugs converge on the same faulting instruction: the
interpreter dereferencing a word that is not a capability.
""",

"MPY-T13_write-callback-grows-buffer": """MPY-T13 / upstream 19075

  stock MicroPython at pin 2e3304a : SIGSEGV
  Capstone domain under QEMU       : returns, prints "T13 survived", retval 0x005da053

UNTRAPPED, BUT THE CORRUPTION IS NOT VERIFIED. This row is deliberately weaker
than MPY-T09 and MPY-T10 and must not be quoted as if it were the same result.

The quiet return looked exactly like a test that never created its triggering
condition, so it was diagnosed before being recorded. probe-buffer-type.py runs
in the same domain and reports:

  T13type ['bytearray', 'bytearray']

json.dump hands write() a BYTEARRAY in the domain, precisely as it does on stock,
so the in-place mutation happens and the buffer is reallocated under a caller
that still holds the old pointer. The trigger is created and the domain does not
fault.

What was measured is survival. Whether memory was corrupted was NOT measured, and
the difference from stock is unexplained: the same trigger crashes there. The
plausible reading is that the stale write lands harmlessly inside the 384 KiB
heap rather than on an unmapped page, but that is a hypothesis and this file is
not going to record it as a finding.
""",

"MPY-T02_objarray-bytes-self-copy-uaf": """MPY-T02 / CVE-2024-8947, and MPY-T05 / upstream 13283 are the same defect

  stock MicroPython at the pin        : fixed, nothing to measure
  stock at the fix's PARENT ce491ab0d1: runs to completion, no crash, no diagnostic
  Capstone domain                     : not run

MEASURED AT THE PARENT, AND THE DEFECT IS SILENT.

    extend  len 128 head b'AA'
    slice   len 128 head b'BB'
    exit 0

Both cases the fix commit names, extending a bytearray from itself and assigning
to a slice from itself, execute at the pre-fix commit and produce no visible
difference whatsoever. This is not a failed reproduction. It is what this defect
does: m_renew moves the buffer and array_extend keeps using the argument's cached
pointer, but that pointer never surfaces in a Python-visible value.

The fix's author recorded the same thing: in default configurations the bug
"exists but has no impact", and reproducing it required "running the unix port
under valgrind with GC-aware extensions".

So a published CVE against this runtime is invisible to the language, invisible
to a crash, and, per ../../evidence/asan-blindness-2026-08-17.txt, invisible to
AddressSanitizer as well. That is the nested-allocator problem stated by a case
rather than by an argument.

BUILDING THE PARENT: use gcc-12, not the default gcc 15, and do not add
AddressSanitizer. Recipe and reasons in
../../evidence/parent-build-attempt-2026-08-17.txt.
""",

"MPY-T08_stdio-close-then-use": """MPY-T08 / upstream 12670

  stock at the pin                    : fixed, raises ValueError
  stock at the fix's PARENT 3b954698fa: SIGSEGV, exit 139
  Capstone domain                     : not run

REPRODUCED AT THE PARENT. The trigger is two lines, sys.stdout.close(), and at
the pre-fix commit it takes the interpreter down before even the first
sys.stderr.write escapes: stdout empty, stderr empty, exit 139.

The comparison against the pin is what makes that readable. The same script on
the pinned build prints

  T08 survived close
  T08 EXC <class 'ValueError'>

so the fix made the standard streams non-closable and the later write raises
rather than faulting. Two builds, one script, one difference.

A NOTE ON THE FIRST ATTEMPT. The verdict was originally printed with print(),
which writes to the stdout the test has just closed, so the run produced no
output on either build and looked identical. Routing the verdict to stderr is
what separated them. A test whose observable travels through the thing under
test cannot distinguish anything.

BUILDING THE PARENT: CC=gcc-12, and run `make -C ports/unix submodules` in the
new worktree first; mbedtls and berkeley-db are not shared between worktrees and
their absence shows up as a confusing fatal error about missing headers.
""",

"MPY-T15_readblocks-grows-buffer": """MPY-T15 / upstream 17848, with MPY-T14 / 19060 closed as its duplicate

  stock at the pin                    : fixed, raises ValueError
  stock at the fix's PARENT 4e6dc0b569: accepted, runs to completion, exit 0
  Capstone domain                     : not run

THE DEFECT CONDITION IS PRESENT AT THE PARENT, AND IT IS SILENT THERE.

repro.py is the reporter's script verbatim. Its readblocks callback assigns
bytearray(1 + SEC_SIZE) into a buffer of SEC_SIZE, which enlarges the buffer the
caller handed it and reallocates under a pointer the caller still holds.

  parent: the assignment is accepted, readblocks returns, vfs.VfsFat(bdev)
          completes, exit 0, nothing on stderr
  pin:    ValueError: lhs and rhs should be compatible, at the assignment

So the fix is a size check at the boundary, and the parent build is what still
lets the enlargement through.

WHAT WAS NOT REPRODUCED. The issue is titled "crash", and no crash was observed.
An extended attempt that also calls vfs.mount() gets an ordinary OSError, because
the sparse device has no valid filesystem, and still exits 0. The row therefore
says silent-no-effect rather than crash-sigsegv. The reporter presumably reached
a crash through more filesystem activity than the published script performs;
recording it as a crash on that assumption would be recording their title rather
than a measurement.
""",

"MPY-T25_stringio-subclass-print": """MPY-T25 / upstream 10402

  stock MicroPython at pin 2e3304a : SIGSEGV
  Capstone domain under QEMU       : FAULT, cause 24, pc 0x10166007c

FAULTED, BUT NOT FOR TEMPORAL SAFETY. Same class as MPY-T11 and MPY-T12, at a
DIFFERENT pc, so these are distinct instructions failing the same check rather
than one shared crash site.
""",
}


def main():
    n = 0
    for d, text in RESULTS.items():
        p = os.path.join(CASES, d, "RESULT.txt")
        if not os.path.isdir(os.path.join(CASES, d)):
            print(f"missing case dir: {d}")
            return 1
        open(p, "w").write(text.rstrip() + "\n" + COMMON)
        n += 1
    print(f"wrote {n} RESULT.txt files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
