#!/usr/bin/env python3
"""Label an mruby ladder run's return values.

    python3 tools/report-ladder.py <log>

LABEL BY POSITION, not by marker tag. Most rungs return a raw 32-bit number and
only some carry the 0x6D52 tag, so a tag-driven reader silently drops exactly the
rungs that carry the measurement. Position works because the calls are made in
order, and a rung that faults simply ends the list.
"""
import re
import sys

# EXACTLY what md_snap() in port/md_probe.c writes, in its order. Keep the two in
# step: a label list that has drifted mislabels every number without ever looking
# wrong.
WORDS = [
    "ci->stack length",
    "ci->stack cursor - base",
    "nregs",
    "stack_keep",
    "stbase length",
    "ci->stack base - stbase base",
    "ci->stack base - heap base",
    "probe stack addr - heap base",
]

FIXED = {
    0: "anchor &domain_main",
    1: "entry + cap-init",
    2: "outer allocator",
    3: "malloc(64) capability LENGTH",
    4: "PROBE SELF-TEST (1 = the predicate fires)",
    5: "mrb_open_core",
    54: "probe calls (mrb_vm_run reached the clear)",
    55: "probe violations",
    56: "RE-READ DIFFERED (frames where c->ci->stack changed)",
    57: "mrb_gc_add_region",
    58: "run bytecode",
    59: "knobs<<16 | site(md_first)<<8 | site(md_last); knob 1=SKIP_CLEAR 2=FORCE_STACK; site 1=mrb_vm_run 2=exec_irep",
}

ARRAYS = [("md_first", 6, "the first call"),
          ("md_last", 22, "the most recent call seen"),
          ("md_viol", 38, "the frame that changed under the probe, or the first bad one")]


def label(i):
    if i in FIXED:
        return FIXED[i]
    if 6 <= i <= 53:
        name = ARRAYS[(i - 6) // 16][0]
        w, half = divmod((i - 6) % 16, 2)
        return "%-8s[%d] %s %s" % (name, w, WORDS[w], "high" if half else "low")
    return "call %d" % i


def sequences(text):
    """Split the log into per-domain-instance call sequences.

    The loader smoke-tests the image with its own single invocation BEFORE running
    the guest command, and that is a separate domain instance whose static call
    counter starts at zero again. Reading the log as one flat list therefore shifts
    every label by one and makes the loader's anchor look like the ladder's -- two
    different load bases that both look plausible. The "(N-th time)" index is what
    separates them: N resetting to 1 starts a new instance.
    """
    runs, cur = [], []
    for n, v in re.findall(r"Called dom \((\d+)-th time\) retval = (\d+)", text):
        if int(n) == 1 and cur:
            runs.append(cur)
            cur = []
        cur.append(int(v) & 0xFFFFFFFF)
    if cur:
        runs.append(cur)
    return runs


def main(path):
    text = open(path, "rb").read().decode("utf8", "replace")
    runs = sequences(text)
    vals = runs[-1] if runs else []
    if len(runs) > 1:
        print("  %d call sequences in the log; reporting the last (%d calls). "
              "Earlier: %s" % (len(runs), len(vals), [len(r) for r in runs[:-1]]))

    print()
    if not vals:
        # "no data" is an ERROR, not a zero: a boot flake and a wedge look
        # identical from here, and only the log tells them apart.
        print("   NO RETVAL AT ALL -- read %s before calling this a failure" % path)

    for i, v in enumerate(vals):
        note = ""
        if (v >> 16) == 0x6D52:
            note = "   marker stage %d code 0x%02X" % ((v >> 8) & 0xFF, v & 0xFF)
            if (v & 0xFF) == 0x77:
                note += "  <- the probe JUMPED OUT; md_viol below is the frame"
        print("  %3d  0x%08X  %-12d  %s%s" % (i, v, v, label(i), note))

    if len(vals) > 4:
        print("\n    probe self-test: %s" % (
            "PASSES, the predicate fires on a frame it must reject" if vals[4] == 1
            else "DOES NOT FIRE (%d) -- every verdict below is void" % vals[4]))

    for name, first, what in ARRAYS:
        if len(vals) < first + 16:
            continue
        print("\n    %s -- %s" % (name, what))
        for w in range(8):
            lo, hi = vals[first + 2 * w], vals[first + 1 + 2 * w]
            full = (hi << 32) | lo
            signed = full - (1 << 64) if full >= (1 << 63) else full
            print("      [%d] %-30s = %d  (0x%X)" % (w, WORDS[w], signed, full))

    if len(vals) >= 56:
        calls, viol = vals[54], vals[55]
        print("\n    mrb_vm_run reached the clear %d times, %d of them out of bounds"
              % (calls, viol))
        # The instrument's own honesty check. Both of these read as a clean result
        # and are not one.
        if calls and not viol:
            print("      -> md_viol is zeros that MEAN zero: no clear went out of"
                  " bounds. md_first still carries real data.")
        if not calls:
            print("      -> mrb_vm_run was NEVER reached, so BOTH arrays are"
                  " meaningless, not merely empty")

    for line in text.splitlines():
        if "capability fault" in line or "Cap mem access OOB" in line:
            print("   " + line.strip()[:150])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__))
