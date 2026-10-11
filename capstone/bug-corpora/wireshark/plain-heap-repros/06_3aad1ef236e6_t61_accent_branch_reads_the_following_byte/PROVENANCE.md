# 3aad1ef236e6 — epan: Add a boundary check to get_t61_string.

## The defect

The accent-composition branch is entered on `(*c & 0xf0) == 0xc0` without first checking that a following byte exists, and then dereferences `c[1]`. At the last iteration — `i == length - 1` — that is `ptr[length]`, one byte past the end of the input buffer.

## Upstream defect

- **Fix:** `3aad1ef236e6`, *"epan: Add a boundary check to get_t61_string."*, `epan/charsets.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The pin carries the index check.

## The vulnerable code, quoted from the fix's parent

```c
    for (i = 0, c = ptr; i < length; c++, i++) {
        if (!t61_tab[*c]) {
            wmem_strbuf_append_unichar(strbuf, UNREPL);
        } else if ((*c & 0xf0) == 0xc0) {
            gint j = *c & 0x0f;
            if ((!c[1] || c[1] == 0x20) && accents[j]) {
```

## The fix

```c
        } else if (i < length - 1 && (*c & 0xf0) == 0xc0) {
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is.

**Reduced:** no tvbuff and no string builder. The input is a bare allocation whose last byte is an accent lead byte, and the second-byte read is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
