# 87803328179 — blf: don't assume that app text is null-terminated in the file.

## The defect

The APP_TEXT payload is allocated with `g_try_malloc0((gsize)apptextheader.textLength)` — exactly the declared length, with no room for a terminator — and `blf_read_bytes()` then fills all `textLength` bytes from the file, overwriting every zero byte the allocator placed. `g_strsplit_set(text, ";", -1)` walks it as a C string, so a file containing no zero byte makes the scan run off the end.

## Upstream defect

- **Fix:** `87803328179`, *"blf: don't assume that app text is null-terminated in the file."*, `wiretap/blf.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The pin allocates the extra byte.

## The vulnerable code, quoted from the fix's parent

```c
    gchar *text = g_try_malloc0((gsize)apptextheader.textLength);

    if (!blf_read_bytes(params, data_start + sizeof(apptextheader), text,
                        apptextheader.textLength, err, err_info)) {
    ...
    /* returns a NULL terminated array of NULL terminates strings */
    gchar **tokens = g_strsplit_set(text, ";", -1);
```

## The fix

```c
    /* Add an extra byte for a terminating '\0' */
    gchar *text = g_try_malloc((gsize)apptextheader.textLength + 1);
    ...
    text[apptextheader.textLength] = '\0'; /* Here's the '\0' */
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is.

**Reduced:** no BLF file and no tokeniser. The buffer is filled with non-zero bytes as the file would, and the scan is reduced to the loop that leaves the allocation.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
