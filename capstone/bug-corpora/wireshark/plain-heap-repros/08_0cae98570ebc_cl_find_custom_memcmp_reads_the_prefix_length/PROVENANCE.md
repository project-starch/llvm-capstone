# 0cae98570ebc — Fix buffer overrun when applying prefs with command line prefs present

## The defect

`cl_find_custom` compared with `memcmp(elem_data, search_data, strlen(search_data))` — an unconditional read of the **prefix's** length from both operands. Any stored `-o` option shorter than the prefix being searched for is read past its end. `memcmp` is specified to read `n` bytes from both objects regardless of content, so the defect holds even where a byte-wise implementation would stop at the terminator; the fix uses `strncmp`, which stops there by contract.

## Upstream defect

- **Fix:** `0cae98570ebc`, *"Fix buffer overrun when applying prefs with command line prefs present"*, `ui/commandline.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The pin uses strncmp.

## The vulnerable code, quoted from the fix's parent

```c
static int cl_find_custom(const void *elem_data, const void *search_data) {
    return memcmp(elem_data, search_data, strlen((char *)search_data));
}
```

## The fix

```c
static int cl_find_custom(const void *elem_data, const void *search_data) {
    const char *prefix = (const char *)search_data;
    const char *opt_and_val = (const char *)elem_data;

    return strncmp(opt_and_val, prefix, strlen(prefix));
}
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is.

**Reduced:** no command line and no preference modules. The stored option is a bare allocation holding a proper C string, the prefix is longer, and the comparison is a byte loop so the crossing is attributable to the labelled probe rather than to libc's memcmp.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
