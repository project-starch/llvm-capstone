# 381681583b -- pcapng put_nrb_option NRB custom-string over-copy

## The defect

A pcapng custom-string option's **value** is a 4-byte private enterprise number followed by the
string, so `put_nrb_option` computes `size = sizeof(uint32_t) + stringlen`. It writes the PEN
separately and then copied `size` bytes **out of the string**, which holds only `stringlen` bytes
and its terminator. The copy therefore reads `sizeof(uint32_t)` bytes beyond the string, of which
three lie past the end of its allocation.

## Upstream defect

- **Fix:** `381681583b`, *"wiretap: pcapng: fix put_nrb_option NRB custom-string over-copy"*,
  `wiretap/pcapng.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The pin copies `stringlen`. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
    case OPT_CUSTOM_STR_COPY:
        /* String options don't consider pad bytes part of the length */
        stringlen = strlen(optval->custom_stringval.string);
        size = sizeof(uint32_t) + stringlen;
        ...
        memcpy(*opt_ptrp, &optval->custom_stringval.pen, sizeof(uint32_t));
        *opt_ptrp += sizeof(uint32_t);
        memcpy(*opt_ptrp, optval->custom_stringval.string, size);
        *opt_ptrp += size;
```

## The fix

```c
        memcpy(*opt_ptrp, optval->custom_stringval.string, stringlen);
        *opt_ptrp += stringlen;
```

## What is real here, and what is reduced

**Real:** `size = sizeof(uint32_t) + stringlen`, the separate PEN write that makes `size` the wrong
length for the string copy, and the fix's replacement of `size` by `stringlen`. The buffer is a
plain allocation because upstream's is: this code is in `wiretap`, outside `epan`'s wmem scopes.

**Reduced:** no capture file, no block writer, no option table. The destination is a bare buffer and
the copy is a byte loop so the crossing is attributable to the labelled probe rather than to libc's
`memcpy`.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential -- the buggy arm reads
past the string's allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions. Nor
upstream reachability: writing a custom-string NRB option is the trigger, and this case does not
show that a capture file can request one.
