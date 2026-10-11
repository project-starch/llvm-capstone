# 07ffcf90426b — extcap: Avoid double free of help.

## The defect

`interfaces_cb()` duplicated the help text once per EXTCAP sentence and then assigned that single pointer into **each** `extcap_interface` the loop produced. Every one of those structs is later destroyed by `extcap_free_interface()`, which frees `->help`, so a tool exposing N interfaces frees one string N times. The fix duplicates at the assignment instead.

## Upstream defect

- **Fix:** `07ffcf90426b`, *"extcap: Avoid double free of help."*, `extcap.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The pin duplicates per interface.

## The vulnerable code, quoted from the fix's parent

```c
            help = g_strdup(int_iter->help);
            ...
            int_iter->help = help;
```

## The fix

```c
            help = int_iter->help;
            ...
            int_iter->help = g_strdup(help);
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows. The object is a plain allocation because upstream's is
— this code is outside `epan`'s wmem scopes.

**Reduced:** no extcap tool and no interface list. Two owners stand for two interfaces, the duplication is each arm's own, and the second owner's release is reduced to a read through its pointer.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing. Nor upstream reachability of
the specific sequence chosen here.
