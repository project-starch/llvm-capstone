# 8dc7d164dcdb — prefs: prevent double-free on changing prefs

## The defect

`prefs_reset()` releases `prefs.saved_at_version` but leaves the struct field pointing at it. Two resets without an intervening `read_prefs()` therefore free one allocation twice — which is exactly what changing preferences in the GUI does.

## Upstream defect

- **Fix:** `8dc7d164dcdb`, *"prefs: prevent double-free on changing prefs"*, `epan/prefs.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The pin clears the field.

## The vulnerable code, quoted from the fix's parent

```c
void
prefs_reset(void)
{
  prefs_initialized = FALSE;
  g_free(prefs.saved_at_version);
```

## The fix

```c
  prefs_initialized = FALSE;
  g_free(prefs.saved_at_version);
  prefs.saved_at_version = NULL;
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows. The object is a plain allocation because upstream's is
— this code is outside `epan`'s wmem scopes.

**Reduced:** no preferences and no GUI. One allocation, one struct field, and the second reset's use of the field is the labelled probe rather than a second free.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing. Nor upstream reachability of
the specific sequence chosen here.
