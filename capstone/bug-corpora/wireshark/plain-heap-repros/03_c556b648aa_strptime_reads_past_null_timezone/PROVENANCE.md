# c556b648aa -- strptime reads a character past a null timezone

## The defect

The timezone branch of `ws_strptime`'s format walk switches on the current character. It had arms
for `'+'`, `'-'` and a `default:` that falls through to the named-zone path, but **no arm for the
string terminator**. On an empty timezone the NUL was treated like any other character and the scan
advanced past it -- one byte beyond an allocation that holds exactly the terminator.

## Upstream defect

- **Fix:** `c556b648aa`, *"strptime: Don't read a character past a null timezone"*,
  `wsutil/ws_strptime.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The pin carries the `'\0'` arm. A fix-reversal.

## The vulnerable code, quoted from the fix's parent

```c
			case '-':
				neg = 1;
				break;
			default:
namedzone:
				bp = zname;
```

The terminator falls through to `default:`.

## The fix

```c
			case '\0':
				goto out;
```

## What is real here, and what is reduced

**Real:** the missing arm, and that the bound being crossed is a terminator rather than a length.
The buffer is a plain allocation because upstream's is: `wsutil` is outside `epan`'s wmem scopes.

**Reduced:** no format string, no `struct tm`, no named-zone table. The scan is reduced to the loop
that steps past the terminator, so the fault is attributable to the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential -- the buggy arm reads the
byte after the terminator, the fixed arm stops at it, and the arms differ by exactly the fix's arm.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions. Nor
that a caller passes a heap-allocated empty timezone -- the reduction supplies one so the crossing
has an allocation boundary to cross.
