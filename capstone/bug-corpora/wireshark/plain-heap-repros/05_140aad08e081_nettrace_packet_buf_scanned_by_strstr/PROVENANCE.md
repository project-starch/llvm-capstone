# 140aad08e081 — nettrace_3gpp_32_423 Protect from buffer overun.

## The defect

`packet_buf` is allocated `packet_size + 12` bytes: twelve header bytes written by hand plus `packet_size` bytes read straight from the file. All of it is written, leaving no terminator, and the same function then does `strstr(curr_pos, "<fileHeader")` over it — so a file whose bytes never match and never contain a zero byte makes the search read past the end.

## Upstream defect

- **Fix:** `140aad08e081`, *"nettrace_3gpp_32_423 Protect from buffer overun."*, `wiretap/nettrace_3gpp_32_423.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The construct was removed rather than fixed at the pin.

## The vulnerable code, quoted from the fix's parent

```c
	packet_buf = (guint8 *)g_malloc(packet_size + 12);
	...
	curr_pos = packet_buf + 12;
	/* Find the file header */
	curr_pos = strstr(curr_pos, "<fileHeader");
```

## The fix

```c
	packet_buf = (guint8 *)g_malloc(packet_size + 12+1);
    /* Terminate buffer*/
    packet_buf[packet_size + 12] = 0;
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is.

**Reduced:** no nettrace file and no XML. The buffer is filled with non-matching, non-zero bytes and the search is reduced to the scan that leaves the allocation.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
