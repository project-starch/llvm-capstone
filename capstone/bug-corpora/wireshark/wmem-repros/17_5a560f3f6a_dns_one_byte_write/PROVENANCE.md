# 5a560f3f6a — the DNS name buffer is given a size one larger than itself

## The defect

`expand_dns_name` builds the printable name into a packet-scope buffer of `maxname` bytes and then
hands `g_snprintf` a size of `maxname + 1`, so the terminator can land **one byte past the chunk**.
**The smallest crossing in the corpus: exactly one byte, and a write.**

## Upstream defect

- **Fix:** `5a560f3f6a`, *"dns: fix off-by-one buffer overflow (write)"*. The same three calls with
  `maxname` instead of `maxname + 1`.
- **CVE:** none assigned.
- **Live in our pin: NO** — a fix-reversal.

## The vulnerable code, quoted from the fix's parent

`5a560f3f6a^:epan/dissectors/packet-dns.c`, three sites:

```c
np = (guchar *)wmem_alloc(wmem_packet_scope(), maxname);
...
print_len = g_snprintf(np, maxname + 1, "\\[x");
print_len = g_snprintf(np, maxname + 1, "%02x", ...);
print_len = g_snprintf(np, maxname + 1, "/%d]", bit_count);
```

## This row is a worked example of why a liveness PROBE is not a liveness TEST

The triage script reported this one `DEFECT-LIVE`, and it is not. Upstream renamed `g_snprintf` to
`snprintf`, so **none of the fix's added lines matches the pinned file verbatim** even though the
fix is there. Reading the pin settles it: `v4.6.8:epan/dissectors/packet-dns.c` has
`snprintf(np, maxname, …)` at `:1677`, `:1689` and `:1703` — the fixed form — with the allocation
`np=(char *)wmem_alloc(scope, maxname)` at `:1619`, and the vulnerable `maxname + 1` form has **0
occurrences**. This is the fifth measured instance of that false-live mode; see
`docs/ref/memcached-spatial-defect-triage.md` §5, which quantifies it at about 20%.

## What is real here, and what is reduced

**Real:** the allocator. `wmem`'s core and its allocators from the pinned 4.6.8 release, so the
chunk geometry the case turns on — a block from one `g_malloc`, chunks carved inside it — is a
property of Wireshark's allocator and not of this driver.

**Reduced:** no dissector, no capture file, no conversation state; reaching the site in place needs
all of `epan`. The stale access is reduced to the single byte the labelled probe touches.

**The triggering condition is created, not assumed.** The case asserts the geometry *before*
`wm_mark()` — that the crossing really leaves the chunk, and that it stays inside the block — so a
run on different allocator geometry exits 75 as an infrastructure failure rather than quietly
measuring nothing.

Reduced to the single byte at index `maxname` — the one a size of `maxname + 1` permits and a size
of `maxname` does not. The name expansion itself is dropped.

## What the run establishes, and what it does not

**Establishes:** the defect is real, reduces cleanly, and a bounds fault (cause 5) lands on the
labelled probe.

**Does NOT establish** that a nested allocator hides the *extent* from `malloc`. Every arm of this
harness narrows a wmem allocation to its request
(`ports/wireshark/wmem/src/shared/wmem-port-hooks.h:11-15`, `wm_narrow()`), so there is no
malloc-granular arm here to contrast against. That contrast lives in the tshark **app** port, whose
measured fx12 length ladder is `level0` 41 908 912, `shrink` 8 388 560, `sublet` 1 048 528,
`chunks` 64. Case 13's own run established this, and these rows are pre-registered with the
correction already applied.

**Does not establish reachability.** That a capture can deliver the triggering input to this site is
not proven here.

**Says nothing about silicon.** Cause 5 under capstone-qemu is a bounds diagnostic; the deployed
silicon is a separate question.
