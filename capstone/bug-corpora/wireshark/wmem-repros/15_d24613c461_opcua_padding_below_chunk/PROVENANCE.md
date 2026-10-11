# d24613c461 — OPC UA reads a packet-controlled distance BELOW its chunk

## The defect

`verify_padding` receives a pointer near the **end** of the decrypted chunk and a length it reads
**from the packet**, then indexes backwards by it. The corpus's only downward crossing.

**This defect was already triaged once and rejected ON SCOPE, not on merit.**
`docs/ref/wireshark-wmem-defect-triage.md:147` calls it *"an out-of-bounds read inside a
`wmem_alloc`'d buffer"* and declined it because that file's scope was allocator **lifetime**. Its
subject says `heap-use-after-free`; the diff says spatial. Trust the diff.

## Upstream defect

- **Fix:** `d24613c461`. It passes an `available` count and rejects `pad_len > available`.
- **CVE:** none assigned.
- **Live in our pin: YES**, read two-sided from the pinned tree.

## The vulnerable code, quoted from the PIN

`v4.6.8:plugins/epan/opcua/opcua.c:324-332`:

```c
static int verify_padding(const uint8_t *padding)      /* :324 -- no bound is passed in */
{
    uint8_t pad_len;
    uint8_t i;

    pad_len = *padding;                                /* read FROM THE PACKET */

    for (i = 0; i < pad_len; ++i) {
        if (padding[-pad_len + i] != pad_len) return -1;   /* :332 -- negative index */
```

`padding` points into `plaintext`, allocated at `:642` with
`wmem_alloc(pinfo->pool, plaintext_len)`. `pad_len` is attacker-chosen up to 255.
**Liveness, two-sided:** the negative-index read is present at `:332`, `verify_padding` still has
the single-argument signature at `:324`, and the fix's guard `pad_len > available` has **0
occurrences**.

## Class, stated with its limit

**Class B while the chunk has a predecessor inside the block; class A when the chunk is first in a
fresh block**, because then the same read leaves the `g_malloc`'d region. The case allocates a
predecessor first and asserts it is below, so what it measures is the **chunk** bound. **Which of
the two a real capture produces is not established here.**

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

`sig_len` is reduced to 0, so the cursor is the chunk's last byte, and the loop is reduced to its
first iteration — the one that crosses.

## What the run establishes, and what it does not

**Establishes:** the defect is real, reduces cleanly, and a bounds fault (cause 5) lands on the
labelled probe.

**Does NOT establish** that, under a nested allocator, the *extent* is invisible to `malloc`. Every arm of this
harness narrows a wmem allocation to its request
(`ports/wireshark/wmem/src/shared/wmem-port-hooks.h:11-15`, `wm_narrow()`), so there is no
malloc-granular arm here to contrast against. That contrast lives in the tshark **app** port, whose
measured fx12 length ladder is `sysalloc-none` (`level0`) 41 908 912, `sysalloc-bounds` (`shrink`) 8 388 560,
`sysalloc-sublet` (`sublet`) 1 048 528, `chunks` 64. Case 13's own run established this, and these rows are pre-registered with the
correction already applied.

**Does not establish reachability.** That a capture can deliver the triggering input to this site is
not proven here.

**Says nothing about silicon.** Cause 5 under capstone-qemu is a bounds diagnostic; the deployed
silicon is a separate question.
