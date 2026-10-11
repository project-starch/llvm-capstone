# e8ef9df09d — the ETSI DCP Reed-Solomon decode writes parity past its buffer

## The defect

The in-place RS decode writes its parity bytes at a **fixed** offset of 207 while the output cursor
advances by `rsk`, which can be well under 207 — and the buffer is sized `fcount*plen` with no room
for parity at all. **The corpus's first row whose stale access is a WRITE.**

## Upstream defect

- **Fix:** `e8ef9df09d`, *"ETSI DCP: Fix heap buffer overflow"*. It renames the swapped constants,
  widens the allocation to `decoded_size + PFT_RS_N - rsk`, and says so in its own comment: the
  output *"does require that output have extra space at the end for the parity bytes (PFT_RS_P) and
  any extra zeros (PFT_RS_K_MAX - rsk)"*.
- **CVE:** none assigned.
- **Live in our pin: NO** — a fix-reversal, like every other row in this corpus bar 14 and 15.

## The vulnerable code, quoted from the fix's parent

`e8ef9df09d^:epan/dissectors/packet-dcp-etsi.c`:

```c
#define PFT_RS_N_MAX 207                                                       /* :235 */
#define PFT_RS_K 255                                                           /* :236 */
#define PFT_RS_P (PFT_RS_K - PFT_RS_N_MAX)          -- 255 - 207 = 48          /* :237 */
...
decoded_size = fcount*plen;                                                    /* :304 */
uint8_t *output = (uint8_t*) wmem_alloc(pinfo->pool, decoded_size);            /* :376 */
...
memcpy(output+index_out+PFT_RS_N_MAX, deinterleaved+index_coded, PFT_RS_P);    /* :264 */
index_coded += PFT_RS_P;
...
index_out += rsk;                                                              /* :270 */
```

`N` and `K` had been swapped relative to RS(255,207). **Liveness, two-sided:** the vulnerable
`memcpy(output+index_out+PFT_RS_N_MAX, …)` has **0 occurrences** in `v4.6.8`, while the fix's
renamed constant `PFT_RS_K_MAX` has **6**, and the pin allocates
`wmem_alloc(pinfo->pool, decoded_size + PFT_RS_N - rsk)`.

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

The decode is reduced to the **first byte** its parity copy writes past the buffer
(`index_out = 0`, offset `PFT_RS_N_MAX`); `eras_dec_rs` and the deinterleave are dropped because
they do not touch the bound.

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
