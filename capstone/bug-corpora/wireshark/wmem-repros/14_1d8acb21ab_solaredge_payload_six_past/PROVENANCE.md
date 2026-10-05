# 1d8acb21ab — the SolarEdge decrypt loop reads six bytes past its chunk

## The defect

`solaredge_decrypt` carves **two consecutive chunks** of `payload_length` from the packet scope and
then, for `i < payload_length`, reads `intermediate_decrypted_payload[i + 6]`. The last iteration
reads index `payload_length + 5` — **six bytes past the chunk**, into storage the same block handed
out next. At `malloc` granularity nothing is wrong: one `g_malloc` owns the whole block.

## Upstream defect

- **Fix:** `1d8acb21ab`, *"SolarEdge: Fix buffer overflow"*. It adds the guard
  `length < SOLAREDGE_ENCRYPTION_KEY_LENGTH + 6` and then `payload_length -= 6`, exactly
  compensating the `+6`.
- **CVE:** none assigned.
- **Live in our pin: YES**, and read two-sided from the pinned tree rather than from a probe's label.

## The vulnerable code, quoted from the PIN

`v4.6.8:epan/dissectors/packet-solaredge.c:1005-1029`:

```c
int payload_length = length - SOLAREDGE_ENCRYPTION_KEY_LENGTH;
uint8_t *payload = (uint8_t *) wmem_alloc(scratch, payload_length);                      /* :1007 */
uint8_t *intermediate_decrypted_payload = (uint8_t *) wmem_alloc(scratch, payload_length); /* :1008 */
...
for (i  = 0; i < payload_length; i++) {
        out[i] = intermediate_decrypted_payload[i + 6] ^ intermediate_decrypted_payload[2+(i&3)];
}                                                                                        /* :1029 */
```

`scratch` is `pinfo->pool` at the `:1292` call site. **Liveness, two-sided:** the `[i + 6]` read is
present at `:1029`; the fix's marker `payload_length -= 6` has **0 occurrences** in the pinned file.

**A second, separate defect in the same function** is recorded and not claimed here: the pin has no
guard on `length` at all, so `payload_length` can go negative.

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

The decrypt arithmetic is reduced to the loop's **last** read, index `payload_length - 1 + 6`, the
one that leaves the chunk; the cipher calls are dropped because they do not touch the bound.

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
