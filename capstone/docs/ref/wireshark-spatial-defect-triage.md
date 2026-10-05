# Wireshark: triage for SPATIAL defects whose overflow stays inside a wmem block

The temporal hunt could not have found these. `wireshark-wmem-defect-triage.md:23` makes filter 1
*"a lifetime defect … **and not as an overflow**"*, so every spatial defect in this program was
discarded before triage, and 29 further rows of the 4.6.9 security tracker were rejected en bloc as
*"spatial or availability"* (`:178`). This file is the mirror instrument. See the retraction in
`spatial-vs-temporal-three-programs.md:20` for why the earlier "0 spatial" was a statement about the
search, not about Wireshark.

**Verdict: 3 class-B defects found, 2 of them LIVE at the `v4.6.8` pin.** All three are real
upstream instances of the shape that, until now, existed in this project only as the **synthetic**
fixture `tsapp` fx12 `wmem_neighbour`.

## The instrument

| | |
|---|---|
| clone | `/tmp/capstone/upstream/wireshark`, 118,543 commits |
| pin | tag **`v4.6.8`** |
| population | `v4.6.8..origin/master` = **4,321** commits (the widened population, as `wireshark-wmem-defect-triage.md:112` uses) |
| filter 1 | spatial wording in the subject — **31 survive**. Integer/refcount overflow, leaks, DoS and stack overflow are excluded explicitly: an integer overflow here is a *temporal* driver, which is the trap memcached case 03 sits in |
| filter 2 | **which bound does the overflow cross?** Read from the allocation site in the fix's parent, never inferred from the subsystem |
| filter 3 | liveness **by reading the pinned source**, not by ancestry — see below |
| script | `bug-corpora/tools/spatial-triage.py --self-test` |

### Filter 2 is the whole point: the class decides the value

| class | the overflow crosses | who faults | value |
|---|---|---|---|
| **A** | the `g_malloc` bound | `shrink`, `sublet`, and CHERI alike | a tie row; no contribution |
| **B** | a **wmem chunk** bound, inside the one block `g_malloc` handed wmem | **only a chunk-bounding port** | the contribution |
| **STACK** | a stack array | no allocator involved | out of class |

### Filter 3: ancestry is WRONG here, and it cost a false verdict

`merge-base --is-ancestor <master fix> v4.6.8` called **21 of these candidates live**. Reading the
pinned source shows the fix is already there, backported **under a different hash** — which
`wireshark-wmem-defect-triage.md:22` warns about in as many words (*"master carries the same fixes
under different hashes"*), and whose own rule is the one to follow: *"Liveness proved by reading the
pinned tree, not by the fix date"* (`:50`).

`e8ef9df09d` is the case that exposed it: not an ancestor of `v4.6.8`, yet `v4.6.8` already carries
`PFT_RS_K_MAX` and the widened allocation. So filter 3 now tests whether the **fix's own added lines**
appear in the pinned file, and the self-test keeps that honest two-sided — it must catch
`e8ef9df09d` as `FIX-IN-PIN` (6/6 added lines present) **and** still be able to say `DEFECT-LIVE`.

The script's controls, all of which fire: filter 2 separates a hand-adjudicated B / A / STACK triple;
a nonexistent sha reads `UNRESOLVED` rather than either verdict (`--is-ancestor` on a bad object
exits 128, which reads exactly like "live"); a tag resolves; and filter 1 rejects
`"refcount overflow"`. The self-test **failed twice for real reasons** while being written, which is
the only evidence that it can.

## The three class-B defects

All three overflow a chunk obtained from **`pinfo->pool`**, the per-packet wmem scope. One `g_malloc`
hands that scope a block and every chunk carved from it inherits the block's bounds, so on `level0`,
`shrink` and `sublet` the access is **in bounds and legal**. That is the gap, and it is the same one
`tsapp` fx12 demonstrates synthetically.

### 1. `1d8acb21ab` — SolarEdge, 6 bytes PAST a chunk. **LIVE at the pin.**

Quoted from the pinned tree, `v4.6.8:epan/dissectors/packet-solaredge.c:1005-1029`:

```c
int payload_length = length - SOLAREDGE_ENCRYPTION_KEY_LENGTH;
uint8_t *payload = (uint8_t *) wmem_alloc(scratch, payload_length);
uint8_t *intermediate_decrypted_payload = (uint8_t *) wmem_alloc(scratch, payload_length);
...
for (i  = 0; i < payload_length; i++) {
        out[i] = intermediate_decrypted_payload[i + 6] ^ intermediate_decrypted_payload[2+(i&3)];
}
```

`i` runs to `payload_length - 1`, so `[i + 6]` reads indices up to `payload_length + 5`: **six bytes
past a `payload_length`-byte chunk**. `scratch` is `pinfo->pool` at `:1292`. The two `wmem_alloc`
calls are consecutive in the same scope, so the overread crosses into the neighbouring chunk —
**fixture 12's shape, with a real dissector**.

- **Liveness, two-sided:** the `[i + 6]` read is present at the pin (`:1029`); the fix's marker
  `payload_length -= 6` has **0 occurrences** in the pinned file. The fix adds the guard
  `length < SOLAREDGE_ENCRYPTION_KEY_LENGTH + 6` and then `payload_length -= 6`, exactly
  compensating the `+6`.
- **Note** the pin's version has **no guard on `length` at all**, so `payload_length` can also go
  negative; that is a second, separate defect in the same function and is not what this case claims.

### 2. `d24613c461` — opcua, a read BELOW a chunk. **LIVE at the pin.**

This one was **already triaged and rejected on scope**, not on merit —
`wireshark-wmem-defect-triage.md:147`: *"the defect is an **out-of-bounds read inside a
`wmem_alloc`'d buffer**. A bounds bug, and the scope here is allocator lifetime."* It is in class B
and in scope here.

At the fix's parent, `plugins/epan/opcua/opcua.c`:

```c
static int verify_padding(const uint8_t *padding)
{
    pad_len = *padding;
    for (i = 0; i < pad_len; ++i) {
        if (padding[-pad_len + i] != pad_len) return -1;   /* NEGATIVE index */
```

`padding` points into `plaintext`, allocated at `:652` with
`wmem_alloc(pinfo->pool, plaintext_len)`, and `pad_len` is **attacker-controlled data read from the
packet** (`pad_len = *padding`). So `padding[-pad_len + i]` reads an unbounded distance *below* the
chunk. The fix passes an `available` count and rejects `pad_len > available`.

**Its subject says `heap-use-after-free`, and the defect is spatial.** Trust the diff, not the
wording — the existing triage doc reached the same conclusion from the other direction.

### 3. `e8ef9df09d` — ETSI DCP, a write PAST a chunk. Fix already in the pin.

Not live (`FIX-IN-PIN`, 6/6), so this is a **fix-reversal** case, which is what every memcached and
FFmpeg corpus case already is. At the fix's parent, `epan/dissectors/packet-dcp-etsi.c`:

```c
#define PFT_RS_N_MAX 207
#define PFT_RS_K 255
#define PFT_RS_P (PFT_RS_K - PFT_RS_N_MAX)          /* = 48 */
...
uint8_t *output = (uint8_t*) wmem_alloc(pinfo->pool, decoded_size);   /* :376, = fcount*plen */
...
memcpy(output+index_out+PFT_RS_N_MAX, deinterleaved+index_coded, PFT_RS_P);   /* :264 */
index_out += rsk;                                                            /* :270, rsk < 207 */
```

The in-place Reed-Solomon decode writes 48 parity bytes at `+207` while `index_out` advances by
`rsk`, which can be well under 207, and `output` is only `fcount*plen` bytes. The fix says so itself:
*"it does require that output have extra space at the end for the parity bytes (PFT_RS_P) and any
extra zeros (PFT_RS_K_MAX - rsk)"*, and the pin's allocation is now
`wmem_alloc(pinfo->pool, decoded_size + PFT_RS_N - rsk)`. The constants `N` and `K` had been
swapped relative to RS(255,207).

## The finding that matters for the port, and it is a gap

**All three allocate from `pinfo->pool`, the packet scope, which is a wmem `BLOCK_FAST` allocator —
and the `chunks` port deliberately does not narrow `BLOCK_FAST`.** That is measured, not assumed:
`ports/wireshark/app/results/2026-10-03-qemu-wmem-chunks-arm/README.md` shows fixture 10, a
BLOCK_FAST case, still carrying the whole block (`len=1048528`) on `chunks`, while BLOCK chunks come
back at 64 bytes.

So as the port stands today, **these three real defects would not be caught by any arm we have** —
`level0`/`shrink`/`sublet` see one block, and `chunks` leaves this scope alone. The synthetic fx12
uses a **BLOCK** allocator (file scope), which is why it discriminates.

This is the concrete, actionable consequence of the hunt: **extending the chunk adapter to
`BLOCK_FAST` is what converts three real upstream spatial defects into detections.** Whether that is
worth doing is a port-scope decision; that it is the blocker is now measured rather than guessed.

## Rejected, with reasons

| candidate | reason |
|---|---|
| `c8c396cf23` OBEX "off by one going up a path level" | **Not a memory-safety defect.** `wmem_strndup(scope, current_path, i_path - current_path - 1)` — the `-1` is in a *length* argument, so the copy is one byte SHORT and the path is truncated. A logic off-by-one that reads and writes in bounds. Filter 1 matched the wording; the mechanism is not spatial. A matching shape is not a matching mechanism |
| `46c813e7ad` DCT2000, `be813ede9d` etwdump, and 12 others | **`FIX-IN-PIN`** — the fix is already in `v4.6.8`, backported under another hash. Usable only as fix-reversals, and none is class B |
| `be813ede9d` etwdump, `f207d25f4b` / `830cf562a0` wiretap | **class A** — `g_malloc`/`g_strdup`, so `shrink` and `sublet` already fault and CHERI does too. A tie row |
| `76a6c36`-shaped stack cases | **STACK** — no allocator is involved, so the nested-allocator argument does not apply |
| the 29 rows of the 4.6.9 security tracker | not re-triaged here. `wireshark-wmem-defect-triage.md:178` names their subsystems; they are the obvious next population |

## What this instrument cannot see

- **A defect never fixed upstream.** The population is fixes; an unfixed overflow leaves no commit.
- **A defect whose fix does not word itself spatially.** Filter 1's cost, the same cost
  `ffmpeg-live-defect-triage.md:191` records for the temporal side: no wording filter catches a
  subject that says only "clear the ER picture".
- **The allocation site when it is more than one alias hop from the fix's hunks.** 24 of 31
  candidates came back class `?` for this reason. A `?` is *unresolved*, **not** class A — reading it
  as "no nested defects here" would be the clean-zero mistake again.
- **Reachability.** That a dissector path exists in the pin does not prove a capture can reach it.
  None of the three carries a reachability proof yet.
- **Whether `tshark` as our port builds it** compiles the dissector at all; `solaredge`, `opcua` and
  `dcp-etsi` still need checking against the port's dissector set.
