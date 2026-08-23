# S-10 acceptance arms — build manifest and reading rule

Built 2026-08-21 from `capstone/tests/runtime-qemu/silicon-ladder/wbuf_fpga_app.c` +
`wbuf_kernel.h` via `build-ladder-domain.sh`, silicon config (`-capstone-gp-captable`, gp-free
call/ret, shrink OFF).

**Every arm is byte-distinct — 10 files, 10 distinct sha256.** This matters more than it sounds:
five of them share the identical byte *size* (10,224), so a size check would have passed while
byte-identical arms silently tested nothing.

| arm | sha256 (first 16) | bytes | carves | build flags |
|---|---|---|---|---|
| `wb0` | `63b5296361b967dd` | 10224 | 2 | `-DWBUF_ARM=0` |
| `wb1` | `c526e0b11996f9d7` | 10224 | 2 | `-DWBUF_ARM=1` |
| `wb2` | `e9d425efd3495c41` | 10224 | 2 | `-DWBUF_ARM=2` |
| `wb3` | `d4da7177e92466a3` | 10304 | 3 | `-DWBUF_ARM=3` |
| `wb4` | `3cc9982bfb04e5e7` | 10224 | 2 | `-DWBUF_ARM=4` |
| `wf1` | `32c81c7bfe5a067c` | 10360 | 2 | `-DWBUF_ARM=1 -DWBUF_FIELDS=1` |
| `wf5` | `14f2a1ee9a379ed9` | 10360 | 2 | `-DWBUF_ARM=5 -DWBUF_FIELDS=1` |
| `wr6` | `a387389befc103f6` | 10224 | 2 | `-DWBUF_ARM=6` |
| `wr7` | `c80aec6406b2b48d` | 10304 | 3 | `-DWBUF_ARM=7` |
| `wr8` | `a630a0193883e478` | 10424 | 4 | `-DWBUF_ARM=8` |

Carve cost is 2–4 per arm against a **1000-entry pool budget**. **CORRECTION 2026-08-23: I
originally wrote "arms run one at a time, so the worst case is 4, not 40." That reasoning is
wrong.** `capstone_rev_node.anvil:79` allocates with `set head := *head + 16'd1` — a monotone bump
with **no reclamation**, and nothing in the monitor resets it on `create_dom`. So the head is
**cumulative across every arm in a boot** and only clears on a power cycle. Ten arms at 2–4 each is
20–40 for the boot, not 4.

The conclusion survives — 40 against a 1000-entry budget is still no pressure — but the reasoning
did not, and the corrected version matters for anyone reading this to size a longer ladder. **A
boot's arms share the pool; they do not each get a fresh one.** (Counted from `gp-carve-count.py`'s labelled `carve count`
field. Reading the last number on its output instead gives `1000`, the *budget*, which would make
every arm look pinned at the limit.)

---

## Reading rule, per arm

Three outcomes for every arm, never two. **"The arm tested nothing" is not a pass and not a
failure** — it means the result carries no information and must not be reported as either.

| arm | PASS | FAIL (defect present) | TESTED NOTHING |
|---|---|---|---|
| **wb0** control, no plain store at all | `loss == 0` | — | any non-zero. Not a defect finding: the detector is reporting loss where no plain store exists, so the instrument is broken and **nothing else in the boot may be read** |
| **wb2** POSITIVE CONTROL, `stc G` then plain store `G+8` | `loss == N` (every slot) | — | `loss == 0`, or any value `< N`. This arm is architecturally *required* to lose the tag; if it does not, the detector cannot report loss at all and **every other arm in the boot is void** |
| **wb1** TEST, plain store `G+8` then `stc G` | `loss == 0` — program order honoured, S-10 holding | `loss > 0` — a reorder; the tag S-10 was meant to protect was overwritten | any result at all, if `wb0 != 0` or `wb2 != N` in the same boot |
| **wb4** neighbour granule, store to `G+16` | `loss == 0` — the effect is granule-scoped | `loss > 0` — the effect is not granule-scoped, which contradicts the mechanism and means the model is wrong, not just the fix | as `wb1` |
| **wb3** SPACED, ~64 unrelated stores between | `loss == 0` — the buffer drained | `loss > 0` — loss survives drainage, so residency is not the variable | as `wb1` |
| **wr6** tight, check immediately | `loss == 0` | `loss > 0` | as `wb1` |
| **wr7** as `wr6` with a 300-iteration drain delay | `loss == 0` | `loss > 0` | if `wr6` and `wr7` agree exactly, the drain delay changed nothing and the pair discriminates nothing — report both as uninformative rather than as two passes |
| **wf1** / **wf5** `WBUF_FIELDS=1`, also checks start/end/perm/cursor | `loss == 0` **and** no field mismatch | either a tag loss or a field mismatch | as `wb1`. A field mismatch with `loss == 0` is a **different** defect and should be reported separately, not folded into S-10 |
| **wr8** FORCED EVICTION — the only arm that can fail an L1-only fix | ratio materially **> 16** and corrupt count `== 0` | ratio materially **> 16** and corrupt count `> 0` — the fix repairs L1 while DRAM keeps the stale tag | **ratio == 16**: cold and warm were indistinguishable, the eviction walk did not evict, and the corrupt count means nothing. This is the arm's own built-in void signal — do not read its count |

**The discriminator that carries the most weight:** `wb1` losing while `wb3` does not. That pair is
the buffer-residency result, and it is the strongest single thing this ladder can produce.

**A monotone ladder is not evidence.** Arms that all return the same value can equally mean every
arm hit the same upstream failure. Treat uniform results as suspect until `wb2` has fired.

---

## Boot plan — FOUR domains per boot, and why

`tests/preflight-board-run.sh:270` hard-blocks above four domains per boot: the monitor's
`split_out_cap` middle exact-fit case spins at roughly the 5th `create_dom`, so **slots 5+ carry no
verdict and whatever occupies them gets blamed**. A seven-domain ladder would put the positive
control in a void slot, and the run would fail its own validity test through slot exhaustion rather
than through anything about S-10.

So: **two boots, each self-validating, each led by a known-good control** (which itself fails
roughly 1 in 5 for infrastructure reasons — a boot whose control fails is VOID and carries no
verdict about anything).

**Boot A — the verdict boot. Self-contained: everything `wb1`'s reading rule depends on is here.**

    1. known-good control domain
    2. wb0     must be 0     (no false positive)
    3. wb2     must be N     (the detector can report loss)
    4. wb1     THE VERDICT   (only readable if 2 and 3 held)

**Boot B — scope and eviction.**

    1. known-good control domain
    2. wb4     must be 0     (granule-scoped)
    3. wb3     must be 0     (drained)
    4. wr8     THE S-10 DISCRIMINATOR (ratio 16 = void)

`wf1`, `wf5`, `wr6`, `wr7` are a third boot if wanted; nothing in boots A or B depends on them.

**Read no further than the first failure in a boot.** The drivers do not reboot between programs
and a wedged program takes the core, so everything after a failure is collateral rather than
result. These arms are documented as unable to wedge — a lost tag is counted via `lcc` selector 1, the
type query, which is **total**: it answers instead of raising on a `NOT_CAP` operand. **I have
verified the total-ness in the generated netlist, NOT the specific answer.** `$1083 = $1080 &&
$1082` is a genuine AND of two 1-bit comparisons with both arms consumed
(`EVENTS1[352]` raise, `EVENTS1[340]` proceed), so selector 1 does not raise — but I could not
locate the `cap_type - 3'd1` computation that is supposed to make the answer 7, and **an earlier
version of this file asserted 7 as fact.** Do not key anything on the value 7; key on "differs
from the healthy baseline", which is decidable from an observed run. So every arm should return a
number. **An arm that returns nothing is itself the finding.**
