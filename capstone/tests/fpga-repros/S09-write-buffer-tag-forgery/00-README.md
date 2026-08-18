# S-09 — the write buffer FORGES capability tags over scalar data

**A capability store and a plain store to the same 16-byte granule are reordered, so a plain
store's data can end up carrying a VALID CAPABILITY TAG.** Measured on silicon at **7.27% per
opportunity**.

This is a **soundness** defect, not an availability one: it fabricates a usable capability over
data the program wrote as scalars. It is not a denial of service — it is the capability model
failing open.

> **Sibling issues, so a reader with the wrong symptom is redirected immediately.**
> `S07-capability-untagged-on-reload/` is the **opposite direction of the same mechanism** — a
> capability's tag is LOST, the program faults with mcause 25, and the domain wedges. Same root
> cause, same fix, different consequence. `S06-untagged-ldc-stc-high-half/` is a distinct,
> already-fixed defect about untagged `ldc`/`stc` and is not this.

## The defect

The write buffer hits at **64-bit word** granularity, so the two halves of a 16-byte granule
occupy **separate entries** — but every entry writes the **whole granule's single tag bit** when
it drains, and drain order is `rr_arb_tree` rotation, **not program order**.

    wt_dcache_wbuffer.sv:444   hit compare on wtag == {address_tag, address_index[11:3]}   -> 8-byte
    wt_dcache_wbuffer.sv:410   wr_idx_o  = wr_paddr[11:4]                                 -> 16-byte
    wt_dcache_wbuffer.sv:416   wr_ctag_o = wbuffer_q[rtrn_ptr].ctag
    wt_dcache_mem.sv:459       cap_tag_q[wr_idx_i][j] <= wr_ctag_i
    wt_axi_adapter.sv:158      tag byte address = (paddr - DATA_MEM_BASE) >> 4

So an older `stc` to `G+0` can drain **after** a younger plain store to `G+8`, and the tag the
plain store should have cleared stays set — over the plain store's data.

## Reproduction

`capstone/tests/runtime-qemu/silicon-ladder/wbuf_kernel.h`, arm 2, staged as `wb2.dom`:

    per slot, back to back:
        wbuf_slots[i] = base;        /* stc to G+0  */
        *(volatile unsigned long *)((char *)&wbuf_slots[i] + 8) = value;   /* plain store G+8 */
    then reload and query the type with lcc field 1

`lcc` field 1 is the TOTAL type query, which returns 7 for NOT_CAP **without raising**
(`capstone_dyn_unit.anvil:195`), so the run always returns a number and never wedges.

**Measured, 16384 slots, one boot:**

    expected (program order): 16384 tags cleared by the trailing plain store
    observed:                 15193 cleared
    ** 1191 slots (7.27%) kept a VALID TAG over scalar data **

## Controls, in the same boot

| arm | sequence | lost | meaning |
|---|---|---|---|
| wb0 | `stc G` only | 0 | the round trip alone is clean |
| wb4 | plain `G+16`; `stc G` | 0 | effect is granule-scoped, not "a nearby store" |
| wb3 | plain `G+8`; **64 stores**; `stc G` | 0 | **draining the buffer removes the effect** |
| wb1 | plain `G+8`; `stc G` | 1107 (6.76%) | the opposite direction — see S-07 |

**wb1 vs wb3 is the decisive pair:** identical stores, identical granule, differing only by
intervening traffic that drains the buffer. 1107 versus 0.

## QEMU cannot reproduce it, by construction

QEMU's capability store is one atomic 16-byte-plus-tag operation with no write buffer, no
per-word entries and no drain arbiter. Arms 0/1/3/4 return zero loss under emulation. **The
silicon/emulator difference is the mechanism**, and no amount of emulator testing could have
found this.

## Why it matters more than the availability direction

A lost tag faults loudly and stops the program. A **forged** tag does not: it produces a
capability the hardware will honour, over bytes the program wrote as ordinary data, with no trap
and no indication. Any code that writes scalars into a granule adjacent to a capability field can
manufacture one.

## Status

Mechanism identified by the RTL lane from the sources above; confirmed on silicon by the directed
test in this folder. The fix is open. Whether the `ctag`/`cap_tag_q` path predates the S-06 work
is under independent audit and does not change the defect.
