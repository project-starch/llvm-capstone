# S-06 — proposed RTL fixes

Written by the software side. It is a **proposal, not a patch**: it changes how a capability is
represented in the register file, which is the hardware owner's call. Everything below is cited to
`file:line` so it can be checked rather than taken on trust.

Two options are given. **Option B is much smaller and solves the software problem**; Option A is
the general fix. They are not exclusive — B unblocks the toolchain, A closes the architectural
hole.

---

## 1. What is broken, precisely

Two independent gates, in sequence, both in `core/cache_subsystem/wt_dcache_mem.sv`:

**The load discards the bytes.**

```systemverilog
:310   ruser = cap_tag_hit ? ruser_cl[rd_hit_idx] : '0;
```

`ruser_cl` is bank 1's *real* SRAM content. When the line's 1-bit shadow tag `cap_tag_q` (`:135`)
is clear, it is replaced by a literal `'0`. Any ordinary store clears that tag (`:417-421`
writes `st_wr_cap`), so a buffer written with `sd` always reads back through `ldc` with a zeroed
high half. The refill path does the same with a different condition (`:306`,
`(|wr_cl_user_i[7:0])`).

**The store then never writes the high half.**

```systemverilog
:138   assign st_wr_cap = |wr_user_i;
:229   if(!(st_wr_cap)) begin  ...only the bank matching wr_off_i...  end else begin bank_req='1; end
```

`st_wr_cap` is gated on metadata **content**, not on the opcode. With the metadata now zero, an
`stc` requests only the bank matching its own offset, so `dst+8..15` is never written — it keeps
its prior content (zero on a fresh line, **stale** otherwise).

The load's force-to-zero *creates* the all-zero metadata that *causes* the store's bank-skip.
Neither side is individually "the bug".

---

## 2. Why the obvious one-line fix is a security hole

The tempting patch is to delete the `: '0` at `:310`. **That zeroing is the safety mechanism.**

`cap_metadata_t` (`core/include/ariane_pkg.sv:632-637`) is exactly 64 bits and fully packed —
`revnode_id[30] + perm[3] + cap_type[3] + bounds[28]` — and validity is encoded *inside* it as
`cap_type == NOT_CAP == 3'b000` (`:650`). `cap_pack_t` is that plus a 64-bit cursor: **128 bits
total, with no separate tag bit anywhere in the register or pipeline path.** The only real tag in
the design is `cap_tag_q`, inside the D-cache.

So returning the raw high half makes arbitrary data decode as a capability. Using the actual bytes
from this package's measurement:

| lost high half | would decode as |
|---|---|
| `0xcfcecdcccbcac9c8` | `cap_type=4 (UNINIT)`, `perm=1`, `revnode_id=871609203` |
| `0xdfdedddcdbdad9d8` | `cap_type=5 (SEALED)`, `perm=1`, `revnode_id=938981239` |

Seven of the eight `cap_type` encodings mean "is a capability", so ordinary data forges one with
probability ~7/8 per 16-byte chunk, with attacker-chosen bounds and permissions, **out of a
`memcpy`**. That trades silent data loss for collapse of the memory-safety model.

The narrower variant — return raw bits but force `cap_type` to `000` — preserves 61 of 64 bits and
still corrupts 3 bits inside byte 3 of every high half. Same bug, harder to notice.

---

## 3. Option B — a tag-preserving 16-byte memory-to-memory copy (RECOMMENDED FIRST)

**Idea.** The value never enters the register file, so no register representation has to change.
The cache already holds both the data and the tag; give it one instruction that moves both.

Roughly: `ccpy rd_cap, rs_cap` — copy the 16-byte line at `rs` to the line at `rd`, carrying
bank 0, bank 1 **and** `cap_tag_q` unchanged, with the ordinary bounds/permission checks on both
operands.

**Why this is attractive**

* It does not touch `cap_pack_t`, the register file, forwarding, the scoreboard, or
  `capstone_dom_switcher` — the whole datapath question in Option A disappears.
* It is exactly, and only, what software needs. Every failure in this package is a *copy*: our
  `memcpy`'s aligned loop, and the compiler's aggregate-copy lowering. Neither needs the value in
  a register; both need "move these 16 bytes faithfully".
* The cache is already the component that knows the truth: `cap_tag_q` is right there.

**Cost:** a decoder entry, a small functional unit or LSU sequence, and a compiler pattern. No
change to how capabilities are represented.

**Limitation, stated plainly:** it fixes copying. It does **not** make a general
`ldc` → register → `stc` round trip faithful for plain data, so Option A is still the correct
long-term answer.

---

## 4. Option A — give the register representation a tag bit

The general fix, and what QEMU already does: `capstone-qemu target/riscv/cap.h:79-94` carries a
`scalar_hi` field plus a `tag`, precisely so an untagged `ldc`/`stc` round trip is bit-exact.

1. **Add `logic tag` to `cap_pack_t`** (`ariane_pkg.sv:644`). When `tag == 0` the 128 bits are raw
   data: the cursor holds the low word and the metadata field holds the high word verbatim.
2. **Load:** deliver bank 1 **ungated** and set `tag <= cap_tag_hit`. The `: '0` at `:310` and the
   equivalent at `:306` are then unnecessary — the stale-forwarding concern they exist for is
   handled by the tag travelling with the data instead of by erasing it.
3. **Capability checks become tag checks.** Everywhere that currently infers "this is a
   capability" from `cap_type != NOT_CAP` must test `tag` instead. **This is the audit that makes
   the change non-trivial and it should drive the estimate** — `NOT_CAP` stays a legal
   `cap_type` for genuine null capabilities, so the two notions must be separated carefully.
4. **Store:** make `st_wr_cap` opcode-gated (STC / domain-switch / rev-node update) rather than
   `|wr_user_i`, always write both banks, and set `cap_tag_q <= reg.tag`.
5. **Widen the user lanes.** `data_wuser`/`data_ruser` carry the metadata and need one more bit for
   the tag.
6. **`capstone_dom_switcher`.** It serialises the register file to memory on every domain switch,
   so the saved-context format must carry the tag. If it writes through the same path as `stc`,
   the existing shadow tag covers it — worth confirming early, as it is the most likely place for
   this change to grow.

Items 3 and 6 are where the real work is; 1, 2, 4 and 5 are mechanical.

---

## 5. Why software cannot work around this — all three routes measured, not assumed

This is the part worth reading if the question is "can you just avoid it in the compiler?" We
tried all three, on the board.

**Route 1 — stop using the capability-grained copy.** Impossible: that path is the *only* one
that preserves tags. A byte-wise `memcpy` strips them, SQLite then dereferences untagged pointers,
and the core wedges. Measured, recorded under issue S-04.

**Route 2 — copy plain when the chunk holds no capability, capability-grained when it does.**
Impossible: software cannot ask. `LCC` raises `UNEXPECTED_OPERAND` on a `NOT_CAP` operand *before*
it looks at the requested field (`core/anvil_build/capstone_dyn_unit.anvil`, `func LCC`), so the
obvious `if (cap_get_tag(v))` faults on exactly the plain data it is meant to detect — and a
capability fault inside a domain wedges rather than trapping.

**Route 3 — write the line as plain data first, then lay the `ldc`/`stc` on top.** This is the
only construction that is correct on paper, and it *is* correct in isolation: it exploits the
mechanism in §1, because for a real capability the metadata is non-zero so the `stc` writes both
banks and restores the tag, while for plain data the `stc` degrades to a single-bank store that
leaves the plain-written high half intact. It passes in RTL simulation
(`sim/untagged-ldc-stc-fixup.S` arm E) and **passes on silicon at the primitive level** — the
`s06agg`/`s06aggf` pair in this package returns 66 unfixed and 64 fixed, four observations across
two boots in both slot orders.

**And it destabilises a real workload.** Applied to SQLite it turns a diagnosable error return
into a wedge with `mcause 25 = INVALID_CAPABILITY`. Matched pair in one boot, control green, two
builds differing only by this workaround, both running an identical small `CREATE TABLE`:
workaround off returns `rc=11` twice, workaround on wedges. Two candidate causes were found and
eliminated — integer pointer arithmetic stripping the capability (real, fixed, and not this), and
write-before-read destroying a tag on a self-copy (reordered; the wedge persists).

**A hypothesis for §4's benefit, explicitly not established:** the workaround writes each line
three times, tripling store traffic and cache pressure, and the refill gate (`:306`) and the
single-word gate (`:310`) use *different* conditions. If a line whose tag was set by an `stc`
loses it across an eviction and refill, a later `ldc` sees untagged data and the
revocation-validity query fails — which is exactly `INVALID_CAPABILITY`. That fits every
observation: it passes in simulation where there is no eviction pressure, passes on a 10 KB rung
where the line stays hot, and fails only at scale. Nobody has measured a tag surviving an
eviction; if it turns out the two gates disagree, that is a second defect worth its own fix.

**Conclusion:** there is no safe software workaround. The correct copy sequence is correct, and
the hardware does not sustain it under load.

---

## 6. Acceptance

Both options are checked by what is already in this folder, and neither needs SQLite:

* `./run.sh sim` — RTL simulation, 499 cycles, carries its own control and fails loudly if that
  control is wrong. Passes when the high half reads `0xfedcba9876543210` instead of zero.
* `./run.sh rung` — on hardware. `s06copy` must return **32** instead of 16, and `s06agg` **64**
  instead of 66. `s06agg` is the important one for Option B: it contains no `memcpy` at all, only
  an ordinary struct assignment, which is the case a library cannot reach.

For Option B specifically, the compiler side is a one-line change on our end — the aggregate-copy
lowering already exists and would emit the new instruction instead of the `ldc`/`stc` pair.
