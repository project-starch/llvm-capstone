# S-06 — proposed RTL fixes

Written by the software side. It is a **proposal, not a patch**: it changes how a capability is
represented in the register file, which is the hardware owner's call. Everything below is cited to
`file:line` so it can be checked rather than taken on trust.

Two options are given. **Option B is much smaller and solves the software problem**; Option A is
the general fix. They are not exclusive — B unblocks the toolchain, A closes the architectural
hole.

> **Revision note.** An earlier draft of Option B proposed a copy instruction that carried the
> capability **tag** along with the data. That version was **withdrawn** during a security review:
> it would have duplicated linear capabilities, which no instruction in this architecture may do.
> §3 below is the replacement, restricted to untagged lines. §3.1 records why the original failed,
> because the idea is an attractive one and will otherwise be re-proposed.

---

## 1. What is broken, precisely

Two independent gates, in sequence, both in `core/cache_subsystem/wt_dcache_mem.sv`:

**The load discards the bytes.**

```systemverilog
:308   ruser = cap_tag_hit ? ruser_cl[rd_hit_idx] : '0;
```

`ruser_cl` is bank 1's *real* SRAM content. When the line's 1-bit shadow tag `cap_tag_q` (`:135`)
is clear, it is replaced by a literal `'0`. Any ordinary store clears that tag (`:419`,
`cap_tag_q[wr_idx_i][j] <= st_wr_cap`), so a buffer written with `sd` always reads back through
`ldc` with a zeroed high half. The refill path does the same at `:304`, spelled
`(|wr_cl_user_i[7:0])` — a different encoding of the same "is this a capability" test, not a
different condition. (`:412` sets the tag on refill from the same eight bits.)

**The store then never writes the high half.**

```systemverilog
:138   assign st_wr_cap = |wr_user_i;
:228   if(!(st_wr_cap)) begin  ...only the bank matching wr_off_i...  end else begin bank_req='1; end
```

`st_wr_cap` is gated on metadata **content**, not on the opcode. With the metadata now zero, an
`stc` requests only the bank matching its own offset, so `dst+8..15` is never written — it keeps
its prior content (zero on a fresh line, **stale** otherwise).

The load's force-to-zero *creates* the all-zero metadata that *causes* the store's bank-skip.
Neither side is individually "the bug".

---

## 2. Why the obvious one-line fix is a security hole

The tempting patch is to delete the `: '0` at `:308`. **That zeroing is the safety mechanism.**

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

## 3. Option B — a 16-byte copy restricted to UNTAGGED lines (RECOMMENDED FIRST)

**Idea.** The value never enters the register file, so no register representation has to change.
The cache already holds both the data and the tag; give it one instruction that moves the data
**only when there is no tag to carry**.

`ccpy16 rd, rs1, rs2` — if the source granule at `rs2` is untagged, copy its 16 bytes to the
granule at `rs1` and leave the destination untagged; if the source **is** tagged, copy nothing and
report that in `rd` as a plain integer. `rs1` and `rs2` are address capabilities and get the
ordinary bounds, permission and validity checks; `rd` is an ordinary integer register.

Software branches on `rd`: `memcpy` and the compiler's aggregate-copy lowering use `ccpy16` for
the common case and fall back to today's `ldc`/`stc` pair for tagged chunks, so capability copying
keeps its present semantics **including the source clear**. Every failure in this package is
untagged, so this covers all of them.

**Why this is attractive**

* It does not touch `cap_pack_t`, the register file, forwarding, the scoreboard, or
  `capstone_dom_switcher` — the whole datapath question in Option A disappears.
* **No capability of any type ever passes through it.** It never propagates a tag and never moves
  a capability, so linearity, sealing, revocation and monotonicity are untouched by construction
  rather than by argument. That is the property §3.1 shows the earlier draft could not claim.
* The cache is already the component that knows the truth: `cap_tag_q` is right there.

**Cost:** a decoder entry, a small functional unit or LSU sequence, and a compiler pattern. No
change to how capabilities are represented.

**Two things to review rather than wave through.**

* `rd` is a **non-destructive tag probe**, which this ISA does not currently offer: `LCC` faults on
  a `NOT_CAP` operand, and `ldc` of a linear capability consumes the source. It grants no
  authority — one bit about a granule the caller already holds read authority over — but it is a
  new observation primitive and should be signed off as one.
* The tag bit it branches on is the D-cache's `cap_tag_q`. If that bit is ever wrong the failure
  modes are: a capability seen as untagged (copied as data, tag lost — safe-fail, and no worse
  than S-06 today), or plain data seen as tagged (the copy is skipped and software falls back to
  `ldc`/`stc`, i.e. S-06 returns for that chunk). Neither is a security hole.

**Limitation, stated plainly:** it fixes copying. It does **not** make a general
`ldc` → register → `stc` round trip faithful for plain data, so Option A is still the correct
long-term answer.

### 3.1 Why the tag-preserving version of this instruction was withdrawn

The obvious and more general form of Option B is "copy the 16 bytes **and** the tag". It must not
be built, and the reason is not a detail of this defect but the central invariant of the
architecture.

`intro.adoc:58-61` states it normatively: *"instructions can only **move, but not copy**, linear
capabilities between general-purpose registers."* Only `NONLIN` is copyable. `ldc` enforces this
by clearing its source for **five** types — `LINEAR`, `REVOKE`, `UNINIT`, `SEALED`, `SEALEDRET`
(`core/load_unit.sv:447-453`, condition duplicated at `core/commit_stage.sv:332-335`). A
tag-preserving copy is by definition a *copy*, so it would break the invariant for all five.

`SEALED`/`SEALEDRET` is the case that makes this more than a memory-safety bug: a duplicated
`SEALEDRET` grants `ldc`/`stc` over the callee's saved register file and is retained across the
RETURN, which is domain confusion.

Two arguments that look sufficient and are not, recorded so they are not re-made:

* *"It performs the same checks as `ldc`/`stc` on both operands."* Hollow. Every type, permission
  and bounds check in `LDC`/`STC` is on **`rs1`, the address operand**; neither instruction
  inspects the type of the value being moved. The only value-dependent behaviour in the pair is
  the linear clear — precisely the check the proposal dropped while keeping the ones that were
  never load-bearing.
* *"Revocation would catch an escaped duplicate."* It would not, but neither is it broken: there
  is no reference count, and all aliases share a `revnode_id`, so one `revoke` invalidates every
  copy. Duplication defeats **exclusivity**, not revocation. Do not offer revocation as a
  mitigation.

The restriction to untagged lines in §3 is what removes the problem, rather than mitigating it:
an instruction that only ever moves `tag == 0` granules is not moving a capability at all.

---

## 4. Option A — give the register representation a tag bit

The general fix, and what QEMU already does: `capstone-qemu target/riscv/cap.h:79-94` carries a
`scalar_hi` field plus a `tag`, precisely so an untagged `ldc`/`stc` round trip is bit-exact.

1. **Add `logic tag` to `cap_pack_t`** (`ariane_pkg.sv:644`). When `tag == 0` the 128 bits are raw
   data: the cursor holds the low word and the metadata field holds the high word verbatim.
2. **Load:** deliver bank 1 **ungated** and set `tag <= cap_tag_hit`. The `: '0` at `:308` and the
   equivalent at `:304` are then unnecessary — the stale-forwarding concern they exist for is
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

**Why the wedge happens is UNEXPLAINED.** An earlier draft of this section proposed a mechanism —
that the refill gate (`:304`) and the single-word gate (`:308`) use different conditions, so a tag
set by an `stc` could be lost across an eviction and refill, making a later `ldc` fail its
revocation-validity query with `INVALID_CAPABILITY`. **That mechanism has since been refuted and
should not be repeated.** `:304` tests `|wr_cl_user_i[7:0]`, and on the refill path those eight
bits are not capability metadata: `core/cache_subsystem/wt_axi_adapter.sv:441-442` zeroes the word
and writes a single byte of `tag_wr_value_q`, which is `is_cap_req = |dcache_data.user` (`:196`,
`:402`), so the byte read back at `:731-734` is `0x00` or `0x01`. The two gates are the same
predicate over different encodings. The AXI USER sideband carries nothing (`:204`,
`axi_wr_user[0] = '0;`). The cache is also write-through with no dirty writeback, so a tag written
by an `stc` reaches the shadow-tag region directly.

So the workaround's failure at scale is a genuine open question, not a suspected second defect. It
does not gate this proposal: §3 and §4 both remove the need for the workaround entirely.

**Conclusion:** there is no safe software workaround. The correct copy sequence is correct, and
the hardware does not sustain it under load.

---

## 6. Acceptance

Three functional criteria, checked by what is already in this folder and needing no SQLite, plus
one security criterion that has to be added.

* `./run.sh sim` — RTL simulation, 499 cycles, carries its own control and fails loudly if that
  control is wrong. Passes when the high half reads `0xfedcba9876543210` instead of zero.
* `./run.sh rung` — on hardware. `s06copy` must return **32** instead of 16, and `s06agg` **64**
  instead of 66. `s06agg` is the important one for Option B: it contains no `memcpy` at all, only
  an ordinary struct assignment, which is the case a library cannot reach.

**A LINEARITY CONTROL IS REQUIRED, and none of the three above is one.** All three would pass
unchanged for an instruction that duplicates linear capabilities — they measure only whether plain
bytes survive. That gap is what §3.1 was written about, so the criterion must be explicit:

* **Duplication check.** Build a `LINEAR` capability, place it in a 16-byte granule, run the new
  instruction with that granule as source, then read the source granule back. **PASS = the source
  is unchanged and the destination did not receive a capability** (Option B copies nothing from a
  tagged line and reports it in `rd`). **FAIL = the destination holds a usable capability**,
  regardless of whether the copy also "worked". Run the same sequence over an untagged granule as
  the matched control: it must copy, or the check has not been shown to fire and proves nothing.

The last clause is not boilerplate. A duplication check that has never produced its own positive
result is indistinguishable from one that is silently inert.

For Option B specifically, the compiler side is small on our end — the aggregate-copy lowering
already exists; it emits the new instruction plus a branch to the existing `ldc`/`stc` pair for
the tagged case.
