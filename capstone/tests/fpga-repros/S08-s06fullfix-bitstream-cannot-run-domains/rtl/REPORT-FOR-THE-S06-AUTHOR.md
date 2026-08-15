# S-08 — report for whoever wrote the S-06 RTL fix

**Written 2026-08-15, the day `caplifive_s06fullfix.bit` was flashed.**

## The one-paragraph version

Your S-06 fix is **untested, not failing** — we have never been able to read the acceptance gate on
this bitstream, because a separate regression stops anything from running. A **userspace `ecall` is
no longer delegated to S-mode**: it reaches M-mode's unhandled trap handler, so every syscall the
test program makes dies. We have a mechanism hypothesis that points at the dom-switcher lane your
P5b widened, with the code path quoted below. We may be wrong, and we say where we would be.

---

## 1. What we measured

Four boots, every one void at its control arm. The monitor's unhandled-trap dump, after we added
`mstatus` capture:

```
EXCX:0000E002    the monitor took a trap it does not handle at all
MCAU:00000008    ECALL from U-mode
MTVL:00000073    the ecall encoding
MSTA:00004022    MPP = 0  ->  U-MODE
MEPC:AB88EC88    therefore a USER VIRTUAL address
```

The second signature, on the other driver, is `MCAU:0000000C` (instruction page fault) — which by
your own RTL can only be taken with translation on (`cva6_mmu.sv:384,433,460` are the only cause-12
write sites, all under `enable_translation_i`; `csr_regfile.sv:2905-2907` gates translation on
`priv_lvl_o != PRIV_LVL_M`). So both signatures are **traps from below M-mode arriving at M-mode's
unhandled `default:`** — i.e. delegation, not capabilities.

**The discriminator that isolates the bitstream:** `EXCX:0000E002` fires **4 of 4** boots on
`caplifive_s06fullfix.bit` and **0 of 14** on `caplifive_12august.bit`. The instrument is
unconditionally compiled and present in *the same firmware bytes* on both sides — we booted the
byte-identical stored image `fw_93aa9a2426bc.bin` on both bitstreams (the console stores images by
content hash, so this is provably the same firmware). Same bytes, old bitstream: runs. New
bitstream: `EXCX`.

**The capability machinery itself works.** The domain is created, two regions are made and mapped,
and a full annotated share completes — `SHA5` (leaving M-mode), `SHA6` (domain returned), `ECSZ`
(handler returned) are all present, byte-for-byte the old-bitstream success sequence. The trap comes
*immediately after that*.

## 2. Our hypothesis, and why we think it

**`medeleg` is restored by the dom-switcher, and P5b changed that lane.**

```systemverilog
core/csr_regfile.sv:1915     7'd5: medeleg_d = dom_switch_reg_req_i.data[63:0];
core/csr_regfile.sv:411      7'd5: dom_switch_reg_resp_o = {64'd0, medeleg_q};
```

Delegation state is part of the domain-switch context. And your commit `25035c4c0` widened exactly
the lane it travels on:

```diff
core/include/ariane_pkg.sv
     logic [6:0] reg_id;
-    logic [127:0] data;
+    logic [128:0] data;      // S-06 fix (P5b): {tag, metadata, cursor}
   } dom_switch_reg_req_t;
```

(and the same change to `dom_switch_data_req_t`), with the Anvil dom-switcher rebuilt to match.

The timing fits exactly: the failure lands at the first instant after the dom-switcher has restored
context. If the Anvil producer and the SystemVerilog consumer disagree about **where the tag sits in
that 129-bit lane**, then `data[63:0]` is no longer the scalar, `medeleg` comes back shifted, and
delegation for cause 8 is lost — which is precisely the measured symptom.

Your own plan flagged this area as the risky one: *"the dom-switcher anvilh must widen and Anvil be
rebuilt — it moves metadata through internal registers a side tag can't bypass."*

**We have NOT verified the Anvil-side packing.** This is a hypothesis with a mechanism, not a root
cause.

## 3. What would settle it, cheapest first

1. **Assert that a dom-switch save/restore round-trips `medeleg` unchanged**, in simulation. If it
   does, our hypothesis is dead and we will say so.
2. Read `medeleg` before and after a domain switch and compare.
3. If it is shifted, check the bit order agreement on `{tag, metadata, cursor}` between the Anvil
   dom-switcher and `csr_regfile.sv`'s `data[63:0]` consumers (`:1915` and the other `7'd*` arms in
   the same block — every one of them takes `[63:0]` of the widened lane).

## 4. What is NOT established, so nobody chases it

* **The mechanism.** See above.
* **Whether this is a defect or a required change on our side.** Your P5a makes CPMP grants require
  a tag (`pmp/src/pmp_data_if.sv:45-46`, every CPMP gate ANDs `cpmp_tag_i[i]`). If our monitor were
  populating something untagged, the RTL would be right and our monitor would need updating. We
  could not find such a site — our CPMP writes come from `__linear` caps — but we cannot exclude it.
* **Your S-06 fix.** `s06agg` has never been read on this bitstream. Expected 15 (was 5); we have no
  value in either direction. **Please do not read S-08 as evidence about S-06.**

## 5. Two claims we made and then retracted — listed so you do not act on stale versions

* **"`MEPC:800072D0` symbolises to `sanitize_domain+0x188`."** Wrong twice over. That decode came
  from `fw_jump.elf`, which is not the binary that boots; against the actual `fw_payload.elf` the
  address is not even an instruction boundary (it is the second halfword of a 32-bit `bgeu`), and
  nothing in `.text` branches to it. With `MPP=0` now measured, it is a **user virtual address** and
  symbolising it against firmware names nothing. Also, `0xffffffff800072cc` — four bytes away — is
  the latched mepc in **9 old-bitstream logs**, so the address is not distinctive either.
* **"The bitstream cannot run any capability domain."** Too strong. The domain runs; the trap comes
  after its first share returns.

## 6. State of our side, so you know what the board is running

* **The board firmware was stale, and we fixed the cause.** `caplifive-system/sw/buildroot/Makefile`
  generated the monitor's `.c.S` with a rule depending only on a one-line `#include` wrapper, not on
  `sbi_capstone.c`. The flashed firmware was built from **Aug-6 assembly** while the source was
  Aug-12, so trace markers added for an earlier investigation had never shipped. Fixed by adding the
  source as a prerequisite; verified by a positive-controlled scan of the generated assembly
  (`MSTA`/`DBAS`/`DENT` now present, `MTVL`/`BASE` as controls). **Any board conclusion of ours that
  relied on instrumentation added after 2026-08-06 should be treated as suspect.**
* We added `MSTA` (mstatus) to the unhandled-trap dump. That is the datum this whole diagnosis turned
  on, and its absence is why we misread the fault twice.

## 7. Reproducing

```
cd capstone/tests/rtl-smoke
export FPGA_URL=<FPGA-CONSOLE-URL>
export FPGA_FW=<...>/fw_payload.bin
export FPGA_IMG_NAME=fw_93aa9a2426bc.bin    # boots the stored known-good image; skips the upload
export SQLITE_STAGE_DOMS="/test-domains/L2.dom,/test-domains/G6.dom"
python3 -m fpga_driver.run_sqlite_stages_fpga
```

Look for `EXCX:0000E002` followed by `MSTA:` in the run-scoped transcript. `L2.dom` enters and
returns; it did so on 14 of 14 old-bitstream boots and 0 of 4 here.

## 8. What we would find most useful back

Either a corrected bitstream, or a note that our monitor needs a matching change and which one. If
it is the latter we will make it — we would rather change the monitor than have you weaken a fix
that closes a real forgery path.
