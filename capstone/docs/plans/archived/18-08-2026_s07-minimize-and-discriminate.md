# S-07 board plan: discriminate H-mem vs H-load, and shrink the repro

Written 2026-08-18, for the **current** bitstream `caplifive_s06s08fix_s07tag2_618f4ce.bit`.
Needs no reflash. One boot, six domains.

## The two open questions

1. **H-mem vs H-load.** Memory genuinely holds nonzero untagged residue, versus memory holds
   correct zeros and the load path returned wrong data. These have *identical* observables in
   every wedge so far — same mepc, same `sw=204 = 0x00`, same clean QEMU.
2. **How small is the repro?** Today it is the entire SQLite workload in a ~1.5 MB domain.

## The design constraint that decides everything

**No software channel survives a wedge.** A wedge is an M-mode wedge that takes the whole core,
so the retval dies, the `output_text` buffer dies (the host only reads it on return), and a shared
region dies with the host. Only the hardware latches survive — `mepc` (196-203), the trap log
(255), and the displacement counter (204).

Therefore **every probe must complete and RETURN before any wedge can occur**, which rules out the
obvious design. Probing inside `sqlite3BackupRestart` or `pagerFreeMapHdrs` looks right and is
wrong twice over:

* it violates the "stop the FLOW, not a leaf" rule — an early return there lands back in
  `pager_reset`, which carries on;
* and per `00-README.md` §3, instrumenting a site just **moves the death** to the next uncovered
  one, which then destroys the report.

**So all probing happens at TOP LEVEL, in `domain_main`.** This is possible because the Pager is
reachable from a `sqlite3*` without touching SQLite internals — `sqlite3BtreePager` is linked
(`ob.dis`), `struct Db.pBt` is visible (`sqlite3-capstone.c:18507`):

```c
Pager *pP = sqlite3BtreePager(db->aDb[0].pBt);
unsigned long v = *(volatile unsigned long *)(void *)&pP->pBackup;   /* plain ld, never ldc */
```

Reading with a plain 64-bit load is the point: it is the *same* instruction the compiled NULL
guard uses (`ld a0, 0x0(a0)`), so it observes exactly what the guard observed, and it cannot
itself raise mcause 25.

## The discriminator: read the same address 8 times, EACH FROM MEMORY

**Revised 2026-08-18 after the RTL lane refuted the first version.** The original design took 8
consecutive plain loads of one address and read agreement as "the value is in memory". That is
unsound, and it fails in the row most likely to be hit:

> the first load may miss and refill; loads 2-8 then **HIT IN L1** and never reach memory. One bad
> value installed in the line and read back seven times is perfectly self-consistent — so the
> cache makes **H-load look exactly like H-mem**, which is the single confusion this probe exists
> to prevent.

Sharper still: the untagged response actually measured came back `src=1`, **MISS REFILL** — the
refill path is the suspect, and the naive design exercised the L1-**hit** path in seven samples
out of eight.

**Fix: force every sample to miss.** Geometry verified from primary source, not memory —
`cv64a6_imafdc_sv39_config_pkg.sv:42-44`: `DcacheByteSize 32768`, `DcacheSetAssoc 8`,
`DcacheLineWidth 128` bits.

| quantity | value |
|---|---|
| line | 16 B |
| way size | 32768 / 8 = **4096 B** |
| sets | 4096 / 16 = 256 |
| **conflict stride** | **4096 B** |

Index bits are `[11:4]`, entirely inside the 4 KiB page offset, so a *virtual* stride of 4096
selects the same set with no aliasing question. Between samples, touch **≥ 8** addresses at
4096-byte stride (8-way set) to evict the line under test.

**And the eviction gets its own positive control, because an under-filled eviction loop has
already silently tested nothing on this project.** Eviction is unobservable by construction — it
looks like a successful probe either way — so it is timed with `mcycle`: a post-evict load must be
measurably slower than a known-cached load. If the ratio is not clearly above threshold, the
domain returns `0x5107_EE00` and **every sample that boot is discarded**.

| 8 samples, each forced to miss | reading |
|---|---|
| all 8 identical and **nonzero** | the value is **in DRAM** → H-mem |
| the 8 **disagree** | the load path returns inconsistent data → **H-load**, demonstrated |
| all 8 **zero** | memory is clean at this phase |

Without the eviction fix, agreement must be reported as *"consistent, cache-masked, NO VERDICT"* —
never as H-mem. `V=2` was always sound: a cache serving one line cannot manufacture disagreement.

### Two more controls, both for silent-failure modes

* **Confirm the 8 loads survive into the artifact, before spending the boot.** A
  repeat-the-load-N-times ladder on this project was once CSE'd into ONE `ldc` regardless of N,
  and memory barriers did not stop it; the whole set tested nothing and reported with total
  confidence. `volatile` should hold where a barrier did not, but "should" is what that set
  relied on. One `llvm-objdump` and count the loads per sample site.
* **`V=2` needs its own positive control, and the selftest does not provide one.** Writing a
  nonzero pattern and reading it back proves the probe can report `V=1`. It proves nothing about
  the comparator, and `V=2` is the outcome that would end the ambiguity. If the comparison is
  subtly wrong — compares `sample[0]` to itself, wrong index, compares after overwriting —
  disagreement can *never* be reported and the boot returns a confident `V=1` having tested
  nothing. So: run the identical comparator over a seeded array with one element deliberately
  different and require `V=2` before any real sample is taken.

### One assumption now verified rather than assumed

`lcc` selector 1 on a NOT_CAP is total and will not raise: `capstone_dyn_unit.anvil:195` raises
only when `cap_type == NOT_CAP && zimm != 1`, and the result path computes `cap_type - 3'd1`, so
NOT_CAP(0) wraps to `3'b111` = 7. The second guard that could have caught it on the way past,
`check_LCC_invalid_multiplexing` (`capstone_unit.anvilh:469`), fires only for selectors 2, 4, 5
and 7. (Checked by the RTL lane.)

## The boot: six domains, ordered so every wedge point is itself an answer

Boots have run 9 domains, but `SILICON-BLOCKER.md:5130-5136` documents a **~6-run ceiling**
independent of the exact-fit spin, so slot 6 is treated as expendable.

| # | domain | what it does | expected |
|---|---|---|---|
| 1 | `S7T` | known-good control | **must pass, or the boot is VOID** |
| 2 | `S7Q` | top-level phase probe (below) | RETURNS by construction |
| 3 | `S7R` | same probe, reps x3, to catch a rare phase | RETURNS by construction |
| 4 | `MRO` | minimal candidate A: open `:memory:`, close. Nothing else | may wedge |
| 5 | `MRR` | minimal candidate B: open, one rolled-back transaction, close | may wedge |
| 6 | `XU` | historical reproducer | expected to wedge; expendable |

Ordering rationale: 1-3 cannot wedge, so they always yield data. 4 and 5 are ordered
least-to-most likely to wedge, and **whichever first fails to return IS the minimisation
result** — if `MRO` wedges the repro is "open and close a database", if `MRO` returns and `MRR`
wedges it is "a rollback". Losing `XU` costs nothing; it is already a known reproducer.

Only one domain in the boot is *expected* to wedge, and it is last, as the rule requires.

## `S7Q` — the phase probe

In `domain_main`, at three top-level points, sample `pPager->pBackup` and
`pPager->pMmapFreelist` 8 times each:

* **P1** immediately after `sqlite3_open`
* **P2** after the workload statements
* **P3** immediately before `sqlite3_close`

On the first nonzero reading it **returns immediately** — flow stops at the top level by
construction, nothing downstream can wedge and destroy the answer.

Sentinel `0x5107_PSVT` (the `0x5107` space is unused; `0x9Exx`, `0x5A6E`, `0xDEAD`, `0xBADA5` are
taken):

```
P = phase 1..3, or 0 = all three phases clean
S = which field: 1 pBackup, 2 pMmapFreelist
V = 0 all-8-agree-and-zero, 1 all-8-agree-nonzero, 2 THE 8 DISAGREE
T = lcc type of the granule (7 = NOT_CAP)
```

## Pre-registered readings — written before the boot, not after

* **`V=2` at any phase** — the 8 reads disagree. **H-load demonstrated directly**, no RTL probe
  needed to establish it. This is the outcome that would end the ambiguity outright.
* **`V=1` at P1** — the field is nonzero straight out of `sqlite3MallocZero`. A zeroing gap on a
  path QEMU does not exercise; the repro shrinks to "open a database", and it is a software bug.
* **`V=1` first at P2 or P3** — the field was born zero and became nonzero. A wild store, and the
  interval is bisected to one phase.
* **`P=0`, all clean** — the field is genuinely zero right up to the last instruction before
  `sqlite3_close`. Combined with the wedge happening inside close, that localises the corruption
  to close itself, or supports H-load. **This is a real result, not a null one** — but only
  because the same probe reports `V=1`/`V=2` when they occur, which is what makes the zero
  meaningful.
* **`MRO` or `MRR` wedges** — the minimal repro is found and drops from ~1.5 MB of SQLite to a
  few dozen lines.
* **`S7T` fails** — boot VOID, nothing else in it carries a verdict.

## Positive control — the probe must be shown to fire

Per the CLEAN-result rule, a probe that has never produced a nonzero reading is unproven. `S7Q`
is built with `S07Q_SELFTEST=1` writing a known nonzero pattern into a scratch granule and
sampling *that* through the identical code path first. If the selftest does not report
`V=1, T=7`, the domain returns a distinct `0x5107_FFFF` and **every zero it reports that boot is
discarded**. This is the check that the S-07 tag-history probe lacked, which is why that probe
produced nothing usable on two boots.

## Explicitly dropped

`CapstoneLdcRetry` Phase A. At these sites there is no valid capability to recover, so the retry
re-reads NOT_CAP and the wedge proceeds under **both** hypotheses — it discriminates nothing, and
the old decision table would have mis-scored "still wedges" as a verdict.

## Cost

One boot, about 5 minutes of board time, plus one firmware rebuild covering all six domains.
No reflash, no RTL change, no dependency on the pending synthesis.

## The black-box recorder — accepted as an idea, NOT taken as a domain slot

The RTL lane's strongest counter-proposal: my constraint is right for *software* and may be wrong
for the *system*. Nothing executes after the wedge, so retval, `output_text` and any host-read
region are dead — but **a store that COMMITTED before the wedge is already in DRAM, and DRAM is
not cleared by a core reset.** A recorder at a fixed physical address, written as the probe goes
and read by a domain on the *next* boot, would not need the core to survive. If it works, the
constraint that shapes this whole plan dissolves and probes can sit **at the faulting `ldc`
itself**, turning every wedge into data rather than one bit. That is a much bigger prize than this
boot.

**It is not, however, a spare domain slot, and slot 6 stays `XU`.** Two obstacles:

1. **A domain cannot reach an arbitrary physical address.** Domains run on carved capabilities and
   cannot fabricate one, so someone has to mint a capability covering the recorder — the monitor
   at carve time, or the kernel module through the existing `REGION_SHARE` path. The RTL lane's
   three preconditions (reserve via the DT memory node, survive the JTAG load, retain across
   reset) are all real, but they are not sufficient: capability delivery is a fourth.
2. **The cheap shortcut — reuse the existing shared region and look for the pattern next boot — is
   DEAD, and this is the load-bearing new fact.** Every region allocation in
   `modcapstone/module/capstone.c` (lines 113, 142, 190) passes **`__GFP_ZERO`**. Linux zeroes the
   page on allocation, so the pattern is guaranteed absent next boot whether or not DRAM retained
   it. That experiment cannot produce a positive, and its negative would be uninformative — a test
   that cannot fire.

**Cheapest version that can actually answer it**, proposed rather than started because it touches
shared firmware: do the retention test **in the monitor**, in M-mode, before Linux exists. A few
lines in `sbi_capstone.c` that at boot read a magic from a fixed high physical address, print it,
then write a fresh pattern. One firmware rebuild, no DT change, no module change, no capability
plumbing, and it is its own positive control — write, reset, read back. If the pattern survives,
retention and JTAG-clobber are both answered at once and the recorder becomes worth building
properly (DT reservation + monitor-minted capability). If it does not survive, the idea is dead
for one rebuild and no board time beyond a normal boot.

Sequencing: this is **independent** of the six-domain boot and should not delay it.

## Addendum: the recorder is blocked by neither DRAM nor DDR init — source read, no rebuild

Two corrections land here, and both were established by reading source rather than by spending
board time or a rebuild.

**1. Cross-boot retention is impossible on the current flow, so the monitor retention test I
proposed above would have been VOID.** `run_ladder_base_fpga.py:76-77` does
`console.power(False); time.sleep(POWER_CYCLE_OFF); console.power(True)` with
`POWER_CYCLE_OFF = 8.0` (`run_rtl_smoke.py:65`). The board is **unpowered for 8 seconds every
boot**. DRAM holds charge for milliseconds unrefreshed, so a negative result would have meant only
"the board was switched off" — not a fact about retention. That test could not fire. (Credit to
the RTL lane for catching it before the rebuild.)

**2. The stated reason for power-cycling is NOT supported by the firmware source.** The harness
comment (`run_ladder_base_fpga.py:73-74`) says a warm `monitor reset halt` fails because "the
fw_payload OpenSBI cannot re-run its one-time hart/DDR init". Checked:

* **The DDR half does not exist.** `platform/fpga/ariane/platform.c:65-69` — `ariane_early_init`
  is literally `/* For now nothing to do. */ return 0;`, and `ariane_final_init` only calls
  `fdt_fixups` on cold boot. A grep for DDR/MIG/SDRAM across `platform/`, `firmware/` and
  `sbi_init.c` returns nothing but unrelated address offsets. DDR is brought up by the
  **bitstream's MIG at FPGA configuration time**, not by software, so there is no DDR init to
  re-run.
* **The hart half is real, but it is stale `.data`, not hardware state.** `fw_base.S:507-511`
  defines `_relocate_lottery` and `_boot_status` as writable words initialised to 0, and
  `fw_base.S:61-64` does `amoadd.w` on the lottery and branches to `_wait_relocate_copy_done` if
  the prior value was non-zero. On a warm restart **without reloading the image**, the lottery is
  already 1, so the boot hart waits forever for a relocation that will never happen — a dead boot
  that looks exactly like the comment describes.

**Consequence: a JTAG image reload rewrites `.data` and therefore clears the lottery.** So
`monitor reset halt` **followed by a full image reload** should boot, with **no firmware change at
all** — and the harness already does a reload after reset; it just also power-cycles first. The
experiment is therefore free: on a boot we are running anyway, try reset + reload **without** the
power cycle and see whether it comes up.

If it does, the recorder becomes viable for free: the board is never unpowered so the DDR
controller keeps refreshing, and the reload writes from `0x80000000` upward for the image length,
leaving a recorder placed **high** in DRAM untouched.

**Still unproven, and not to be trusted until it is:** that the recorder's store actually
COMMITS before the wedge. A store sitting in the store buffer when the core dies never reaches
DRAM, and the failure mode is a silent zero. The RTL lane is establishing that in simulation.
Until it holds, the channel is not a channel.

Sequencing unchanged: **the six-domain boot goes first and independently** — it needs none of
this.

## Addendum 2: the store-drain caveat is answered — verified against the RTL

The remaining hardware risk was that a recorder store still sitting in the write buffer when the
core wedges never reaches DRAM — failing as a **silent zero**. It is answered. Every claim below
was re-checked against the source here rather than accepted from the report, and the evidence is
ordered with the *measured* fact first.

1. **The clock is alive across a real wedge — measured on the failing runs, not read off source.**
   In the `OB` boot the selftest, a counter increment in an `always_ff`, fired *after* the wedge:
   `obcombo1.txt:1525-1529`, `post-204 = 0x41  OK: ldc_seen set and count moved by exactly 1`.
   This is the only item established on the actual failure, and it is the precondition for the
   rest.
2. **The drain is autonomous.** `wt_dcache_wbuffer.sv:269` —
   `assign miss_req_o = (|dirty) && free_tx_slots;`. A pure function of the buffer's own dirty
   state and TX-slot availability: no commit signal, no pipeline liveness, no core involvement.
3. **Every wbuffer flush port is tied off** — `wt_dcache_wbuffer.sv:320` `.flush_i (1'b0)`, and
   `:370`, `:486`, `:504` all `.flush_i('0)`.
4. **No flush path exists at all in this build.** `controller.sv:130-133` asserts `flush_dcache`
   only under `CVA6Cfg.DcacheFlushOnFence`, and that is **false in our config**:
   `capstone_cv64a6_imafdc_sv39_config_pkg.sv:52` `localparam CVA6ConfigDcacheFlushOnFence =
   1'b0`. So the dcache flush is never asserted — not by a trap, and not by a fence either. The
   reason is sound: the cache is write-through, so a fence has nothing to write back. The write
   buffer cannot be emptied by anything the core does; only its own drain empties it.

**Explicitly NOT part of this chain:** the `flushed cache implies flushed wbuffer` assertion at
`wt_dcache.sv:425-428`. It sits inside `//pragma translate_off` **and** `` `ifndef VERILATOR ``,
so our flow does not compile it and it has **never executed once** in this project. A check that
has never run reads exactly like a check that passes — the same shape as the lint gate that could
not fire. It is recorded here only so nobody cites it later; it is not evidence of behaviour.

**What remains is a program property, not a hardware unknown:** stores enter the write buffer at
COMMIT, so anything still speculative in the store unit when the core dies is lost. The recorder's
store must be placed **well ahead of the faulting site** — with the eviction walk between them,
which interposes 8+ committed loads and makes "still speculative at the fault" impossible rather
than merely unlikely. A concrete placement can be confirmed from the RVFI trace in ~15 s of
directed sim on the RTL side, once a real probe exists.

**Geometry re-verified in the config we actually build.** The stride figures above were first read
from `cv64a6_imafdc_sv39_config_pkg.sv`, a *sibling* of our target — the same wrong-file mistake
made earlier today with the amalgamation. Re-checked in
`capstone_cv64a6_imafdc_sv39_config_pkg.sv:48-50`: `32768 / 8 / 128`, identical, so the 4096-byte
conflict stride stands.

## A standalone lesson from this exchange: a code comment is a claim

The entire boot procedure — a full 8-second power cycle on every run — rested on a comment saying
OpenSBI "cannot re-run its one-time hart/DDR init". Half of that names a thing **that does not
exist on this platform**, and the half that is real is stale `.data` cleared by the image reload
the harness already performs. Nobody re-derived it because the sentence was plausible and
load-bearing. Comments are claims; the ones holding up a procedure deserve the same treatment as a
result.
