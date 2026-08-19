# S-07 — a capability read back from memory comes back UNTAGGED, sporadically

> **PROVISIONAL — the silicon numbers below are pending a timing attribution (2026-08-20).**
> The build that produced `caplifive_s07fix.bit` reports post-route **WNS −10.629 ns**, with
> **96727 of 246476 endpoints failing setup** (hold and pulse width are fine: WHS +0.054, WPWS
> +0.062). The **mechanism is NOT affected** — it rests on the RTL text, a Verilator matched pair
> and an assertion, none of which involve a bitstream. The **measured silicon numbers are**, and
> *all* of them, pre-fix as well as post-fix: the timing environment is byte-identical across both
> builds, so this caveat cannot honestly be scoped to the fix run. It is expected to resolve in
> favour of the numbers, because every result here is a **differential** between arms that differ
> by exactly one thing, and the entire design delta against the last known-healthy build
> (`618f4ce36`) is one module-internal file — `wt_dcache_wbuffer.sv`, +146/−1, no port changes.
> That is an argument, not a measurement. Two artifacts settle it and neither needs the board:
> the **per-clock Intra Clock Table** from the routed timing summary, and a grep of
> `ariane.timing_WORST_100.rpt` for `i_wt_dcache_wbuffer`. If the worst paths do run through the
> write buffer, the differential argument collapses and this becomes a candidate regression.
> Repeatability is deliberately **not** offered as evidence: a setup-failing path at fixed voltage
> and temperature can fail deterministically. Full record: `agent-handoff/ref/RATE-RULE.md`.


> # ROOT CAUSE FOUND AND CONFIRMED ON SILICON — 2026-08-19. THIS SUPERSEDES EVERYTHING BELOW.
>
> **The write buffer reorders two stores to the same 16-byte granule, and the loser's tag wins.**
>
> The buffer hits at **64-bit word** granularity, so a granule's two halves occupy **separate
> entries** — but each entry writes the **whole granule's single tag bit** on drain, and drain
> order is `rr_arb_tree` rotation, **not program order**:
>
>     wt_dcache_wbuffer.sv:444   hit compare on wtag == {address_tag, address_index[11:3]}  -> 8-byte
>     wt_dcache_wbuffer.sv:410   wr_idx_o  = wr_paddr[11:4]                                 -> 16-byte
>     wt_dcache_wbuffer.sv:416   wr_ctag_o = wbuffer_q[rtrn_ptr].ctag
>     wt_dcache_mem.sv:459       cap_tag_q[wr_idx_i][j] <= wr_ctag_i
>
> So an older plain store to `G+8` can drain **after** a younger `stc` to `G+0`, overwriting the
> tag the `stc` set. The capability is then reloaded untagged and faults on first use.
>
> **Measured on silicon**, 16384 slots per arm, one boot (`wbuf_kernel.h`, this repo):
>
> | arm | sequence | lost | expected |
> |---|---|---|---|
> | wb0 | `stc G` only | **0** | 0 |
> | wb4 | plain `G+16`; `stc G` | **0** | 0 — granule-scoped |
> | wb3 | plain `G+8`; **64 stores**; `stc G` | **0** | 0 — buffer drained |
> | wb1 | plain `G+8`; `stc G` | **1107 (6.76%)** | **0** — REORDER |
>
> **wb1 vs wb3 is the decisive pair**: identical stores, identical granule, differing only by
> intervening traffic that drains the buffer. 1107 versus 0.
>
> ### The SQLite trigger, end to end
>
> `sqlite3JournalOpen` does `memset(p, 0, sizeof(MemJournal))` and, twelve source lines later,
> `pJfd->pMethods = ...`. `pMethods` is the **first** member of `sqlite3_file`, so it occupies
> `[p+0, p+16)` — one granule — and the memset's plain stores hit its high word. The reorder
> clears the tag the `stc` set; `sqlite3OsRead+0x48` (`ldc a4, 0x0(a0)`) reloads `pMethods`
> untagged; `+0x4c` (`ldc a4, 0x20(a4)`) then faults with **mcause 25 = UNEXPECTED_OPERAND**,
> because its operand is NOT_CAP. Confirmed by the latched trap mepc at **11 wedges across 7
> distinct `DBAS` values**, all at offset `0x2a83c`.
>
> ### WHY THE "2.1M RELOADS, ZERO LOST" RESULT BELOW MISSED IT — and it is not a wrong measurement
>
> `tagsweep` stores **every** slot and only **then** clobbers a few. By the time its clobber
> issues, the `stc` entry has long since drained, so the two stores are **never in the buffer
> simultaneously** and the triggering condition is never created. The 2,097,160 reloads are real
> and the instrument was genuinely proven — it simply measured a different experiment.
>
> This is the failure mode the project rules name explicitly: *directed tests that come back clean
> without ever creating the triggering condition.* The exclusion below should be read as
> "eviction and bulk reload are not the mechanism", which remains true, and **not** as evidence
> against tag loss.
>
> ### The defect is BIDIRECTIONAL — see the sibling issue (severity since corrected)
>
> The same reorder also runs the other way. **It does NOT produce a tag over scalar data** — that
> was the first reading and it is retracted. Measured with field checking, the
> CORRUPTED-BUT-TAGGED bucket is **empty** (0 of 3840 in both directions): when the capability
> entry drains last it writes BOTH words of the granule, so the capability survives INTACT and the
> program's **plain store is silently dropped** instead. That is an integrity bug, not a
> capability-model soundness hole. Folder: **`S09-write-buffer-tag-forgery/`** (name retained so
> existing links resolve; its README leads with the correction).
>
> QEMU cannot reproduce either direction: its capability store is one atomic 16-byte-plus-tag
> operation with no write buffer, no per-word entries and no drain arbiter.
>
> ### MECHANISM CORRECTED 2026-08-19 — it is not tag ordering, and that matters for the fix
>
> The description above ("the loser's tag wins") is a **symptom**. The root is that
> **`is_cap` entries span TWO words but are tracked and merged as ONE**:
>
>     wt_dcache_mem.sv:241-250
>       if (!(st_wr_cap)) begin
>         bank_req |= dcache_cl_bin2oh(wr_off_i[...]);   // ONE word
>         bank_we  =  dcache_cl_bin2oh(wr_off_i[...]);
>       end else begin
>         bank_req = '1;                                  // BOTH words
>         bank_we  = '1;
>       end
>
> A capability entry writes **both** words of the granule; a plain entry writes one. They overlap
> on the high word — the **metadata** half. But the write buffer's hit/merge compare is on the
> **word** address (`wt_dcache_wbuffer.sv:444`), so it cannot see that a cap entry at `G+0`
> already covers `G+8`. Two entries end up writing the same physical word in arbitrary order.
> The tag disagreement is a consequence of that, not the disease.
>
> **THE OBVIOUS FIX IS WRONG, AND WRONG IN THE DANGEROUS DIRECTION.** Propagating the youngest
> store's tag to any co-resident same-granule entry — so drain order stops mattering — leaves the
> older plain entry still writing its stale scalar over the **metadata** half. Giving it `ctag=1`
> as well would produce a capability with a **valid tag over corrupted metadata**, converting the
> loss case into a forge case. It would have turned an availability bug into a soundness bug, and
> **the directed test would have gone green, because it only checks the tag.**
>
> **That is a limitation of the test in this folder and it is stated rather than discovered
> later:** `wbuf_kernel.h` verifies the TAG via `lcc` field 1 and does **not** verify the
> capability's bounds, permissions or cursor. It can distinguish tagged from untagged. It cannot
> distinguish a correct capability from a tagged one with corrupted metadata. Any candidate fix
> must be validated against metadata integrity, not against this test alone.
>
> **Two real options, neither built yet:**
>
> * **(A) make the merge granule-aware for capability entries** — a plain store to `G+8` merges
>   INTO a resident cap entry at `G+0`, its bytes routed to that entry's metadata half, `ctag`
>   last-writer-wins. One entry, one writeback, order-independent. Correct and complete, but real
>   surgery on the merge path.
> * **(B) forbid co-residency** — refuse to allocate an entry that conflicts at GRANULE level with
>   a resident entry when either side is `is_cap`, stalling until it drains. Small and obviously
>   correct, but it adds a term to `rdy`, which feeds the grant path, and that cone is on the
>   standing `UNOPTFLAT` list where synthesis has twice gone pathological.
>
> **The `wb3` arm is the evidence that (B) works:** 64 unrelated stores between the pair gave 0
> losses out of 16384 — co-residency broken by drain rather than by design.
>
> Choosing between them trades correctness completeness against synthesis risk in a cone that has
> already cost two blowups, and the forgery arm proves this subsystem can produce soundness
> failures. That is a design decision for the project lead, not a late-session patch.
>
> **Status:** mechanism identified from the RTL, confirmed on silicon by directed test. The fix is
> open. Everything below this box is the investigation that preceded the root cause and is kept
> for its exclusions, which remain valid.

> ## UPDATE 2026-08-18 — 2.1M capability reloads through DRAM on silicon, ZERO tags lost. READ THIS FIRST.
>
> Sibling issues, so a reader who arrived with the wrong symptom is redirected immediately: S-06
> (untagged 128-bit `ldc`/`stc` high half) and S-08 (dom-switch CSR clobber) are both FIXED in
> silicon and verified; their folders are resolved. This folder is the one open silicon issue.
>
> Bitstream `caplifive_s06s08fix_s07tag2_618f4ce.bit`. No RTL change was involved in anything below.
>
> ### A. NEW EXCLUSION, and it is the strongest one in this folder
>
> `tagsweep` is a **standalone rung, 10-39 KB, no SQLite**. It stores capabilities into memory,
> reloads them, and asks each one's type with `lcc` field 1 — total, returns 7 for NOT_CAP
> **without raising** — so tag loss is **COUNTED, never fatal**. The run always returns a number
> instead of wedging and destroying its own reporting channel.
>
> | rung | footprint | checks | retval |
> |---|---|---|---|
> | `k800` | — | — | `4` — control PASS, **boot valid** |
> | `ts1` | 128 B | 8 | `0xA5000000` |
> | `tsml` | 8 KiB, cache-resident | 1 048 576 | `0xA5000000` |
> | `tagsweep` | 64 KiB, exceeds the 32 KiB D-cache | 1 048 576 | `0xA5000000` |
>
> **Zero unseeded tag losses in 2 097 160 reloads.**
>
> **The instrument is PROVEN, which is the only reason that zero is evidence.** Each board arm
> deliberately clobbers `SEED = 3` slots per rep with a scalar store, dropping their tags. The
> domain returns `TAGSWEEP_OK | (lost - seeded_lost)` **only if** `seeded_lost == SEED * REPS`
> exactly, and `0xEE000000` otherwise. So `0xA5000000` certifies the counter detected **6144**
> deliberately-untagged granules in `tsml` and **768** in `tagsweep`, and found no others.
>
> **What it refutes.** The D-cache is write-through, no-write-allocate
> (`capstone_cv64a6_imafdc_sv39_config_pkg.sv:48-50`), so a `stc` does not allocate: after the
> store pass the slots are in DRAM and not cached, and every first reload is a genuine **miss
> refill** — the path this folder's own measurement implicates (`src=1, MISS REFILL`). Over 2.1M of
> them not one tag was lost. **S-07 is therefore not a generic property of storing a capability to
> memory and reloading it**, at any rate above ~1 in 2.1M, and "the refill path erases tags" is
> refuted at that rate.
>
> ### B. CORRECTION — `s07evict` was VOID, not negative, and the summary built on it was wrong
>
> This folder says every synthetic shape is excluded. That overstated the evidence. The one arm
> that targeted the **memory** path, `s07evict`, is recorded as returning 0, but it assumed
> **64-byte cache lines**; the real geometry is 128-bit (16-byte) lines, 4 KiB per way, 256 sets.
> Its eviction never happened, so it tested nothing and must not be counted as an exclusion. The
> `tagsweep` result above is what that arm was trying to be, with a control it did not have.
>
> ### C. RETRACTION — §5's "the memory held wrong data" claims more than the evidence
>
> §5 says a non-zero cursor in a should-be-NULL slot "means **the memory held wrong data**". It
> does not. It means **the guard's load RETURNED non-zero**. Memory-holds-residue and
> load-returned-wrong-data have identical observables at that site: if memory held a correct zero
> and only the guard's load misread, the following `ldc` still yields an untagged value and the
> next use still faults at the same `mepc`, with the same `sw=204 = 0x00` and the same clean QEMU.
> The sentence is corrected in place below.
>
> ### D. THE §0 CAVEAT ON THE STICKY BIT IS NOW CLOSED
>
> §0 says the displacement sticky bit "has never been observed to SET on silicon", so a `0x00` read
> could not be fully separated from a dead detector. **It has now been made to set on hardware.**
> A boot on this bitstream reported `SELFTEST post-204 = 0x41  OK: ldc_seen set and count moved by
> exactly 1`, then `SELFTEST PASS`. Every `sw=204 = 0x00` on that boot is therefore a **controlled**
> negative, and displacement (case a) is excluded for those wedges with a proven instrument.
>
> ### E. WHERE THIS LEAVES THE DIAGNOSIS
>
> Still open, and narrowed rather than answered: **memory-holds-residue vs load-returns-wrong-data**
> remains undecided. What A adds is that a bulk store→DRAM→reload sweep is **not sensitive enough**
> to trigger it, so the mechanism depends on something the sweep does not reproduce. The most
> useful constraint for whoever picks this up is already in this folder and is easy to miss: **for
> a given image the fault site is FIXED**, and what varies run to run is only whether it fires. A
> statistical sweep is the wrong shape for a defect that is structural per image; behavioural
> minimization of the failing image is the right one.
>
> ### F. TWO HARNESS DEFECTS ON OUR SIDE, recorded because they cost three boots
>
> Neither is a silicon issue; they are here so a reader does not repeat them.
> 1. **`DOMAIN_WINDOW` is unusable without a monitor rebuild.** The rung build will relocate a
>    domain's globals, but the monitor hardcodes `GPFREE_GLOBALS_OFFSET 0x1000`; the two then
>    disagree about where the blob starts and the domain hangs before it runs.
> 2. **"NO SHA6 → ENTRY STALL" is a misdiagnosis for a SLOW domain.** `SHA6` prints only when a
>    domain RETURNS, so one still running is indistinguishable from one that never entered — yet
>    the rung driver asserts R-16 and advises not to retry the binary. Three boots went into
>    rebuilds before a deliberately trivial arm (`ts1`, 8 checks) returned instantly and showed the
>    domain had been entering and running correctly all along, just slower than estimated. **When a
>    run yields no result, build one arm that must finish in microseconds before rebuilding
>    anything.**

> ## UPDATE 2026-08-16 — four boots on `caplifive_s07diag.bit`. READ THIS FIRST.
>
> Sibling issues, so a reader who arrived with the wrong symptom is redirected now: S-06
> (untagged 128-bit `ldc`/`stc` high half) and S-08 (dom-switch CSR clobber) are both FIXED in
> silicon and verified; their folders are resolved. This folder is the one open silicon issue.
>
> ### 0. STATUS AND WHAT THE NEXT BOOT DOES
>
> **The discriminator now exists in hardware.** The RTL lane synthesized the scoreboard's
> displacement condition as a **sticky bit on the debug mux** — bank `3'b110` reg `5'b01100`, i.e.
> switch **204**, byte `{stc_seen, ldc_seen, ldc_count[5:0]}` with the count saturating at 63,
> cleared only by reset so it survives the wedge. They positive-controlled it by forcing the syncer
> to bypass. It needs no reproducer, no `mtval` and no working trap dump.
>
> **Read rule.** Non-zero → a capability op's response was displaced onto a scalar writeback port,
> which zeroes `cap_result` at writeback: **the value was still intact in memory** (case a).
> All-zero → nothing was displaced, so the NOT_CAP came from memory/tag state (case b).
>
> **Three things that make a naive read wrong**, all handled in
> `tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py`:
> 1. **It is boot-cumulative.** Cleared only by reset, so the byte at a wedge covers the control,
>    every earlier domain, and the monitor's own traffic. The driver samples after *every* domain;
>    the **count delta** is the attributable number, the seen-bits alone are not.
> 2. **Switch 191 is a blind window.** The logging `always_ff` is `if (clear) … else <record>`, so
>    while the switches sit at the trap-log clear nothing is recorded — and a missed displacement
>    looks exactly like case (b). `SQLITE_TRAPLOG_CLEAR=first` gives the runs under test a zero
>    blind window; the trap latch is last-writer-wins, so a wedge still latches its own mcause-25,
>    and a latch unchanged since before the domain is reported STALE.
> 3. **The encoding is closed, so the readout is self-checking.** Legal bytes are `0x00`, `0x80`,
>    and `ldc_seen` with count ≥ 1 (`0x41-0x7F`, `0xC1-0xFF`). Anything else — including `0x40` and
>    `0xC0` — is an instrument fault, not a finding. This matters because a mis-aimed read returns
>    the mux default `0x00`, which is *also* the legal quiescent value, so a wrongly-pointed probe
>    is otherwise indistinguishable from a clean case-(b) verdict.
>
> **Asymmetry to hold to.** A non-zero delta is self-supporting. An all-zero byte means "not case
> (a) on this run", **not** "case (b) established" — get a second wedge first.
>
> **Boot recipe:** `SQLITE_TRAPLOG_CLEAR=first`, control first, then the uninstrumented full
> workload repeated until it wedges, four domains maximum. **Requires the bitstream built from the
> commit carrying the sticky bit** — on anything older switch 204 reads whatever that slot decoded
> to before.
>
> ### 0. RESULT: TWO WEDGES, BOTH WITH THE DISPLACEMENT BYTE AT ZERO
>
> | boot | rep | DBAS | latched mepc | image VA | site | `sw=204` |
> |---|---|---|---|---|---|---|
> | 5 | 3 of 3 | `0x84400000` | `0x8442a83c` | `0x3a83c` | `sqlite3OsRead+0x4c` | **`0x00`** |
> | 8 | 2 of 3 | `0x84000000` | `0x8402a83c` | `0x3a83c` | `sqlite3OsRead+0x4c` | **`0x00`** |
>
> Independent boots, different domain placements, different rep indices, same instruction:
> `ldc a4, 0x20(a4)` whose rs1 came from the immediately preceding `ldc` — the invariant. Both
> control-validated, both with `SQ: G/enter` (so genuinely executed), both mcause 25 (in the
> capability range 24..39, so the latch is the domain's own).
>
> **CONCLUSION: on both wedges nothing was displaced onto a scalar writeback port.** The
> syncer/LOAD_WB path — the leading candidate on both the board and RTL sides — **is not the
> mechanism**. That leaves (b) the load genuinely returned `tag=0`, and (c) the granule was never
> tagged.
>
> #### The caveat that bounds this, and it is not small
>
> **The sticky bit has never been observed to SET on silicon.** Its positive control was done in
> *simulation* (the syncer matcher forced to bypass makes it set; the 79-test sweep leaves it
> silent). On hardware it has only ever read `0x00`. By this project's own hardest rule — *a CLEAN
> result is not evidence until the check is known to fire* — a silicon-side demonstration that the
> bit **can** set is still missing, and until then "no displacement" and "the detector does not
> work in this bitstream" are not fully separated.
>
> What makes this weaker than the usual version of that error: the detector is not a bespoke
> counter but a direct read of a condition the sweep exercises, and it is validated in the same RTL
> that was synthesized. What would close it: a debug-triggered synthetic displacement, so the bit
> can be made to set on demand on hardware. That is one line of RTL and belongs in the same
> reflash batch as the tag-history probe.
>
> ### 0a. FIRST MEASUREMENT: a wedge with the displacement byte at ZERO (boot 5, 2026-08-17)
>
> On `caplifive_s06s08fix_s07probe_a2ef8eb.bit`, a control-validated boot in which all four
> domains entered:
>
> | domain | result | `sw=204` after |
> |---|---|---|
> | `S7T` control | RETURNED `0x57070703` | `0x00` |
> | `XU` rep 1 | RETURNED, workload passed | `0x00` |
> | `XU` rep 2 | RETURNED, workload passed | `0x00` |
> | `XU` rep 3 | `SQ: G/enter`, then **WEDGED** | `0x00` |
>
> At the wedge: `sw=255 = 0x99` (seen, **mcause 25**), **`sw=204 = 0x00`**, latched
> `mepc = 0x8442a83c` with `DBAS 0x84400000` → image VA `0x3a83c`, symbolised **in the `XU` binary
> that ran** as `sqlite3OsRead+0x4c`:
>
> ```
>   3a834: ldc a0, 0x0(a0)
>   3a838: ldc a4, 0x0(a0)
>   3a83c: ldc a4, 0x20(a4)   <== mcause 25
> ```
>
> i.e. the canonical instance of the invariant, at the same site as the historical `XF`
> reproducer (`XU` is byte-identical to it, `f1214600`).
>
> **Validity:** control passed; `SQ: G/enter` present so the domain genuinely executed rather than
> entry-stalling; mcause 25 is in the capability range 24..39 so the latch is this domain's own and
> not stale kernel traffic; `0x00` is in the legal encoding set so the integrity check correctly
> stays silent; and the byte read `0x00` twice — post-domain and again at the wedge.
>
> **CLAIMED: on this wedge nothing was displaced onto a scalar writeback port, so case (a) is not
> supported here. NOT CLAIMED: that case (b) is established** — see the stopping rule below and the
> third case now on the table.
>
> ### 0a-bis. THERE IS A CASE (c), AND IT MAKES A RETRY PROBE ACTIVELY MISLEADING
>
> The fork was (a) syncer displacement vs (b) the load returning `tag=0`. Both assume the granule
> **was tagged when it was stored**. The faulting site is a capability *spilled to a stack slot and
> reloaded*, so there is a third possibility:
>
> **(c) the granule was never tagged.** Post-S-06 an `stc` writes `ctag` from the rs2 register tag,
> so an untagged register produces an honestly untagged granule and the reload returning NOT_CAP is
> **correct behaviour**. The fault would then be upstream of both memory and the syncer.
>
> The displacement detector cannot see (c) — and neither can a retry: re-issuing the load re-reads a
> granule that is honestly untagged and stays untagged, which is **indistinguishable from (b)**. So
> the obvious next probe, a compiler-emitted `ldc` retry, would have returned a confident (b)
> verdict that was wrong. It is not being built for that reason (the pass exists at
> `llvm/lib/Target/Capstone/CapstoneLdcRetry.cpp`, off by default, and stays off).
>
> **The discriminator that separates all three** is a store/load tag history keyed on address:
> `(paddr, ctag)` of the most recent `stc`, and `(paddr, source)` of the first `ldc` whose response
> tag is 0, with `source` distinguishing L1 hit / refill / write-buffer forward — the three legs of
> `rd_ctag_o`. Then: stored `ctag=1` but loaded 0 → genuine (b), and `source` says where; stored
> `ctag=0` → (c), and the hunt moves upstream; no matching `stc` → the granule came from a copy or
> a context restore, itself informative. That needs a bitstream, so it is **ask-first** and batched.
>
> ### 0b. PRE-REGISTERED STOPPING RULE (written before the data, 2026-08-17)
>
> **There is no validated wedge rate for this bitstream.** The familiar "~1 in 3" comes from
> `caplifive_s06fixs08fix`, and a reflash invalidates prior silicon numbers — that rule exists
> precisely because rates are what shift. So `(2/3)³ ≈ 0.30` for a clean 3-rep boot is arithmetic
> against a **stale prior**, useful as a rough guide and nothing more.
>
> **Why the rate can move even though the sticky commit changes no behaviour.** That statement is
> true of RTL *semantics* — the detector is read-only, no backpressure, no path into any existing
> decision — but not of the *implementation*: it adds registers and a mux leg, so place-and-route
> differs, and this would be a different bitstream even if the RTL were byte-identical. If S-07 is
> timing-marginal, resynthesis can move the window **either way**. A lower rate here is therefore
> not evidence the defect is gone, and a higher rate is not evidence anything broke.
>
> **The threshold, fixed in advance: at THREE consecutive clean boots, stop hunting and report a
> possible rate change rather than continuing.** Against the stale prior that is ≈0.026, which is
> not decisive — the point is not the p-value but that beyond it "run another boot" stops being
> cheap and the alternative explanations become worth more than another repetition. An instrumented
> bitstream that no longer reproduces at all is a real outcome for a timing-marginal defect, and
> the answer to it is a different attack, not more grinding.
>
> **Boundary that must stay attached to a `0x00` reading.** The counter only moves for an LDC/STC
> displaced onto LOAD_WB/STORE_WB. A persistent zero is equally consistent with "no displacement
> happens" and with "the mechanism is case (b), which this detector cannot see by design". It rules
> out *routine* displacement; it does not rule *in* case (b).
>
> ### 1. The invariant: ONE instruction shape, three unrelated functions
>
> Three wedges, three different builds, three different source functions, three different
> addresses — and byte-for-byte the same shape. **Two ADJACENT `ldc` instructions where the
> second's rs1 is the first's rd:**
>
> ```
> sqlite3OsRead+0x4c   (S7P)      pagerFreeMapHdrs+0x4c (S7C)     sqlite3BackupRestart+0x5c (S7B)
>   3a8d0: ldc a0, 0x0(a0)          43368: ldc a1, 0x0(a0)          40bc0: ldc a0, 0x0(a1)
>   3a8d4: ldc a4, 0x0(a0)                                          
>   3a8d8: ldc a4, 0x20(a4)  <==    4336c: ldc a1, 0x40(a1) <==     40bc4: ldc a0, 0x70(a0) <==
> ```
>
> A **fourth** site, `whereLoopOutputAdjust+0x200` (`S7B`, boot 5), is the purest instance — three
> consecutive identical loads, a bare pointer chase through one register, no arithmetic between:
>
> ```
>   115884: ldc a0, 0x0(a0)
>   115888: ldc a0, 0x0(a0)   <== mcause 25
>   11588c: ldc a0, 0x0(a0)
> ```
>
> `ldc`'s guard is rs1-only (`capstone_dyn_unit.anvil:327-330`), so in every case the value
> **produced by the preceding `ldc`** arrived NOT_CAP. This is the back-to-back dependent
> capability-load pair, and it is a far sharper statement than any site name.
>
> ### 2. "The site wanders" is WITHDRAWN — it was an artefact of comparing builds
>
> Of 8 mcause-25 wedges, 6 have a recoverable image VA. **Five of those six are
> `sqlite3OsRead+0x4c` in five DIFFERENT builds** — different link addresses for the same source
> site (`0x3a2f8`, `0x3a83c`, `0x3a8d8`, `0x3a9d0`, `0x3aa74`). Earlier wording (ours, and an
> auditor's) compared raw `mepc` values across builds and read that as wandering. It is not.
>
> ### 3. Instrumenting a site does not fix it — it MOVES the death
>
> | build | site instrumented | wedged at |
> |---|---|---|
> | `S7C` | `sqlite3OsRead` | `pagerFreeMapHdrs+0x4c` |
> | `S7P` | `pagerFreeMapHdrs` | `sqlite3OsRead+0x4c` |
> | `S7B` | BOTH | `sqlite3BackupRestart+0x5c` |
>
> Each build dies at whichever vulnerable `ldc` pair the previous one had covered. **A software
> probe can therefore never be the thing that fires** — the uncovered site always kills the run
> first. Combined with the fact that a wedge discards the retval AND everything `output_text`
> buffered (the host only reads that buffer when the domain RETURNS), a wedging run has no
> reporting channel at all.
>
> ### 4. `mtval` is written and UNREADABLE on this path
>
> The RTL lane's diagnostic puts the faulting rs1 cursor in `mtval`, to be read from the monitor's
> trap dump. That dump never runs: a capability fault inside a domain wedges at exception commit
> instead of trapping to `mtvec` — `capstone-ariane core/cva6.sv:1228-1231` says so in as many
> words. Measured, matched pair, same capture code and board:
>
> | latched cause | `EXCX` | `MCAU` | `MTVL` | other live monitor markers |
> |---|---|---|---|---|
> | mcause **8** (3 runs) | 1 | `00000008` | 1 | 3 |
> | mcause **25** (6 runs) | **0** | — | **0** | 6–18 |
>
> The mcause-8 rows are the fired positive control. The debug latch carries no `tval` either
> (`cva6.sv:994-996`, `:1097-1099`).
>
> **Nor can GDB read it — measured, not assumed.** Halting the wedged core against a latched
> `mcause=25, mepc=0x84105888` returns `mcause=2 mepc=2 mtval=0`: the CSRs are already destroyed by
> a nested trap. `mtval=0` taken at face value reads as "the operand was NULL", i.e. a confident
> H2 verdict that is simply wrong — our reader discards it because gdb's mcause/mepc do not match
> the latch. **So the latch is the ONLY remaining channel.**
> **The ask is in `rtl/MESSAGE-TO-THE-RTL-LANE.md` §1.**
>
> ### 5. H1/H2 — the fork was incomplete, and H1 is NOT established
>
> At `pagerFreeMapHdrs+0x4c`, **H2 is refuted by control flow**: the loop condition two
> instructions earlier reads the SAME stack slot with a plain integer `ld` and branches away if
> zero, with `a0` rederived from `s0` and no intervening store, so reaching the fault proves the
> cursor was non-zero. But in that build the field can hold no legitimate capability
> (`SQLITE_MAX_MMAP_SIZE` is 0 at `sqlite3-capstone.c:16156`; `pagerAcquireMapPage`, the only
> setter of `PGHDR_MMAP`, is ABSENT from the binary — the writer at `:63849` IS compiled in and
> callable, gated only by that flag, so the accurate phrase is **"no reachable writer under intact
> data"**). A non-zero cursor in such a slot means **the guard's load RETURNED non-zero**; whether
> memory held non-zero is UNDETERMINED (corrected 2026-08-18, see block C at the top). Either
> way it is not the same as a lost tag. **"A real capability arrived NOT_CAP" is NOT claimed.**
>
> ### 6. What is solid
>
> * the selftest control returned its exact PASS value `0x57070703` on **every** boot, so the
>   `ld`-based instrument is proven on this bitstream, zero-reads included;
> * `sqlite3OsRead` is **never called in a clean run** (`calls=0`, full extended workload passing),
>   so it is reachable only after an upstream error and was always the SECOND fault;
> * clean reps are NOT evidence of suppression — at the observed rate, 3 clean is p≈0.30.
>
> ### 6b. TWO MORE MECHANISMS REFUTED (2026-08-16, by source inspection)
>
> Both were chased to the point of writing a test, and both died on reading the RTL. Recorded so
> nobody re-derives them:
>
> * **Operand forwarding (LOAD_WB outranking CAP_WB).** The priority premise is real — CAP_WB is the
>   lowest-priority of the five writeback ports and ports 1/2/3 tie `cap_data` to `'0`, so a grant
>   there gives a correct cursor with NOT_CAP metadata. But `stall_waw_rs1`
>   (`issue_read_operands.sv:1434-1436`, `:1453-1455`, `:576-578`) stops a capability op issuing
>   while its rs1 is an in-flight destination, and that holds until **commit** — so a dependent
>   `ldc` pair never consults the writeback arbiter.
> * **The D-cache refill "tag inference".** `wt_dcache_mem.sv:431` looks like the `|user|` heuristic
>   the same file calls defect D3/D7 at `:144-147`, left behind when the S-06 fix replaced it on the
>   store path at `:441`. It is not: on the refill the dcache sees, `user` carries the **tag byte**
>   from the tag memory, not metadata (`wt_axi_adapter.sv:812-822` + `:743-747` — the tag R-beat
>   zeroes the user register, writes the tag byte, does *not* shift data, and only then signals the
>   return).
>
> **What survives.** Because the dependent `ldc` cannot issue until its producer commits, a NOT_CAP
> rs1 means the *first* `ldc` retired NOT_CAP — either (a) the syncer bypassed it to LOAD_WB, where
> `scoreboard.sv:246` zeroes `cap_result` (memory would still be fine), or (b) the load genuinely
> returned `tag=0`. The detector for (a) already exists as a sim-only `$error`
> (`scoreboard.sv:326-347`) and is absent from the bitstream, which is exactly why four boots could
> not separate them. The ask is now **one sticky bit on the debug mux** carrying that condition —
> see `rtl/MESSAGE-TO-THE-RTL-LANE.md`.
>
> ### 7. Operational
>
> `split_out_cap`'s unimplemented exact-fit case caps a boot at **~4 domains**; the 5th spins at
> `SPLB` with no `SQ: A/dom-ok`, and the monitor's own comment records that this once
> "manufactured a confident, entirely false localization of a SQLite function that never
> executed". Discriminator: `SQ: G/enter` present, plus the latch cause.

> ### CORRECTION 2026-08-15 — the "domain boundary / hostcall VFS" claim is WITHDRAWN
>
> We wrote that `sqlite3OsRead` reaches a hostcall-based VFS and therefore crosses the domain
> boundary, and offered that as the ingredient the rungs lack. **That is wrong.** The database is
> opened with `sqlite3_open(":memory:")` — SQLite's in-memory backend, entirely inside the domain.
> There is no file I/O and no boundary crossing on that path at all.
>
> So the distinguishing ingredient is **unknown**, not "the boundary". What remains different
> between the failing site and the passing rungs: a much larger working set and cache footprint, a
> capability chain rooted in heap-allocated structures rather than a static array, and far more
> capability traffic overall. Cache/working-set pressure was the leading remaining candidate.
>
> **Since tested and NOT supported as the distinguishing ingredient** (see the 2026-08-16 update at
> the top): the `s07evict` rung that was to test it is void — it assumed a 64-byte cache line where
> `DcacheLineWidth` is 128 **bits** = 16 B, and the cache is write-through no-write-allocate, so it
> walked at most 4 of 16 slots per set and its spills never allocated. The sharper description that
> replaced this line of enquiry is the adjacent-dependent-`ldc` invariant.

> ## THE PATTERN, MEASURED THREE TIMES IN THREE INDEPENDENT BUILDS (2026-08-16)
>
> Every instrumented build wedges at the same shape, and it is the sharpest description we have:
>
> | build | sha | fault VA | site |
> |---|---|---|---|
> | `XF` (uninstrumented, full workload) | `f1214600` | `0x3a83c` | `sqlite3OsRead+0x4c` |
> | `RT` (retry probe) | `debe064a` | `0x3a9d0` | `sqlite3OsRead+0x144` |
> | `RN` (retry + null discriminator) | `9dc935cf` | `0x3aa74` | `sqlite3OsRead+0x180` |
>
> In all three the emitted code is the same triple: a capability is loaded, **spilled to a stack
> slot**, **reloaded**, and the immediately dependent `ldc` through it raises mcause 25. `ldc`'s
> guard is rs1-only, so the reloaded value is genuinely NOT_CAP.
>
> ### What our instrumentation CANNOT settle, and why — please do not assume we just did not try
>
> The open question is whether that reloaded value is **(H1)** a real capability that lost its tag,
> or **(H2)** a legitimately NULL `pMethods`, in which case mcause 25 is the architecturally correct
> rendering of a NULL dereference and this site is not a silicon defect at all.
>
> We built three successive probes to answer it and none can:
>
> * LCC field 1 answers **7 for a lost tag and 7 for integer 0**, so the type query alone cannot
>   distinguish them. We added a null-cursor check for that.
> * A checked value does not stay checked. At `-O0` the compiler **spills and reloads between the
>   check and the dereference** — visible in the emitted code: our point-of-use guard is present
>   (the `SQLITE_IOERR_READ` early return is emitted) but the guarded path still does
>   `ldc a0, 0x0(a0)` from the stack again before the deref. So the value we verify is never the
>   value that faults.
> * A wedged domain **cannot report**. Our counters are returned at the end of the run, and there is
>   no live channel out of a domain that has already taken the fault.
>
> Closing this from the software side needs the check and the dereference in **one inline-asm block**
> with no spill between them. That is buildable; we are flagging it rather than doing it because the
> RTL side can read the faulting register directly and would settle H1-vs-H2 in one observation.
>
> **This is also why the "same construct in two binaries" result must be read carefully.** It shows
> the same *death site*, which is only the same *defect* under H1.
>
> ## RETRACTED 2026-08-16: the rung exclusions below are VOID on this bitstream
>
> We wrote that every synthetic shape is excluded and that the reproducer is therefore SQLite. **All
> of it is withdrawn**, on three independent grounds, each verified against the sources:
>
> 1. **No positive control has ever fired on this bitstream.** Every rung ran exactly once and not a
>    single `*_SELFTEST` arm was run since the reflash. So every `65535` here is a number from an
>    instrument never shown able to return anything else — the precise failure these rungs' own
>    headers warn about.
> 2. **`s07evict` did not evict.** The L1 line is **16 bytes**
>    (`capstone_cv64a6_imafdc_sv39_config_pkg.sv:50`, `DcacheLineWidth = 128` bits), not the 64 the
>    rung assumed, so its 64-byte stride touched one line in four — 64 of 256 sets — while the 16
>    spill slots sit in 16 *consecutive* sets. At most 4 of 16 checks were genuine eviction tests.
>    And the cache is **write-through with no write-allocate**
>    (`wt_dcache_wbuffer.sv:43-44`), so an `stc` spill never allocates the line in the first place.
> 3. **The rungs have no statistical power.** 16-48 samples bound the per-operation failure rate only
>    at p > 6-19% (rule of three). SQLite wedges at roughly 1-in-3 per execution over ~10⁶ capability
>    operations, i.e. a per-operation rate near 10⁻⁶. "The shape is sound" overstates by about five
>    orders of magnitude. The honest statement is **these rungs cannot see a defect at the rate this
>    one occurs.**
>
> Also corrected: the `s06spill` run recorded as an R-16 entry stall was nothing of the kind — the
> UART says `ladder-perf: open .dom failed`; the domain was never staged, and our classifier called
> a missing artifact an RTL entry stall.
>
> **And the SQLite failure is NOT deterministic on this bitstream**: the identical `L2.dom` passed at
> 17:45, passed at 18:15, and wedged at 18:26. Any bisection at one run per stage would call a
> failing stage clean about two times in three. The downward-bisection plan is withdrawn with the
> rest.
>
> ## (WITHDRAWN) EVERY SYNTHETIC SHAPE IS NOW EXCLUDED ON THE CURRENT BITSTREAM
>
> Seven rung experiments on `caplifive_s06fixs08fix.bit`, all control-validated (`k800` = 4), while
> the SQLite domain wedges reliably on the same silicon:
>
> | rung | the shape it adds | result |
> |---|---|---|
> | `s06spill` ×3 redraws | spill a capability, reload it | **65535** |
> | `s06bnds` | ...are the BOUNDS intact? | **65535** |
> | `s06wr` | ...surviving byte stores written THROUGH it | **65535** |
> | `s06pld` | ...surviving a scalar load of its granule | **65535** |
> | `s07evict` | ...**plus a 48 KiB walk to EVICT it before reload** | **65535** |
> | `s07chase` | dependent `ldc` chain | 0 — and **VOID**, see below |
> | `s07indep` | independent `ldc` burst | 0 — and **VOID**, see below |
>
> `s07evict` is the one that matters most, because the board had localized the fault to a spill and
> reload and the obvious missing ingredient was cache pressure. It adds exactly that variable and
> nothing else, and the eviction walk was **verified in the disassembly** to sit between the store
> and the reload before the boot was spent. It still returns 65535.
>
> **So no construct we can build in isolation reproduces S-07.** The smallest thing that does is
> still SQLite. We are stopping rung construction here: five shapes excluded and two void attempts
> is enough to say the approach is not converging, and continuing would be stubbornness rather than
> method.
>
> **What we would do next, and would rather hand over than guess at:** bisect the SQLite workload
> downward — reduce the failing domain until it stops wedging — instead of building synthetic
> shapes upward. That attacks the one artifact that *does* fail, and every reduction step is a
> measurement rather than a hypothesis.
>
> ## THE FOUR EXCLUSIONS RE-VERIFIED ON THE CURRENT BITSTREAM, 2026-08-15
>
> Every `0xFFFF` in this folder was measured two bitstreams ago and was therefore
> baseline-invalid. All four have now been re-run on `caplifive_s06fixs08fix.bit`, in
> control-validated boots (`k800` = 4):
>
> | rung | asks | current silicon |
> |---|---|---|
> | `s06bnds` | do BOUNDS survive a spill/reload? | **65535** |
> | `s06wr` | does it survive byte stores written THROUGH it? | **65535** |
> | `s06pld` | does it survive a scalar load of its own granule? | **65535** |
> | `s06spill` ×3 redraws | does a spilled capability come back TAGGED? | **65535, 65535, 65535** |
>
> `s06spill` first hit an R-16 entry stall and carried no verdict; it was REDRAWN three times
> (a padding knob varying image size while leaving the tested loop byte-identical, all three
> hashes distinct) and all three draws pass. That is 48 spill/reload round trips, every one
> tagged.
>
> **So the bare `stc` → `ldc` stack round trip is SOUND on this silicon**, and the sequence
> localized below is necessary but **not sufficient**. Something the SQLite site supplies is a
> required ingredient — the candidates being its surrounding capability traffic, its cache
> footprint, and the hostcall/domain-boundary path that no rung crosses.
>
> ## BOARD ANSWER TO YOUR QUESTION, 2026-08-15 — the tag dies on a STACK SPILL/RELOAD
>
> You asked for the datum that separates the syncer-mismatch path from the shadow-tag refill path.
> The debug mux cannot show a writeback port without a reflash, so we instrumented the failing site
> instead: query `pMethods` with LCC (total) and, if untagged, re-load the same address.
>
> **The probe never fired — and where it wedged is the answer.** Control-validated boot (`L2`
> returned), one RT arm returned, the next wedged at `sqlite3OsRead+0x144`:
>
> ```
>   3a9b8:  ldc a0, 0x0(a0)     ; the probe loads pMethods -- LCC query PASSES, counters stay 0
>   3a9c0:  stc a0, 0x0(a1)     ; the compiler SPILLS it to a stack slot
>   3a9cc:  ldc a0, 0x0(a1)     ; reloads from that same stack slot
>   3a9d0:  ldc a4, 0x20(a0)    ; <== mcause 25: a0 is NOT_CAP
> ```
>
> So the capability was **verified TAGGED by an LCC query**, then `stc`-ed to a stack slot and
> `ldc`-ed back **three instructions later**, and the reload is NOT_CAP. The vtable memory it
> originally came from was never in question — the probe passed on it.
>
> **This is a `stc` → `ldc` round trip through a stack slot, with a source proven tagged at the
> moment of the store.** That is a far smaller sequence than anything previously offered, it needs
> no SQLite, and it should be directly simulable.
>
> **Two honest caveats.** (1) It is still not every execution — one RT arm returned before this one
> wedged. (2) `s06spill` tests exactly this shape and returned `0xFFFF` — but on the PREVIOUS
> bitstream, so that exclusion is baseline-invalid and this shape is now **un-excluded**, not
> contradicted. Re-running `s06spill` on the current bitstream is the obvious cheap next step and we
> will do it.
>
> ## RTL-LANE ANSWER v2, 2026-08-16: `rtl/ANSWER-FROM-THE-RTL-LANE.md`
>
> Answers your 2026-08-16 reframing. **An RTL instrument now makes MTVL the H1/H2
> discriminator** (commit 45bd5a3ee): on cause 25, mtval carries the faulting operand's rs1
> CURSOR — 0 => H2 (legitimate NULL, not a defect, hunt upstream in sqlite3_step); nonzero
> => H1 (real tag loss). Validated four ways (nonzero/zero x DYN/FLU). This supersedes the
> powered-rung ask: boot the failing workload once and read MTVL from the dump you already
> print. **Source pre-check predicts H1**: readDbPage asserts !MEMDB, so the only reachable
> OsRead on :memory: is the memjournal path whose pMethods is a static const (never NULL).
> Also: S-07 does NOT reproduce in sim (dyn unit serializes cap loads — A-1 overwrite and
> hit-under-miss both impossible; the 8-entry-vector fix would be dead code), so nobody can
> read the faulting register in sim; the instrument is the substitute. Please fold it into
> the next synthesis (reflash stays yours, ask-first).

> ## RTL LANE: START WITH `rtl/MESSAGE-TO-THE-RTL-LANE.md`
>
> That is the handover — what is solid, **what of ours you should not trust** (both our reproducer
> rungs are void and A-1 is unmeasured, not downgraded), the three things we'd ask you to take, and
> a proposed split of work. Your S-06 and S-08 fixes are both **confirmed on silicon**; S-07
> survives and is easier to hit than before.


> ## S-07 SURVIVES THE S-06 FIX — measured 2026-08-15 on `caplifive_s06fixs08fix.bit`
>
> The full extended SQLite workload, built with **no software workarounds at all** (no granule
> guard, no library memcpy fixup, no instrumentation), wedged on its **first** execution in a
> control-validated boot. `sw=255 = 0x99` → seen, **mcause 25**.
>
> **The new instance is RS1-UNAMBIGUOUS and is the cleanest evidence in this folder.** `mepc`
> `0x8342a83c` decodes to domain VA `0x3a83c` = `sqlite3OsRead+0x4c` — a fourth function, unrelated
> to the earlier three:
>
> ```
>   3a834:  ldc  a0, 0x0(a0)
>   3a838:  ldc  a4, 0x0(a0)      ; a4 is loaded BY AN ldc
>   3a83c:  ldc  a4, 0x20(a4)     ; <== mcause 25: a4 is NOT_CAP
> ```
>
> The fault is **at an `ldc`**, whose guard is rs1-only (`capstone_dyn_unit.anvil:327-330`). There is
> no two-armed `cincoffset` here to explain it away: **a capability produced by an `ldc` arrived
> untagged**, and the immediately dependent `ldc` raised on it. A-family is confirmed; the rs2
> reading cannot account for this site.
>
> **It also matches the A-1 mechanism's prediction.** Two capability loads are in flight
> back-to-back and the second *depends on the first* — precisely the shape that displaces a
> one-deep outstanding-request tracker (see `rtl/MECHANISMS-AND-PATCH-PROPOSAL.md`, A-1). We are not
> claiming that as proof; we are pointing out that the site is the construct A-1 predicts, and that
> it is now reproducible on the first try rather than 1-in-4.
>
> ### THE SAME SOURCE CONSTRUCT, IN TWO INDEPENDENT BINARIES
>
> `L2.dom` (sha `fd0445cf…`) and `XF.dom` (sha `f1214600…`) are different builds at different
> addresses. Both wedge at **`sqlite3OsRead+0x4c`**, on the byte-identical triple:
>
> ```
> L2   3a2f0: ldc a0,0x0(a0)   3a2f4: ldc a4,0x0(a0)   3a2f8: ldc a4,0x20(a4)   <== mcause 25
> XF   3a834: ldc a0,0x0(a0)   3a838: ldc a4,0x0(a0)   3a83c: ldc a4,0x20(a4)   <== mcause 25
> ```
>
> So this is **the construct, not the image layout** — the failure follows the source, not the
> address. `L2` is the truncation arm that returned in every surviving pre-fix transcript we have
> and had never wedged in any surviving transcript; on this one it wedges. We have **two** post-fix
> wedges, so we do NOT claim a rate — only that a domain which never failed before now does.
>
> ### TWO MINIMAL REPRODUCERS TRIED, BOTH RETURN 0 — please do not re-run these
>
> | rung | what it does | silicon |
> |---|---|---|
> | `s07chase` | 20 000 hops of `ldc` → **dependent** `ldc`, no query between, burst of 8 | **0** |
> | `s07indep` | 8 **independent** capability loads back to back, no address dependencies | **0** |
>
> Both have firing positive controls (selftest arm returns 256 under QEMU) and both ran in
> control-validated boots (`k800` = 4).
>
> **The first one was also structurally incapable of testing A-1, and that is worth knowing.** A
> *dependent* `ldc` cannot issue while its predecessor is outstanding — it needs that result as its
> base address — so a pointer chase can never put two capability loads in flight at once. `s07indep`
> was written to fix exactly that. It also returns 0.
>
> **Consequence for A-1:** if displacement required two capability loads in flight, `s07indep`
> should have hit it in a tight loop. It did not. And note the faulting SQLite triple is itself a
> *dependent* pair, so whatever overlaps it must come from elsewhere. A-1 is not refuted — its
> window may need traffic a rung cannot create — but it no longer explains the evidence on its own.
>
> ### WHAT THE FAULTING SITE HAS THAT THE RUNGS DO NOT
>
> `sqlite3OsRead` dispatches through `id->pMethods->xRead`, and our VFS is implemented by
> **HOSTCALLS — the domain boundary is crossed** on that path. The rungs cross nothing. Other
> differences: a 256 KiB heap and a warm cache versus a 64-entry ring, and a vtable capability
> loaded from the heap versus a static array.
>
> **The domain-boundary crossing is the one we would look at first**, and it is not idle
> speculation: the S-08 bug fixed the same day lived in the dom-switcher's context save/restore, and
> this fault lands on a capability loaded on a path that has just crossed that boundary.
>
> **There is a failing control again**, reliably, which every discriminator has been blocked on.
>
> Everything below predates the S-06 fix and the reflash, and its rate table is baseline-invalid.

**Status: OPEN. Silicon defect, not root-caused. Software workarounds do not address it.**

**WHICH SILICON.** The current evidence (the `sqlite3OsRead+0x4c` instances, and both rung runs) is
on **`caplifive_s06fixs08fix.bit`**, which carries your S-06 fix `25035c4c0` and the S-08 fix
`9fd5507b` — **both in-tree**, so unlike the previous bitstream this one IS reconstructible. Older
material below is on `caplifive_12august.bit` and is marked baseline-invalid; a reflash invalidates
measurements, and there have been two.

> ## START HERE IF YOU OWN THE RTL
>
> * **`rtl/MECHANISMS-AND-PATCH-PROPOSAL.md`** — the three candidate mechanisms with quoted
>   evidence, everything we ruled out so you do not re-derive it, one concrete patch proposal, and
>   the single board-free experiment we would run first.
> * **`board/G6P-DISCRIMINATOR.md`** — a 4-byte binary patch, built and verified, that separates
>   "the capability lost its tag" from "the offset gained one". Never produced a verdict; waiting
>   for reproduction.
> * **`board/fault-sites.md`** — the raw latched trap state and decoded fault sites.
>
> **REPRODUCTION STATUS — CURRENT.** On the current bitstream `caplifive_s06fixs08fix.bit` (which
> carries your S-06 fix `25035c4c0` and the S-08 fix `9fd5507b`, both in-tree) this defect
> **reproduces**: the full SQLite workload wedged on its first execution, and a second boot wedged
> its control domain. Two wedges, two different binaries, one construct.
>
> The paragraph that used to stand here said the opposite — "has not reproduced since … the 23%
> figure came from a lucky hour". That was true of the PREVIOUS bitstream and is now withdrawn. Any
> rate figure below the line marked *baseline-invalid* belongs to a bitstream that no longer exists.

**This is NOT S-06, and merging the two would be a mistake.** S-06 is *plain, untagged* data losing
its high 64 bits on an `ldc`/`stc` round trip — it corrupts data and raises nothing. S-07 is a
*genuine capability* coming back from memory with **no tag**, so the next instruction that requires
one raises **mcause 25 (UNEXPECTED_OPERAND)**. Different symptom, different cause code, and a fix
for one should not be assumed to touch the other. Sibling issues: `S06-untagged-ldc-stc-high-half/`,
`S01-image-perturbation-hang/`, `R20-stc-rs1-cursor-forward-x10/`.

---

## The signature

A capability is stored to memory, read back with `ldc`, and the value that comes back is NOT_CAP —
the next capability consumer raises mcause 25. Three instances, in three unrelated functions, none
of which share a caller.

> ### CORRECTION, 2026-08-14 — read before instances 1 and 2
>
> **For instances 1 and 2 the measurement does not establish which operand was wrong.** Both fault at
> a `cincoffset`, and that guard has TWO arms:
>
> ```
> core/anvil_build/capstone_flu_unit.anvil:29-31
> func CINCOFFSET(data){
>     if((data.cap_rs1.metadata.cap_type==cap_type_t::NOT_CAP)||(data.cap_rs2.metadata.cap_type!=cap_type_t::NOT_CAP)){
>         call raise_exception(data.trans_id,ex_code::UNEXPECTED_OPERAND)
> ```
>
> mcause 25 is raised if **rs1 is NOT_CAP _or_ if rs2 is anything other than NOT_CAP**. In both
> instances rs2 is an integer produced by a plain `ld` one to four instructions earlier. So "the
> reloaded capability lost its tag" and "the integer offset gained one" are indistinguishable in the
> data we have.
>
> **TWO observations are unambiguous** and together anchor the thesis:
>
> 1. instance 3, which faults *at* the `ldc` — its guard is rs1-only
>    (`capstone_dyn_unit.anvil:327-330`);
> 2. `sqlite3_strnicmp+0x134`, faulting at `cincoffsetimm a0, a0, 1` — the **immediate** form, whose
>    guard also has no rs2 arm (`capstone_flu_unit.anvil:57-61`). Recorded in
>    `src/s06spill_kernel.h:9-16`.
>
> So "a register that should hold a capability is NOT_CAP" **is established**. What remains open is
> whether the two `cincoffset` instances are the same mechanism or a second, rs2-side one.
>
> Sentences below reading `<== mcause 25: aN is NOT_CAP` for instances 1 and 2 are therefore an
> INTERPRETATION that was stated as a measurement. The discriminating query is cheap and is in
> "What would settle it".



**1. In our `memcpy`'s byte tail loop** — the most precisely characterised instance:

```
memcpy+0x2a8:
    lhu           a0, 0x24(a0)     ; a SCALAR load off the pointer      -- SUCCEEDS
    cincoffsetimm a2, s0, -0x60
    ldc           a2, 0x0(a2)      ; reload the dest pointer from its stack slot -- SUCCEEDS
    cincoffset    a1, a2, a1       <== mcause 25: a2 is NOT_CAP
    sb            a0, 0x0(a1)
```

**2. In the domain's own output writer** — nothing to do with SQLite:

```
output_text+0xdc:
    ld          a2, 0x0(a4)        ; the payload length
    sd          a3, 0x0(a4)
    cincoffset  a1, a1, a2         <== mcause 25: a1, the SHARED-REGION PAYLOAD capability, is NOT_CAP
    sb          a0, 0x0(a1)
```

**3. In SQLite's allocator** — where a full workload run dies:

```
sqlite3DbMallocRawNN+0xd8:
    ldc  a0, 0x2a0(a0)             ; db->lookaside.pSmallFree; mcause 25
```

The common factor is **a capability read back from memory**, not any particular caller, primitive,
or data structure.

## It is SPORADIC, and that is part of the signature

The same binary (`G6.dom`, sha256 `f93a9188a9a4433c`, kept across boots and **not** rebuilt —
verified by hashing the initramfs cpio members, not the staging directory) both passes and wedges.
Measured deliberately, 2026-08-14: a control domain then eight repetitions per boot, three boots,
all three controls passing.

| source | genuine executions | passed | **wedged** | entry stalls (excluded) |
|---|---|---|---|---|
| earlier record (its one wedge was at `output_text+0xdc`, the same instruction) | 5 | 4 | 1 | 1 |
| boot 1 | 2 | 1 | 1 | 0 |
| boot 2 | 4 | 4 | 0 | 1 |
| boot 3 | 1 | 0 | 1 | 0 |
| **total** | **12** | **9** | **3** | **2** |

**The "entry stalls" column is MISLABELLED and it is not R-16.** Those arms stop far earlier, at
`SQ: id=5` with `RGNO:0000E00C` / `RGNN:00000020` — deterministic **monitor region-pool exhaustion**
(32 regions) during setup, before the domain is entered. The signature is identical in every boot in
both measurement windows, so the exclusion is symmetric and cannot bias the comparison — but it does
mean **every boot is structurally capped at about 4 genuine `G6` executions**, which is the real
reason accumulating samples is slow.

**p(wedge) = 3/12 = 25% per execution.** An R-16 entry stall is excluded from both numerator and
denominator — an image that never entered says nothing about the code in it, so counting one as a
failure would be wrong. Each boot stops at its first failure, so these are censored run-lengths,
not 8+8+8 independent trials.

**ALL THREE WEDGES ARE AT THE SAME INSTRUCTION**, `output_text+0xdc`. Not most of them — all,
across three boots. Boot 1 latched `mepc = 0x839416a8` and boot 3 `mepc = 0x835416a8`: different
4 MiB physical placements (two independent `__get_free_pages` allocations), both decoding to
domain VA `0x1516a8`.

So for a given image **the site is fixed and only the firing is sporadic**. This is the single most
useful thing in this folder: it names one `ldc`/`cincoffset` pair to look at rather than a class of
construct. The three instances listed above came from three different builds, which is consistent —
the site moves with the image, not between runs of one image.

One thing this does NOT show, because the overstatement is close by: the low 22 bits of the two
`mepc` values are identical, so every cache set index is the same in both. A **set-dependent**
mechanism is not excluded by this data.

**"Isn't that just the hottest loop?"** — the first fair objection, and no. `output_text` writes
the domain's output one byte per iteration, so it looks like a hot loop, but per execution it
writes only ~278 characters (3 result rows plus 15 `SQ:` markers) — on the order of 2 000
instructions, against a basic SQLite workload of at least a hundred thousand. That is under ~2% of
the run. Three independent wedges all landing inside a ≲2% region is p ≈ 10⁻⁵ under a uniform
fault; the concentration is real, not a sampling artifact of instruction frequency.

The same objection, answered the other way: if the trigger were something time-based rather than
site-based — an interrupt landing between the `ldc` and its consumer, say, with the domain context
save/restore losing a tag — the wedges would scatter across the workload in proportion to execution
time. They do not.

Any experiment on this defect needs repetition: a single passing boot proves nothing, and a single
wedge does not establish a deterministic trigger.

> ### THE RATE IS A PROPERTY OF THE WHOLE IMAGE, NOT OF `G6.dom` — added 2026-08-14
>
> **If you build only this domain and run it, you may see nothing at all.** Two independent
> measurements the same day show the defect responds to things outside the domain binary:
>
> * Adding ~85 instructions to `output_text` (an in-place probe) turned a working `CREATE` into
>   `rc=11` (malformed schema) — a completely different and much earlier failure. Verified as a
>   matched pair in ONE boot: the uninstrumented binary printed its three rows twice while the
>   instrumented one failed twice.
> * **THE DEFECT MAY HAVE STOPPED REPRODUCING — but this is NOT established, and an earlier version
>   of this note overstated it.** Since the rate was measured, `G6.dom` (byte-identical throughout)
>   has wedged **0 times in 18 further genuine executions** (4 of them on the byte-identical
>   *firmware*, recovered from the console's content-addressed image store). On the
>   **like-for-like** comparison — the same initramfs, the only configuration ever observed to
>   wedge — it is **0 in 8**: Fisher exact **p = 0.24, no evidence of a change at all**. Pooling all
>   18 gives p = 0.054, suggestive and no more.
>
>   The previously published "0 in 25, p = 0.0015, Fisher 0.034" is **WITHDRAWN**. It pooled in 11
>   executions of a *patched* binary built specifically under a hypothesis that predicts it will not
>   wedge; those arms are predicted not to wedge by both live explanations, so they cannot
>   discriminate between them, and pooling them to reach significance was circular.
>
>   A live alternative that the data does not exclude: **burstiness**. All three wedges fall inside
>   a single 17:27-18:40 window, and one boot *inside* that window was itself 0-in-4.
>
> An earlier version of this note attributed the suppression to three unrelated domains having
> been added to the initramfs. **That is RETRACTED.** Removing them again and rebuilding to a
> byte-size-identical cpio, with all 14 original domains byte-identical, did NOT bring the wedge
> back: 0 in 8 on the restored image. Image composition is therefore not the explanation, and
> physical placement is NOT promoted by this evidence — the earlier paragraph claiming so was
> written before the restoration test and was wrong.
>
> **What this means for reproduction.** The 23% figure is what was measured in one window on
> 2026-08-14. The defect is not reproducing now, on the same binary and an equivalent image, and
> the cause of the change is unidentified. Do not treat 23% as a rate you can rely on seeing.
> Candidates not yet separated: the several firmware rebuilds in between (content-identical monitor,
> relinked), some board-state or thermal effect after a long session, or genuine clustering that
> makes 3/13 a less stable estimate than it looked.

## What has been EXCLUDED, with positive controls that fire

Four ladder rungs, on this silicon, each returning `0xFFFF` — all sixteen slots intact:

| rung | question | selftest build | gives |
|---|---|---|---|
| `s06spill` | does a spilled capability come back TAGGED? | `-DS06SPILL_SELFTEST` | 0 |
| `s06bnds` | ...with its BOUNDS intact? | `-DS06BNDS_SELFTEST` | 0 |
| `s06wr` | ...surviving byte stores written THROUGH it? | `-DS06WR_SELFTEST` | 0 |
| `s06pld` | ...surviving a scalar load of its own granule? | `-DS06PLD_SELFTEST` | 0 |

**Every rung carries a positive control and every one has been shown to fire**, because `0xFFFF`
from a query that cannot return anything else is not a measurement. The selftest build feeds the
same LCC query a value that is not a capability and requires the mask to collapse to 0; all four
do. The controls are exercised under QEMU, whose LCC field-1 is total with the same encoding
(`capstone-qemu/target/riscv/op_helper.c:713-716` returns 7 for an untagged operand), and the
control sits behind an `#ifdef` so the clean build is byte-identical to the one measured on
silicon.

`s06spill`'s control was added on 2026-08-14, after its silicon run — it had shipped without one
while the three rungs written after it all had one. Its 65535 stands (same bytes), but until that
date it was an unproven instrument, and this table said otherwise.

Plus, in the SQLite domain (which owns a 256 KiB heap a rung cannot): a capability held live across
a walk touching **every line of that heap** comes back with type and cursor unchanged — so a plain
evict-and-refill does not lose the tag either.

**Ruled out from disassembly, without a boot — but the argument differs PER INSTANCE, and the folder
previously gave only the instance-1 form:**

* **Instance 1 (`memcpy`)**: every instruction touching the faulting granule is `stc`, one plain `ld`
  and three `ldc` — **zero plain stores**. Neither correct tag-clearing on a partial overwrite, nor
  the write-buffer `.user` clobber (`wt_dcache_wbuffer.sv:602` writes `.user` unconditionally
  whole-word while `.data` is byte-gated), which needs a coalescing plain STORE to the same word.
* **Instance 2 (`output_text`, the thrice-measured site)**: this loop DOES execute a plain
  `sd a3, 0x0(a4)` on every iteration, so the "zero plain stores" argument does not apply here at all.
  The exclusion still holds, for a different reason: the write-buffer hit compares the full 64-bit
  word address (`wt_dcache_mem.sv:276`, `wt_dcache_wbuffer.sv:444`), and the scalar at `s0-0x48` is a
  different word *and* a different 16-byte granule from the capability at `s0-0x40`.

**Ruled out previously — please do not re-run these** (recorded in `agent-handoff/ref/ISSUES.md`):

* **Rev-node pool exhaustion** — the pool holds 65536; the heads observed at wedges were ~250-600.
* **Rev-node tag loss zeroing `valid`** — refuted by rung `s06rev` (returns 11, both arms, control
  green). `valid` sits in `data_rdata`, not in `ruser`, so zeroing `ruser` cannot clear it. That
  rung also covers evict-and-refill of a capability round-tripped through memory **with** the
  validity queries `ldc`/`stc` perform.
* **The entire revocation-validity family, arithmetically** — those sites raise
  `INVALID_CAPABILITY` = mcause **26**, and this is **25**.
* **The S-06 fixup's store pattern** — `s06sfix` returns 2048 at 64 KB scale.
* **That it is specific to the `CREATE INDEX` statement** — refuted with a matched control that
  substitutes `SELECT count(*)` and wedges at the *identical* instruction. Table in
  `board/fault-sites.md`.

## Not reproducible under QEMU, structurally

QEMU is instruction-atomic with no cache, no write buffer and no eviction, and keeps a
full-precision bounds side-table for tagged loads (`cap_mem_map.h`). Its silence is not evidence.

## Two questions for the hardware side

1. **An R-20 analogue on another register — WE HAVE NOW LARGELY ANSWERED THIS OURSELVES; it is here
   so you do not re-derive it.** `f623c48a1` is an ancestor of every candidate synthesis tree and was
   never reverted. R-20's signature is incompatible with S-07 anyway: it was x10-specific, silent, and
   trapped nothing. The entire hand-written core contains exactly three register-literal special
   cases, all CAPENTER/x10-x11 (`issue_read_operands.sv:573`, `scoreboard.sv:236-238`,
   `decoder.sv:1287`) — none names x11 as an operand and none names x12 at all, while our instances
   fault on `a1`(x11) and `a2`(x12). We no longer think this is the mechanism.

   Two workload facts that close whole branches, measured by disassembling the domain (327 860
   instructions): it contains **zero** `amo*`/`lr.*`/`sc.*` and **zero** hardware `mul`/`div`/`rem`
   (soft routines instead). Any hypothesis resting on the atomic path or the multiplier is dead
   without a boot.
2. **Capability TYPE.** Every rung above spills a pointer to a static array — **NONLIN**. `stc`
   writes cnull back into rs2 for LINEAR/UNINIT/SEALED (`capstone_dyn_unit.anvil:458-461`), and
   `beebs_freestanding_string.c` already carries a `BEEBS_STRING_LINEAR_SAFE` knob because linearity
   has bitten these primitives before. Can a LINEAR or UNINIT capability round-trip through memory
   and come back untagged where a NONLIN one does not? We can build a rung for it given the shape
   worth testing.

## What would settle it

**Two experiments, and the first one alone is not enough.**

1. *The memory path.* An RTL simulation of the instance-1 sequence — `stc` to a stack slot, a plain
   `ld` of its low half, then `ldc` of the same slot — with the shadow tag `cap_tag_q` and the AXI tag
   byte instrumented, for both a NONLIN and a LINEAR source.

2. *The register-delivery path — please do not skip this one.* An `ldc`'s metadata reaches the
   register file only via the CAP_WB port (`cva6.sv:1379-1380`, `:1401-1408`). If a response is
   bypassed to LOAD_WB instead, that port carries no capability (`scoreboard.sv:320-324` ties
   `wb[1..3].cap_data` to `'0`) and the scoreboard erases the entry's `cap_result`
   (`scoreboard.sv:242-246`); commit then writes metadata `'0` (`commit_stage.sv:279`) into the
   metadata regfile under the **plain GPR** write enable (`issue_read_operands.sv:1578`
   `we_pack[i] = we_gpr_i[i]`). That produces a NOT_CAP register with a correct cursor, having never
   touched memory. Instrument the CAP_WB/LOAD_WB routing of the `ldc`'s `trans_id`
   (`ex_stage.sv:933`).

**A clean result from experiment 1 excludes nothing about experiment 2's path** — and it would read as
exoneration, which is why both are listed. It is sporadic on the board, so one clean simulation does
not exonerate either path; the question is whether it can *ever* happen.

**We are also running a board experiment that discriminates these directly**: query the type of BOTH
`cincoffset` operands at the failing site, and on a lost tag re-`ldc` the same address. If the retry
comes back TAGGED, memory was never wrong and the fault is in register delivery.

## Impact

The **basic** workload — CREATE / INSERT / SELECT returning all three rows / finalize — runs to
completion on silicon **75%** of the time (9 of 12), and wedges at `output_text+0xdc` the rest. The
**full** workload wedges at instance 3. So SQLite does execute on this hardware; what it does not
do is execute reliably, and the failure is in the domain's own output writer rather than in the
database engine.

## Files

* `rtl/MECHANISMS-AND-PATCH-PROPOSAL.md` — **the handover document.** Ranked candidate mechanisms
  with quoted `file:line` evidence, a corrected account of the transaction-ID question (the ID is
  NOT under-width — that framing is withdrawn, and the real concern is tracker depth), a concrete
  proposed change with an acceptance criterion that must FAIL before the fix, and a table of what is
  already ruled out.
* `board/G6P-DISCRIMINATOR.md` — the 4-byte patch that separates the two readings of mcause 25,
  with its stopping rule and the reason a recompiled probe cannot be used instead.
* `board/fault-sites.md` — the latched trap state and the decoded fault sites for each instance.
* `src/` — the four exclusion rungs (`s06spill`, `s06bnds`, `s06wr`, `s06pld`), each self-checking
  with a `*_SELFTEST` build that must return 0.
* `run.sh` — rebuilds and stages the exclusion rungs and prints what each should return.

**Not in this folder, and how to get it.** The domain binaries the board measurements were taken on
(`G6.dom`, sha256 `f93a9188a9a4433c…`, and the patched `G6P.dom`, `8f77d68dbb780dfb…`) are **not
shipped here** — they are ~1.6 MB SQLite domains. Ask and we will send them, or the disassembly
window around the faulting instruction. The harness that produced every run, and the definitions of
"genuine execution", "wedge" and "entry stall" as used above, is
`capstone/tests/rtl-smoke/fpga_driver/run_sqlite_stages_fpga.py`. `src/s06pld_kernel.h` ends with
"See s06evict" — that rung is not in this folder either; ask if you want it.

**A DIFFERENT SIGNATURE ALSO EXISTS, and if that is what brought you here, this is the wrong
folder.** `src/s06bnds_kernel.h:12-21` records five wedges from the same board on the same day with
**mcause 29 (OUT_OF_BOUNDS)**, at `vdbeMemClearExternAndSetNull+0x3c` — there the reloaded value is
**TAGGED** and it is the bounds that are wrong. That is a separate phenomenon from S-07's mcause 25
and is not analysed here.

**On the exclusion rungs' positive controls.** The `0xFFFF` results are from silicon; the controls
that prove the query can report NOT_CAP were exercised **under QEMU**, and on a different build —
`run.sh` notes the board runs the `_fpga_app.c` variant, which is not the same bytes as the `_app.c`
variant QEMU runs (24-byte result region instead of 8). The one control that *was* demonstrated on
silicon is the operand probe's, which returned `0x5B400000` on two separate boots
(`board/G6P-DISCRIMINATOR.md`). Treat the rung controls as strong-but-simulated rather than proven
on hardware.

**What we are asking for.** Not agreement — a check. Two of the three mechanisms we could not settle
from the sources, and each has an experiment that kills it. The one we would run first is board-free
and is named at the end of the mechanisms document.

Full investigation trail:
`agent-handoff/history/14-08-2026_02-30-00_sqlite-wedge-is-out-of-bounds-on-Mem.md`.

**This folder is the whole report.** An earlier draft of the same material lived in
`agent-handoff/ref/RTL-QUESTION-mcause25-tag-loss.md`; it was deleted rather than kept in sync,
because two documents for one issue is precisely how a live page ends up contradicting itself.
