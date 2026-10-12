# Current Capstone next step

## Read this first (2026-10-13)

This file was 1,560 lines, and it is a next-step file: a superseded
instruction here is not merely stale, it is an instruction. Everything dated
before 2026-10-01 is now in
[`history/13-10-2026_01-00-00_next-step-journal-through-2026-09-29.md`](../history/13-10-2026_01-00-00_next-step-journal-through-2026-09-29.md),
verbatim and with an index of all 20 sections -- including the five separate
blocks headed `0. CURRENT`, each current when it was written. Do not take a
next step from that file. The cut is by date alone; the dated entries below
are what is in force.

**The four open fronts, as of 2026-10-13.**

1. **Land the allocator migration.** `ports/nested-allocators-rebased`
   (PR #219, draft, conflict-free, native CI green) carries the eight ports on
   managed lifetimes and the physical Sublet heap's removal. Its one open check
   is **memcached fixture 24**: both page moves complete with the reversal flag
   set, and the protected arm returns `1800021` instead of faulting. The
   fixture now has to prove the orphan-header dereference actually happens
   before its temporal-fault prediction is touched; the gate stays recorded
   FAIL, not normalised to PASS. See the
   [cleanup qualification](../../runtime/virtual/results/managed-mallocng/README.md).
2. **Build the eight ports fresh, and gate them.** Seven of the eight were
   last *relinked* from cached virtual objects rather than rebuilt from source
   (`relink.json` names them). All eight now build through one entry,
   `ports/common/application/build-virtual.sh`, with one selector
   (`CAPSTONE_VIRTUAL_LIFETIMES`); a rebuilt SDK now invalidates a port's build
   tree instead of leaving stale objects behind. What is owed is the fresh
   builds themselves, and QEMU protection gates running regularly rather than
   only the host/native CI.
3. **The next bitstream.** `next-bitstream-r51-r52` (capstone-ariane
   `0568f93a9`) is approved and in synthesis. Owed, in order: the synth lane's
   routed numbers to the board lane; the reflash on the lead's own word; the
   three pre-registered boots. The next RTL candidate after it is **R-44**
   (the CPMP adopts unseen ids), which needs an evidence feed plus a seed for
   `cpmp(0..2)` or the board does not boot --
   `docs/plans/r44-cpmp-evidence-tracker.md` is the proposal awaiting the
   lead's decision.
4. **Corpora and ports housekeeping**, detailed in the 2026-10-04 section
   below: `ffmpeg/plane-repros` case 0 is the one unmeasured CheriBSD arm of
   54 and needs libavutil cross-built with the CHERI SDK; `check-ports.py` is
   blocked on one real problem (`ports/sqlite/cheribsd` declares 3.22.0
   against a `pin_source` carrying no version) and no gate runs it; 36
   committed files carry foreign home-directory paths, which is the lead's
   call. Run the corpus gates from a tree at `dev`'s tip -- the shared clone
   is ~170 commits behind and reports 14 corpora / 108 cases against the real
   23 / 218.

2026-10-10 — **CheriBSD PoisonCap is removed** (branch `nested/poisoncap-removal`; PostgreSQL's arm
goes with its rewrite). Every PoisonCap next step below is void, and its results stay as history.
See `current-state.md`.

2026-10-12 — **Sublet ISA is merged; allocator cleanup has fresh qualification.**
QEMU #21 and runtime #216 are integrated, with the superproject pinned to QEMU
`7dbfdd534eea`. The migration moves the eight nested allocator ports to the
managed lifetime operations and retires the physical Sublet heap and active
PoisonCap tooling. Review the per-port code and its recorded controls together.
Historical measurements identify their original builds; they are not results
from the rebased tree. The PostgreSQL hosted runner retains the per-case
negative controls introduced by #218, and the fresh paired run detects all
five defects while all safe variants complete. The
[cleanup qualification](../../runtime/virtual/results/managed-mallocng/README.md)
records allocator, runtime, corpus and application checks. Review and land the
migration branch with this evidence. Next investigate memcached fixture 24:
both page moves complete, but the protected run returns instead of faulting.
Establish whether the orphan-head access occurs before changing its prediction.
The previously mentioned CPython pop discrepancy is absent from the current
sharded archive: recomputing all 2,183 outcomes matches its zero-difference
stock/protected summary. Earlier failed runs and differences from native
remain recorded; this audit is not a new suite run.

2026-10-10 — **The next bitstream is approved and in synthesis: `next-bitstream-r51-r52` = capstone-ariane
`0568f93a9` (776d9d859 + R-52 + R-51). The lead's word: "yes, go, approve synth".**

Content: R-52 (546807884, the FPR clobber predicate un-negated; every hard-float native result on silicon was stale) and
R-51 (ef8900e2d + f4e4d6051, RETURN parks the domain's PC capability, so a resumed domain is bounds-checked). The board
lane's cherry-pick d15a836fa has the identical tree. Validation on that tree, all on apollo: lint gate PASS at the
committed baseline with the negative control firing; 95-test sweep vs 776d9d859 90/90 identical incl. cycles; the 79 arms
shared by both testlist_sup lists 76/76 identical (3 without a baseline log); 25 directed arms PASS by tohost (12 rv64ud,
s18 x2, r51 x2, the sup ladder); the six stcsb variants read their reference resume counts.

Predictions on record with the synth and board lanes before the build: routed WNS inside −13.335..−5.855; loop
membership by stem unchanged; LUTLP-1 0; R-52 flips one polarity term per scoreboard entry (an earlier "a few hundred LUTs
restored" was WITHDRAWN: the pre-fix line was a live mirror of the integer side, nothing was pruned); the discriminator is
whether failing endpoints move into issue_read_operands or the scoreboard. R-52's registry mechanism was corrected on dev
the same day (ecaa3691ae3c).

The audit's trap-entry question is measured (branch `r51-trap-after-yield` c2e2ed44b, tests only): after the fix a resumed
domain's handler runs under the domain's code capability as at a first entry; a vector of 0, the monitor's bare-domain
shape, storms ILLEGAL_INSTR identically on both trees, and the debug latch filters cause 2, so the R-43 list's a10 is
predicted unchanged (25 at DBAS+0x4354); the refuting signature is mepc 0 with cause 28.

Next: the synth lane's routed numbers to the board lane; the reflash on the lead's own word; the board lane's three
pre-registered boots (bare acceptance incl. r51 28-after-yield; the R-43 list + C5u; b0..b3 with b3-clock last). The next
RTL candidate after this cycle is R-44 (the CPMP revnode tracker adopts unseen ids: S/U-mode enforcement, live now that
native Linux processes run on silicon); it needs an evidence feed plus a seed for cpmp(0..2) or the board does not boot. Design proposal for the lead's
decision: `docs/plans/r44-cpmp-evidence-tracker.md` (fail-closed on install, the allocator's reserved indices 0..2 as the
seed, evidence from the rev-node fill taps, one install-time probe through the LSU's R-43 probe block).

2026-10-08 — **The virtual stack, node growth and local mallocng are all merged. Two items below
are corrected: one was DONE, one was FALSE.**

Landed today, each with its gates run rather than its PR body taken on trust:
`capstone-qemu` `virtual-capstone` = `1a6dd20732` (#17 paged node stores, #19 exact-bounds
profile, #20 PCC/access liveness cache); `llvm-capstone` `dev` = `4c8a3f12155e` (#194 build
repairs, #195 node growth, #197 local mallocng). Zero PRs open in any repo at that point.

**The submodule pin needed correcting on BOTH runtime PRs, and this is now a pattern, not an
accident.** #195 pinned `736f4a7a4e7a` and #197 pinned `24b1e95b1dac`; both are review-branch
heads, reachable from no integration branch. Each was repinned forward to the landed
`virtual-capstone` commit, with source-tree identity checked (#17's merge and `736f4a7a` share the
tree `713ffa8074`). #197's pin conflicted three ways precisely *because* dev's pin had already been
corrected, so taking either side mechanically would have pointed dev at an unreachable commit.
**Before landing anything that touches `capstone/capstone-qemu`, check whether the pin names a
commit on `virtual-capstone`.** The full trail is in
`docs/history/07-10-2026_16-11-23_virtual-capstone-stack-audit-and-landing.md`.

Corrections to the 2026-10-04 list below:

- **Item 3 (INDEX/index.json consolidated regeneration) is DONE.** One regeneration, `build-index.py
  --check` now reports current. It also closed ten gap lines by fixing the declarations behind them
  rather than the index: `corpora` on the three full-application ports, and `inventory` on
  `wmem-repros`, `allocator-repros` and `subobject-repros`, whose triage documents already existed
  and each say in their own opening lines that they ARE that inventory.
- **Item 4 is FALSE and is withdrawn.** It says CheriBSD "is absent from this host. Not a gap that
  work here can close." A stock CheriBSD purecap vehicle has been built here since 2026-10-05, and
  **53 of the 54 cases across the three programs now carry a `cheribsd-revocation` arm with status
  `measured`**, over ten committed bundles. The claim was written before those runs and never
  revisited. **PoisonCap remains genuinely absent** — that half stands, and the distinction matters:
  *not yet done* is not *cannot be done*.

Open, with the blocker named rather than implied:

- **`ffmpeg/plane-repros` case 0 is the ONE remaining unmeasured CheriBSD arm** of the 54. Its probe
  is now an external symbol so `supervise` can resolve it (it was header-`static`, which would have
  made every arm read UNRESOLVED). The real blocker is that this corpus links **real `libavutil`**,
  unlike its plain-heap siblings, so a purecap run needs libavutil cross-built with the CHERI SDK.
  Predicted reading, recorded before any run: **completes, not caught** — the crossing stays inside
  the frame's single `AVBuffer` (measured slack 1024 bytes), which is exactly why this is the
  nested-discriminating cell.
- **`plane-repros` must NOT be grown.** `docs/ref/ffmpeg-spatial-defect-triage.md` records the
  measurement that settles it: per-plane pool slack of 1024 against a logical 160 forced the
  retraction of this corpus's discriminating-arm claim, and the plane/row shape is "exhausted at one
  case". FFmpeg's nested-spatial cell standing at 1 is a finding about FFmpeg's allocation
  discipline, not a shortfall in the search. The same document names the two candidates that ARE
  open, both NOT nested and both for `subobject-repros`: `01701bdcd5` (its own "next candidate") and
  the re-filing of `30c6667dad`.
- **`check-ports.py` is BLOCKED on dev with one problem, and no gate runs it** — it is named in four
  READMEs and in `build-index.py`'s docstring, and in no script. Registering `capstone-virtual` (a
  target the virtual stack added) and the CheriBSD `arms` field took it 3 problems to 1; the
  remaining one was *unmasked* by that fix and is real: `ports/sqlite/cheribsd` declares version
  3.22.0 against a `pin_source` line that carries no version.
- **Committed files carry foreign home-directory paths** across 36 files, including one
  collaborator's account name several hundred times, and `precommit-scan.sh` has no `/home/<user>`
  pattern so the gate cannot see them. Whether an account handle counts under the no-names rule is
  the project lead's call; evidence is under `/tmp/capstone/`, deliberately not committed.

One instrument note for anyone re-running the corpus gates: **run them from a tree at dev's tip.**
The shared clone is ~170 commits behind and reports 14 corpora / 108 cases there against the real
23 / 218 — a reading from it looks plausible and is wrong.

2026-10-07 — **Review the growing node tables on `review/virtual-node-growth`.**

## Local virtual mallocng review

Use `virtual-musl-local` for the local allocator architecture, based on
`review/virtual-node-growth`. Merge node growth before the exact-bounds QEMU
PR #19, then the matching runtime. Keep QEMU and SDK/application ABI v5
in sync; ordinary malloc/free run in Capstone. Review the representation and
lifetime hooks against upstream policy, the VM remap transaction, and the
[qualification](../../runtime/virtual/results/local-mallocng/README.md).

Further work remains separate: metadata-cost tuning, full benchmark timing,
CPython inner and FFmpeg/tshark corpus requalification, and a hardware storage
contract for exact bounds. The extra-shadow profile is a QEMU prototype.

The original virtual stack is merged into `dev`. Merge
[QEMU #17](https://github.com/project-starch/capstone-qemu/pull/17) into
`virtual-capstone` and [build repairs #194](https://github.com/project-starch/llvm-capstone/pull/194)
into `dev`, then retarget the stacked runtime PR from
`review/virtual-build-repairs` to `dev`. If the repairs are squashed, rebase
only the runtime commits onto the landed `dev` first. Preserve
the QEMU pin's tested source when moving it to the landed processor commit.

For new work, branch from the runtime review head and initialize its pinned
QEMU. The [runtime README](../../runtime/virtual/README.md) gives build and
qualification commands; the [ISA chapter](../design/virtual-capstone/isa.md#lifetime-storage-and-encoding)
defines publication and complete-sweep reuse. The separate malloc block-record
ceiling, empty-page reclamation, multi-hart and virtual RTL remain future work.
Historical physical/silicon tasks below retain their recorded scope.

2026-10-05, night: **R-29 AND S-10b HAVE A LINT-CLEAN RTL FIX IN SIMULATION** (capstone-ariane `sup-call` 776d9d859; the
dated blocks in ISSUES R-29 and S-10b). A read whose granule has a conflicting store in flight -- a plain store for a 16-byte
read, a capability store for a plain high-word read -- now waits in the dcache read controller until the store has drained,
seeing the stores in BOTH buffers from flops: the write buffer's entries and the store buffer's queues exported as a shadow.
The store buffer's address checker is untouched (its widening failed synthesis on 2026-08-21, S-10b); nothing in the memory's
response path changes.
- 25 directed arms read correctly with the hold where one is due, the set-index-9, MISS-face and S-10b controls among
  them; the clean tree reads torn on every one that should. Sweep: 95-test directed sweep (testlist_capstone.yaml, seed 20260922, memory delay 12) on the fixed tree against the clean HEAD: 89 identical in exceptions, readings and retired count with zero cycle-count differences; 3 random-generator entries log-less on both sides as always; untagged-ldc-stc-fixup retires one instruction more and reads the same values -- its repair branch, which re-wrote the destination's high word after the stale read, is skipped now that the read is ordered (its own comment anticipated this); and linear-clear-audit / linear-clear-residue failed by their codes 2 and 3 for the reason above and pass with the updated expectation (one hold each, a plain high-word read behind the clear's capability entry in the write buffer), while the clean tree fails them. Audit: claim-auditor, two passes. The first (against the version that widened the store buffer's compare) REFUTED it on the S-10b record (c867dfcbb's LUTLP-1 loop in the same cone), found that the "R-51" mirror shape was S-10b's data route already registered, SUPPORTED the write-buffer hold's correctness/key/coverage/interactions, and called the UNOPTFLAT 40 -> 35 drop uninformative. The second (this shape) found no error by reading -- deadlock freedom, visibility timing (zero slack: the store appears in speculative_queue_q exactly in the load's earliest decision cycle; depends on NrStorePipeRegs = 0), the S-10b tag-route release, lint structure and the run results all SUPPORTED or plausible -- and named the deadlock rule's missing positive control, which r29-sb-younger and its mutant now provide. Residual it named: dom-switch reads re-check against lsu_ctrl in WAIT_PAGE_OFFSET (an inherited gap; whether the switcher writes and reads one granule within a switch is UNRESOLVED).
- Mechanism sharpened: at zero distance the high-word face was the STORE buffer (word-granular check), not the write
  buffer; the write-buffer phase begins 2-4 instructions later.
- **SYNTHESIZED (the synth lane, 4 h 57 m): all three pre-registered items PASS** -- no LUTLP of any kind, the one routed
  TIMING-23 loop is the standing loop-4 cluster with its cut moved one cell and nothing of the hold on the cycle (settled
  from the netlist), WNS -8.760 in band (+1.19 vs 715bdd1fe). Bitstream `ariane_xilinx.bit` sha256 3c91335acc7c1065…
  (11,443,722 B); the full numbers are in ISSUES R-29.
- **FLASHED on the lead's word (16:58) and FIXED ON SILICON** (board-supmon 3cdddc4727ec, dev f432fca10db1): controls unchanged
  (S-16 bare 18/18, R-43 a1..a10, C5u +0.680 %); b0-memcpy unguarded 94/96 -> 0/96, guarded 0/96; the R-29 folder's rung
  s06agg 66 -> 64; s10b-storebuf-primed bare 8 of 8 legs trap. ("The R29 rung 31/32 -> 0/32" written here before the run was
  b0-memcpy's per-face numbers mislabeled.) The hardware no longer needs the memcpy guard or W-12 for this hazard; dropping
  them is the runtime lane's call after its own measurement.
- **R-52 (was S-18): every floating-point consumer on silicon reads its operand STALE** -- the issue stage never marked an
  FP destination as clobbered (one negated predicate, a regression of the 2026-04-24 issue-stage commit, in every bitstream
  since; delegated domains are soft-float and unaffected). Reproduced in simulation with the board lane's sequences; the
  rv64ud ISA suite fails 11 of 12 on the current RTL and passes 12 of 12 with the one-line fix on branch `s18-fpr-clobber`
  (546807884; lint at baseline, sweep 95-test directed sweep (testlist_capstone.yaml, seed 20260922, memory delay 12) against the 776d9d859 tree's own sweep). NOT synthesized: **a candidate for the lead's next
  bitstream, with R-51's.** Until then no hard-float native process on silicon can be trusted; the registry entry has the
  details.
- **R-51** (the board lane's entry: after a RETURN yield the resumed domain runs with an untagged PC) is confirmed on silicon
  with this lane's `r51-return-pcc.S` and has a candidate fix on branch `r51-return-pcc` (ef8900e2d; QEMU's swap_pc semantics;
  lint at baseline, sweep identical) -- NOT synthesized: **whether RETURN restores the PC capability is the lead's decision.**
- The TLB-miss ordering case is closed by construction (a store that misses the TLB does not pop the LSU bypass, so the
  following load is not presented until the store has pushed); the remaining stated assumption is NrStorePipeRegs = 0.
- memcached on silicon never waited for it; with the fix measured, the guard and W-12 are a performance question, not a correctness one.

2026-10-05: **B0 PASSES ON SILICON.** The gp-captable delegated hello-world runs byte-exact on
`caplifive_supcall_715bdd1fe.bit`, N=2. It is launched by capstone-exec through the process-ABI module and stepped by
the FPGA monitor's supervised CALL, with no fabricated gp. Preflight was off, for a documented reason.
- Full record: `docs/plans/b0-silicon-delegated-runtime.md`.
- Two silicon-only defects were found and fixed, both QEMU-silent:
  - **C-76:** capstone-c caller-saved a `__rev` argument with `stc`, and silicon's STC moves it, so the monitor's
    step wedged the board. Fixed by `managed_reinit`.
  - **A delegate-runtime copy loss:** a freshly built `{offset, len}` pair lost its length in a granule copy. Fixed
    with 8-byte copies.
    - This is R-29's family. The first attribution, to R-29's recorded stale-refill mechanism, was retracted. The
      RTL lane then reproduced the real path in simulation: write-buffer residency, with the high half taken from
      a plain store's `.user = 0`.
    - musl's memcpy has the same exposure, so memcached on silicon needs a plain-data guard or the RTL fix.
- **M-13:** a trap inside the FPGA monitor always reads as cause 24 at `_cap_trap_entry`+4.
- A module built outside buildroot needs `-march=rv64g` on the FPGA; the board-run skill has the module rules.
- **Next:**
  - the monitor fix is on caplifive-sbi `capstone-bootstrap` 91c8aa2. The QEMU A/B shows no regression. The
    opensbi and buildroot pins are not bumped: QEMU behaviour is unchanged, and the FPGA line carries the fix;
  - the Part 3 silicon monitor line (supervised CALL under CSR events, B0.1) is still on lane branches;
  - **B0.8 done:** the runtime memcpy's R-29 guard holds on silicon (0/96; the unguarded control 94/96).
  - then B1 toward memcached on silicon (`docs/plans/memcached-on-silicon.md`): contexts and threads stepped by the
    monitor, the W-12 aggregate-copy guard in the application build, non-empty TLS and init arrays.

2026-10-04, evening: `caplifive_supcall_715bdd1fe.bit` is FLASHED and ACCEPTED.
- Bare S-16 images: 18 of 18, the 14 previously-hanging S-16 arms among them.
- The R-43 list: unchanged.
- C5u without the fence: 1,277 preemptions at +0.69 %.
- **S-16 is fixed on silicon.** The audited wording is in ISSUES S-16.
- Set `FPGA_BITSTREAM=caplifive_supcall_715bdd1fe.bit` for every board run.
- **Next:**
  - B0 on the board (branch `b0-silicon-runtime`: the gp-captable delegated hello-world, which passes in QEMU without
    fabricated gp);
  - S-17 stays watched through apertures 219..222.

## APP PORTS AND BUG CORPORA — 2026-10-04. **Nothing is owed on memcached, tshark or FFmpeg.** All 24
registered-but-unmeasured cells are measured and committed: FFmpeg pool corpus 40-47 on both app arms
(16/16, `ports/ffmpeg/app/results/20261004-qemu-pool-corpus-40-47/`), case 3's component pair as probe 39
(2/2), tshark `chunks` 2-9 (9/9, `ports/wireshark/app/results/20261004-qemu-chunks-2-9/`), memcached's
corpus 5x2 plus its negative control (`ports/memcached/allocators/results/20261004-qemu-corpus-defects/`).
**22 of 22 upstream cases across the three programs now carry a measured capability arm.**

**NEXT, and none of it is this lane's to start:**
1. **PostgreSQL's 8 `spatial`/`sublet` cells.** The paper prints Sublet catching 8 of 8 while all 8
   `bug-corpora/postgres/mmgr-repros/*/case.json` say that arm has **NOT been run**. It is the one place a
   printed number is denied by its own corpus, and the highest-value single measurement for the paper. Out
   of scope by the lead's decision (report only). The runner pattern now exists:
   `ports/common/application/run-fixtures-9p.py`.
2. **The paper's S3 numbers are the lead's call.** All five totals print as `\targetmeasured` while S3 is
   "Pending" in three places and `experiments/results/S3/` does not exist; "every comparator now runs the
   whole denominator" is false (22 of 57). Report, not committed:
   `/tmp/capstone/2026-10-04-paper-defect-evidence-audit.md`. The paper's own appendix authorises either
   running the arms or narrowing the claim.
3. **`INDEX.md`/`index.json` need ONE consolidated regeneration.** They are stale on `dev` from the board
   lane's fpga-repros and ISSUES additions (S-16/S-17; 28 -> 30 folders). This lane applied only its own
   lines, deliberately, to avoid republishing another lane's content. The lead asked the two lanes to
   squash and merge a single consolidated version.
4. **PoisonCap and CheriBSD for memcached and FFmpeg** stay unmeasured: that platform (SDK, purecap
   sysroot, image) is absent from this host. Not a gap that work here can close.

**New instrument worth knowing about:** `ports/common/application/run-fixtures-9p.py` runs app-SDK domain
images over 9p + the serial console when `capstone-vm` is unavailable (it needs ssh; no riscv64 dropbear
exists here). It issues the same guest command `capstone-vm` does, merges the launcher's fault record as
`capstone-vm`'s host side does, and judges with the committed oracles. Negative-tested.

2026-10-04: the FPGA monitor's resume hang (S-16) is root-caused and fixed on
capstone-ariane `sup-call` (192a5e624, registry R-49: the switcher's first SAVE
write pushed into a FULL store-buffer commit queue), together with R-50
(429c60b32, the load unit's flush exemption). Sweep: 95-entry sweep on the committed tree against the clean baseline, seed 20260922, memory delay 12: 92 identical in taken exceptions, CAPPRINT readings and retired-instruction counts, 0 differ; the 3 random-generator entries produce no log on either side, as in every previous sweep. FLASHED AND ACCEPTED: `sup-call` 715bdd1fe (caplifive_supcall_715bdd1fe.bit, sha256 a7add4c0...1311) --
18/18 bare images, R-43 a1..a10 unchanged, the C5u monitor without a fence at 1,278 preemptions where
36a641e0b hung after 212 (plan step-11 acceptance; R-49 FIXED ON SILICON). NEXT for the RTL lane: the
runtime lane's supervised_invoke work needs nothing more from the RTL; keep the fallback sup-call-b2 parked;
the audit's cheap checks (byte 220 at idle = 0x40 as the bitstream's fingerprint; a sticky mis-check flag for
the next bitstream) go into the next aperture batch, not a cycle of their own. (Before the reflash, on 36a641e0b, a `fence` before every CALL covered only that CALL's switch and a
fence before every RETURN as well covered the non-preemptive twins, the quantum escape staying exposed --
the board lane, bare; superseded by the flashed fix.) S-17 on silicon stays open; the apertures that read its state (219..222) are in the candidate (R-50).

Supervised CALL (2026-10-01): the RTL is on capstone-ariane `sup-call`
([plan, revision 1.2](../plans/supervised-call-silicon.md)); every simulation
gate is done (the after-audit's four findings fixed and re-measured, the
92-test neutrality run clean with `revocation`'s phantom traps as the only
difference). Synthesized and FLASHED 2026-10-02 (sealed, exit 0, every binding item in band -- the plan's step-9 result; `caplifive_supcall_36a641e0b.bit`, sha256 016219b2..., also at /tmp/capstone/_bitstreams; flashed 15:55 by the board lane on the lead's direct confirmation; the R-43 acceptance list passed unchanged and sup-quantum / sup-escape / call-retpc ran bare M-mode on the board bit-identical to simulation; next the first supervised-CALL program needs the runtime lane's FPGA monitor built to the plan's contract); the runtime lane builds the FPGA monitor's
`supervised_invoke` against the CSR-based events (csupctl.resume before a
resume CALL; status 3 means an unread event). Before silicon: why an unguarded
`mret` trapped instead of escaping (mutant-only). (The earlier line here, "a
SEALED capability does not survive an STC/LDC round trip in simulation", is
RETRACTED: the test had overwritten the seal with an integer; repaired, it
survives and the dead-node status is measured.) R-47 (CALL's parked return pc
after a jump or branch) is fixed on the same branch and must land with it;
until then, never place a branch right after a CALL. Two decisions taken on
the lead's delegation, audited, in the same bitstream: a CALL of another seal
while armed fails CLOSED (illegal-instruction trap at the CALL; the arm
persists across the trap and across every exception on a CALL head -- the
monitor's `supervised_invoke` CALLs the seal it armed, and its trap path
forgets with `cssupervise x0`, the silicon forget form; QEMU's `cssupervise
rd, d, x0` is a refusal here), and S-11 is fixed at 1024 bytes / 16-byte
alignment with R-32's off-by-one corrected and the sealed cursor set to the
region's start (a SEALED capability's base is its cursor) -- every known
sealer complies; the runtime lane should raise QEMU's `CAP_SEALED_SIZE_MIN`
from 528 to 1024. The cursor fix is its own commit (c0c546542, only the bracket test above it), so the
lead can veto it alone (a veto costs one revert and a 75-minute re-sweep).

Application ports (2026-09-29): the seven application recipes now require
the [shared delegated SDK](../../ports/common/application/README.md), with
v1 rejection qualified. Use the recorded compiler and memory settings when
rebuilding; do not reintroduce the removed private launchers or fake process
inputs. The protected PostgreSQL qualification needs more than 65,536
revocation nodes. Investigate the larger mruby GC stress's cause-30 exhaustion
before treating that workload as qualified; it also fails at 262,144 nodes.
Keep FFmpeg's C-70 flag workaround explicit until a reduced compiler fix is
qualified. These are separate from the successful application migration.

Delegated signals (2026-09-29): the `delegation-signals` branch passes its
[26-mode synchronous contract](../../runtime/tests/application/results/20260929-signal-contract.json)
plus 25 native tests, the application gate, binfmt and
Perl `t/base`. The asynchronous bell remains later, as do the deviations in
[the signal plan](../plans/delegation-signals.md). Continue broader process
group and region-grant qualification before calling the four-step runtime
stack accepted.

Delegated runtime (2026-09-29): the reviewed stack has
[checked syscall/process fixes](../../runtime/applications.md#review-verification-2026-09-29).
Guest binfmt_misc now passes direct shell execution and Perl's full `t/base`
(9 files, 493 assertions), with all 45 previous libc-test passes retained and
`sscanf_long` newly green (46 total). Complete the
outstanding process-group gates. Region-grant memory
is still step 5. The current rdtime sample does not discharge the planned
icount/per-step measurement. Do not label all four steps accepted yet.

Review and implementation priorities (2026-09-28): the completed runtime,
measurement tools, application integrations and evidence are now organized for
stacked PR review. The [current baseline](current-state.md#review-baseline-2026-09-28)
records branch dependencies. Historical requests below to withhold PRs are
superseded. Continue with Perl inner SV lifetimes and reuse qualification;
publication does not remove the remaining workload and memory-accounting gates.

Next application (2026-09-28): five complete four-arm inner-reuse comparisons
are available in the [updated first figure](../../experiments/study/results/reuse-five-applications-20260928/README.md).
CPython's VM-map panic and per-free sweep bottleneck are resolved for the
qualified JSON/GC workload; its six new CheriBSD processes pass with outer
revocation enabled. Preserve the explicit failure on unresolved VM probes,
matched metadata, transferred thresholds, final drains and archived identities.
Broader swap-pressure/concurrent-VM qualification remains a separate kernel task.

Perl now has a reproducible CheriBSD 5.36.3 interpreter and a native-matched
17-section smoke under both libc policy switches. Next implement the inner
SV-head/body lifetime adapters for the complete Capstone and PoisonCap
interpreters, plus the common observed-reuse histogram. The existing Capstone
Sublet switch covers outer malloc only. Select explicit pinned `Porting/bench.pl`
cases, qualify their outputs, then collect three processes in all four arms.
Do not add Perl to the reuse violin until those gates pass. Keep this study off
a PR. Equal-scope total-memory ledgers and recognized broader workloads remain
later milestones; the present CPython and PostgreSQL inputs are qualification
workloads. Earlier records below describe the sequence of resolved blockers.

PostgreSQL reuse follow-up (2026-09-28): [four-arm qualification](../../experiments/study/results/postgres-reuse-four-arm-20260928/README.md)
now validates 12/12 complete SQL processes and the
[first-plot extension](../../experiments/study/results/reuse-four-applications-20260928/README.md)
adds PostgreSQL to SQLite/mruby/FFmpeg. The blocking recursion was metadata
exhaustion; the PIE-offset parser diagnosis was wrong. Preserve the larger,
equal within-pair metadata capacities, transferred thresholds, verified outer
process revocation, disabled guest-service default and raw repeat variation.
CPython still needs the revoker VM-map locking repair plus deferred pymalloc
publication; Perl needs the complete CheriBSD inner-allocator port. Keep this
study off a PR (superseded 2026-09-28: the review stack is merged into dev). Standard pgbench workloads, equal-scope total memory ledgers,
and the fixed-follow-up reuse metric remain later milestones.

CPython reuse follow-up (2026-09-28): the
[three-arm raw qualification](../../experiments/study/results/cpython-reuse-three-arm-20260928/README.md)
now has three valid complete-interpreter processes per available arm, with
runner-checked inner histograms. Complete the protected CheriBSD PoisonCap
oracle and repeat it three times before drawing or admitting the fourth
violin. The present observer indexes successful new lifetimes; instrument
failed attempts and fixed-follow-up retirement cohorts before claiming the
full metric from `memory-metrics.md`. Capstone runs require an explicit
262,144-node capacity gate; the smaller-capacity attempt is excluded.

Published-threshold memory follow-up (2026-09-28): the
[fresh mruby/FFmpeg campaign](../../experiments/study/results/published-policy-20260928/README.md)
passes all 48 processes with outer defaults enabled. Preserve its exact
policy labels, destructor-sweep counters, corrected runtime identities and
retention countercosts. SQLite's six repeated outer-default processes also
pass and pair with six archived Capstone controls; the combined figure set
validates 60 processes (54 new). The [review index](../../experiments/study/results/published-policy-20260928/index.html)
now groups the same measurements by metric, with a secondary application
view and an explicit metric-coverage table. The [campaign presentation](../plans/application-memory-campaign.md#cross-application-presentation-and-paper-precedents)
now defines common cross-application endpoints using inspected paper precedents.
An exploratory reuse overview exists; shared byte ledgers, matched baseline
kinds and identical A–C schedules are still required before a common memory-cost
ranking. Do not relabel the old outer-disabled SQLite
data. The next wider
milestone remains complete qualified PoisonCap arms for PostgreSQL, CPython
and Perl, followed by normalized compiler settings and equal-scope full
memory accounting. Keep the study off a PR (superseded 2026-09-28: the review stack is merged into dev).

Three-application paper figures (2026-09-28): the [checked figure set](../../experiments/study/results/application-memory-paper-20260928/README.md)
contains common four-arm reuse/difference panels, SQLite/mruby/FFmpeg memory
companions and all eight workloads in a supplement. All 96 raw reuse records
reproduce exactly; 12 separate SQLite memory processes pass their raw oracle
and ledger checks. No new guest measurements were run. The next memory
milestone is equal-scope allocator/metadata accounting and original-layout
controls for mruby/FFmpeg, followed by consistent optimized-build campaigns.
Keep the measured reuse advantage separate from total-memory claims. The
instruction to keep the study off a PR is superseded 2026-09-28: the review stack is merged into dev; the measurement gaps it named
remain.

FFmpeg FATE follow-up (2026-09-28): the [archived four-arm qualification](../../experiments/study/results/ffmpeg-fate-four-arm-20260928/README.md)
passes 24/24 complete decoder processes on two adapted FATE inputs, each with
three exact-oracle repetitions. The original published CheriBSD kernel's
`Poison probe missing page` panic was caused by RISC-V
`pmap_extract_and_hold()` omitting valid L2 superpages. A narrow source patch
now holds the constituent page; both CheriBSD spatial and protected arms were
rerun on the same patched kernel. All four arms have identical 32-bin pool
lease-reuse histograms per input. The selective PoisonCap temporal adapter
peaks at 224.4 and 167.1 KiB of snapshot backing; this selected component is
not a total-memory ranking. Preserve the adapted-FATE label and the original
panic as excluded evidence. Next build a common requested/live/backing ledger
and an original-layout CheriBSD control before ranking full adaptation cost.

PostgreSQL next (2026-09-28): revise the protected PoisonCap 17.5
queue/sweep policy, then finish its `work.sql` run. Both fresh `-O1` Capstone
images now pass 3/3 complete SQL oracles each with explicit `sysv` DSM.
The pristine 16-byte-MAXALIGN cluster builder and [archived 6/6 campaign](../../experiments/study/results/postgres-pristine-20260928/README.md)
are verified by matching tree hashes and SQL oracles. Charge the separate
Sublet context region and platform node storage before comparing memory. The
PoisonCap chunk queue has passed `SELECT 1`
but a full-workload diagnostic was stopped during its first INSERT after
53 minutes of guest CPU and 7,154 sweeps, with no completed SQL oracle.
It is excluded from paper data. The existing memory contexts reissue slots
promptly. Preserve queued bytes and sweep counts when tuning its
threshold; do not quote the old per-free-sweep adapter as a PoisonCap memory
lower bound. Before a paper plot, use a common phase/work-size schedule and
account for live payload, reusable and withheld backing, metadata, and
Capstone node storage in each pair. `work.sql` is a qualification workload,
not pgbench. Keep this study off a PR until the broader experiment set is
ready (superseded 2026-09-28: the review stack is merged into dev; the experiment set is still incomplete). See the [complete-backend port](../../ports/postgres/app/README.md).

A fresh-guest policy diagnostic on 2026-09-28 tested bounded chunk and whole-block
quarantines at the same 17.5 `work.sql` input. The mode-0 control from the
same binary passed all 22 native-matched rows in about 9 seconds. The protected
mode did not complete: a 256-entry chunk queue recorded 949 sweeps after about
six minutes before an oracle result. Raising the chunk limit to 4,096 entries
and queuing whole blocks reached the first aggregate SELECT, then exhausted
the fixed chunk metadata table. **Correction:** the earlier attribution to
`flatten_grouping_sets` omitted the PIE load bias. The stack instead recursed
through allocator failure and error-message allocation. The new adapter
reports explicit capacity exhaustion and provisions both metadata tables
independently of the published quarantine. Scratch diagnostics remain
excluded; the qualified replacement is linked above.


CPython next (2026-09-28): the [archived complete-interpreter qualification](../../experiments/study/results/cpython-objects-qualification-20260928/README.md)
now passes Capstone spatial/Sublet 3/3 each and CheriBSD PoisonCap-adapter
spatial 3/3 at `-O1`, all with the same JSON/GC oracle. The PoisonCap
component is integrated into the complete interpreter, but protected mode 1
triggers the kernel's `share->excl` VM-map lock panic both after a
control and when run first in a fresh guest; it still panics on the patched
FFmpeg kernel. A temporary trap-PC diagnostic resolved the faulting instruction
to RISC-V `fupoison` in the PoisonCap test predicate: a target user page faults
while the revoker already holds the user VM-map read lock. Skipping such faults
in a **diagnostic-only** kernel avoided the panic but timed out before the
first workload marker after 120 seconds; that bypass may miss poison in paged
out data and is not a valid benchmark arm. The diagnostic kernel changes were
removed. A follow-up direct-map probe build showed why a general skip is
unsafe: the first missing page had no backing object, but a later CPython
probe targeted a backed page absent from the page table (`0x406b9b60`). The
fail-closed build stopped there with `Poison probe backed page missing`; it
was excluded and its source edits were removed. Further scratch diagnostics
showed that this particular backed page was a capability-free vnode page;
another missed page was swap-backed but had no tag at its index. An isolated,
fail-closed trap build that skipped only these provably empty probe classes
reached CPython's `startup` phase without a panic, but timed out before
`baseline` at 120 seconds. This is still not a protected application oracle.
The general case of a nonresident, tag-bearing target needs a resumable VM
probe rather than a silent skip. A separate, unqualified CPython external
free-list/quarantine attempt passed the spatial JSON/GC control, but protected
mode still timed out before `baseline`: in a 40-second diagnostic it had over
100 sweeps, including 84 flushes on normal block issue, zero on `realloc`,
and eight on region transitions. The fixed 512-block/1 MiB quarantine did
not prevent prompt reissue from cached pools. The diagnostic and allocator
edits were removed from the branch; scratch evidence remains under
`/tmp/capstone/poisoncap-nofault-probe/` and
`/tmp/capstone/cpython-external-free-list/`. Next design a bounded pool-level
quarantine with explicit retained-pool accounting and complete the VM probe
before accepting a protected process. Then add inner-pymalloc
issue/release, backing and metadata ledgers; the present outer counters cannot
support a four-arm CPython memory figure. The [CheriBSD port](../../ports/cpython/app/cheribsd/README.md)
records the build and runtime selection.

Nested application memory study (2026-09-27): the [normalized SQLite campaign](../../experiments/study/results/sqlite-normalized-memory-20260927/README.md)
now passes the build gate with an original-layout CheriBSD denominator and
all 12 repeated-work attempts. Sublet has a reproducible address-reuse advantage
(1.00× baseline versus PoisonCap's 3.99×), but higher selected peak allocator
footprint once tables are charged (4.68× versus 4.27×). Keep both results.
Do not substitute the old capacity-selected pilot for this matched-pool result.

Next reduce/account for Sublet's per-atom table cost, add requested live bytes
before fragmentation claims, and measure platform/node storage before total
memory or working-set claims. Move SQLite from the legacy host to the shared
supervised runner: the legacy campaign needed 4,194,304 provisioned nodes and
fresh guests between processes. Work units already preserve allocator state
inside one process. Apply the same unit/oracle/ledger contract to other ports;
no cross-application generalization is established by SQLite alone. Use the
[metric specification](../../experiments/study/memory-metrics.md) and preserve
all burst failures and the metadata counterexample. Keep this study off a PR
until the broader experiment set is ready (superseded 2026-09-28: the review stack is merged into dev).

The [measured four-arm SQLite reuse CDF](../../experiments/study/results/sqlite-reuse-gaps-20260927/README.md)
fills the release-gap slot for memsys5, and the [FFmpeg whole-decoder
follow-up](../../experiments/study/results/ffmpeg-reuse-gaps-20260927/README.md)
fills it for actual pool leases. FFmpeg's four arms have equal reuse bins
over 36 passing full-application processes; its selective PoisonCap adapter
still performs explicit per-granule poison/clear and copy operations. The
[mruby AO GC-slot campaign](../../experiments/study/results/mruby-gc-memory-20260927/README.md)
now admits a third complete boundary with 24/24 native-PPM-matched processes
at two AO sizes. Next extend the recognized AO work-size range toward the
upstream default, qualify a broader FFmpeg input
set, and bring another full application allocator into the paired comparison.
Do not substitute
outer-malloc address data for inner-pool reuse. Before a total-memory claim,
measure Capstone node and PoisonCap revocation storage in comparable units;
the current SQLite CDF runs provision 4,194,304 emulator nodes and omit those
bytes. Qualify an optimized release build before a final paper result.

Use the [application figure contract](../plans/application-memory-figures.md)
for the next measurements: complete applications, paired original controls,
release-to-reissue gaps, full metadata accounting and a qualified common burst.
The linked SQLite layout preview reformats existing complete repeated-work
data only; it does not qualify the failed burst or fill missing metrics.

The [memory follow-up](../../experiments/study/results/memory-followup-20260927/README.md)
adds full-oracle budget attempts and a size-2 scaling check. Resolve the
published-kernel `Poison probe missing page` panic in protected PoisonCap
SQLite at 7.5 MiB (and size 2 at 16 MiB), plus the Capstone Sublet
`sqlite3GenerateConstraintChecks` bounds fault at size 2, before estimating
protected minimum capacity or scaling. Repeat fixed budgets and count complete
platform memory once both arms can finish. The [matched SQLite 3.22.0
pilot](../../experiments/study/results/sqlite-322-memory-20260927/README.md)
now covers all 32 phases, native-matched SQL results, memsys5-only Sublet,
effective lookaside/heap settings, and both published and corrected PoisonCap
quarantine paths. Next diagnose the published-kernel `Poison probe missing page`
panic across attempted budgets and move the legacy SQLite domain host to the
shared persistent runner. Extend that
runner rather than adding VM scripts. The SQLite PoisonCap planner binding
still preserves unqualified cells until it moves to the shared runner; the
qualified mruby binding cannot substitute ordinary CheriBSD binaries.

The [whole FFmpeg decoder pool pilot](../../experiments/study/results/ffmpeg-pool-memory-20260927/README.md)
adds a second real four-arm nested application: 1/4/16 independent 30-frame
streams. The [selective adapter follow-up](../../experiments/study/results/memory-followup-20260927/README.md)
shows the original full-copy snapshot overhead was adapter-specific. The
[lease-gap campaign](../../experiments/study/results/ffmpeg-reuse-gaps-20260927/README.md)
now repeats every PoisonCap cell three times. Next widen the input corpus
toward FFmpeg FATE and report complete platform metadata.
The plot keeps Capstone outer-heap and PoisonCap jemalloc ledgers separate.

mruby now has a full-interpreter GC-slot Sublet/PoisonCap comparison on upstream
AO at widths 8 and 16, with exact PPM output and per-slot gap counters. Its Sublet
metadata and post-render page retention remain a counterexample to a blanket
memory-footprint win. Measure multiple AO sizes and selected physical backing,
then admit lists or another upstream benchmark under the same four-arm
contract. The generic `study.py` binding now emits all mruby four-arm cells
from pinned artifacts; the next application's adapter still needs registration
after its shared runner can validate the inner policy.
Retain the CheriBSD on/off reference, original failures and no-timing limits.

Application memory measurements (2026-09-27): use the [checked reuse and retention metrics](../../experiments/applications/memory-behavior.md)
and [72 passing application attempts](../../experiments/applications/results/20260927-reuse/README.md).
Extend the same input/oracle and default-CheriBSD method to Perl, CPython and
SQLite before generalizing beyond the two measured applications. Keep the
large-retained-graph counterexample and all reservations visible. For longer
runs, provision and account for the address-history observer before running:
the rejected 128-batch mruby attempt exhausted its fixed table. Replace the
QEMU software sweep with the intended reclaimer before metadata-capacity or
hardware-cost claims; preserve the phase-equality controls across that change.

Application workload discovery (2026-09-27): the
[`application-memory-experiments` lane](../../experiments/applications/README.md)
has checked workloads across six applications and prepared tshark inputs.
The [in-process node-reuse follow-up](../../runtime/tests/application/results/20260927-node-reuse/README.md)
now passes all 27 original Capstone repeats at 65,536 nodes. Extend the checked
workload sizes and remaining application ports; record live-node demand separately
from cumulative allocations. The QEMU collector performs a software tag sweep.
PostgreSQL's larger workload needs FileFallocate
support; tshark's full build is #128's, under `/tmp/capstone/delegation-ports/tshark-source`,
relinked and green on the current stack (`ports/wireshark/app/results/2026-09-30-qemu-tshark-current-stack`). Matching default
CheriBSD runs now cover FFmpeg and mruby: use the [shared comparison
contract](../../experiments/applications/comparison.md) for requested bytes,
address reuse and allocator retention. Extend this default-CheriBSD reference method to
the remaining ports alongside the separate nested comparison. Account for complete reservations before claiming a total
memory advantage. Keep the original six mruby failures and larger-node control
visible as historical data, alongside the separately identified fixed-QEMU runs.
QEMU timings do not establish performance; inspect
the paper branch `eval/application-memory` for the exploratory plots and failures.

Application execution (2026-09-26): the persistent **one-hart QEMU** lifecycle,
trusted preemption/fault return, owned resource reclamation, installed Buildroot
guest and common application SDK are implemented and verified. The combined
gate passes resource exhaustion/recovery and 1,008 subsequent mixed starts in
one boot with stable retained resources. Use the shared launcher and ordinary
upstream test runners for new application work; do not add per-port VM runners.

The next port work is to migrate each real application's upstream build to the
CMake target or SDK, keeping only necessary source/configuration adaptations.
Perl is migrated and mruby links with the same SDK. The [current complete Perl
`t/base` run](../../ports/perl/musl/results/2026-09-26/base-tests-rebased-qemu.txt)
has eight passing files; only `term.t` test 2 fails because its backtick command
needs target subprocess creation, and clone syscall 220 is unserved. The prior
`lex.t` fault was a Perl pointer-to-UV regex-save round trip, now patched;
`rs.t` passes with effective `CAPSTONE_GP_NONLIN=1` or the already-merged C-46
compiler fix. Next decide how application subprocesses map to protected domains,
then extend upstream coverage. Keep older hardware/ABI gates until their
callers migrate. The legacy snapshot's null_blk and borrowed-region INIT failures
reproduce before these changes and remain separate work. An FPGA equivalent
requires architectural support for protected continuations and reclamation;
this QEMU result does not establish it. [Commands](../../runtime/applications.md),
[acceptance](../../runtime/tests/application/results/20260926-qemu-rebased.json),
[architecture and scope](../plans/domain-process-runtime.md).

Port integration (2026-09-19): follow the [cross-repository plan](../plans/port-stack-integration.md)
for the allocator, corpus and cooperative fault-recovery PRs. Shared runtime
and state-file merge conflicts remain integration work; the board milestones
below are a separate track. Build entry points are in the [port catalog](../../ports/README.md).
PoisonCap pymalloc: the [first pilot](../../ports/cpython/pymalloc/results/20260919-poisoncap/README.md)
passes the complete 33-process suite and both modes of a 115-event native
recording. Next validate a larger complete recording and account for matched
payload, authority records, snapshot retention and kernel scan work. Evaluate
quarantine/batching as a separate policy; the initial synchronous sweeps are
not an architectural lower bound. Preserve the unwritten-reuse regression and
explicit automatic-libc-off configuration. The full interpreter, existing
defect corpus and hostile nested-manager protection remain outside the
validated scope. See the [integration plan](../plans/poisoncap-pymalloc.md).

PoisonCap FFmpeg: the [first pilot](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-poisoncap-pilot/README.md)
passes a full 29-process suite with the guest's automatic libc-revocation
default disabled; explicit adapter revocation remains active. Three replays
match exactly. Extend this documented configuration with matched memory
accounting, and separately isolate the kernel VM-locking panic seen with the
guest default preserved. Retain failed attempts. Measure selective state
preservation and quarantine as explicit
adapter policies; the initial all-payload snapshots are not an architectural
minimum. See the [integration plan](../plans/poisoncap-ffmpeg.md).

CheriBSD allocator ports: the four components now have a
[shared build/link/run workflow](../../ports/common/host/cheribsd/README.md).
Use these libraries and explicit protection scopes for matching replay
campaigns. Next collect equivalent per-component memory accounting and add
inner-lifetime adapters only with paired stale/live controls. Compilation and
successful examples do not establish Sublet-equivalent temporal protection.

Protected pool replay: the [A1 matrix](../../ports/ffmpeg/buffer-pool/results/measurements/20260919-alias-scatter/README.md)
verifies Capstone ancestor revocation across five alias locations, with
same-address reuse and unaffected-sibling controls. Next define matched
parent/child semantics for a cross-system hierarchy experiment; A1's no-revoke
arm is not a CHERI or PoisonCap implementation. Vary child fan-out and
account for required manager bookkeeping before making a scaling claim.
Keep trusted adaptation
separate from protection of untrusted nested managers. Before making a
reclamation-space claim, use a Capstone model implementing the intended node
reclaimer; the pinned QEMU accumulates nodes in these traces. Account for
physical metadata and adapter snapshot storage separately from payload and
ID counts. Keep spatial and protected arms separate.

Replay measurements: the first FFmpeg paired matrix is complete. Next extend
the [measurement plan](../plans/replay-memory-measurements.md) with full
allocator/driver/node accounting and separate payload/metadata budget sweeps.
The initial carving watermarks must not be used as total protection overhead.
Validated repeatable epochs and other allocator adapters remain follow-ups.

Trace tooling: use the [shared readers and format-adapter contract](../../ports/common/host/port_trace/README.md)
for new allocator traces. The development branch includes the pending port PRs;
the [upstream integration plan](../plans/port-stack-integration.md) remains relevant
when those independent PRs land. Historical milestone entries follow.

---

Everything before 2026-10-01 is in
[`history/13-10-2026_01-00-00_next-step-journal-through-2026-09-29.md`](../history/13-10-2026_01-00-00_next-step-journal-through-2026-09-29.md).
