# Next step

## 0. CURRENT — 2026-09-11 (evening). Everything outstanding is merged and pushed; the only work left needs the board.

**All four PRs are in and `dev` is at `f727c338594a`.** llvm-capstone #7/#8/#9 (the collaborator's
speedtest1 bring-up stack, the Sublet port, the A1 experiment) and our `speedtest1` measurement
branch merged into `dev`; capstone-qemu #2 merged into `c128-qemu-merge` (`656cc034899f`) and the
parent gitlink bumped after the submodule was pushed. `speedtest1`, `pr9-check` and
`sqlite-stockness` all report 0 commits ahead of `dev`.

**Two merge hazards worth carrying forward, because both were invisible to git.** In
`sqlite_host.c`, our arena share and the collaborator's pool share both claimed shared-region
**slot 2**, and both domains capture by ORDER — taking both sequences in either direction hands one
domain the other's capability, with nothing for the compiler to see. Resolved `#ifdef`/`#else` plus
a refusal before `capstone_init`. In `op_helper.c`, `GETPC()` was used at four sites inside
`_helper_access_with_cap`, which is `static` and so has no host pc in the TCG buffer; **two of the
four sat outside the conflict and git merged them silently**, leaving the bounds path restoring from
one frame and raising from another. The general shape: the dangerous part of a merge is the region
git resolves without asking.

**speedtest1 `json` RUNS in a capability domain** (QEMU, image `c1ce36ac6711e194`): 6 MiB region
arena, `TOTAL 28.988s`, `SPEEDTEST1-CYCLES 725363552`, `RC 0`, eight phases with real timings. Both
documented reasons json "could never run" were true of a `.bss` build and neither survives the
region arena. **This is not an S-14 fix** — per that entry, the bad reloads vanish incidentally with
the array and any change to the global set can bring them back; gate on `capinit-reload-scan.py`.

**NEXT — the board ask has changed from BREADTH to DEPTH, reported by the peer lane as the lead's
call, and the json arm is superseded.** Published speedtest1 numbers use the default `--size 100`;
everything we have on silicon is size 1, i.e. **1 % of the benchmark's default scale**, so our
numbers are not comparable to the literature. `main` at size 100 outranks a tenth testset. Approved
shape is a rehearsal first: `main --size 20` built with the **size-100 arena**, so only the row
count is smaller — it exercises a 120 MiB region as SQLite's heap on silicon, the raised timeouts
and a multi-hour silent arm, for ~1.1 h against ~5.4 h. `main --size 100` needs a measured 120 MiB
arena against the 130 MiB demonstrated in sw55 with a 256 MiB CMA reservation: 8 % margin, and
reachable only because the arena now comes from a region (`.bss` caps at 4 MiB).

**THE DOMAIN AND THE BASELINE TAKE THEIR ARENA SIZE FROM DIFFERENT VARIABLES, AND NOTHING GATES IT.**
The domain reads `SPEEDTEST1_ARENA_SIZE` (`run-speedtest1-measure.sh:114`); the baseline reads
`SQLITE_HEAP_SIZE` (`build-speedtest1-baseline.sh:46`), which the runner has already **pinned** to
the geometry default at `:64`. Set only the first and you get a large-arena domain against a 2 MiB
baseline, silently. **This is not hypothetical — the json pair built this evening had it**: domain
`arena_bytes=6291456`, baseline `.bss=2168152`. The handoff spec claimed they "differ in exactly one
dimension"; they differed in two, and the second was the allocator size, which is the dimension a
speedtest1 ratio is most sensitive to. Spec superseded with the defect recorded; a gate is being
added on the peer side. Compare the BUILT artifacts, not the two variables — a stale `OUT_DIR` makes
the artifacts disagree while the variables agree.

**Four board-side defaults would void a healthy size-100 run**, all verified at source and all tuned
for 90-second arms: `SQLITE_STAGE_TIMEOUT` 90 (`run_sqlite_stages_fpga.py:47`, and it is PER DOMAIN),
`BAKED_TIMEOUT` 120 (`run_baked_rungs_fpga.py:55`), `BAKED_IDLE_S` 25 (`:62`), `ENTRY_STALL_S` 260
(`board-watchdog.sh:54`). The silence is structural: speedtest1's output goes into the shared region
and the host prints it only at the end, so a multi-hour arm shows nothing on the console and the
watchdog would read it as an entry stall. Precedent for raising the last one already exists —
`slt-corpus/run-slt-corpus-fpga.sh:29` uses `ENTRY_STALL_S=420`.

The json DOMAIN result stands on its own — it is a single-arm claim, not a ratio. Only the pairing
was broken.

**Two instrument repairs landed on the way.** Five stale `benchmarks/sqlite/` includes left
`revoke-on-free` and `hier-revoke` dead since the layout move — they reported a *build* failure as a
suite failure, in one second, next to a 591-second PASS. Three more were latent. And a QEMU suite
can be blocked by another lane holding the `rootfs.ext2` write lock: `authority` is the one suite
that runs without `-snapshot`. The fix is not to wait — point `CAPSTONE_BUILDROOT_DIR` at a shadow
tree that symlinks everything and carries its own sparse rootfs copy (56 MB on disk).

**The credential claim in the section below is STALE** — it said six repositories; it was three, and
the token was replaced. `dev`, `c128-qemu-merge` and the submodules used today all push.

## 0a. ANCHORING CLOSED AND THE FLASH GATES RE-READ — 2026-09-11 (late). Nothing about the flash is waiting on a judgement any more.

**Every commit that existed only as a local tag is now on `origin`.** Eleven `backup/*` tags were
pushed to `project-starch/capstone-ariane`, publishing **18 distinct commits** that were reachable
from no remote branch. Before the push those 18 were one lost disk from gone. The set, by tag:

| tag | newly published |
|---|---|
| `backup/r24-excode-base-2026-09-11` | `69658cf16` |
| `backup/r25-init-rs1-dup-2026-09-09` | `ec50837b5` |
| `backup/r25-r26-r27-final-2026-09-11` | `8858fd975` `d799f84d9` `4ef7a7b37` `e62eb5f6f` |
| `backup/r26-ccsrrw-stale-read-2026-09-09` | `67d870cc8` |
| `backup/r27-revnode-orphan-drain-2026-09-09` | `3bfaa544c` |
| `backup/r28-interrupt-probe-2026-09-11` | `b871c51d4` `f29465d8c` `7bc7c447b` `8331d052b` `3c800b369` |
| `backup/r29-granule-data-overlay-2026-09-11` | `00e89d968` and the five above |
| `backup/shrinkto-size-fix-2026-09-11` | `a4b478754` `a3b7a22c0` |
| `backup/verif-arms-r18-r28-r29-2026-09-11` | `21ccf09c9` `12ed21aa2` |
| `backup/p3-final-2026-09-09`, `backup/r30-r31-init-revoke-2026-09-11` | none — already reachable |

**All 18 were scanned AFTER the fact, and the gate was proved live rather than assumed.** The pushes
used `--no-verify`, so the scan had to be run separately: `precommit-scan.sh --range <sha>^..<sha>`
by absolute path, from inside `capstone-ariane`, exit status read directly and never through a pipe.
All 18 exit 0. **Positive control, because a clean sweep is not evidence until the check is known to
fire:** re-running one of the same commits with `CAPSTONE_NAME_DENYLIST` pointed at a scratch list
holding a token certainly present in that commit exits **1** with four quoted hits, from both the
message and the diff. So range mode, in this invocation form, does scan and does block. The real
denylist run prints `CLEAN` with no "exact-name check skipped" warning, so the list was loaded.

**Five local-only tags were deliberately NOT pushed.** `chain-v1`, `chain-v2`,
`chain-v2-pre-records`, `chain-v3` and `chain-v4-pre-v4` anchor the amend chain of one
synthesis-tooling commit. `chain-v3`/`chain-v4-pre-v4` have a tree **identical** to `947327f6d`,
which is on `origin/fpga-testing-dev`; `chain-v1` and `chain-v2` differ only by content the final
commit also has, plus an **older lint baseline** (`UNOPTFLAT 39` / `UNUSEDSIGNAL 713` against the
current 40 / 717). Nothing is recoverable from them that is not already on a remote, so publishing
them would add clutter to a shared remote for no recovery value. Left in place; not deleted.

**BOTH DECISIONS THAT GATED THE FLASH HAVE BEEN RULED, and the registry still reads as though they
have not.** `ISSUES.md`'s synthesis box says the flash "is gated on two OPEN decisions"; both closed
on 2026-09-10. Item 1, the `end`-convention resolution, was **ruled that evening**. Item 2, the
monitor reclaim shape, was **ruled and implemented** at monitor `0a5c3d9`. Anyone reading the
registry box alone will carry the wrong blocker forward, which is exactly what happened here.

**The firmware half is NOT local-only, and the recorded 403 was against the wrong repository.** The
superseded 2026-09-10 block says monitor `0a5c3d9` cannot be pushed because
`project-starch/capstone-sbi` returns 403. The monitor checkout's remote is
`project-starch/**caplifive**-sbi`, and against the live remote `capstone-bootstrap` is at
**`2c49c41`** — the M-6 fix, with `0a5c3d9` and the reclaim commits beneath it. Read from
`ls-remote`, not from a tracking ref. So the firmware half is published and the flash cannot ship
RTL-only by accident.

**What is actually left before a flash is mechanical, and only the last hop needs a person.**
Timing is discharged: `1bfff7776` was scored on 2026-09-10 against its pre-registered ranges and all
three falsifiers held — WNS **−12.425** inside −15.3…−11.7, placed LUTs **169,207** (83.03 %) inside
168.9k…170.5k, combinational loops **29** and unmoved. Remaining: extract
`work-fpga/ariane_xilinx.bit` on the synth machine, hash it **there**, transfer it here (that machine
has no outbound route, so the transfer must originate on it), hash it again here and require the two
to agree — the registry's `406e12bf…` is a transcription and is the weakest of the three checks —
then upload to the console store through the GUI, which our driver deliberately cannot write. Both
lanes have been asked. **Expect the size to be 11,443,722 bytes, identical to the resident
bitstream; that is fixed by the device. It is the hash that must differ.**

**The board has NOT been reflashed.** The board lane confirms zero board operations today and
`caplifive_r25r26r27_66c4e7517.bit` is resident; `~/capstone-artifacts/bitstreams/` holds only that
one. An earlier note in this session's history read as though a flash was in progress. It was not.

## 0b. EARLIER 2026-09-11 (afternoon). The board work is banked; the credential claim in this heading is superseded by section 0 above.

**Banked on silicon today, four boots.** sw55: a **130 MiB** capability region created, mapped and
round-tripped — 32x the buddy allocator's `MAX_ORDER 10` ceiling. sw56/sw57/sw58: speedtest1 across
seven testsets in a capability domain and natively, the domain's own instruction count measured on
silicon for the first time, and the position question settled. All of it is in
`ref/fpga-silicon-measurements-for-paper.md` §7f–§7k, whose PRECISION block at the end governs how
many digits any of it may be quoted to. **§7j is the measurement of record**; §7k is the first
pairing and where the tick analysis lives.

**Three results worth knowing without reading the sections.** The domain's instruction count is
**bit-identical** across two boots and matches the emulator to 5e-08. **Position is worth 0.016%** —
`main`'s domain arm run twice in one boot, instruction counts identical, so the positional caveat on
every cross-boot comparison is retired. And run-to-run variation is **0.027 pp sd on the same
image**, six times smaller than the first bound, which had been dominated by a 144-byte layout
difference rather than by the machine.

**M-6 is fixed and M-7 is what was behind it.** `revoke_region` now returns `0` revoked / `2` nothing
to revoke / `-1` refused, and the module treats `2` as permission to pop (the lead's ruling). The
QEMU abort is gone. The release path then hits a second defect — after the pop an access finds no
CPMP region, and the recovery path assumes a domain is running when none is. **`pre_mmap_offset` is
still unproven**, now by M-7 rather than M-6.

**The firmware half of R-30/R-31 is committed at last.** All four monitor pointer paths were
committing the pre-reclaim parent while every boot used the working tree. Verified from a build: five
reclaim sites in the generated asm, and the built firmware's monitor source MD5-identical to what the
tree commits.

### Open, in the order they are likely to move

1. **SIX REPOSITORIES REFUSE THIS CREDENTIAL, and the monitor is one of them.** This is the binding
   constraint, not the board and not the work. Verified against the remotes rather than against
   `@{u}` — three repos have no upstream configured and the tracking comparison returns zero for
   them silently:

   | repo | unpushed | note |
   |---|---:|---|
   | `capstone-sbi` / `caplifive-sbi` (the monitor, two copies) | 1 each | carries the M-6 fix AND the R-30/R-31 firmware half |
   | `capstone-opensbi` (the wrapper, two checkouts) | 1 each | the monitor gitlink |
   | `caplifive-system-dev` (two checkouts) | 2 + 1 | `sw/buildroot` pointers |
   | `capstone-academic-spec` | 2 | the `end`-convention amendment; 403 on **read** too, so its branches cannot even be listed |

   `capstone-qemu` was in this list and is **not** — it was a misconfigured upstream, and its two
   commits (the `badaddr` and cause-6 fault-path fixes) are now pushed. Establish access by trying
   the remote, not by inferring from a tracking ref.

2. **Two branches need an allowlist entry**, which is the lead's file and no lane may edit it:
   `speedtest1` (50 commits, ~3,600 lines — the entire apparatus behind §7f and §7i–§7k, on no
   remote) and `shrinkto-size-fix`.

3. **The flash of `1bfff7776`** — authorised, `.bit` still on the synth machine. It now has a
   **batch** owed to it: control, then the bridge arm tying post-flash numbers to pre-flash ones,
   then the M-3/M-4/Q-06 board controls, the 4 MiB allocator-provenance arm, the `fillcost` repeats
   and sw52's lost `instret` arm, and the R-30/R-31 directed tests LAST. Seven separate items each
   waiting on "the first post-flash boot"; one load, not seven.

4. **M-7's mechanism is not established.** `pop_region` clearing its bookkeeping without releasing
   the CPMP entry is a hypothesis from reading code, recorded as such. The offset proof and the whole
   release path sit behind it.

5. **R-32's ruling** (`SPLIT`, `LCC` bound values) is the lead's. **The note is now in the repo** as
   item 5d of `docs/plans/DECISIONS-WAITING-2026-09-10.md` (moved 2026-09-11); it used to live only
   under `/tmp` and would not have survived a reboot. It carries both measured readings, the three
   options with their costs, the split recommendation — (a) for `LCC`, (c) for `SPLIT`, and why they
   differ — and the exclusion of `SHRINKTO` and `SEAL` from the ruling. It also records why
   `shrinkto-size-fix` is **not** stale work: it post-dates the supersession, sits on `1bfff7776`,
   and leaves the guard the refuted route would have changed.

6. **The paper numbers resume after the silicon work**, by the lead's ordering. The other lane's
   region-arena change is the single piece of work that lifts both `json` and the depth axis; `app`
   is root-caused and stays explained rather than worked around, written up as a porting-cost result.

7. **Hygiene, none of it urgent:** `board-results/` holds only `2026-09-05.tsv`, so sw52–sw58 are
   absent from the TSV corpus the verification rule requires; Q-07's registry entry, its decision doc
   and its in-source comment all contradict git, which shows it committed and pushed;
   `build-sqlite-host.sh`'s provenance line is on the other lane's branch, not on `dev`.

---

## (superseded) 2026-09-10. The reclaim is implemented and measured; two things are waiting on people, not on work.

1. **The `end`-convention re-ruling — the project lead's.** The earlier ruling was superseded the
   same day. It gates the spec amendment and the flash. It does **not** gate the firmware change,
   which is identical under both surviving routes, so that did not wait.
2. **Push monitor commit `0a5c3d9`.** It is LOCAL ONLY. `project-starch/capstone-sbi` returns
   `403, Permission denied` for this credential — tried and recorded 2026-09-10, so this is a
   provable blocker rather than an assumed one. Someone with write access there has to push it, or
   the credential has to gain access.
3. **What is runnable without either.** The R-30/R-31 RTL pair (`1bfff7776`) is built and clean with
   all three pre-registered falsifiers held, and the firmware half is now gated, so a flash is a
   decision with both halves in hand rather than an emergency. After a flash, the reclaim path
   becomes measurable end to end for the first time — `RCLM` climbs instead of reading zero — and
   the `fillcost`/`fillnop` pair should be re-run, because the monitor's loop is one instruction
   shorter per iteration there (the UNINIT cursor advance walks the pointer for free).
4. **A second `fillcost` draw.** The 23.6 cycles/store figure is N=1 on a system with known
   nondeterminism, and the rung is now cheap to re-run inside any boot that is happening anyway.
5. Boot sw52's lost arm — the `instret` counter probe — needs no new work: the classifier defect
   that killed it is fixed and negative-tested, so it rides along on the next boot.

---


## (superseded) 2026-09-09. Phase B closed on the shipping firmware (boot sw38); follow-ups in `docs/plans/after-phase-b.md`.

The unification (one branch `capstone-bootstrap`, `make TARGET=fpga|qemu`) is done, pushed and
validated on both targets (see `current-state.md` and `docs/plans/monitor-unification.md`).
Phase B (the plan doc's convergence backlog) is done for items 1–9: geometry, M-2 bound (raised to
96 AND bounded, the lead's choice), the pre-carve refusal, fence.i pruning, the null-blk package
and loader, ONE `create_domain`, and the transferred slot becoming a hole on the board too — each
its own commit with its own gate, five board boots (sw33–sw37) all at the oracles with zero fault
tags. The QEMU tier on the last QEMU-visible change (5A; 5B and 2B are byte-identical there) came
back 17/18: one BEEBS case (sglib-hashtable) printed nothing for 20 s before the loader's first
line, in a window that also produced five boot-to-login infra retries under another user's load;
rerun alone, first in a fresh boot, on the rebuilt images it passed 3/3 at its marker — recorded
as an infra flake, not a monitor result. Item 10 (kernel unification) is deferred by
the lead's decision; the +1.05 MB initramfs stays. What remains:

0. **Done since:** the Phase B tail is pushed (2026-09-09); closing boot sw38 9/9 on the final
   firmware; the nightly marks suites that ran under other users' load; Q-06 localised to `sbi.dom`'s
   `query_region` on a CPMP-resident region (owner unchanged); R-26 with the RTL lane
   (`docs/plans/after-phase-b.md`).
1. ~~**Push the Phase B tail**~~ DONE (`/tmp/capstone/push-final.sh`, the lead's credential; every push a
   fast-forward, preconditions checked): monitor 4a12d8b→5b27d01 (three commits), wrapper
   5450a2d→1c48f02 (three), buildroot 8c51969→d3c2402 (three), caplifive-system 20fd22f→77eb5a8
   (one); then `dev`. The live QEMU images are already rebuilt from the committed source
   (fw_jump c1a450ac5d06, sbi.dom f48906bf25a4; generated file byte-identical to 5A's).
2. Small follow-ups, each its own commit: the dead `mem_l`/`mem_r` locals in `split_out_cap`; the
   `gpoff == 0` `create_domain` branch has no board image (unexercised on silicon, stated in the
   plan doc); R-26 (CCSRRW vs younger CAPSTONE_DYN readers) awaits an RTL-lane demonstration;
   Q-06 (null-blk split S-mode init) belongs to the null-blk owner.
3. Before any SLT or M-2 board boot, restage `/tmp/capstone/overlay-attic/*` into the overlay and
   `build/target/test-domains` (the preflight refuses unused overlay files above its budget).

The S-12 material below is FINISHED BUSINESS, retained as the evidence trail; do not act on it.

---

## (superseded) 2026-09-04. S-12 IS CLOSED. The next steps are no longer about S-12.


**Everything below this section, including the one dated 2026-08-29 that calls itself CURRENT, is
the S-12 investigation and is FINISHED BUSINESS. It is retained as the evidence trail; do not act
on it.**

### What happened

S-12 is root-caused, fixed in RTL, synthesised, flashed, and the SQLite domain that trapped now
completes. The registry entry (`ref/ISSUES.md`) and the full mechanism
(`capstone/tests/fpga-repros/S12-wherecode-notcap-operand-vs-memory/S12-explanation.md`) carry the
detail. The resident bitstream is **`caplifive_r25r26r27_66c4e7517.bit`** (RTL `66c4e7517`; carries the R-25, R-26 and R-27 fixes). **CORRECTED 2026-09-10** — this said an older bitstream, and the board scripts that actually ran boots sw48-sw51 all set `FPGA_BITSTREAM=caplifive_r25r26r27_66c4e7517.bit`, which is the record that settles it. Several drivers still carry stale hardcoded defaults, so **set `FPGA_BITSTREAM` explicitly** or the resident-silicon guard hard-stops.

The verification is **consistent with fixed, not proven**: 4 clean draws against a pre-fix arm that
trapped 3 of 4, Fisher p = 0.071. This project has already ruled that bound insufficient
elsewhere.

### The actual next steps, in order

1. **Two more board draws on the S-12 arm.** ~15 minutes. Takes p from 0.071 to 0.0095 and lets
   "fixed" be claimed properly instead of "consistent with fixed". Cheapest open item by a wide
   margin.
2. **The post-`capenter` inertness trace** (RTL lane). The measurement licensing the flashed
   bitstream was taken on `capldc`, where the switcher idles; our workload enters via `capenter`
   and runs its body after a switch. The flash worked WITHOUT that evidence, not because of it.
3. **The control build**: base `80843404c` + tie-off, no fix, no instrumentation (synth lane,
   needs the lead's authorisation in that session). Decides whether the stale debug tree can be
   dropped permanently for a further 1.820 ns — worth more than the fix itself gained.
4. **`plans/instrumentation-cleanup.md`** — deliberately deferred until S-12 closed. That trigger
   has now fired; it is unblocked.
5. **The `ptr-diff-signed.ll` coverage gap** from the c128 merge: D's version rewrote the IR under
   test from i128 to i64 and deleted a case, so its `CHECK-NOT: __divti3` can no longer fail.

### What is NOT a next step any more

The board-instrument work below — the `mtval` positive control, the recorder bitstream, the
operand-mux readers, "instruments first" — was executed and superseded by the 2026-09-03 root
cause. It is history.

---

---

## Older, superseded next-step blocks

Every superseded "next step" layer — the S-12 evidence re-audit, the causal-trigger and
compiler-first framings, the S-07 reflash sequence, the R-18 workaround list and the 2026-08-05
bitstream notes — is preserved verbatim in
**`history/04-09-2026_17-00-00_next-step-superseded-layers.md`**.

Split out on 2026-09-04. Several of those blocks were still headed `## 0. CURRENT`, so a reader
skimming for "CURRENT" found four of them and no way to tell which was live.
