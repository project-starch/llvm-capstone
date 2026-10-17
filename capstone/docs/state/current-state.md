# Current Capstone state

## Read this first (2026-10-13)

This file was 2,030 lines and its companion 1,560, most of it instructions and
results that had been superseded for weeks. Everything dated before 2026-10-01
is now in
[`history/13-10-2026_01-00-00_state-journal-through-2026-09-29.md`](../history/13-10-2026_01-00-00_state-journal-through-2026-09-29.md),
verbatim and with an index of all 58 sections. Nothing was rewritten or
dropped. The cut is by date alone, so a section that stayed is not thereby
current and a section that moved is not thereby wrong -- the dated entries
below say which is which, and for any individual defect `ref/ISSUES.md`
outranks this file.

**Where the work stands, as of 2026-10-12.** Four things are live at once:

- **Allocators.** The Sublet lifetime extension (`CDERIVE`, two-operand
  `CREVOKE`, node-scoped MANAGE) is qualified in QEMU and merged; QEMU #21 is
  `7dbfdd534eea` and runtime #216 pins it. The migration of all eight nested
  allocator ports onto managed lifetimes, and the removal of the physical
  Sublet heap, sit on `ports/nested-allocators-rebased` with fresh
  qualification -- 19/19 allocator, 43/43 runtime, 16/16 application checks --
  and are **not** merged into `dev`. One check is open: memcached fixture 24.
- **The virtual address space.** musl mallocng is the one system allocator; it
  executes in the application's virtual C context. The physical Sublet heap is
  retired, and virtual builds reject physical heap selectors.
- **Silicon (RTL lane).** The resident bitstream carries the S-16 fix; a
  gp-captable delegated application runs byte-exact on it. `fpga-testing-dev`
  was rebuilt as one linear history on 2026-10-10.
- **CheriBSD.** Our PoisonCap adapter is removed from every port except
  PostgreSQL, which goes with its rewrite. Stock CheriBSD remains the
  comparison vehicle.

What to do next is [`current-next-step.md`](current-next-step.md), not this
file. This one records what was established and when.

## 2026-10-17 — The virtual profile's RTL is complete for the two-bitstream plan; both builds wait on the lead's go

vp1 (`virtual-profile` 9ed9a2594: translation in C mode) and vp2 (`virtual-profile-2-d` bf3db3175: the flat per-context
lifetime table, CSRUNV with one saved context and FP state, CSRETIRE/CSNODE, the policy list, the collection strobe, the
reserve, Q-15's representability refusal, Q-18) are gated in simulation (lint at baseline by name, the 95-test sweep, the
152-test supervised list, every virtual-profile arm as pre-registered), pre-registered on dev before their handoffs, and
handed to the synth lane, which launched both in parallel at 03:23 UTC on 2026-10-18 on the lead's direct go. The
monitor's CSMINT emulation (the board lane's routine, included at a pinned hash) reads every pre-registered value on the
vp2 RTL through the shipped trap entry. Silicon still runs `caplifive_sublet_d93a8d66b.bit`. Details and the next step:
[`current-next-step.md`](current-next-step.md) and the plan's rows for items A, B, D and E.

## 2026-10-12 — Virtual allocator cleanup qualified on the migration branch

`ports/nested-allocators-rebased` uses managed lifetimes in outer musl mallocng
too. Its allocation policy is unchanged; linear heap loans and their final
carved-corpus consumer are removed. Virtual build metadata names `mallocng`,
physical heap selectors are rejected, and cache keys include the musl
representation sources. The [qualification](../../runtime/virtual/results/managed-mallocng/README.md)
records 19/19 allocator checks, 43/43 runtime checks, 16/16 normal port checks,
paired FFmpeg 13/13 and PostgreSQL 5/5 corpus results with safe controls.
memcached passes 81/81 outer-only checks and 82/83 with protected pools;
fixture 24 remains unresolved. Recomputed CPython archived sharded outcomes
agree for all 2,183 stock/protected tests. Full upstream campaigns and older
measurements retain their original scope. These cleanup commits are on the
migration branch, not claimed merged into `dev`.

## 2026-10-10 — CheriBSD PoisonCap removed (branch `nested/poisoncap-removal`)

Our CheriBSD PoisonCap adapter is gone from every port except PostgreSQL, which goes with
its rewrite: the FFmpeg, wmem, memcached, APR and Perl SV-head backends, their runners,
the `poisoncap-*` arms of the fourteen FFmpeg, tshark, memcached and httpd corpora, the
two PoisonCap columns of `catch-tables.py`, and the study's `nested-poisoncap` planner
and build paths. The platform recipe (`ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/`)
and its libc fix stay, because the stock CheriBSD runners of the httpd and memcached corpora
boot that image. Every PoisonCap entry below is history; its results stay in `results/`.

## 2026-10-12 — Sublet lifetime extension merged; allocator migration in review

`CDERIVE`/two-operand `CREVOKE`, node-scoped MANAGE and explicit parent links
are implemented and qualified in QEMU with C wrappers. The format has
30-bit IDs, four rights bits and 16-byte node records. Rebuild the Linux
adapter with the new header; its capacity limits change too. Compiler
backend changes are unnecessary. See the [qualification and commands](../design/virtual-capstone/development.md#sublet-lifetime-qualification).
QEMU #21 landed as `7dbfdd534eea`; runtime #216 landed as `baa1ab2faded`,
which pins that integrated QEMU commit. The allocator migration branch builds
on this baseline. Its archived port and workload measurements retain their
original source and binary identities; rebasing does not requalify those runs.
This is QEMU evidence, with no new performance or RTL claim.

## 2026-10-10 — `fpga-testing-dev` rebuilt as one linear history (RTL lane)

- capstone-ariane's default branch is now 44 commits above the fork `e1b3db6ba`, one per logical change, tip `5382266b8`;
  its RTL is byte-identical to `776d9d859` (the current bitstream). Old tip `0bf09b1d6` = tag
  `backup/fpga-testing-dev-2026-10-10`. Lane branches, the parent's submodule pointer and R-51/R-52's branches are
  untouched. Method, table of hashes and the carried arms' readings:
  `docs/history/10-10-2026_08-40-00_fpga-testing-dev-rebuilt-linear.md`.

## 2026-10-08 — mallocng executes locally in virtual Capstone

Branch `virtual-musl-local` replaces the virtual buddy heap with Capstone-compiled
musl 1.2.5 mallocng. malloc/free and ownership operations execute in the
application's virtual C context. Actual syscall/VM, node-pressure, fault and
scheduling paths cross the existing trusted Linux adapter. Upstream allocation
policy is retained; patch 0009 ports capability representation and lifetime
hooks. The fixed allocator block/arena arrays are removed.

ABI v5 requires QEMU's opt-in exact physical-shadow bounds profile, pinned at
`24b1e95b1dac`. This is extra QEMU metadata, not a tag-bit-only or RTL design.
The default physical processor profile and physical allocator selection retain
their previous behavior. Linux core and firmware are unchanged.

The [qualification](../../runtime/virtual/results/local-mallocng/README.md)
records 19/19 allocator, 42/42 runtime, 10/10 pthread, 16/16 other normal-port
and 73/73 memcached checks, plus 202 processor checks. It includes 200,000
lifetimes, 70,000 live objects, exact-site spatial/temporal denials and a
control detecting an inserted allocation-loop service. All eight application
ports are relinked with the new SDK. The compiler and platform are reused;
the two long sparse runtime cases and the CPython inner / FFmpeg-tshark safety
corpora are not requalified here. The memcached fixture-17 setup now searches
for actual reissue; its original temporal-fault prediction is unchanged.

The withdrawn native-heap branches are not the integration target. Merge the
node-growth stack, then QEMU [#19](https://github.com/project-starch/capstone-qemu/pull/19),
then the matching local-mallocng runtime. Rebuild SDK and applications together.
The [allocator contract](../../runtime/virtual/mallocng.md) explains the
representation, metadata cost, VM remap transaction and remaining limits.

## 2026-10-07 — growing virtual node tables

The `review/virtual-node-growth` branch builds on `dev` at `08cf07d75ad2`
and the application build repairs in `review/virtual-build-repairs`. Its QEMU
pin includes the `b649187a989c` UNINIT diagnostic fix and a paged guest node
store with a stable physical root,
unchanged 31-bit IDs and 16-byte records, and chained collection requests.
The trusted Linux module allocates base pages on demand and reuses IDs only
after the existing complete stale-tag sweep. The SDK requests node capacity
before heap allocation. Linux core, firmware, upstream musl and the legacy
host-tree algorithm are unchanged. Applications should be relinked with the
new SDK to use proactive growth.

The [node gate](../../runtime/virtual/results/node-growth/nodes.json) passes
15/15 checks: two rounds of 70,000 simultaneously live structural nodes,
140,004 reclaimed IDs, stable capacity at 70,656 slots (1,142,784 metadata
bytes), stale denial, bounded quota failures and teardown. Processor gates
pass 69/69 virtual-runtime, 60/60 M1 and 69/69 U-access checks. The actual C
node backend matches the forest model for 8,949 prefixes per binding
(host, flat guest, paged guest), with 19 directed paging checks and mutation
controls. The [pthread gate](../../runtime/virtual/results/node-growth/pthreads.json)
passes 9/9.

The [complete runtime gate](../../runtime/virtual/results/node-growth/runtime.json)
passes 46/46, including both 200,000-allocation recycling cases, exact-site
stale denial, spatial checks, SQLite persistence, mruby and Perl smoke. The
sparse positive case uses 348 KiB of node metadata, reclaims 188,631 IDs and
leaves unused payload pages unpopulated. All processor and runtime gates use
the same QEMU executable. See the [qualification manifest](../../runtime/virtual/results/node-growth/qualification.json)
for hashes, reused compiler/musl/platform inputs and application relink scope.
The rebase also passes 87/87 native CTest checks under ASan/UBSan. An additional
physical LINEAR/UNINIT corpus attempt fails at `create_dom` before its probes;
the cause remains unresolved and physical compatibility is not requalified.

This is one-hart QEMU evidence. The 65,536 allocator block-record limit is
separate and unchanged. Node capacity is bounded by RAM, an optional module
quota and the ID field; individual published pages are retained until exit.
See the [runtime mechanism and reproduction commands](../../runtime/virtual/README.md).

## 2026-10-05 — B0: a gp-captable delegated application runs byte-exact on silicon

- **b0-hello passes on `caplifive_supcall_715bdd1fe.bit`, byte-exact under a `cmp` oracle, N=2** (boots 13 and
  13b, preflight off for a documented reason).
  - Four delegated rounds: HELLO, `ioctl` -> -25, `writev` -> 51, `exit_group`.
  - Identity was verified by hash. fw e411395bde2c embeds Image d5ab7fad, whose initramfs carries b0run.sh
    cc858ec4, b0-hello.dom cdd82e56, capstone-exec 1dca5b19 and module 0332e2d6.
- **What runs:**
  - a full-LTO gp-captable musl application, launched by capstone-exec through the process-ABI module;
  - stepped by the FPGA monitor's supervised CALL under CSR events;
  - with no fabricated gp;
  - with its domain torn down at exit (census `live_domains 0`).
- **Found on the way, all QEMU-silent:**
  - **C-76:** capstone-c's `stc` spill of a `__rev` argument, moved to cnull by silicon's STC. It wedged every
    process-ABI step.
  - **A delegate-runtime copy loss** in R-29's family, worked around with 8-byte copies; an RTL fix for R-29 and S-10b is
    FIXED ON SILICON (2026-10-05, caplifive_supcall_776d9d859.bit: b0-memcpy unguarded 94/96 -> 0/96, s06agg 66 -> 64; ISSUES R-29). The first attribution
    (R-29's stale-refill mechanism) was retracted. The RTL lane then reproduced the actual path in simulation: a
    granule copy of a plain-store pair still in the write buffer reads the high half as 0.
  - **M-13:** a monitor-internal trap reads as cause 24 at `_cap_trap_entry`+4.
  - **The board kernel cannot unload modules**, and a module built out of tree needs `-march=rv64g`.
- Record: `docs/plans/b0-silicon-delegated-runtime.md`. The module's `vm_flags_set` port is on caplifive-buildroot
  `capstone-bootstrap` (3f485c4).

## 2026-10-04, evening — the S-16 fix bitstream on silicon: S-16 fixed, the R-43 list unchanged, and the real workload preempts 1,277 times at +0.69 %

- **The resident bitstream is `caplifive_sublet_d93a8d66b.bit` since 2026-10-15 23:54** (sha256
  89ba6d816c23b1115585ff6bcc2d5ceeb4523967731a84baf4f8955256077430; routed WNS -15.856, outside the band, and it runs:
  the S-16 bare set identical to the FP respin's). It carries the Sublet extension on top of fp-fixes 4a4261bb4: the
  capability word {revnode_id 30 | perm 4 | cap_type 3 | bounds 27} (MANAGE = perm bit 3), CDERIVE / CREVOKE with the
  node parent link, MREV deepening, CREVOKE in R-45's flush, the non-linear SPLIT keeping its node. Its decoder does
  not read the old word: FP-era images need rebuilding under the full format. Confirmed on silicon 2026-10-16 by the
  bare floor (capstone/tests/rtl-smoke/sublet-bare-2026-10-15, five arms, every reading as pre-registered): the Sublet
  instructions and their refusal causes, deepening, the shared-node split, a supervised domain deriving and ending
  children, and that every load and store through a revoked capability is refused with cause 25. It needs the R-55
  monitor (explicit cursors after every carve) like the FP respin. Registered fallback: `caplifive_fpb_afcd19bea.bit`.
- **Earlier resident: `caplifive_fpb_afcd19bea.bit`, 2026-10-15 01:31 to 23:54** (sha256 d8d2b641da70207c74d41a69ef60a561426fc573b6767fd03bd6d3c28a5fb3f8; routed WNS -15.310, outside the -13.335..-5.855 band, and it runs: every FP acceptance row as pre-registered). It carries fp-fixes afcd19bea: R-53, the R-52 residual, R-54's renormalisation in residue form, the encoder without the cursorless scheme. The full-scheme encoder needs the R-55 monitor (caplifive-opensbi df0e9a8 / capstone-sbi 6dad416 on the board line): explicit cursors after every carve. Registered fallback: `caplifive_r51r52_0568f93a9.bit` (resident 2026-10-11 to 10-15). The paragraph below describes the earlier resident 715bdd1fe and is kept for its workaround history.
- **Earlier resident: `caplifive_supcall_715bdd1fe.bit`.**
  - capstone-ariane sup-call 715bdd1fe: R-49, R-50 and the S-17 apertures 219..222.
  - Flashed 18:41-18:43 on the lead's direct word with `drivers/flash-bitstream.py`. The sha256 matched the sealed
    hash, and the name was read back after the power cycle.
  - Set `FPGA_BITSTREAM=caplifive_supcall_715bdd1fe.bit` for every board run. The previous resident,
    caplifive_supcall_36a641e0b.bit, stays registered on the console.
- **S-16 is fixed on silicon.** Audited wording and limits are in the S-16 entry in ISSUES and in its folder.
  - Bare: 14 of 14 previously-hanging arms complete without a fence.
  - **The FPGA monitor's supervised SQLite speedtest:** the C5u firmware, byte-identical and with no fence, completes
    1,277 preemptions with the oracle hash at **+0.69 %** over the plain call (quantum 2,000,000). On 36a641e0b it
    hung after 212. Preemptive supervision of a real workload now works on silicon.
- **The R-43 acceptance list passes unchanged**, a1..a10 on the same images
  (`tests/fpga-repros/R43-revocation-cache-false-deny/results/board-715bdd1fe.result-lines.txt`).
  - Identity readings are exact: instret 17/17, hashes, Sublet counts, a10's trap and refusal record.
  - Cycles are within noise.
- **S-17 does not reproduce** (3 of 3), mechanism unknown. The new apertures are armed.
- **The trap log's 0x83 residue on bare hangs is the JTAG load's debug-module `ebreak`** (mepc 0x35c). It means no
  trap since the load.
- **Next:** B0, the delegated hello-world, on this bitstream (branch `b0-silicon-runtime`). It already passes in QEMU
  with no fabricated gp.

## 2026-10-04 — the app ports' and corpora's last 24 registered cells measured, two retractions, and the ledgers closed

- **What was asked:** audit the defect evidence for memcached/tshark/FFmpeg against the paper, then
  finalize everything.
- **The audit's answer: count was never the problem.** 22 upstream corpus cases across the three, **all
  22 nested-allocator**, now **22/22 with a measured capability arm**; plus 59 app-port fixtures carrying
  24 measured nested-discriminating cells. Only **1 of 22** is live at the pin, which matches the paper's
  own wording ("previously fixed bugs"). The gap was **printed numbers without backing runs** — see
  `/tmp/capstone/2026-10-04-paper-defect-evidence-audit.md` (not committed; the paper is the lead's).
- **Measured, all as pre-registered** (predictions committed before every run):
  - **FFmpeg pool corpus, fixtures 40-47, both app-port arms: 16/16 cells.**
    `ports/ffmpeg/app/results/20261004-qemu-pool-corpus-40-47/`. `poolsublet` faults cause 24 at
    `case.c:85 / :48 / :33 / :104`, each with non-zero `value_hi`; `poolstock` completes every defect
    with DEFECT-REPRODUCED. **Case 3's `sublet-port` arm was the last unmeasured arm of the 22.**
  - **FFmpeg case 3's component-port pair, probe slot 39: 2/2.**
    `ports/ffmpeg/buffer-pool/results/measurements/20261004-vvc-case3-probe39/`.
  - **tshark `chunks` fixtures 2-9: 9/9** (fixture 1 as the in-boot control).
    `ports/wireshark/app/results/20261004-qemu-chunks-2-9/`. Read as a consistency check: these are
    plain-heap probes on a nested arm, not new nested evidence.
  - **memcached corpus, 5 cases × spatial/sublet: 10/10, plus the suite-wide negative control** (exit 0 =
    every oracle fired). `ports/memcached/allocators/results/20261004-qemu-corpus-defects/` — the
    corpus's first committed bundle; it gitignores its own `results/` by policy.
  - **ASan blindness to pooled reuse, 4 arms**, with upstream's own `e6255fb822` as the positive control
    and arm C showing the `free_entry_cb` gate leaves the MPVPicture pool uncovered even on master.
    `bug-corpora/ffmpeg/pool-repros/results/2026-10-03-native-asan-pool-blindness/`.
- **New instrument: `ports/common/application/run-fixtures-9p.py`.** `capstone-vm` is the app ports' only
  runner and needs ssh; no riscv64 dropbear exists here to give it. This stages images and the pinned
  launcher/module over 9p, issues **the same guest command `capstone-vm` issues**, merges the launcher's
  fault record into the result as `capstone-vm`'s host side does, and judges with the **committed**
  oracles (`check-safety.py`'s `classify()`/`matches()`, or an external `--verdict` script). Negative-
  tested: forced against a wrong row it reports DIFFERS and exits 1.
- **TWO RETRACTIONS, both mine, both from generalising a partial enumeration:**
  1. *"The app SDK gate refuses every toolchain on this host"* (`5208789e4b9e`). It does not: the
     `llvm-capstone-cc` worktree's `build-release`, commit `7d01722aab88`, passes it end to end, and all
     16 FFmpeg app images were built through the gate with it. I had globbed `*/llvm/cmake-build-*`;
     `ffmpeg-live-defect-triage.md:281-284` already named the qualified compiler.
  2. *"No replacement harness exists"* (`dedab28c4a2c`). One existed on disk and had run app-SDK
     fixtures the previous day; the sibling bundle said so outright. Propagated into the four files that
     still carried the first claim (`49c4903ca277`).
  Rule captured as the `blocked-is-an-enumeration-claim` memory.
- **Also corrected:** `dedab28c4a2c`'s `g_refpool` caveat named the **wrong bundles** — the affected one
  is `bug-corpora/ffmpeg/pool-repros/results/20260929-qemu-sublet-port/`, which now carries it in-tree
  rather than only in a commit message; and that port bundle's claim that the oracle's FAULT branch is
  "structurally unreachable" (it is reachable — the host merges the fault record).
- **Still open, deliberately:**
  - **PostgreSQL's 8 spatial/sublet cells.** The paper prints Sublet catching 8 of 8 while all 8
    `case.json` say the arm has **not** been run. Highest-value single measurement for the paper; out of
    scope by the lead's decision (report only).
  - **`INDEX.md` / `index.json` are stale on `dev`** from the board lane's fpga-repros and ISSUES
    additions. Only this lane's own lines were applied; the lead asked that the two lanes squash and
    merge one consolidated regeneration.
  - PoisonCap and CheriBSD for memcached and FFmpeg: that platform is absent from this host.

## 2026-10-04 — Supervised CALL under the FPGA monitor on silicon: a real workload preempts and resumes hundreds of times, then hits S-16 (localised, mechanism traced, an RTL fix needed); S-17 found beside it

- **The FPGA monitor runs supervised CALL on silicon.**
  - Where it lives: capstone-sbi `monitor/supcall-fpga`, behind the opt-in `CAPSTONE_SUPERVISOR_CSR_EVENTS`.
  - The C5 boot hang was the monitor's 21.5 KiB of globals carved from an 8 KiB FPGA `dom_stack`. It is fixed in the
    wrapper and gated in `build-fpga-fw.sh`.
  - The merged monitor with supervision compiled out passed C3.
  - Folder: `tests/rtl-smoke/supmon-2026-10-03/`.
- **Silicon supervision forbids every M-level CSR inside a domain, by design.**
  - The legacy my_first_domain-family glue saves and restores mcause/mtval, and every cycle bracket reads mcycle.
  - `sup-static-audit.py` finds every such instruction (controls both ways).
  - `patch-sup-glue.py` replaces exactly those words. The patched speedtest gives the oracle on a plain call at
    -0.04 % cycles.
  - The musl glue is clean.
- **The real workload under supervision.**
  - The SQLite speedtest resumed 212 times (C5u), then the switch hung.
  - With a fence before every domcall, it resumed 552 times (C5f), then hung again, on the escape side.
- **S-16 (`tests/fpga-repros/S16-supervised-switch-never-finishes/`).** A domain switch that starts while the
  store buffer's commit queue is full never finishes.
  - The switcher's walk-idx-7 write is never acknowledged; commit count 4, WAIT_STORE_READY.
  - Reproduced bare within 16 resumes.
  - 12 stores before the CALL complete; 18 and 24 hang.
  - The RTL lane's simulation matches every silicon read: the store unit acknowledges the switcher's write on the
    previous store's queue ready, so the write is pushed over the full commit queue's head. **The oldest committed
    store is lost**, and the 4th later write starves.
  - **Every switch kind is exposed (bare, 2026-10-04, the RTL lane's `sup-s16-stores.S`).**
    - The armed CALL, a supervised RETURN right after a domain store, and an escape hang at walk idx 7.
    - A PLAIN CALL hangs too, at idx 4, its exchange write.
  - **A fence before the CALL covers only that CALL's own switch.** The RETURN and escape arms hang with it.
  - **A fence before every CALL AND every RETURN completed all four twins that hung**, so runs without quantum
    preemption have a software workaround.
  - A quantum escape lands wherever the domain is. C5f's hang was on the escape side, and so is bare
    `s16st-esc-n8-retfence`. **Preemptive supervision on this bitstream needs the RTL fix**; that lane has one
    passing in simulation.
  - **Withdrawn:** the memcached plan's "unsupervised contexts do not depend on S-16".
- **About 22,000 bare supervised resumes without the swap sequence never hung.**
- **S-17 (`tests/fpga-repros/S17-ldc-after-supervised-switch-lsu-stuck/`).**
  - After a finished switch, an LDC right behind `ccsrrw sp <- cscratch` leaves the LSU not ready. A scalar `ld` there
    completes 8,552 escapes.
  - A plain CALL is affected too.
  - The simulation hangs at the same LDC in a different state; open.
- **Instrument added:** the bare runner `sup-resume-2026-10-03/run_sup_bare_wedge.py` reads the switcher and LSU
  apertures (192-195, 224-229, 238-240) after a hang. The commit-pc aperture during a stuck switch is stale scoreboard
  slot 0, shown by a positive control.
- **Corrections along the way, recorded in the lane PREREGs:**
  - the k800 entry path;
  - "the hot loop never misses";
  - "MSWAP is void" (my x26/s10 register clash);
  - the S-17 prediction ldsync = 1;
  - the C5f prediction "all supervised runs complete".
- **Still open:**
  - the RTL fix for S-16 and its bitstream. A reflash is the lead's call.
  - S-17 on silicon.
  - Supervised contexts on silicon (memcached Part 4b) stand behind S-16.

## 2026-10-03 — Atomics through a capability address work on silicon: the first memcached prerequisite

- **The question.** memcached.dom links 171 atomics: 80 `lr.w`, 80 `sc.w` (musl's `a_cas`), 9 `amoadd`, 2 `amoswap`. No
  image proven on silicon contained any, and no RTL directed test ran one through a capability in capability mode.
- **The test.** `tests/rtl-smoke/cap-atomics-2026-10-03/`, bare on `caplifive_supcall_36a641e0b.bit`, pre-registered
  and pushed before the boot (49d7f98067c3). It has 22 architectural readings.
- **PASS, all 22 exact**, cited by image hash:
  - the control, call-retpc a05ca464, exact (N = 4 on this bitstream);
  - amoadd/amoswap .d/.w, including the 32-bit wrap;
  - lr/sc .w/.d success;
  - SC failure with no reservation and on another granule;
  - a musl `a_cas` loop in one attempt.
- **Not covered:**
  - AMO bounds or permission enforcement;
  - the I4 tag residual (an AMO over a capability granule keeps its tag; the RTL lane's
    `s06sec-amo-no-resurrect.S`);
  - multi-hart contention.

## 2026-10-04 — S-16 root-caused: the switcher's first SAVE write pushed into a FULL store-buffer commit queue; fixed in RTL with a second switch-path defect (R-49, R-50); the resident bitstream needs a `fence` before every CALL and every RETURN, and preemption needs the reflash

- **The resume hang under the FPGA monitor (C5u/C5q, the board lane's S-16) is a store-path defect, not the adapter.**
  The store unit acknowledges a switcher write on the room of the PREVIOUS store's queue, so with four committed
  stores still queued the write lands over the commit queue's head: one committed store is lost and the queue
  deadlocks four writes later, at walk id 7. The board lane's wedge reads (idx 7, WAIT_STORE_READY, commit count 4,
  the write request valid and unacknowledged) and the simulation reproducer `sup-s16-stores.S` (N = 32 stores before
  an armed CALL, memory delay 12 and 40) agree on every aperture. Fixed on capstone-ariane `sup-call` 192a5e624
  (registry R-49). **Workaround on the resident bitstream** (the board lane, bare on silicon, narrowed 2026-10-04):
  - A `fence` before every CALL covers that CALL's own switch only.
  - A fence before every RETURN as well completed every non-preemptive twin that hung.
  - A quantum escape stays exposed. See the S-16 entry above.
- **A second defect, found on the way (R-50, 429c60b32):** the load unit's dom-switch flush exemption outlives the
  switcher's read, so the first LDC after an escape, when dispatched in the swap-in CCSRRW's flush cycle, runs
  un-killed and its stale completion pops the re-issued LDC unexecuted; the DYN unit waits forever. Deterministic in
  `sup-mswap-noploop` on the resident logic; fixed by adding the entry's valid bit to the exemption. Not the silicon
  S-17 (different aperture signature; S-17 stays open).
- **Synthesized and sealed (2026-10-04 evening, the synth lane):** `sup-call` 715bdd1fe (the two fixes plus the
  S-17/S-16 LED apertures 219..222) -- route clean in ~21 min, LUTLP-1 0, WNS −9.949 in band, one TIMING-23 loop
  (arc relocated within the same cluster), census unchanged; TNS +49% with the LED apertures carrying the worst-path
  shift (0.36 ns from WNS) but not the endpoint growth. Bitstream sha256
  a7add4c019d3067627ba984634e96de564f023bcec50e2c29fd0f668c0a81311. The board lane's acceptance batch (17 bare images
  plus the monitor without the fence) is staged; the reflash is the lead's word. Details in the plan's step-11 result.
- **Verification so far:** matched pairs for both fixes; `sup-mswap-noploop/base/postread` 78 switches on the fixed
  tree; lint gate at the committed baseline; 95-test sweep: 95-entry sweep on the committed tree against the clean baseline, seed 20260922, memory delay 12: 92 identical in taken exceptions, CAPPRINT readings and retired-instruction counts, 0 differ; the 3 random-generator entries produce no log on either side, as in every previous sweep. Sim-only tracers (every retirement, write-back,
  LSU queue event, load/store-unit transition, the switcher's busy edges, and the board's apertures emulated every
  4000 cycles) in b576635be/d92093828. **Next:** the synthesis candidate is `sup-call` HEAD; synthesis and the
  reflash are the lead's word; then the board's S-16 arms (armdep-nt, s16sd24) must complete without the fence, and
  the FPGA monitor's supervised speedtest runs again.

## 2026-10-03 — Supervised-CALL ladder complete on silicon: 12 images exact to simulation, the real timer escape, S-11's refusal; N = 3 on the key tests

- **The rest of the RTL lane's ladder ran bare on the board** (`caplifive_supcall_36a641e0b.bit`), one session per image,
  pre-registered and pushed before the boot (`tests/rtl-smoke/sup-bare-2026-10-03-ladder/`, 73093f0bd558).
  - **Equal to simulation, every reading:** fullswitch (the 67-slot full exchange), arm, armclose, guards (84/84), and
    all seven S-11 arms. Below 1024 B or off 16-byte alignment the seal is refused with cause 29; at 1024/2048 B
    aligned it seals with every canary intact.
  - **sup-mtip:** the real CLINT timer interrupt escapes a supervised domain (kind 1, cause 7) twice, with MTIP still
    pending in the monitor, and the quantum escape follows. Exact except the reference spin count, a board timing
    quantity (122 against simulation's 243).
  - **sup-strip x5:** exploratory. It completes within its invariants. Its in-switch detector exists only in
    simulation.
  - **Repeats:** sup-escape v2, sup-quantum v2 (q16, q64) and call-retpc now stand at N = 3, identical every time;
    74 preemptions at both quanta.
- **Still open on silicon:**
  - sup-hostile (it needs a register-preserving recorder);
  - anything through the OpenSBI monitor or Linux. Next: the FPGA monitor's supervised_invoke, in the board lane.

## 2026-10-03 — upstream memory defects for the three app ports: 4 measured on the arms, 18/18 cells as pre-registered, 3 retractions

- **What was asked:** find memory defects (plain `malloc` **and** nested allocators) for memcached,
  tshark and FFmpeg on the upstream trackers, run them through the emulator, check they are caught.
- **Measured, all exactly as pre-registered** (predictions pushed before every build):
  - **tshark** fixtures 14/15 — ZigBee ZCL Touchlink (CVE-2026-95391) and http2 `GRegex` — on
    `level0`/`shrink`/`sublet`/`chunks`: **8/8 cells**. `ports/wireshark/app/results/2026-10-03-qemu-upstream-live-defects/`.
  - **memcached** fixtures 17/18 — `204019d` realloc cursor and `e779381` logger write-after-free — on
    all five arms: **10/10 cells**. `ports/memcached/app/results/2026-10-03-qemu-plain-heap-contrast/`.
- **The memcached pair is the one that changes an argument.** Fixtures 12–16 end their objects inside
  memcached's own allocators, so the runtime's `free` is never called and `sublet` is registered as the
  negative control that must RETURN — that silence is the nested-allocator blindness claim. 17/18 end
  theirs with the platform's `free`/`realloc`, and `sublet` faults. Same heap, same runtime, opposite
  outcome: the variable is **where the lifetime ends**, which makes 12–16's silence attributable.
- **Three retractions, all surfaced and recorded:**
  1. **`ops_dispatch` (FFmpeg fixture 25) is NOT live at our pin** — backported as `716d2a47c5`. The
     retracted method was *"Ancestry decides; the pinned tree proves"*: ancestry is insufficient for
     cherry-picking release tags, and the "proof" was a grep for a line *shape* that matched a site
     where the object is never freed. Replacement test, with both controls:
     `git log <tag> --grep='cherry picked from commit <sha>'` plus reading the enclosing function.
  2. **Neither tshark defect is reachable in the port's own binary** — http2's block is inside
     `#ifdef HAVE_NGHTTP2` which the port undefines (and it is preference-gated upstream), and the
     zbee dissector is not linked (0 symbols against 353). They are live in the **v4.6.8 source**;
     "live in the version we compile" and the triage doc's "reachable and not latent" are withdrawn.
  3. **Three attribution/control overstatements in the tshark bundle**: the load base was given as
     `pc − symbol` (`0xa01f0014`) and is `0xa01f0000`; a symbol-delta argument was **vacuous** (an
     algebraic identity); and "both controls fired in every boot" was false — fixture 5 ran in none of
     the four decisive FAULT boots.
- **Search yield, honestly:** Wireshark `v4.6.8..origin/master` is **4321** commits (I had searched 82
  on the release branch) → 24 lifetime-worded → 5 live in source → **1** new usable. FFmpeg: all 10
  remaining candidates read, **none usable**; its nested column is empty because the only nested-class
  commit was backported, and the four "HW data freed early" pool commits are destruction-ordering on a
  **VkSemaphore**, not pool storage. memcached: **0** fixed-upstream-after-the-pin, now verified
  two-sided (`origin/next`, which memcached actually develops on, is one unrelated commit ahead).
- **Instrument lesson worth carrying:** a lifetime-vocabulary grep over memcached's history returns 13
  commits and is **not** a superset of the 9 the subsystem search found — intersection 2, and three of
  the five built corpus cases would never have been found by wording. Treat such a filter as a sampler.
- **Still open:** memcached's nested column needs port work (`--disable-proxy --disable-extstore`
  everywhere, page mover not hooked); no native/ASan arm for the four new fixtures; the "revocation on
  free" mechanism is inferred rather than counted in the faulting runs, and the missing control is named
  in the tshark bundle.

## 2026-10-02 — Supervised CALL RUNS ON SILICON: quantum preemption with exactly-once resume, and fault-event delivery (bare M-mode, one run each)

- **The RTL lane's directed tests ran on the board** (`caplifive_supcall_36a641e0b.bit`), bare M-mode, JTAG-loaded at
  `0x80000000` with no OpenSBI. The tests are byte-identical to capstone-ariane `36a641e0b`; only a harness header
  differs. All three board reading vectors are **bit-identical to the RTL lane's simulation of the same tree**
  (113/113, 67/67, 8/8):
  - `sup-quantum`: 53 quantum preemptions, each resumed. The domain's 400 instructions net to **exactly 400**. The
    domain's x1, the monitor's resume count and its mie survived, and no trap was taken.
  - `sup-escape`: ecall, illegal, `csrw mepc` (CSR gate), `mret` (decoder guard) and a misaligned load each deliver a
    fault EVENT to the monitor with the right cause, epc and tval. The seal comes back SEALED.
    - It does NOT show the trap stripped: the no-strip mutant reads the same 67 values, because mepc, mcause and
      mstatus are walked. **Settled by the v2 tests** (capstone-ariane 7564c0945, repaired by the RTL lane;
      `tests/rtl-smoke/sup-bare-2026-10-02-v2/`, pre-registration committed before the boot): sup-escape v2 reads
      the domain's saved mcause slot as 0 in all five arms, and equals simulation 72/72. **The trap is stripped on
      silicon.** sup-quantum v2 (an acc := acc*7 + i ladder) is exact at both quanta. At q16 it equals simulation
      155/155. At q64 the board took 74 preemptions against simulation's 73: a pre-registration miss on a timing
      count, with every correctness reading exact.
  - `call-retpc`: correct values. This is not a demonstration of R-47 fixed on silicon, because the board's timing
    was not shown to create the trigger.
  Folder: `tests/rtl-smoke/sup-bare-2026-10-02/` (self-contained harness, images, pre-registration, audited result
  lines).
- **These two tests discriminate** against any bitstream without the extension, where `cssupervise` and the event
  CSRs do not exist. This is the first silicon evidence for the mechanism the delegated runtime needs.
- **Not yet on silicon:**
  - repeatability (one run each);
  - the MTIP escape, the guards, the hostile domain, and S-11's refusal;
  - anything through the OpenSBI monitor or Linux.
  The delegated runtime on silicon still needs the runtime lane's FPGA monitor, built to the plan's contract.
- **Instrument:** the runner's GDB halt fallback does not work on this board (`dmstatus` allrunning; the halt is not
  honoured). The UART report is the only channel.

## 2026-10-01 — Supervised CALL implemented in RTL (capstone-ariane `sup-call`), verified through step 7 in simulation; R-47 found and fixed; S-11 amended

- **The supervised-CALL platform extension** (the primitive the delegated application runtime needs on silicon)
  is implemented on `capstone-ariane` branch `sup-call` per `docs/plans/supervised-call-silicon.md` (revisions 1.1
  and 1.2): `cssupervise`, the escape at commit, the SAVE/RESTORE walks around the existing 8-register exchange,
  the quantum, the guards, the event CSRs (csupquantum 0x7C3, csupctl 0x7C4, csupstatus..csnodefree 0xFC0..0xFC4).
  Steps 2-7 of the ladder read as pre-registered in simulation (five synchronous faults, exactly-once resume
  under two quanta, a real timer interrupt escaping the domain twice, all ten guards, a hostile domain and a
  foreign RETURN), with ten mutant positive controls, an adversarial after-audit whose four findings are fixed
  with discriminating arms, and the 92-test neutrality sweep on the committed tree (91 identical, `revocation`'s
  phantom mid-switch traps gone). Lint at baseline. Commits: d38887426, 1dbf379b1, 727ea6e93, 03b70667e on
  `sup-call`, pushed, not landed on the submodule's shared branch. **Not synthesized, not on silicon**: synthesis
  is the lead's call; the FPGA monitor build (`supervised_invoke` with the CSR-based events, csupctl.resume
  before a resume, interrupts masked between cssupervise and CALL) is the runtime lane's.
- **R-47 (new, fixed on `sup-call`, OPEN on every bitstream):** CALL parks the wrong return pc when a jump or
  branch reaches issue before the dyn unit responds -- the caller resumes past that instruction, or
  mid-instruction after a compressed one. Nothing resident is hit only because the monitor's and the runtime's
  CALLs are followed by an `ldc` that holds issue. Registry entry on dev.
- **S-11 amended on dev** (the Anvil relational-precedence fold, read from the compiler's own grammar): QEMU now
  raises on an undersized SEAL (since 2026-09-26, minimum 528 B); the switcher's instance of the fold is fixed
  on `sup-call` and the full-mode slot map measured; a 64-byte seal makes the switch write 24 bytes past the
  capability's end (measured); `cssupervise` CANNOT check a seal's size (a SEALED capability carries no end
  bound in a register), so the 1 KiB minimum is SEAL's alone -- the resident monitor's 1536-byte seals are the
  guarantee until S-11 is fixed. A deterministic lint for the fold is on `sup-call`.
- **Two more decisions on the lead's delegation, same bitstream, audited:** a CALL of another seal while armed
  fails closed (trap, the arm persists across the trap and across every exception on a CALL head, as QEMU; one
  predicate for the block and the trap, debug mode included); S-11 is fixed in RTL at 1024 B / 16-B aligned with
  R-32's off-by-one corrected AND the sealed cursor set to the region's start -- the audit showed a SEALED
  capability's base IS its cursor, so the size check alone bounded nothing (every known sealer complies; QEMU's
  528 should be raised). Both tested with mutant controls; sweep4 on the final tree 91/92 identical to the unmodified RTL (revocation -3 phantom traps, as before) and 92/92 identical to the tree before the audit's changes. RETRACTED
  with it: "a SEALED capability reads back untagged after STC/LDC in simulation" and "status 1 is unreachable" --
  `sup-arm.S` had overwritten the seal with an integer before the store; repaired, the seal survives the round
  trip and the dead-node status (1) is measured.
- **Supervised CALL synthesized (2026-10-02):** `sup-call` 36a641e0b sealed by the synth lane, exit 0; routed loop arc
  identical to the resident's, LUTLP-1 = 0, WNS -8.189 in band, FF +1,090 explained row by row, two ORDER-like
  pre-registration misses with mechanisms (plan, step-9 result). Bitstream sha256 016219b2..., NOT flashed.
- **Supervised CALL confirmed on silicon (2026-10-03, the board lane):** the whole plan ladder except the hostile domain
  ran bare on the resident bitstream exact to simulation -- full switch, arm, the fail-closed arm, guards, all seven
  seal-size arms (S-11 fixed on silicon), MTIP, strip x5, N=3 repeats of escape/quantum/call-retpc; folder
  `tests/rtl-smoke/sup-bare-2026-10-03-ladder/`, plan step-10 result. Next: the FPGA monitor's `supervised_invoke`
  against the plan's CSR contract (the board lane), then the first `capstone-exec` program.
- **Resident bitstream: `caplifive_supcall_36a641e0b.bit`** (sha256 016219b2...5d045), flashed 2026-10-02 15:55 by the
  board lane on the lead's direct confirmation, name read back after the power cycle; the previous resident was
  `caplifive_r43_8f6a0af.bit` (the name the console reports). **ACCEPTED 2026-10-02 16:12-18:05: the R-43
  acceptance list a1..a10 reads as pre-registered on every row, on the same images as 2026-09-29**
  (`tests/fpga-repros/R43-revocation-cache-false-deny/results/board-36a641e0b.result-lines.txt`; audited, every number
  re-derived from the primary logs). Highlights:
  - live512 sums 17408 (R-43 holds), and P1 cell 6 completes with 112006 38bb59fd and the exact Sublet counts;
  - the R-35 probe still traps 25 at +0x4354 with the refusal record LATCHED, id 95, probe-DEAD;
  - the capability ladder's instret is identical on 17/17 rungs, and P1 moves -0.01..-0.05 %.
  **Identity shown by behaviour, not only by the console label:** a one-instruction domain reading csnodefree (CSR
  0xFC4, which exists only in 36a641e0b; on 8f6a0af98 the read is an illegal instruction) RETURNED 0xFFCD, inside the
  pre-registered range. That is the first silicon execution of the census CSR. **Read the acceptance as "no observable
  change on these ten workloads", NOT "inert until cssupervise"** (that wording was retracted before this landed).
  Several changes run without cssupervise, every boot:
  - commit drops exceptions while any domain switch is busy;
  - every CALL parks its pc per R-47;
  - every SEAL applies S-11's size/alignment check and sets cursor := start; the monitor's seals passed.
  The supervised path itself, and any input that would trip S-11 or R-47, have not run on silicon. Set
  `FPGA_BITSTREAM=caplifive_supcall_36a641e0b.bit` for every board run. Every Part A bundle so far (M1 v2, S1,
  repro322) was measured on 8f6a0af98, before this flash.

Minimal snapshot. Read first in every session.


---

Everything before 2026-10-01 is in
[`history/13-10-2026_01-00-00_state-journal-through-2026-09-29.md`](../history/13-10-2026_01-00-00_state-journal-through-2026-09-29.md).
