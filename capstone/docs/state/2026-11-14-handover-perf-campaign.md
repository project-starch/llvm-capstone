# Handover: the Capstone performance campaign (SPEC on silicon, nested allocators, RTL, compiler)

Written 2026-11-14 16:55 +08 by the SPEC/runtime lane (Claude session `runtime`). It exists so that the campaign can
be continued by someone with no access to the conversations behind it. It covers every lane. Everything here is a
pointer to a record that is committed; where a fact lives only on the measurement host, the path is given.

**Path convention.** `$H` is the measurement root on host *apollo*: `/mnt/hdd/<account>/capstone`. The account
directory is not written out because the pre-commit name scan forbids it. Every `$H/...` path below is on apollo.

---

## 1. Where to look first

| What | Where |
|---|---|
| Live SPEC results table (3 arms, per-row config, status, caveats) | llvm-capstone branch `spec-silicon`: `capstone/ports/spec-cpu2017/results/2026-11-14-three-arms/README.md` |
| Pre-registration and run-by-run trail (Addenda 1-112) | same branch: `capstone/ports/spec-cpu2017/results/2026-11-08-train/PREREG.md` |
| Raw reading log, one line per board reading | `$H/perf/natstage/RESULTS` |
| Nested-allocator workloads (separate lane) | branch `nested-workloads`: `capstone/docs/ref/nested-allocator-workloads.md` |
| RTL plans (tag-word cache, DYN bypass, stage-2 RVC sketch) | dev 46f8d4c6b38f (the next-bitstream plan + R-74); lane `land/rtl-twc-reach` 0b458f3595a6, `land/rtl-dynbypass`; dev `capstone/docs/plans/2026-11-14-capability-rvc-stage2-decoder-sketch.md`; capstone-ariane perf/twc-f2-build 69d5ac2c0, perf/twc-f2-l1d64 67b9ed442, testharness default b36989efd |
| Compiler gp-relative plan and rules | llvm-capstone branch `gp-relax-prereg` |
| Rules every agent follows | `CLAUDE.md` (repo root), skills `.claude/skills/board-run`, `.claude/skills/rtl-sim` |

---

## 2. The goal and the lead's standing rulings

**Goal (12-14 Nov).** Every SPEC CPU2017 row reaches Capstone temporal / riscv-base < 1.20. gcc is measured at
train01, the rest at test size. perl is reported on its FULL test (makerand + test.pl); makerand-only runs are used
for quick A/B work only.

**Rulings that bind every comparison:**
1. **Baseline = stock CVA6 RTL** (`caplifive_rvbase_d31345cf3`). Never tune its RTL.
2. **Software is identical on both arms**: compiler and flags, static linking, K2N kernel, inputs, measurement, and the
   allocator policy (mapping cache, eviction, in-place realloc). Only the RTL differs. Consequence: a toolchain feature
   that riscv-base could also use (gp relaxation) must be applied to both arms or neither.
3. **No riscv-base rerun without a strong reason.** A software change that both arms share is such a reason.
4. **Capstone is "our processor".** Larger caches, TLBs and the like are allowed, but each must be proven by a
   measurement to help and not hurt.
5. **All optimisations run in parallel.** A new bitstream or SDK runs the problem rows first (perl, gcc, mcf).
   Experiments run only on the best bitstream.
6. **The capability-compressed ISA (stage 2) is tried, but after everything else.**
7. **Three arms are measured:** riscv-base, Capstone temporal (spatial + temporal safety) and Capstone spatial (heap
   lifetimes off).
8. **Results files must hold current numbers** with their bitstream, what is still running, and the context, never an
   empty "pending" table.

**Absolute rules** (CLAUDE.md):
- No personal names in anything committed.
- Never commit or echo the FPGA console URL or token.
- Never edit or push `capstone/paper*`.
- A force-push of a shared branch needs the lead.
- Synthesis only under the memory guard.
- `precommit-scan.sh` by absolute path before every commit and push.

---

## 3. Current results (14 Nov, 17:30; the live table in §1 is always newer)

Ratio = Capstone time / riscv-base time on the board. specdiff PASS on every row. x264 is checked by its 5 decoded-frame
hashes against a plain board run, because the native stage has no imagevalidate.

| Row | riscv-base (s) | Capstone temporal, best (s) | Ratio | Bitstream / images |
|---|---|---|---|---|
| perl full | 12,623.02 | 17,051.48 | 1.351 | fdk / FRCDK |
| gcc train01 | 1,722.35 | 1,989.28 | 1.155 | **F** / FRCDKMS |
| mcf | 5,125.02 | 6,576.13 | 1.283 | **F** / FRCDKMS |
| lbm | 1,077.50 | 994.66 | 0.923 | fdk / FR |
| x264 | 35,613.24 | 34,053.48 | 0.956 | fdk / FR |
| imagick | 11.08 | 12.76 | 1.152 | batch / FRCDKMS |
| nab | 679.92 | 714.92 | 1.051 | batch / FRCDKMS |
| xz | 4,993.81 | 5,196.56 | 1.041 | fdk / FR |
| geomean | | | 1.105 | |

Six rows are below 1.20. **perl (1.351) and mcf (1.283) are above it.** F is the final temporal bitstream
(Addendum 113): against the batch with the same images, mcf -10.8 % and gcc -18.7 %.

**Spatial arm so far:** gcc train01 at 2,431.58 s (1.412) against temporal's 2,446.30 s on the same bitstream, board
and kit. So the temporal increment on gcc is +0.6 %. The other seven spatial rows are twinned and ready (§8).

**History on the problem rows** (temporal ratio):

| Step | perl | gcc | mcf |
|---|---|---|---|
| 13 Nov arm C | 1.551 (makerand) | 1.720 | 1.484 |
| fdk (8-way I$) | 1.347 (makerand); 1.394 full | 1.506 | 1.440 |
| best now | 1.278 (makerand); 1.351 full | 1.155 (F) | 1.283 (F) |

---

## 4. Lanes (Claude sessions on apollo, each in its own tmux)

| Lane (tmux / message name) | Owns | State today |
|---|---|---|
| `runtime` (this one, the SPEC lane) | board queue and chains, pre-registrations, the SPEC table, kits, stage building; capstone_vm/runtime decisions | running the table and the bitstream pairs |
| `compiler` | codegen, SDKs, images, QEMU twins (slots 29-31), kernel profiling (perf/kpin) | gp-relative, test.pl kernel census, spatial/temporal images |
| `rtl` | capstone-ariane RTL, simulation (sim12/13/14, vp8/vp9 containers), plans and gates | tag-word cache F/F2, timing attribution, stage-2 decoder sketch |
| `board` | synthesis on apollo (launch-job.sh), seals, the board-busy hub | builds F/F2/L1D-64K, sealed |
| `synth` (Remote Control, host leffe) | synthesis on leffe; launches need its own user's approval | idle |
| `nested` (started 14 Nov) | nested-allocator workloads (lead's request, §9) | speedtest1 measured; next apps |
| `paper` | the paper; nothing to do until numbers are final | idle |

**Talking to a lane.** Use `SendMessage` to the lane's name, or attach its tmux session. Treat a lane's report as a
claim until checked against its record (CLAUDE.md).

**Starting a lane.** Start it in tmux with `CLAUDE_CONFIG_DIR=... GIT_CONFIG_GLOBAL=<agent creds gitconfig>`. Both are
needed: neither is inherited by tmux, and without the second, git push fails.

---

## 5. Hardware: boards, bitstreams, kits, synthesis

**Boards.** Three FPGA boards (Genesys-2, 25 MHz core).
- Shared, booked through the hub (`/tmp/capstone/lab-hub/lab-book.py`), one run at a time per board.
- The console URL and token live only in `~/.claude-kisp/secrets/` (board 1: `fpga-console-url`; others:
  `fpga-console-url-board2/3`).
- The board-idle watchdog writes `$H/perf/watch/board-idle.log`.

**Bitstreams.**
- **Store:** all sealed, mode 0444, in `$H/bitstreams/`. The board lane's seal job writes there and refuses to
  overwrite, so never pre-place a file under a sealed name.
- **Seal sidecar:** each has one in `$H/perf/natstage/SEAL-<name>.sha256`.
- **Timing:** WNS is measured against the 40 ns operating period (xlnx_clk_gen clk_out1, `tcl/run.tcl:27-32`; the "50 MHz" comment at `ariane_xilinx.sv:1209` is stale). Every bitstream is negative at the signoff corner
  (F -9.66 ns), and bitstreams down to -19 ns have run clean for weeks, but no instrument measures the real margin.

| Short | Name | sha256 | Content | Status |
|---|---|---|---|---|
| riscv-base | `caplifive_rvbase_d31345cf3` | (thp2/SEAL-rvbase) | stock CVA6 | baseline |
| mlp | `caplifive_ids55-native-tagcache-mlp_b98f5a517` | 5e8421a5… | Capstone, 55-bit ids, native route, tag cache, MLP | superseded |
| final | `…-hpmreplay-dynbypass3-dtlb64-shrinkfix_6d0e754d5` | 72bf215d… | + FP fix, HPM, DYN bypass v3, DTLB 64, SHRINKTO fix | superseded |
| fdk | `…-dynbypass3-dtlb64-shrinkfix-ic32k_3c6b9906c` | 141f840e… | final + I$ 32 KiB 8-way | superseded by batch |
| batch | `…-ic32k-r72-stcnoclaim_a47b588a1` | ff36870d… | fdk + R-72 fetch-end check + lever (a) (STC rs1 no-claim) | current table config |
| **F** | `…-r72-stcnoclaim-twc16k-merge-hpmtag_a948e4134` | ec675c2b… | batch + 16 K-entry tag-word cache (2 MB reach), merge-on-tag-write, one-write-port fix, HPM events 24-28 | mcf -10.8 %; expected final |
| B | `…-r72-stcnoclaim-hpmtag_afe4e643e` | 5f8273b4… | F's tree before the one-write-port fix (bbfc6abc1 + b9010ad23 + config), TWC at 256 entries, merge compiled out, + HPM 24-28: the tree control for F (the missing fix is functionally inert without a merge write) and the counter baseline | running on board 2 |
| F2 | `…-twc16k-merge-f2-hpmtag_eeb654104` | 57f8b1e2… | F + targeted fill invalidation | sealed, flash-ready (9/9 keys), not yet flashed |
| L1D-64K | `…-merge-f2-l1d64k-hpmtag_b0f02bac7` | 64d1c569… | F2 + L1D 64 KiB 16-way | identity only: WNS -17.71 (-5.5 ns against F2 alone, N=1: deeper D$ logic plus density at 99.91 % slices) and congestion 5 fail; not flashed. 64 KiB at 8 ways cannot be built (8 KiB per way exceeds the 4 KiB VIPT index limit) |

**Kits.** In `$H/t3-smoke/`. A kit = bitstream + kernel tree + firmware recipe + command line. Its `board.env` and
README record each change.

| Kit | Bitstream | Notes |
|---|---|---|
| bkit73 | batch | standard HPM set |
| bkit74 | batch | + `dom_grant_root/dom_null/dom_local_tmp`, needed by perl's full test.pl |
| bkit75 / bkit76 | F | std HPM / tag-path HPM (firmware wrapper `wt-wrap-hpmtag`, `-DCAPSTONE_HPM_EVENTS_TAG`: hpm3..8 = events 24-28 + L1D miss) |
| bkit77 | B | tag-path HPM |
| bkit78 / bkit79 | F2 | std / tag HPM |
| bkit71 / bkit72 | fdk | std / dom |
| bkit54-k2n-rv | riscv-base | used through `perf/thp2/thrun.sh` |

- **Kernel trees** are under `$H/perf/k2n/`. The current one is `linux-board-k2n-native-hpm2`: capstone_vm from
  llvm-capstone a0ed3250249b. `node_flat_records=262144` comes from the kit command line.
- **Synthesis:**
  - apollo: `$H/../synth/launch-job.sh TAG SHA gate|full` (the `board` lane runs it; 40 GB guard, at most two jobs);
  - leffe: through `synth`.
- **Before a build or a board:**
  - lint (`verif/sim/rtl-lint-gate.sh`);
  - a 95-test sweep (`verif/sim/SWEEP.md`; the testharness crossbar now defaults to the FPGA's CUT_ALL_PORTS);
  - a synthesis-only gate for wide changes. Its mapping-memory key is a delta over the run's own XDC-phase peak +
    500 MB, not an absolute band. Placed-FF keys exclude the retiming-bistable families and count the build's own FD*
    primitive sum.

---

## 6. Compiler, SDKs and images

**Image sets ("arms")** live in `$H/perf/cclever/arms/<ARM>/exes/*.strip.dom`; each arm keeps its record in
`arms/<ARM>/RESULTS`.

| Arm | Compiler | Flags / SDK | Use |
|---|---|---|---|
| FR | llvm-c6 (compiler-global-lo-fold 2f3ff76eba7b) | -capstone-fold-global-lo, padded link.ld, +relax; arms SDK (nat-realloc-evict) | old base |
| FRCDK | + standard RVC and capability-base RVC (llvm-capstone 74080cce29e5), -capstone-disable-cost-per-use | same SDK | perl -3.5 %, gcc -2.0 % on silicon |
| FRCDKMS | llvm-c9 = compiler-capbase-rvc-c8 29027126f39b (+ C8 inline 16-byte memcpy) for the benchmarks; the SDK's libc and runtime are compiled by llvm-c4 | SDK perf/memset (wt-ms beb1bbabe26a = 7166a346d516, the cherry-pick of b68bbde23c79 CAPSTONE_MEMSET_CAP_ZERO, + a test commit) | **current temporal images**, all 8 rows |
| FRCDKMSsp | same | SDK perf/memset-spatial: only `CAPSTONE_HEAP_LIFETIMES=OFF`. By archive member it differs only in libc's four mallocng objects and malloc-mapping.c.obj, and all 774 benchmark objects of the 8 rows are byte-identical to FRCDKMS's (spatial/objcmp.py, textcmp.py) | **spatial arm**, all 8 rows |
| GPC / GPR2 | llvm-c10 = compiler-gp-relax 81193d42edb0 (branch: 784137547b99 + 81193d42edb0; lit 221/223, both failures also on llvm-c9) | gp SDK perf/gpsdk (perf/memset's script on wt-gp = runtime-gp-cursor, built at ea3ab7249002; e120e53b053f differs only in comments). GPC = every gp flag off (control). GPR2 = `-capstone-mark-gp-relaxable -capstone-small-data-limit=16 -global-merge-min-data-size=17` + lld `--relax-gp` (sdk/ld.lld-relax-gp) | gp decision pair |
| RVGP | riscv-base: llvm-c4 codegen + `-msmall-data-limit=8`; link by llvm-c10's lld --relax-gp (wrapper perf/fair/plain-cc-gp) | label plain-rv64gp, `spec/tree-plain`; stages 500.perlbench_r-rv-gp-mk, 502.gcc_r-train01-rv-gp | riscv-base half of gp |

**riscv-base images:** fair3 build (`plain-rv64fre`), stages in `$H/perf/fair/stages/*-rv-fair3`.

**Twins.** QEMU twins (native route): `$H/perf/faultesc/run.sh <slot> <images>`. Slots 25-28 are free for others;
the compiler lane uses 29-31. A twin must use the board's kernel module and parameters, i.e. images-a0ed (Image f463a092e905178e) with
node_flat_records 262144, the dom_* parameters and TWIN_CMDLINE_MAX=1024 (the a0ed command line is 557 bytes; the
runner's default limit is 512). A cross-arm "outputs identical" needs each arm's own specdiff N/N
first.

---

## 7. How a board experiment is run (SPEC lane procedure)

1. **Stage.**
   - `$H/perf/natstage/mk-native-stages.py <exe dir> <out> <rows>` makes a native stage from the fair stages.
   - The kit's `make-train-stage.sh <stage> <board stage>` makes the board stage.
   - perl full is made with `make-native-stage.py` on `fair/stages/500.perlbench_r-cap-fair3`, which gives e4631fa's
     $^X copy.
   - perl makerand-only: copy `stages-fr-mk/.../run.sh` in and drop test.pl from plan.json.
   - Diff the board stage against an existing one: only the executable and plan.json may differ.
2. **List and conf.** Name them `<x>-b<board>.list/.conf`. The list holds `stage timeout booking-min`. The conf holds
   E0, SEAL, TAG, KIT, LIST. Two stages with the same basename must not share a list: the work directory collides.
3. **Check.**
   - `bash stage-check.sh <list>` refuses an empty `train/in`.
   - `NR_DRY=1 NR_OUT=… NR_LOG=… bash nat-run2.sh <board> 1 <conf>` is the stubbed dry run.
4. **Pre-register** an Addendum in PREREG.md with predictions, run `precommit-scan.sh`, commit and push to
   `spec-silicon`.
5. **Launch.**
   - `NR_PREREG=<hash> setsid nohup bash nat-run2.sh <board> <wait pid> <conf> &`. The wait pid chains it behind the
     previous chain; the driver refuses a pre-registration that is not on origin.
   - riscv-base: `NB_PREREG=… bash ../thp2/thrun.sh <board> <wait pid> <conf>`.
6. **Read.**
   - `bash read-all.sh` (out-dir TAG → stage-plan mapping), or `run-stage.py read --stage … --tree $H/spec/tree-boarddevmk/SPEC17 --text board.txt`.
   - Native counters line: `grep run_cycles= board.txt`, with k1/k3/k4, checks, run_cycles, hpm3-8.
   - Append a line to RESULTS, update the three-arms README, commit and push.
7. **Stopping a waiting chain.** Kill its bash pid only, and remove its empty out-dir. A running boot's child
   continues. Unbook the hub booking if a chain dies mid-boot.

---

## 8. What is running and queued right now

| Board | Now | Queued after it (chains, nat-run2.sh) |
|---|---|---|
| 1 | gp pair on F: GPR2 gcc train01 (chain as-gpr2-b1; GPR2 perl makerand done, 262.62 s) | aw-s (spatial gcc, mcf, imagick, nab, lbm, xz on F; bkit75) → aw-sp (spatial full perl on F; bkit80) |
| 2 | idle: the switches are being parked (Addendum 119) | aw-sx (spatial x264 on F, ~10 h), relaunched after the park. aw-tp (temporal full perl) and the first x264 attempt never reached the UART: the 21:06 xz run's final aperture read timed out and its setsw(0) park with it, so the switches stayed at 217 and sw[0] gave the console TX to the tracer. No benchmark ran. The perl row reruns on board 3 |
| 3 | temporal x264 on F (chain av-b3, ~10 h, until ~03:00) | ay-b3 (the nested lane's speedtest1 event-23 ABBA boot on F, kit bkit75n, ~31 min; Addendum 117) → aw-tp on board 3 (the temporal full-perl row, rerun after board 2's failed boot; Addendum 118) → ax-b3 (pool A/B: temporal full perl on F with node_table_pool=4, bkit81, Addendum 115; now same-board against aw-tp) |

**Decisions waiting on readings:**
- **F is final (decided, Addendum 113).** The final-config rows on F are all queued (table above). The F dom kit for
  full perl is bkit80. The tag counters (B against F) are the pre-registered deciding reading for WHY F helps; they
  were read on 14 Nov: they explain F's gain (tag reads and writes halve or better, L1D misses and page fills
  unchanged; the RESULTS lines of 19:33 and 20:15).
- **gp-relative.**
  - **Rule (Addendum 110):** adopt it on both arms iff the perl ratio improves and perl×gcc improves.
  - **riscv-base silicon, RVGP against RVF3:** perl makerand -6.2 %, gcc -0.60 %.
  - **Capstone twins** (shift-1 user instructions, each arm 2/2), GPR2 against GPC: perl makerand -12.7 %, gcc
    +0.67 %, mcf +0.28 M, nab +0.13 %, imagick/lbm/xz flat. The riscv-base twin gives perl -8.7 %, gcc -0.42 %. So
    the twin predicts gcc worsens on Capstone; the board pair on board 1 decides.
  - **Capstone silicon on F, GPR2 against GPC:** perl makerand 287.74 → 262.62 s (-8.7 %), so the makerand ratio
    improves by about 2.7 %. gcc control GPC 1,992.35 s; GPR2 gcc is running. Addendum 114: if gcc's ratio worsens,
    gp is not adopted; otherwise a full-perl pair decides.
  - **GlobalMerge rule:** decided by riscv-base alone, pre-registered, outcome "LLVM default". It is
    `docs/plans/2026-11-14-gp-relax-globalmerge-rule.md` on lane branch `gp-relax-prereg` (707931298371, aa1e91b0628e,
    7125fd54bf4d), not on dev.
  - **If adopted:** both columns are rerun in full. The dev landing uses the local, unpushed branch
    runtime-gp-cursor-dev = origin/dev d07a38aedfaf + d75791670ef1 + bd6cf22e49b0 (the pthread-contract rewrite with 4
    mutants, NOT YET RUN). It needs a dev-matched virtual kit: kit-prof fails dev's own contract, and the QEMU
    0542c5556c kit dies in OpenSBI.
- **F2.** Flash it as its own A/B only if page fills (ev27) are frequent on F/B. Otherwise F2 becomes the base of the
  next set.

---

## 9. The nested-allocator workloads (lane `nested`)

**The lead's request.** For speedtest1, a pyperformance subset, single-user pgbench, tshark on a pcap, an FFmpeg
decode and a memcached load, measure each application's own workload, then explain the cost (record-cache hit rate,
CDERIVE/CREVOKE counts, revoked-subtree size, peak table memory). Give CREVOKE's worst case for subtrees from 1 to
10^6 nodes, including whether it is interruptible, and a real case such as a PostgreSQL context reset.

**Arms, the lead's rule:** exactly two per app, the app's lifetime patch applied vs not applied.
- Everything else is identical: malloc lifetimes are ON in both, and the bitstream is the batch.
- There is no pure-RISC-V arm.
- The lead agreed the plan with the lane directly.

**Record:** branch `nested-workloads`, `capstone/docs/ref/nested-allocator-workloads.md`. The latest commit,
3ae593fa513d, adds the gcc comparison.

**speedtest1 `--size 1`.**
- **Silicon** (board 2, ABBA, batch, bkit73): in-context +1.04 %, wall-clock +1.90 %, 32/32 answers. That is about
  1.7× gcc's SPEC temporal increment (+0.62 % in-domain, +0.61 % wall).
- **Kernel difference, not read:** the unprotected repeats were 2.47 % apart, over the pre-registered 2 %.
- **Twin:**
  - 28,216 CREVOKEs, revoked subtree mean 1.0001 and max 3;
  - the lifetime table is a fixed 4 MiB (node_flat_records 262144), with 4.8 KB of live records inside it;
  - the kernel +3.3 % comes from the SQLite patch's aReq array working through the per-syscall page scan;
  - the lifetimes themselves cost the kernel -0.6 M instructions and the program +0.39 % in-context.
- **Boot 1 (board 1) was VOID:** the empty `train/in` was never baked. It was fixed and redone on board 2.

**Next steps.** Each waits on the lead's word after speedtest1:
- **(a) speedtest1 `--size 100`:** twin, then one silicon pair, about 13 h of board time. The lead decides whether to
  run it.
- **(b) The apps, in order,** each first on the QEMU twin (slot 28: build both arms, check answers, count lifetimes),
  then one ~1 h ABBA silicon boot:

  | App | Patch |
  |---|---|
  | CPython pyperformance subset | cpython-3.13.7-0014, pymalloc lifetimes |
  | PostgreSQL pgbench | postgresql-17.0-0003, memory contexts |
  | tshark | wireshark-4.6.8-0001, wmem |
  | FFmpeg decode | ffmpeg-9.0.1-0003, pools |
  | memcached | memcached-1.6.45-0006, slabs and cache |

  Whether each builds on FRCDKMS native is each app's first step.
- **(c) Board slots:** about 1 h per app on the batch bitstream and bkit73, requested from the SPEC lane with a pushed
  pre-registration and a stage.
- **(d) HPM:**
  - event 23 (verdict-cache miss replays) in place of hpm8. The firmware define is ready: caplifive-opensbi branch
    `hpm-events-nested` 2c846cf7, `-DCAPSTONE_HPM_EVENTS_NESTED`, worktree `$H/wt-wrap-hpmnest`;
  - events 29-31 need a new bitstream and are held for the lead. The lane reads the lead's "record TLB" as the verdict
    cache, since no separate record TLB exists.
- **(e) CREVOKE worst case:** RTL simulation, no board. Also held until the lead has seen speedtest1.

## 10. Optimisations: done, in progress, rejected

**Done and on silicon** (effect on the problem rows):

| Optimisation | Where | Effect |
|---|---|---|
| DYN bypass v3, DTLB 64, SHRINKTO/CAPBOUND fix | bitstream final | perl -7.3 %, gcc -3.4 %, mcf -2.6 % |
| I$ 32 KiB 8-way | fdk | gcc -6.4 % (L1I misses halved); perl and mcf ≈ 0 alone, -3.6 % together with DTLB 64 on perl |
| global-lo fold + call relaxation (FR) | compiler | perl makerand -6.9 % instructions |
| standard + capability-base RVC (FRCDK) | compiler | perl -3.5 %, gcc -2.0 % |
| R-72 + lever (a) (batch) | bitstream | gcc -0.6 %, mcf -0.4 %, perl +0.5 %: the removed issue stalls reappear as scoreboard-full time behind memory |
| C8 (inline aligned 16-byte memcpy) + null-capability memset (FRCDKMS) | compiler + libc | gcc -3.2 %, perl makerand -2.1 %, mcf +0.3 % |
| tag-word cache 16 K + merge (F) | bitstream | mcf -10.8 % (run_cycles -11 %); gcc and perl pending |
| allocator: mapping cache + eviction + in-place realloc | runtime, both arms | basis of arm C |

**In progress:**
- F's remaining readings, and the gp-relative decision pair (§8). F2 is sealed and flash-ready.
- **Kernel cost of perl's full test.pl.**
  - **Caveat, 14 Nov evening.** Every Capstone full-perl twin this week (the FRCDKMS/spatial/gp "q" stages and the
    kernel table) ran with gcc's tuning set: arena_pin=1, resume_quantum=1, step_budget 10 s and retire_exit_sweep=2.
    The board's perl stages write no parameter. So the kernel and page figures below describe that tuned
    configuration, and are withdrawn as board-representative until the board-faithful census (census-bf, queued)
    reads. User instruction counts are believed unaffected, not yet checked. The pool pair is internally one-variable;
    its board effect is what bkit81 measures. (The gcc board stages DO carry that tuning: Addendum 116.)
  - **The measurement.** Capstone runs +4.73 G kernel instructions against riscv-base. That was measured at
    node_flat_records 524288, twice the board's table, on s1-FRCDKMSqk2 vs s1-RVBqk2. Corrected table:
    `$H/perf/kpin/perl-full-cap-vs-rv-v2.txt`.
  - **Correction.** The earlier "OpenSBI +0.56 G" was kernel memset running at the kernel's physical alias. OpenSBI
    itself is +0.3 M. memset is +588 M, of which about 354 M is the node table (318 execs × 8 MiB).
  - **Node table.** Each native exec cma_allocs and memsets a node table of node_flat_records × 16 B, then frees it.
  - **Per-arena census** (kernel a143573f, perf/census; FRCDKMS perl test-sw 2/2, 318 execs):
    - pinned pages swept 530,337 = 91 % of pgalloc;
    - 460 k of them were pinned by adaptive fault-ahead windows (image 126 k, heap 314 k, exchange 20 k), only about
      68 k by faults;
    - per exec: image 403 pages (data+bss is 283), exchange 64, heap 1,190, stack 6.3;
    - the 1 MiB stack is not pre-touched, so it is not a lever.
    - riscv-base's per-exec footprint is being measured; the proposed cut follows from it.
  - **Candidate: a pool of node tables.** rtl-oracle reading of rtl-ids55 afd35e449:
    - no record cache is keyed by table PA;
    - the LSU revocation cache is flushed on an edge of srevroot or the namespace flag (ex_stage.sv:1486);
    - the pool must also reset header words table[2] and table[3] (read on every allocation,
      capstone_rev_node.anvil:284-301);
    - virtual-namespace reads have no index bound, so records above the old bump pointer must be provably never
      written.
- Capstone ISA stage 2 (16-bit capability ops). The decoder sketch is on dev (option A: take the compressed FP-double
  slots; option B: a per-image decode mode). The choice is the lead's, and it comes last.

**Measured and not adopted:**
- arena_pin + image_end: pages +19 % at the board's exec rate.
- The "line-end jump" I$ lever: 0 cycles in a clean sim pair (two retractions on the way).
- C6 (fold constant offsets): about 7 M instructions, and the rest is blocked by the cursor-safety rule.
- C4 (movc) and C3 (spills): no concentrated idiom.
- Allocator trims inside the same policy: about 10 M, below noise.
- The pin-scan lever (pin_remaining/next_resident): 0.01 % of user on gcc, mcf and xz; not built. History doc
  a0ba0ad8130f, lane branch gp-relax-prereg.
- L1D 64 KiB: not timing-closed beside F2 (-5.5 ns, congestion 5). A 48 KiB 12-way variant (b9ab75c41) exists unbuilt,
  and is worth building only if the tag counters say L1D matters.

**Waiting on the lead:** CCSRRW privilege (S/U can write CPMP/ctvec in sim); CAPBOUND option (A) landing (QEMU
`r71-csmint-capbound`, capstone-sbi `vp2/csmint-r71`); the stage-2 ISA option; HPM 29-31.

---

## 11. Open problems and caveats

- **R-74 (open): tag-cache fill-time safety.** It is verified in simulation on ONE window instance per design (base,
  F, F2, B) with the FPGA's crossbar setting, and forging controls fire. It is not a proof for all timings, so no
  safety claim may rest on it yet. F2's credit ran on 325cf3bff; the built eeb654104 computes the same index bits at
  their own width, and tw-fill arm 1 was re-read on eeb654104 with its forging control.
- **R-73 (open): the debug module's system-bus writes bypass the tag adapter.** The exposure is nil iff the board's
  `reset halt` asserts ndmreset. One gdb read in a board gap settles it (`monitor riscv set_mem_access`, the reset
  config).
- **I-16:** the RVFI tap misrecords a store after an STC in sim traces. This is an instrument fault; memory is right.
- **Timing margin:** every bitstream is negative-WNS at signoff (above).
- **Twin vs board:**
  - many twins this week used node_flat_records 524288 against the board's 262144, which doubles the per-exec
    node-table cost;
  - the twin kernel's cmdline limit was 512 against the board's 1024.
  Match the board when a twin is meant to predict it. cmpc.py now WARNs when an arm's own validation is not N/N:
  perl test-sw read 1/2 on every Capstone twin until images-a0ed (dom_grant_root), and that figure was retracted.
- **Cross-board spread** is about 0.9 % on the same image; same-board run-to-run is 0.06 % (perl). perl full and x264
  ratios are same-board (board 3).
- **Simulation traps** (now in the rtl-sim skill):
  - the memory delay wraps mod 16;
  - cva6.py model reuse across define changes is UNPROVEN (both cited cases were the delay wrap). Read the define the
    model was built with from verFiles.dat;
  - the testharness crossbar used to be NO_LATENCY. Cycle baselines from before capstone-ariane b36989efd are on the
    old setting, which `+define+TB_XBAR_NO_LATENCY` restores;
  - the D$ replaces at random, so use the CSR_DCACHE flush, not a sweep, to evict. twc-reach-ld's pass-2 numbers and
    the two s07 cold tests from before 67b9ed442 / 69d5ac2c0 came from partial evictions and are superseded.

---

- **A failed switch park voids the NEXT boot on that board (14 Nov, Addendum 119).** BAKED_FINAL_APERTURES steps
  the switches after the rungs. When a `switch_state` wait times out, the `finally: setsw(0)` park can time out too.
  The next boot then connects at an odd value and transmits nothing. It looks like a dead board. Check the
  connect-time `switch_state` in driver.log: it must be all zero. The board lane is hardening the driver (verified
  park, refuse to cold-boot with sw[0] or sw[1] set).

## 12. Lessons from this campaign (each one cost a run or a retraction)

- A cross-arm "identical outputs" is not a pass: read each arm's own validation.
- A twin must use the board's module, parameters and table sizes.
- Dry-run every wrapper and chain with board calls stubbed. Two stages with one basename collide, and an empty
  `train/in` is not baked; `stage-check.sh` and `nat-run2.sh` now refuse both.
- `local A=x B=$A` in bash leaves B empty.
- Pre-registration timestamps come from `date`, not from memory.
- Count retractions as they happen; this session had several, all surfaced: the line-end jump cost, lever (a)'s first fix,
  the twin's test.pl kernel figure, the riscv-base images used in the lever census, C6.
