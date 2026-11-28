# Line B: finishing the bitstream and the final measurement campaign

Status: plan, 2026-11-25; corrected on 2026-11-26 after a three-part audit of the post-submission programme against
git (hardware, software, ports and measurements), whose load-bearing findings were checked adversarially before
this rewrite. Owner: the runtime lane. It continues the post-submission programme on `spec-silicon`
(`capstone/docs/plans/2026-11-20-post-submission-programme.md`, not on dev). By the lead's rule SPEC train runs only
at the very end.

## Where Line B stands (2026-11-26; each row is cited where it was read)

| Item | State | Record |
|---|---|---|
| Final Full seal **b42c8c644** (a5c45a46d + R-77's PRE fix; R-80's fix) | sealed and graded, clk_out1 +0.605, 150,303 LUT; kits on silicon on two boards as the first boots of each session, graded against the simulated vectors: board 3 b5b2n 21/21, crevoke4b 16/16, wapflu 1/1, Linux boot 3 P1-P4 ok; board 2 b5b2n 21/21, crevoke4b 16/16, wapflu 1/1, misscost both AS PREDICTED | 266e59a9a095; vectors fca456c2aad8, 1258f0833b60; readings board/meas-pw f47b2a686ce2 |
| Spatial-only seal **14f2ef81a** (b42c8c644 + CAPSTONE_SPATIAL_ONLY) | kits on silicon, board 1: every binding image reads its vector except b5b2-cptag position 22 (the H-fetch instrument, intermittent) | 74fe0e8d943e |
| Trap route, deletions, CLDRAW/CSTRAW/CPTAG, one-operand CREVOKE, CRENEW, CSMINT, interruptible CREVOKE | on silicon (CSMINT and 4(b) at N=1) | b5b2 plan; 5d45de462acf; d139d076f84e |
| Growth and failure rules | cause 30 and table growth on silicon at N=2: crevoke4b's b5b2-sbdrain-none/-fill/-fence/-mint read 0x1e 0x0 0x100 0x1, and -sret (U's CDERIVE raises 30, delegated to S, S grows and srets, the retry allocates) reads 0x1 0x0 0x1 0x0, on boards 3 and 2. NOGROW (the no-fault failure on a table that cannot grow) is on silicon too: b5b2-nogrow reads `0x0 0x7 0x100 0x0` beside its control b5b2-nogrow-ctl's `0x1e 0x7 0x100 0x1` (board 2, b42c8c644, 2026-11-26; kit b5b2-nogrow-bare-2026-11-26, pre-registered at fec2f2e16400), the pair differing only in the growth word | crevoke4b expected-vectors-b42c8c644.json; board/meas-pw f47b2a686ce2; NOGROW: board/meas-pw 371f7706e0e5 |
| Record cache (5.2) | model only; the P1 version was refuted. Line B decisions by the semantics owner's rule (BUILD only on a warm gain over 1 %, NO BUILD only under 1 % cold): gcc NO BUILD; perlbench NO BUILD on Full's g4mk row (CF-B5TL1-g4mk e82d42eb, 275.70 s: 0.163 % cold, about 6x under the line; the image traced from its booted bytes to the traced binary); mcf UNDECIDED until B5TL1-g1b's board row | 1c7617108984; perlbench e3fd98c877e5, 7338479886c4 |
| Linux on B5b-2 | boots; fast rdtime 119 cycles per op; msghdr loader on dev (eaee80a0fa70), memcached served on silicon | b5b2-linux kit; 098cae0f858b |
| Final kernels | Full (NO_HZ_IDLE, TRAP_TEST off): boot 13 on b42c8c644 4 of 4 as predicted (rdtime 119.7, getppid 1,267). +Spatial (+ RISCV_CAPSTONE_SPATIAL_ONLY): boot 14 on 14f2ef81a read as predicted. Both recipes and the option are on dev | runtime/b5b2-linux 098479a070eb, ec0ecb7c7427; dev d93ebefc8188, baf238ae8e8a |
| Trap kernel's record table | grows on demand by batches through cause 30; ceiling `sublet.node_max_pages` (0 = the format's 2^31 records); `sublet.table_growths` and `nogrow_sets` readable | `arch/riscv/mm/sublet-table.c:51-58,81-120` of the series |
| SPEC test, Full arms on b42c8c644 (qualification) | every stage read so far passes: B5T-g1b on boards 3 and 2 (mcf completes, 6,828.57 / 6,826.53 s, consistent with R-80 by Amendment 8), B5TL0-g1b, imagick on six g2b rows, gcc on three g3 rows; no cause-25 at an LDC. Owed: B5TL1-g1b (running), the g4mk rows, the three x264b, ldcexc2 | spectest RESULTS (runtime/b5b2-linux cc8f664b1480); Amendments 9-14 |
| SPEC test, +Spatial (SP-B5TL1, five images) | board 1 on 14f2ef81a: g2i imagick, g3 gcc, g4mk perlbench PASS; g1b running, x264b last. Twin: 8 of 8 stages, x264 by its data hashes (Amendment 13) | spectest Amendments 12-14 (4d544800ea81, e7b3046cf5e8, e660fa0215ae); RESULTS |
| One-operand applications | CPython on b42c8c644 +6.8 % (N=1). memcached requalified on b42c8c644 with the final Full kernel and the clock set from /init: 6 of 6, both arms, transcripts equal to native's. That boot was ungated, and it also changed the kernel, so it is not a one-variable rerun of the 0-of-6 boot. The rest are on superseded seals | nested-workloads 098cae0f858b; requalification PREREG 740970700451, RESULT 5e7f421c064c |
| Optimised gcc and perl ports | the Full + ports arm read on board 2, b42c8c644, final Full kernel, N=1, every stage PASS under v2 and v1: gcc t1.c stock 5.55 s, port 5.83 s (1.050); perl makerand.pl stock 275.70 s, port 279.75 s (1.015). The twin had predicted 1.176 and 0.958, so a twin ratio is not a predictor of a board ratio for these ports | nested-workloads PREREG c4732c42a52b, Amendment 1 1ab5179e06d3, RESULT a162a8aefb60 |
| Compiler, this week | pool-base-at-entry refuted as a default (nab +0.382 %, perlbench +0.369 %); LSRH (LSR run-time end + hoist) costs neither and keeps imagick -1.68 M; eight-stage twin gate pre-registered | compiler/imagick-loop-shapes 1430c473c2c6, 8bf6a86af81f |
| Directed simulation on b42c8c644's RTL | R-74 (tag-fill race): reached at every memory delay 0-15. 0 forged as built. Single and paired protection removals name the protection that decides each instance: the no-epoch write inside a fill for early beats (delays 0-1), poison or that write for poisoned ones. Not a proof for every timing. R-83's namespace case AS PREDICTED; its debug-mode case reduced by source, not measured | dev 214e2fd87624, da65421d4fd6 |

**Stale until the 26 Nov rewrite:** this doc named 904cd9c6b/28fcbb646 as current and said +Spatial existed only on P1.
ISSUES R-77 and R-80 now read ON SILICON.

## The arms of the final campaign

| Arm | Bitstream | Firmware | Kernel | Userland |
|---|---|---|---|---|
| Full | b42c8c644 | keep-tags, FW_FAST_RDTIME | the final Full kernel (boot 13) | B5TL1: compiler llvm-c9, the trap runtime with alloc-lean's mallocng series and the CRENEW mapping cache (runtime/trap-alloc-lean) |
| +Spatial | 14f2ef81a | as Full | the final +Spatial kernel (boot 14) | B5TL1 |
| **Full + ports (new)** | b42c8c644 | as Full | as Full | B5TL1 with the nested lane's gcc and perlbench ports in place of the stock stages. gcc's port (all-abc-bm-ls-tfl-phi) has run t1.c on 904cd9c6b, +9.03 % N=1, outside any SPEC stage. Perlbench's bodies port does not build on the one-operand SDK yet |
| Plain | rvbase 8db8f9cda (`caplifive_rvbase-matched-ic32k-dtlb64`), 32K 8-way L1D, the same as Line B's | stock OpenSBI 1.3.1, the same tree with the flag off | Linux 6.1, the same board config edits, NO_HZ_IDLE at HZ=100, no Capstone series (`trapk/plain/build-kernel-plain.sh`) | native RISC-V images with the same mallocng policy and no lifetime operations, built by the same compiler and flags apart from the Capstone ones |
| B5TL0 (ablation, optional) | as Full | as Full | as Full | B5TL1 without mapping reuse |
| LSRH (compiler, conditional) | b42c8c644 | as Full | as Full | B5TL1 rebuilt with LSRH, only if the eight-stage twin gate holds |

**The SPEC test rows now on the boards are qualification rows, not the campaign's timed rows.** They run the pinned
bring-up kernel tree `trapk/fpga-v1` (HZ_PERIODIC, TRAP_TEST on, no msghdr loader) with FW_FAST_RDTIME off, and so
does the +Spatial set (SP-B5TL1, tree `fpga-v1-spatial`). The timed rows use the final kernels and the fast-rdtime
firmware, so every arm's images are rebuilt on them before the campaign: a payload relink per image with the overlay
unchanged, each checked against its predecessor's overlay hashes.

**The matching rules,** each to be checked from the artifact and not from the recipe:
- **Tick.** The bring-up kernel is HZ_PERIODIC, so the final Full kernel moves to NO_HZ_IDLE at HZ=100, the measured
  configuration since PREREG Addendum 66. The bring-up chain stays as built and is labelled bring-up.
- **THP, SMP, KSM and modules** off on both arms: the trap-route kernel requires it (one possible CPU, no THP).
- **The DTS memory node** equal on both arms (`reg = <0x0 0x80000000 0x0 0x3c2d2000>`); read the Plain kit's DTB before
  the first Plain boot.
- **The L1D and I-cache** equal (32 KiB, 8 ways, 128-bit lines), from each bitstream's config package, as the gap audit
  read them for 88db40027, 28fcbb646 and 8db8f9cda.
- **The compiler identity** is printed beside every number.

## The order of work

Board time is the critical path; everything that needs no board runs beside it.

**Without a board, now (in parallel):**
1. runtime: spectest Amendment 12 (the +Spatial rows, from the twin); this doc; land the +Spatial kernel option and
   both final-kernel recipes after boot 14. **Done** (Amendments 12-14; dev d93ebefc8188, baf238ae8e8a).
2. nested + compiler: the perlbench bodies, HE and op-slab port made to build on the one-operand SDK; then the gcc
   and perlbench ports rebuilt on the trap SDK as B5TL1-paired SPEC test stages (make-stage.py), scanned
   for illegal operations, run on the twin at 962 MiB; then their pre-registration. memcached's requalification with
   the clock set before the server starts (twin first).
3. compiler: the LSRH eight-stage twin gate; pool-base built only on paths that use the base.
4. rtl: the record-cache model decision on Line B traces (cache RTL only on a modelled gain); directed simulation for
   R-74, R-81, R-82, R-83, and a core-level NOGROW arm; R-60/68/72/32's kits in simulation on b42c8c644; the assertion
   side-build ported to b42c8c644; ISSUES R-77/R-80 entries.
5. runtime: the campaign images, every arm's stages relinked on the final kernels with FW_FAST_RDTIME, overlays
   byte-identical to the qualification images; then their twin replay at 962 MiB. Images built (CF and CS for five
   groups, CFP for g3 and g4mk; `trapk/build-campaign.done`, `build-campaign-ports.done`). The twin replay is queued
   behind the compiler's LSHOFFR on the shared QEMU lock.
6. runtime: twin memory checks of every train input at 962 MiB, so train boots do not meet the OOM the first +Spatial
   twin replay met at 512 MiB.
7. compiler + board: the Plain SPEC images paired with B5TL1, and the Plain test rows on rvbase, pre-registered.
8. runtime: train stages for Line B. Inputs over the 26 MB image cap need an external initrd, which the trap payload
   lacks (its initramfs is built in). The table-size knob exists (above), so Perl's and gcc's train tables grow on
   demand; set `sublet.node_max_pages` only if a twin run shows a need.
9. runtime: the re-registrations named by Addendum 236 (Addendum 231, the ordering half of 232, t202s2's two rows),
   restated for the Line B arms.

**Added on 26 Nov, after the lead re-read the submitted paper (ebc1700a) against Line B.** These are the paper's
claims and rows that b42c8c644 has not yet answered:
10. compiler + rtl: **stage 2 on Line B**, by the lead's decision to try it and prove a benefit. The mode's home is
    a CSR `scapmode` (0x5C3), because Line B deleted CSRUNV; option B, proposed at compiler/stage2-trap-spec
    3477097bd5af. Order:
    - fetch bytes on the Line B twin (CRVC, ZCB and CRVCZ arms against LSHOFF);
    - a pre-registered cycle band and go/no-go: at least 1 % on perl or gcc, no stage worse than noise;
    - the decoder RTL, directed simulation, the synth-only gate, then a separate seal, b42c8c644 + decoder;
    - on the board, the on/off pair on that seal, plus the off image on both seals as the control.
    It replaces the campaign seal only if it reads AND the lead agrees. **Zcb with a capability base needs no new
    seal:** b42c8c644 decodes Zcb (config pkg:77, compressed_decoder.sv:152-153, rtl). Its pair therefore runs on
    the campaign seal: N=3 per arm, with P1's ZCB9 (perl -0.33 %) as the prior. The mode's home is being settled
    between compiler and rtl: a scapmode CSR, or a flop loaded from bit 0 of the srevroot write, which adds no
    flush for processes without the flag. The decoder output sits in a cone on the UNOPTFLAT list, so the
    synthesis-only gate comes first.
11. nested: **Perl's CRENEW variant on the trap route.** The variant was already measured on 24 Nov in the
    one-operand QEMU environment (oneop), with fixtures (nested-allocator-workloads.md, the three "perlbench item 3"
    sections):
    - heads: -11.68 M image instructions;
    - bodies: -0.31 %;
    - hash entries and op slabs: a net cost on suns.pl.
    What Line B adds is generations, so renewal happens in place, and the B5TL1 build. The arm is perl-fullren on the
    trap-route twin, pre-registered. The liveness-test half stays out: nothing decodes it, and it saves 0.032 % of
    instructions.
12. rtl: three section-5 claims, read at b42c8c644:
    - **Record TLB: present**, 16 entries, fully associative, flushed whole on a namespace flush
      (ex_stage.sv:2218).
    - **Record-path ordering against an in-flight release:** the mechanism is present (load_store_unit.sv:1838-1841
      rvc_set_kill, :1471-1475; commit_stage.sv:951-954), but no directed test covers it. The test being written
      is b5b2-relrace, pre-registered.
    - **Retirement at generation exhaustion:** present (capstone_sublet_life.sv:627-628). The unit-level case covers
      only CRENEW at the limit, and no test releases by CREVOKE at 2^32-2. A core-level test and a
      sublet-life-diff case are being written.
13. rtl + board: the hardware-cost table rebuilt from b42c8c644, 14f2ef81a and rvbase utilization. The spatial-only row
    becomes a measurement. The record-cache row has no hardware.
14. compiler + runtime: the embedded suites (BEEBS, RV8, the CoreMark-derived benchmark) as Line B images, then rows.
15. nested + runtime: the application cost rows (Table 3) on b42c8c644. Table 3's rows are SQLite, CPython,
    PostgreSQL, FFmpeg, mruby, Perl, memcached and PHP, all measured on P1. On b42c8c644:
    - only CPython has a prot-against-stock cost reading (nested-cpymem, +6.8 %, pyperformance on the msghdr kernel);
    - memcached has a qualification (6 of 6), not Table 3's mc-crusher cost;
    - SQLite and mruby have rows only on the superseded seal 904cd9c6b;
    - PostgreSQL and FFmpeg have no Line B row at all.
    The large-subtree revoke time is owed too.

**The long pole is x264.** Its test run needs more than about 12,180 s on b42c8c644 (Amendment 11), so three runs
on each of four arms come to about 40 board-hours for x264 alone. x264b images therefore go into each board's longest
unattended window, one image per boot. Every other stage of an arm fits in one boot of a few hours.

**Boards, in order:**
- Board 3 (b42c8c644): finish the Full SPEC test session (g2b remainder, g3, g1b, g4mk, ldcexc2, x264b).
- Board 2 (b42c8c644): nested-corpus (SQLite and PHP engine repros); boot 13 is read.
- Board 1 (14f2ef81a): boot 14; then the +Spatial SPEC test rows (Amendment 12), x264b last.
- Then, as their pre-registrations land: Full + ports (**read**, board 2: four boots, above); the LSRH board form
  (LSHOFF, LSRH, LSHOFF2 in one session, lbm off); Plain rows on rvbase.
- Every boot runs preflight-board-run.sh first and keeps its exit line. Three boots on 26 Nov did not (nested-mcreq,
  CF-B5TL1-g3, CFP-B5TL1-g3: it started at 07:42:59Z, before the preflight runs at 07:44Z), and their rows say so. A preflight BLOCK is proceeded past only by a ruling recorded in the row, e.g.
  C10's 20 MB budget for 21.66 MB payloads already proven on the same seal.
- Last: three runs per arm, board-constant pairs, drift-balanced order, intervals; then SPEC train for all arms.
  SQLite --size 100 and the Fig. 10 Perl suns.pl row go here. The paper measures gcc on its **train** input, so
  gcc's SPEC test rows here are qualification only, and gcc's paper-footing row comes with train.

## Dispositions written now

These are new: no earlier commit disposes of any of them, and the programme doc (spec-silicon) still lists each as
open.
- **Line A bitstream 3 (the performance batch on P1)** does not go as a P1 bitstream. One lever was promoted into it
  (the 48 KiB L1D, 3990bdb753a5) and then dropped (train Addendum 237). Its other levers have no P1 build: the
  verdict-cache size, the record cache, store-buffer forwarding, record-TLB LRU and pcc_hold. They are decided by the
  Line B model, and any one that shows a modelled gain is built on b42c8c644. On Line B, store-buffer forwarding is
  already read as not relied on (capstone-ariane 88db40027).
- **The I-cache 64-byte-line variant** is rebased on b42c8c644 only if board time remains after the campaign. It is
  not part of the final rows.
- **Stage 2 instructions**: the lead ruled on 26 Nov to try them on a separate seal and prove a benefit (item 10). They stay out of the campaign seal unless that experiment reads and the lead agrees.
- **C2 and C3** stop by the compiler lane's own gates: no safe increment reaches 0.2 % (1c4f472055e6), and C3's
  excess is under its 0.5 % gate (1eaf15006691). These are gate results, not a ruling by the lead.
- **Huge pages** are not a Line B lever, because the trap-route kernel requires THP off (the matching rules above).
  They stay open on P1 (programme: Plain mcf with THP read -0.69 %).
- **The native route's sockets and epoll** are not needed for Line B's server row: memcached is served over the trap
  route on silicon (nested-workloads 098cae0f858b).
- **QEMU runtime cases 29-31** belong to P1's flat-table route (design/virtual-capstone/isa.md:524-526) and do not
  bear on Line B.
- **dev's default build is not the measured configuration.** For an SDK or port build, the measured codegen flags sit
  behind CAPSTONE_APPLICATION_MEASURED_CODEGEN, which is off (redundant-shrink is the exception, on). The measured
  images got their flags from the replayed `make.out` lines. The measured runtime (alloc-lean) is not on dev. Landing
  it is the lead's decision.

## Audit and decisions, 2026-11-28 (the lead: "audit the current state thoroughly, then decide")

Each row was re-read at its source this morning.

| Item | State on 28 Nov | Record |
|---|---|---|
| Campaign (P1-P4) | Phases A, C, C2 and board 2's long rows read: P1 on 42 boots, P2 everywhere, P3 on imagick and gcc (perlbench undecided, inside the noise), P4 within 0.06 % over three boards. The qualification perl gap is not reproduced (1.81 s of 11.02 s). Phase B runs on boards 1 (CS) and 3 (CF) for about a day more | runtime/b5b2-linux cf82e5bc09b5 |
| Item 10, stage 2 | Seal N 796af23d9 built clean (WNS +1.796, LUT 149,891). On board 2: GO by gcc t1.c, −1.62 %; perlbench, imagick, nab flat; outputs equal on every boot (checked here with a positive control); the seal's cost on perlbench is not detected: +1.26 s against an allowance of 3.26 s that is the K reference's own range. N has not run the silicon kits that qualified b42c8c644 (b5b2n, crevoke4b, wapflu) | compiler/stage2-trap-spec 30a2f27261b8, 249d7eeb8d65 |
| Placement control | code placement alone moves perlbench 11.38 s and nab 2.21 s on b42c8c644 | c89f177d0403 |
| Record cache, mcf | LB 1024x8 NO BUILD on silicon's own tick positions (n 7.49 M, 5.45x under the line). RC stays UNDECIDED: no owner of c's upper bound | dev a7038b49974f; runtime/b5b2-linux b341d69fcca6 |
| Item 7, Plain | images built 25 Nov (tests/line-b-plain, dev 22145821659d) and smoke-tested on the twin; **never booted on rvbase**; their four payloads use stock OpenSBI without FW_FAST_RDTIME and put x264 with imagick, so they do not meet this plan's matching rules (the Full arm runs FW_FAST_RDTIME; x264 needs an image of its own). The firmware supports FW_FAST_RDTIME without keep-tags (fw_base.S:932), and the DTS memory nodes are equal | dev 22145821659d |
| Items 6, 8, 9, 14 | no progress since 26 Nov (train memory checks, train stages, the re-registrations, the embedded suites) | git log --all since 25 Nov |
| Item 13 | the hardware-cost table rebuilt from the sealed builds | land/rtl-b5b2-kitsil 26801c60085d |
| Landing | 51 commits on runtime/b5b2-linux are not on dev (kits b5b2-spectest, -campaign, -ticklog and others); the DECLARED-checker fix waits on preflight-declared-multi aca4580de705 | |
| Incidents open | two JTAG debug-module busy timeouts on two boards (N = 2), both recovered by one rerun; unattributed | campaign RESULTS |

**Decisions:**
1. **Stage 2 does not replace b42c8c644 for this campaign.** It reads, but on one stage of four (gcc −1.6 %), with
   perlbench, imagick and nab flat and lbm, mcf, xz and x264 never built for the mode. Replacing the seal would
   discard the Full arm's rows (about 60 board-hours with x264), needs N's own kit qualification, and the campaign
   kernel would have to become K. Stage 2 stays a separate, measured result. It can be revisited after the
   campaign, as its own arm on N, if the lead wants it in the paper.
2. **Board 2 first takes the compiler's x264 bracket on N** (FLASH-N, X-OFF, X-ON, X-OFF2, FLASH-B42, about 13 h),
   once its pre-registration is pushed: the compiler lane reports the lead's answer on stage 2 as "measure more
   first". **The Plain arm goes to the first board free after that** (board 1 or 3 when Phase B ends, about 20:00Z,
   or board 2 after the bracket): the baseline every overhead row needs, and the plan's next board item. Before any
   Plain boot: the Plain-matched images relinked on the Plain kernel with FW_FAST_RDTIME=y (KEEP=n), in the campaign's
   groups (g1b, g2i, g3, g4mk, x264b), overlays byte-identical to the 25 Nov images; a twin replay of all eight
   stages; a pre-registration with predictions. On the board: a flash to rvbase 8db8f9cda, a first boot that is
   the Plain kernel's Tier 1 and the size proof, then the short groups three times and the long groups as board
   time allows. The runtime lane relinks (item 5); the compiler lane, which built the native images, is asked to
   confirm them first.
   - **Changed the same day:** the bracket's pre-registration was not pushed while board 2 stood idle, so the
     Plain arm (pre-registered at runtime/b5b2-linux 932491ce4360) took board 2, and the bracket takes the first
     board free after Phase B. The rule: an idle board goes to the first pushed pre-registration.
3. **Owed records:** the spectest RESULTS x264 rows with the UNGATED disclosure; the campaign, ticklog, placement and
   stage-2 readings into `docs/ref/fpga-silicon-measurements-for-paper.md`; the kits land on dev, squashed per kit,
   after their sessions end.
4. **The JTAG busy timeout** becomes an ISSUES entry at a third instance, or now if one fails its rerun.

## Questions that are the lead's (none of them blocks the order above)

- Landing the measured runtime and the measured codegen default on dev. The runtime lane recommends both.
  - Codegen: dev's compiler with the measured flags reproduces 10 of 11 measured images byte for byte, and gcc by its
    version string only (history 20-11-2026_18-28-19).
  - Runtime: on Line B that means trap/eval plus runtime/trap-alloc-lean (B5TL1). On b42c8c644 it is the faster arm:
    gcc test 5.76 s against 7.13-7.15 s, imagick -1.8 %. Part of trap/eval is the collaborator's series, so it lands
    through its owner.
- Stage 2: option A or B for the mode (proceeding on B), and whether a stage-2 seal may replace the campaign seal if
  it reads.
- Any paper wording that follows from the Line B rows.
