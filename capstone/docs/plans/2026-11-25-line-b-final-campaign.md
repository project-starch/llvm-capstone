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
| Record cache (5.2) | model only; the P1 version was refuted, the Line B port is in progress | 1c7617108984; 68cce3377964, b35fb24d8e97 |
| Linux on B5b-2 | boots; fast rdtime 119 cycles per op; msghdr loader on dev (eaee80a0fa70), memcached served on silicon | b5b2-linux kit; 098cae0f858b |
| Final kernels | Full (NO_HZ_IDLE, TRAP_TEST off): boot 13 on b42c8c644 4 of 4 as predicted (rdtime 119.7, getppid 1,267). +Spatial (+ RISCV_CAPSTONE_SPATIAL_ONLY): boot 14 owed, attempt 1 NO READING (infra) | runtime/b5b2-linux 098479a070eb; board/meas-pw f47b2a686ce2 |
| Trap kernel's record table | grows on demand by batches through cause 30; ceiling `sublet.node_max_pages` (0 = the format's 2^31 records); `sublet.table_growths` and `nogrow_sets` readable | `arch/riscv/mm/sublet-table.c:51-58,81-120` of the series |
| SPEC test, Full arms on b42c8c644 | board 3's session in progress: B5T-g2b imagick PASS; x264 moved to x264b images at 43,200 s (Amendment 11) | spectest PREREG Amendments 9-11 |
| SPEC test, +Spatial (SP-B5TL1, five images) | built; twin 7 of 8 stages rc 0 with specdiff PASS at the board's 962 MiB, x264 running | spectest Amendment 12 (to come) |
| One-operand applications | CPython on b42c8c644 +6.8 % (N=1); memcached served on both arms, 0 of 6 by its criterion because the board has no RTC; the rest on superseded seals | nested-workloads 098cae0f858b |
| Optimised gcc and perl ports | advanced on the twin and P1; **in no arm of this campaign** | nested-allocator-workloads.md |
| Compiler, this week | pool-base-at-entry refuted as a default (nab +0.382 %, perlbench +0.369 %); LSRH (LSR run-time end + hoist) costs neither and keeps imagick -1.68 M; eight-stage twin gate pre-registered | compiler/imagick-loop-shapes 1430c473c2c6, 8bf6a86af81f |

**Stale until this rewrite:** this doc named 904cd9c6b/28fcbb646 as current and said +Spatial existed only on P1.
ISSUES R-77 ("NOT YET SYNTHESISED") and R-80 ("NOT YET SYNTHESISED OR ON SILICON") still need their entries updated
(rtl lane).

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
   both final-kernel recipes after boot 14.
2. nested + compiler: the perlbench bodies, HE and op-slab port made to build on the one-operand SDK; then the gcc
   and perlbench ports rebuilt on the trap SDK as B5TL1-paired SPEC test stages (make-stage.py), scanned
   for illegal operations, run on the twin at 962 MiB; then their pre-registration. memcached's requalification with
   the clock set before the server starts (twin first).
3. compiler: the LSRH eight-stage twin gate; pool-base built only on paths that use the base.
4. rtl: the record-cache model decision on Line B traces (cache RTL only on a modelled gain); directed simulation for
   R-74, R-81, R-82, R-83, and a core-level NOGROW arm; R-60/68/72/32's kits in simulation on b42c8c644; the assertion
   side-build ported to b42c8c644; ISSUES R-77/R-80 entries.
5. runtime: the campaign images, every arm's stages relinked on the final kernels with FW_FAST_RDTIME, overlays
   byte-identical to the qualification images; then their twin replay at 962 MiB.
6. runtime: twin memory checks of every train input at 962 MiB, so train boots do not meet the OOM the first +Spatial
   twin replay met at 512 MiB.
7. compiler + board: the Plain SPEC images paired with B5TL1, and the Plain test rows on rvbase, pre-registered.
8. runtime: train stages for Line B. Inputs over the 26 MB image cap need an external initrd, which the trap payload
   lacks (its initramfs is built in). The table-size knob exists (above), so Perl's and gcc's train tables grow on
   demand; set `sublet.node_max_pages` only if a twin run shows a need.
9. runtime: the re-registrations named by Addendum 236 (Addendum 231, the ordering half of 232, t202s2's two rows),
   restated for the Line B arms.

**The long pole is x264.** Its test run needs more than about 12,180 s on b42c8c644 (Amendment 11), so three runs
on each of four arms come to about 40 board-hours for x264 alone. x264b images therefore go into each board's longest
unattended window, one image per boot. Every other stage of an arm fits in one boot of a few hours.

**Boards, in order:**
- Board 3 (b42c8c644): finish the Full SPEC test session (g2b remainder, g3, g1b, g4mk, ldcexc2, x264b).
- Board 2 (b42c8c644): nested-corpus (SQLite and PHP engine repros); boot 13 is read.
- Board 1 (14f2ef81a): boot 14; then the +Spatial SPEC test rows (Amendment 12), x264b last.
- Then, as their pre-registrations land: Full + ports; the LSRH board form (LSHOFF, LSRH, LSHOFF2 in one session,
  lbm off); Plain rows on rvbase.
- Last: three runs per arm, board-constant pairs, drift-balanced order, intervals; then SPEC train for all arms.
  SQLite --size 100 and the Fig. 10 Perl suns.pl row go here.

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
- **Stage 2 instructions** stay out of the final bitstream unless the lead rules otherwise.
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

## Questions that are the lead's (none of them blocks the order above)

- Landing the measured runtime (alloc-lean) and the measured codegen default on dev.
- Stage 2 instructions in a bitstream.
- Any paper wording that follows from the Line B rows.
