# Untagged CINCOFFSET, CINCOFFSETIMM and SCC executions in the B5b-2 SPEC images: pre-registration (2026-11-25)

**What is counted.** CINCOFFSET, CINCOFFSETIMM and SCC executions whose x[rs1] is untagged (and, for CINCOFFSET and
SCC, whose x[rs2] is untagged too). These are the executions the reference model adds silently. capstone-ariane
904cd9c6b and 88db40027 trap on them in protected U with cause 24, because their FLU applies the rule only under vctx
(capstone_flu_unit.anvil:34/:68/:109).
- The count is per stage, for every image of the three B5b-2 arms (B5T, B5TL0, B5TL1) on the SPEC test inputs.
- The per-stage counts say which chain stages fault on those bitstreams before the board reaches them. They are also
  the number of executions the combined seal 28fcbb646 must add silently.

## Instrument

capstone-qemu sublet/untagged-arith-count a9f9abfdf2, based on 3a9fc39167, the source of the trap-route twin's binary.
- With CAPSTONE_UNTAGGED_ARITH_COUNT=<file> set, a probe before each of the three helpers counts exactly the helper's
  silent branch.
- The file gets one FIRST line per (pc, op) with the running total, and a TOTAL line that a main-loop timer rewrites
  every 100 ms. Unset, the translators emit what they did.

Smoke boot (instrument check, not this run), untag-ctl.dom (bf70220ccaf98c8e):
- argument 0 read all=0;
- argument 1 read all=1750 cinc=500 cincimm=1000 scc=250 u=1750;
- the 300 tagged CINCOFFSETIMM were not counted;
- the first FIRST line was the loop's cincoffsetimm.

## Platform and procedure

The trap-route twin of drive2.py: the trapk linux-6.1 kernel, sbi-keep-run fw-y, -m 963M, sublet.dom_binfmt=1,
the board buildroot rootfs, repacked root-owned.

The per-arm images are the board stages of b5b2spec/board-stages/<ARM> with the boards' run.sh, in the order imagick,
lbm, mcf, nab, xz, perlbench, gcc, x264. x264's ldecod line is replaced by an mv of the host-decoded BuckBunny.yuv,
whose sha256 c3a3d3e2... equals the board's own ldecod output. Each stage is bounded at 1800 s.

There is one boot per arm, each run under the host-wide QEMU lock. Every phase is bracketed by @@UT-BEGIN/@@UT-END
lines with 2 s of guest quiet on each side, and the driver copies the TOTAL line at each marker into boundaries.txt.
A stage's count is END minus BEGIN. A stage cut at its bound gives a LOWER bound.

## Predictions

1. **Control (first phase of every boot):** ctl0 adds 0. ctl1 adds exactly cinc=500 cincimm=1000 scc=250, all in U.
   Otherwise the boot's counts are void.
2. **B5T, the two stages that faulted on board 3:** the twin's first counted execution in the process must be the
   silicon fault pc. vaddr = pc - entry + 0x10000, where entry is the kernel's "enters" line for that process.
   - imagick_r: vaddr 0x4ad62, op 1 (CINCOFFSETIMM);
   - x264_r: vaddr 0x4967a, op 0 (CINCOFFSET).
   Any other first pc falsifies "the board fault is this rule's first execution".
3. **imagick and x264 on all three arms:** count > 0.
4. **Every other stage:** no count is predicted (exploratory). A stage that reads 0 is predicted to pass this rule on
   904cd9c6b/88db40027. A stage that reads > 0 is predicted to fault with cause 24 at its first FIRST pc.

## Amendment 1, before the run (2026-11-25), at the semantics owner's request: less lock

The question is whether each stage has a silent untagged op, and where the first one is. Totals are not needed now:
the combined seal's re-read shows which stages pass.

**What was read under the original procedure before it was stopped.** The B5T boot ran its control and its imagick
stage. It was stopped in lbm, at the semantics owner's request, about 10 minutes into the boot.
- Control: ctl0 +0; ctl1 +1750 (cinc 500, cincimm 1000, scc 250, u 1750). Prediction 1 holds.
- B5T imagick: +1103 (cinc 60, cincimm 1043). The first FIRST line in the process is pc 0x3ff7c3ad62 op 1, and the
  process entered at 0x3ff7c00000. That is vaddr 0x4ad62, CINCOFFSETIMM, the board-3 fault pc, so prediction 2
  holds for imagick.

**Procedure from here.**
- One boot for all three arms. Every arm's lbm, mcf, nab, xz, perlbench and gcc run first, then every arm's imagick,
  then every arm's x264.
- x264's yuv is now copied rather than moved, since three stages share it.
- The host ends a stage by writing KILLSTAGE to the serial console, which a watcher in /init turns into a kill of the
  stage's process group:
  - 1 s after the stage's first counted op;
  - or 600 s after the stage begins with none. That stage is recorded as "0 in the first 600 s (a lower bound on
    clean)".
  The guest's own 660 s bound is the backstop.
- The stage's first op is read from the counter's new LEAD line (capstone-qemu 741be49bb0, binary eadd3a36316605af).
  LEAD logs any counted op that follows at least 1 s without one, so it fires even when FIRST is silent because an
  earlier arm's image used the same pc.
- Smoke, kill path: a stage running untag-ctl.dom then sleeping 700 s was killed (rc 137) 1 s after its LEAD line,
  n=1751. FIRST was silent for it, since the control's earlier run had used the same pc.

**Predictions, unchanged in substance.**
1. The control reads as in prediction 1.
2. B5T x264's first op is vaddr 0x4967a, CINCOFFSET. B5T imagick reads vaddr 0x4ad62 again.
3. imagick and x264 have a first op on all three arms.
4. Each of the other 18 stages reads either a first op (predicted cause 24 at that pc on 904cd9c6b/88db40027) or "0 in
   the first 600 s". No outcome is predicted for them.
