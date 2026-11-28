# Untagged CINCOFFSET, CINCOFFSETIMM and SCC in the B5b-2 SPEC images: results (2026-11-25)

Pre-registered in PREREG.md (1dae2161311f; amendment 1 ac236e3d4c26, pushed before this run). There was one boot for
all three arms on the counting twin (capstone-qemu sublet/untagged-arith-count 741be49bb0, binary eadd3a36316605af,
-m 963M). It held the lock 19:01:44-20:29:54 (5,289 s).

## The control (prediction 1): exact

- ctl0 added 0.
- ctl1 added all=1750, cinc=500, cincimm=1000, scc=250, all in U: exactly untag-ctl.dom's untagged loops.
- Its 300 tagged CINCOFFSETIMM were not counted.

## Per stage

The vaddr is pc - entry + 0x10000, where entry is the kernel's "enters" line for that process. "at" is the
function in the arm's unstripped image.

| stage | reading | first op | vaddr | at |
|---|---|---|---|---|
| B5T-519.lbm_r | 0; ran to its end | - | - | - |
| B5TL0-519.lbm_r | 0; ran to its end | - | - | - |
| B5TL1-519.lbm_r | 0; ran to its end | - | - | - |
| B5T-505.mcf_r | 0; ran to its end | - | - | - |
| B5TL0-505.mcf_r | 0; ran to its end | - | - | - |
| B5TL1-505.mcf_r | 0; ran to its end | - | - | - |
| B5T-544.nab_r | first op; the stage ended 1 s later | CINCOFFSET: `cincoffset a0, a0, a1` | 0x1fc5c | free_ivector+0x2 |
| B5TL0-544.nab_r | first op; the stage ended 1 s later | CINCOFFSET: `cincoffset a0, a0, a1` | 0x1fc28 | free_ivector+0x2 |
| B5TL1-544.nab_r | first op; the stage ended 1 s later | CINCOFFSET: `cincoffset a0, a0, a1` | 0x209c4 | free_ivector+0x2 |
| B5T-557.xz_r | 0 in the first 600 s (a lower bound on clean) | - | - | - |
| B5TL0-557.xz_r | 0; ran to its end | - | - | - |
| B5TL1-557.xz_r | 0; ran to its end | - | - | - |
| B5T-500.perlbench_r | 0 in the first 600 s (a lower bound on clean) | - | - | - |
| B5TL0-500.perlbench_r | 0 in the first 600 s (a lower bound on clean) | - | - | - |
| B5TL1-500.perlbench_r | 0 in the first 600 s (a lower bound on clean) | - | - | - |
| B5T-502.gcc_r | 0; ran to its end | - | - | - |
| B5TL0-502.gcc_r | 0; ran to its end | - | - | - |
| B5TL1-502.gcc_r | 0; ran to its end | - | - | - |
| B5T-538.imagick_r | first op; the stage ended 1 s later | CINCOFFSETIMM: `cincoffsetimm a0, a0, 0x2` | 0x4ad62 | FrameImage+0xc4a |
| B5TL0-538.imagick_r | first op; the stage ended 1 s later | CINCOFFSETIMM: `cincoffsetimm a0, a0, 0x2` | 0x4ad22 | FrameImage+0xc4a |
| B5TL1-538.imagick_r | first op; the stage ended 1 s later | CINCOFFSETIMM: `cincoffsetimm a0, a0, 0x2` | 0x4babe | FrameImage+0xc4a |
| B5T-525.x264_r | first op; the stage ended 1 s later | CINCOFFSET: `cincoffset a3, a3, s4` | 0x4967a | x264_slicetype_mb_cost+0x146 |
| B5TL0-525.x264_r | first op; the stage ended 1 s later | CINCOFFSET: `cincoffset a3, a3, s4` | 0x49646 | x264_slicetype_mb_cost+0x146 |
| B5TL1-525.x264_r | first op; the stage ended 1 s later | CINCOFFSET: `cincoffset a3, a3, s4` | 0x4a3e2 | x264_slicetype_mb_cost+0x146 |

## Against the predictions

- **Prediction 2 holds.** B5T's first ops are the board-3 fault pcs exactly: imagick vaddr 0x4ad62 (CINCOFFSETIMM)
  and x264 vaddr 0x4967a (CINCOFFSET). So the board faults are this rule's first execution in each process.
- **Prediction 3 holds.** imagick and x264 have a first op on all three arms. In every arm it is the same
  instruction of the same function (FrameImage+0xc4a, x264_slicetype_mb_cost+0x146), at each image's own address.
- **Prediction 4 (no outcome predicted).**
  - nab has a first op on all three arms: free_ivector+0x2, `cincoffset a0, a0, a1`, nab's free(v + nl) with v
    NULL. So nab faults with cause 24 on 904cd9c6b/88db40027.
  - lbm, mcf and gcc read 0 on all arms and ran to their ends.
  - xz read 0 and ran to its end on B5TL0 and B5TL1; on B5T it read 0 in the first 600 s.
  - perlbench read 0 in the first 600 s on all three arms.

## What this does and does not say

- A stage reading "0; ran to its end" executes no untagged CINCOFFSET/CINCOFFSETIMM/SCC on the test input. It is
  predicted not to fault from this rule on the vctx-gated FLU.
- "0 in the first 600 s" bounds only the run's start.
- The first-op stages are cut 1 s after their first op, so their totals are not measured. The combined seal's
  re-read (28fcbb646, the rule made reference-exact) shows which stages pass.
- This counts the rule's executions on the reference model. It says nothing about any other fault a stage may meet
  on silicon (for example mcf's wedge on 904cd9c6b).
