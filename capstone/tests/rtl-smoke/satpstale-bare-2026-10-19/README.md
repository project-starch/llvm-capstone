# S-29 and S-23's fetch face as bare board arms: pre-registered vectors for tree A (resident) and tree B

**What this folder is.** Seven M-mode board images that read the stale-translation window of S-29 on silicon, and
with it S-23's fetch-face signature. They are built from one source, `tests/va-satpstale.S`, a copy of
capstone-ariane `virtual-profile-2-c`'s arm, which carries thirteen builds; these are the seven that are useful on
a board. Every vector below was taken by simulating **these images** on each RTL, never from the arm's own
simulation readings, because the recorder's `CAPPRINT` creates its own capability and the board wrapper puts
`_start` a page above RAM_BASE.

**It does not touch `vp2-bare-2026-10-17`.** That folder is live production for the tree A session's arms gate,
which fails unless its table gives exactly 25 arms. These arms have their own folder, their own id characters
130-136 and the prefix `sv-`.

**Why bare arms can do this at all:** eight of vp2's existing 25 bare arms already set `satp` and four already issue
CSRUNV, so translation and virtual contexts are not a new class of arm here. `pick-regs.py` returns x16/x8/x4 for
this source, the same three the `va-runv` family uses, so the arm names none of the recorder's scratch registers.

## The vectors

Format as `results/rtl-sim-vectors.txt`: count then values. `board_rec` is 0x80017000 in all seven.
Every arm ends in a spin loop, so each reaches the 2,000,000-cycle limit and the harness prints FAILED with
tohost 2147483647. That is normal for this set; the vector is the recorder's contents.

| image | tree A `8522d72fe` (RESIDENT, no flush) | tree B `8de3c48cf` (with the flush) |
|---|---|---|
| `sv-stale` | 6 \| 0xaaaa **0xaaaa** 3 0xb **0xaaaa** 0x55 | 6 \| 0xaaaa **0xbbbb** 3 0xb **0xaaaa** 0x55 |
| `sv-stale-f` | 6 \| 0xaaaa **0xaaaa** 3 0xb 0xaaaa 0x55 | 6 \| 0xaaaa **0xbbbb** 3 0xb 0xaaaa 0x55 |
| `sv-stale-b` | 6 \| 0xaaaa 0xbbbb 3 0xb **0xbbbb** 0x55 | 6 \| 0xaaaa 0xbbbb 3 0xb **0xaaaa** 0x55 |
| `sv-stale-bf` | 6 \| 0xaaaa 0xbbbb 3 0xb 0xaaaa 0x55 | 6 \| 0xaaaa 0xbbbb 3 0xb 0xaaaa 0x55 |
| `sv-stale-bn` | 6 \| 0xaaaa 0xcccc 3 0xb 0xaaaa 0x55 | 6 \| 0xaaaa 0xcccc 3 0xb 0xaaaa 0x55 |
| `sv-fetch` | 5 \| 0xaaaa **1** 0x80006000 **0x80016000** 0x55 | 5 \| 0xaaaa **0xc** 0x80006000 **0x80006000** 0x55 |
| `sv-fetch-c` | 5 \| 0xaaaa 0xc 0x80006000 0x80006000 0x55 | 5 \| 0xaaaa 0xc 0x80006000 0x80006000 0x55 |

## How to read them

The second value of an `sv-stale*` arm is what the CONTEXT read; the fifth is what the CALLER read as its first
access after the escape. 0xaaaa is the caller's chain, 0xbbbb the context's, 0xcccc a constant the context stores
when it never touches the probe address at all.

- **The entry direction** is `sv-stale` and `sv-stale-f`: on tree A the context reads **the caller's** page, 0xaaaa,
  where its own table maps 0xbbbb. On tree B it reads 0xbbbb. Those two arms differ only in a fence the caller
  executes AFTER the escape, which cannot affect the context, so they agree with each other on both bitstreams.
- **The escape direction** is `sv-stale-b` against `sv-stale-bf`: the two differ by exactly one `sfence.vma`, the
  caller's own, after the escape. On tree A the unfenced arm reads **the context's** page, 0xbbbb; on tree B it
  reads 0xaaaa. The fenced arm reads 0xaaaa on both, which is what makes the difference a staleness reading rather
  than a mapping mistake.
- **`sv-stale-bn` attributes it:** the context never touches the probe address, so nothing can install an entry for
  it, and the caller reads its own page on both bitstreams. Against `sv-stale-b` on tree A, these two differ in
  exactly one thing, whether the context accessed that address.
- **The fetch face** is `sv-fetch` against `sv-fetch-c`. The caller stays at S throughout and drops to U for its
  first fetch only. On tree A it takes **cause 1**, an instruction access fault, with the third value the faulting
  virtual address and the fourth a **PHYSICAL** address that differs from it: that is S-23's signature, from a stale
  translation alone. On tree B it takes **cause 12** with the fourth value EQUAL to the third, i.e. the virtual
  address, which is the ordinary permission refusal of a clean walk.
  **A reader needs no address table to tell these apart: on tree A the last two values DIFFER, on tree B they are
  EQUAL.**

## What a difference from these vectors means

A value not in the column for the resident bitstream is a finding, not a tolerance. The arms are deterministic:
the same seven images on the same RTL gave these values with `+ntb_random_seed=1` and a fixed memory delay.

**What these arms do NOT test.** They say nothing about S-27, S-28, R-57 or Q-24; those are the Linux steps' and
`pym0`'s business. They also do not reproduce S-23's DATA face, whose refused address is covered by a loaded
region and whose mechanism is open; the candidate there is the revocation-node validity term and it needs its own
arm.

## Rebuilding

    CAPSTONE_LLVM_BIN=<the built llvm bin> bash build.sh

`VP_SRC=<capstone-ariane>/verif/tests/custom/capstone` additionally checks the source against the RTL tree.
Image identity is in `images/SHA256SUMS`; match an arm by hash, never by name.
