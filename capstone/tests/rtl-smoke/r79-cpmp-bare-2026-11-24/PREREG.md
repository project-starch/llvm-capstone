# R-79's CPMP side on silicon: pre-registered 2026-11-24, before any board run

The phase this grades runs on one board after flashing the bitstream built from capstone-ariane p1-r79 7bd984edc
(core = 7214834e3: 1c5d31474 plus R-79's fix, 09b9cf9fd, applied by anchor). It is one session: the control first,
then r79-pmp, then r79-cpmp. Every image is expected to RETURN, so the ordering rule is trivially met.

Each vector is the recorder's words at `board_rec`: the count at board_rec, and the values at board_rec + 8·(1..SR).
- Each was taken by simulating the image itself on that tree's RTL through the Anvil prologue: `rtl_dirty=no`,
  `S12_MEM_DELAY=12`, `+time_out=3000000`, no XDEF.
- rc=2 is the bare image's CAP_PASS spin running to the cycle limit, after the recorder is complete.

| image | sha256 | board_rec | SR | from |
|---|---|---|---|---|
| `va-runv.bin` (control) | `f0ed3fac351b60f3cd92b98ad4e0d6376984e0b08c632e3b13612280e49221a4` | 0x8000c000 | 15 | `r75-bare-2026-11-22` (llvm-capstone 515d5bc97c7b) |
| `r79-pmp.bin` | `8eb72553965cfc1c98fd33b4caa44a0b0f1051e661092b63ed41d1cf4d954ec5` | 0x80007080 | 17 | `r79-pmp-bare-2026-11-23` |
| `images/r79-cpmp.bin` | `c1efe4e23976c6cdb20785140297dbf9f7a5bae10566a62703a0755b5934f036` | 0x80009000 | 21 | this kit |

r79-cpmp was built by the llvm-capstone clang `fa5b23aeb9b8` (binary sha256 660fca7da9067f76…) with `build.sh`. The
same compiler reproduces r79-pmp's committed SHA256SUMS exactly.

**r79-cpmp symbols:** s_code 0x80002000 (CPMP 3, XO), r_page 0x80003000 (CPMP 0, RO), wo_page 0x80004000 (CPMP 1,
WO), w_page 0x80005000 (CPMP 2, RW), res 0x80005800. FR = 0x5a5a5a5a5a5a5a5a fills r_page, FWO = 0xa5a5a5a5a5a5a5a5
fills wo_page.

## The prediction split (the semantics owner's wording, verbatim)

"FIXED rows exact on the new bitstream, and no graded pre-fix value on the old residents".
- A pre-fix row depends on whether a following translation lands in the very next cycle. The ids55 simulation's p8,
  registered as a control, missed for exactly that reason (capstone-ariane 27e75c38a, r79-cpmp.expect). So no pre-fix
  value is graded anywhere in this kit.
- The fixed row does not depend on timing, because every access is judged by its own type.

## Vectors on p1-r79 7bd984edc (graded)

| image | vector (SV, in order) | sim |
|---|---|---|
| va-runv | `0x2 0x2 0x1234 0x2 0xb 0x0 0x0 0x55 0x0 0x77 0x77 0x5 0xb 0x0 0x9` | .sup/r75k-p1, retired 1,379,497 |
| r79-pmp | `0x0 0x0 0x0 0x5a5a5a5a5a5a5a5a 0x7 0x80002000 0x5a5a5a5a5a5a5a5a 0x7 0x80002008 0x5a5a5a5a5a5a5a5a 0x7 0x80002000 0x77 0x77 0x77 0x77 0x3` | .sup/r79p1, retired 1,479,916 |
| r79-cpmp | `0x0 0x0 0x0 0x5a5a5a5a5a5a5a5a 0x7 0x80003000 0x5a5a5a5a5a5a5a5a 0x7 0x80003000 0x5 0x80004000 0x77 0x0 0x0 0x0 0x7 0x80003008 0x4321 0x99 0x5a5a5a5a5a5a5a5a 0x4` | .sup/r79cp1, retired 1,475,779 |

- va-runv's row equals its R-75 kit PREREG row (bitstream 1 and 1c5d31474), retired count included.
- r79-pmp's row is the FIXED row of `r79-pmp-bare-2026-11-23/PREREG.md` (simulated there on d0afdb40f) word for word.
  Its positions are explained in that PREREG.
- r79-cpmp's row is the in-tree arm's FIXED row (capstone-ariane 27e75c38a; on p1-r79, 7bd984edc's run 21/21 EXACT),
  with this image's addresses.

r79-cpmp's positions (0-based):
- **0, p1:** the control store to w_page: 0.
- **1-3, p2:** a legal ld through the RO entry, with an independent store right behind it: no fault, the fill.
- **4-6, p3/p4:** a lone sd 0x1234 through the RO entry: 7 with tval r_page; the fill is kept.
- **7-8, p5:** lr.d through the RO entry: 7 with tval r_page (an AMO takes the write judgement).
- **9-11, p6:** a ld through the W-only entry, with an independent store right behind it: 5 with tval wo_page; the
  register keeps 0x77.
- **12, p7:** a lone legal sd 0x4321 to wo_page + 8: 0.
- **13-14, p8:** a legal sd to wo_page + 16, with an independent sd behind it: 0.
- **15-16, p9:** an sd to r_page + 8, with an independent sd behind it: 7 with tval r_page + 8 (its own address).
- **17-19:** read back through minted capabilities: wo_page + 8 = 0x4321 (p7 landed), wo_page + 16 = 0x99 (p8
  landed), r_page + 8 = FR (p9 refused).
- **20:** traps, 4.

## Information only: 1c5d31474 (R-75 (d), no fix), NOT graded

The same image on 1c5d31474 (.sup/r79cbase, dirty=no, retired 1,475,690), SR 21:
`0x0 0x7 0x80003000 0x77 0x0 0x0 0x1234 0x0 0x0 0x0 0x0 0xa5a5a5a5a5a5a5a5 0x5 0x80004008 0x5 0x7 0x80003008 0xa5a5a5a5a5a5a5a5 0xa5a5a5a5a5a5a5a5 0x5a5a5a5a5a5a5a5a 0x4`
- It differs from the fixed row at 16 of 21 positions (1-14, 17, 18).
- It equals, position for position and with the addresses shifted, the pre-fix ids55 run recorded in r79-cpmp.expect:
  - a legal read refused through RO (p2);
  - a write landing through RO (0x1234);
  - lr.d allowed;
  - a READ through W-only (p6: 0, and FWO read);
  - legal stores through W-only refused as reads (p7, p8: 5).
- This is the positive control that the image arms the defect in simulation. On silicon it is not graded, per the
  split above. A run of this image on a pre-fix resident is information, not a verdict.

## Grading (the p1-r79 seal only)

- **STOP** unless the resident bitstream reads back as the p1-r79 seal's name. r79-cpmp must never run on a Sublet
  build: CAPENTER is illegal there and the CPMP is tied off.
- **VOID** unless va-runv reads SR 15 and its 15 values, and all three images' sha256 match the table.
- **r79-pmp: R-79 (PMP side) FIXED ON SILICON on the P1 line** when it reads its row exactly. **r79-cpmp: R-79's
  CPMP side FIXED ON SILICON** when it reads its row exactly.
- r79-cpmp is VOID unless SR is 21 and position 0 (p1, the control store) reads 0x0. No other position is a validity
  check, because on a tree where the fix did not take, every arm from p2 to p9 can move with the defect. That includes
  p8, which read 5 on the pre-fix tree, and p9, whose 7 there depended on its following store landing in the next
  cycle.
- Any other vector is a finding, graded value by value against the positions above, and goes to the RTL lane before
  any claim. The tvals are addresses, so a partial match is informative.
- Cite readings by image hash, never by label.
