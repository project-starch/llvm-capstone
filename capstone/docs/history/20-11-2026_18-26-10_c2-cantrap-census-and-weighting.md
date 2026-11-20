# C2: which capability increments stay in their loop only because they can trap, and how often they run

Programme item C2 read: "cincoffset is marked trapping everywhere in `CapstoneInstrInfo.cpp` canTrap, so it is
never hoisted: make it non-trapping where bounds are proven, then hoist". Before changing canTrap, this note counts
the sites that rule keeps in a loop and weights them by how often they execute. Result: nothing that can be proven
safe reaches 0.2 % of a benchmark's instructions, so no canTrap change was built. A defect found on the way is filed
as C-87 (debug info changes generated code).

## The premise, read in the source

"Never hoisted" is not exact. `CapstoneInstrInfo::canTrap` returns `trapsOnUntaggedOperand(opcode)`, which is true
for CIncOffset(Imm), LCC, SHRINK, SHRINKTO, SCC, TIGHTEN, SEAL and INIT. MachineLICM (`IsLICMCandidate`) then treats
such an instruction like a load: it hoists it when its block is guaranteed to execute, i.e. dominates every exiting
block of the loop. MachineCSE does not PRE it, and EarlyIfConversion does not speculate it (C-58, C-66). At the IR
level, `isSafeToSpeculativelyExecute` lets a GEP on a known-non-null base speculate (C-19). So what canTrap blocks
is a loop-invariant increment in a block that does NOT run on every iteration.

The opcodes stay trapping because a tagged operand that is sealed or uninitialised raises cause 26, even after the
CHERI rule for untagged operands (2026-10-14). A local proof that an increment cannot trap exists for two operand
origins only: a stack slot (the stack capability is valid and unsealed) and a global (formed from the gp table). An
argument, a loaded capability or a call result can be sealed.

## Instrument 1: the static census

MIR printed right after `early-machinelicm`, from every benchmark's recorded compile lines with the measured flag
set (FRCDKMS: `-capstone-shrinkto-globals`, `-capstone-shrinkto-end-adjust=1`, `-capstone-direct-calls`,
`-fomit-frame-pointer`, `-capstone-drop-redundant-shrink(-declarations)`, `-fjump-tables`,
`-capstone-fold-global-lo`, `-capstone-disable-cost-per-use`, `-capstone-memcpy-align-dispatch`; features `+relax +c
+cap-base-rvc`). Compiler clang 4e3bae5d83fc (code generator library 6962daec0730), 713 files, 0 failures.

For each trapping instruction inside a loop, the census uses the innermost loop containing it (natural loops from
back edges in the printed CFG, dominators computed from it) and assigns one of three classes:
- **blocked**: every register it reads is defined outside that loop, and its block does not dominate every exiting
  block. That is MachineLICM's own guaranteed-to-execute test, the one canTrap sends it to.
- **kept**: invariant and guaranteed, but not hoisted for some other reason.
- **variant**: not loop-invariant.

Each blocked site's operand is traced through copies, increments, bounds changes and PHIs to its origin.

Controls. The three lit tests that pin the guarded cases (`c66-machinelicm-subloop-speculation.ll`,
`no-speculative-cap-arith.ll`, `no-speculative-cap-intrinsics.ll`) report exactly their six known guarded sites, and
the hoisted control instruction of the first is not counted. A self-loop bug in the loop finder put an entry block
inside a loop; it was found by an entry block appearing in a "kept" site and fixed before any count was used. After
the fix, all 24,279 loops found are dominated by their header.

Blocked CIncOffset/CIncOffsetImm sites (total, of which stack slot / other), and blocked SHRINKTO:

| benchmark | blocked cinc | stack slot | argument, load, call, mixed | SHRINKTO (mostly of a global) |
|---|---:|---:|---:|---:|
| 500.perlbench_r | 2,896 | 2,777 | 112 | 624 |
| 502.gcc_r | 6,315 | 6,009 | 292 | 7,638 |
| 505.mcf_r | 30 | 30 | 0 | 14 |
| 519.lbm_r | 9 | 9 | 0 | 3 |
| 525.x264_r | 533 | 444 | 89 | 130 |
| 538.imagick_r | 5,117 | 4,886 | 230 | 434 |
| 544.nab_r | 568 | 555 | 13 | 86 |
| 557.xz_r | 170 | 107 | 63 | 91 |

Almost every blocked increment is `CIncOffsetImm %stack.N, 0`, the address of a stack slot. The "kept" class
(invariant and guaranteed, yet not hoisted) is almost entirely stack slots too. MachineLICM's debug output on a nab
example says why: "Can't remat / high reg-pressure". The instruction is not trivially rematerialisable, so under
register pressure LICM refuses it whether or not canTrap does. For these sites the lever is rematerialisation and
register pressure (C3) as much as canTrap.

The hot spots named in the programme are not here. mcf has no blocked increment outside stack slots, and its 30 are
in `markBaskets` (7), `master` (10), `read_min` (5) and `worker` (8); together they execute 213,064 times (below).
lbm's nine are in `LBM_storeVelocityField` and `LBM_compareVelocityField` (file I/O), none in the stream-collide loop.

## Instrument 2: executions

Profiles: the existing shift-1 (per-instruction) capmix profiles of the FRCDKMS arm, one per benchmark (none for
x264). Each profiled image matches the arm's image by sha256. Compiler: the arm's own, clang 29027126f39b.

Mapping:
1. The arm's recorded compile lines are replayed with `-g`, once to MIR and once to objects and a linked image.
2. The image's executable sections must equal the profiled image's; the tool refuses to weigh otherwise.
3. Each blocked site's `DILocation` (line, column) and function select the PCs in that function's symbol whose
   line-table row has the same line and column and whose instruction is of the site's family.
4. A PC matched by two sites counts once, so every figure is an upper bound.
5. Executions are mapped onto the image by one load offset per image instance.

The mapper reproduces the single-offset mapper exactly on mcf and nab (26,646,288,295 and 7,155,027,703 image
instructions), and maps perl and lbm, where the single-offset search had found no unique offset.

Two of the profiled images could not be rebuilt identically with `-g` (C-87 below). There the objects whose code
changed are linked from the profiled build instead (perl 9, gcc 24, imagick 2), so the image matches. Sites in those
objects have no line table, and like sites without a debug location they are given a bound: every PC of their
family in their function, and for a stack-slot site only increments of sp.

Blocked-site executions as a share of each image's instructions, by operand origin:

| benchmark (input) | image instructions | stack slot | global | argument, load, call | bound with unmapped sites |
|---|---:|---:|---:|---:|---:|
| perlbench test-sw | 3,722,276,972 | 0.006 % | 0.005 % | 0.000 % | 0.143 % |
| gcc train01 | 12,258,808,578 | 0.051 % | 0.085 % | 0.011 % | 0.661 % |
| mcf test | 26,646,288,295 | 0.001 % | 0 | 0 | 0.001 % |
| lbm test | 6,092,487,962 | 0 | 0 | 0 | 0 |
| nab test | 7,155,027,703 | 0 | 0 | 0.172 % | 0.172 % |
| xz test | 49,630,312,159 | 0.009 % | 0.032 % | 0.242 % | 2.812 % |
| imagick test-ref (convert only) | 95,409,008 | 0 | 0 | 0 | 0.012 % |

Instrument slips caught before any figure was reported:
- The metadata lookup was anchored at column 0, but the IR inside a MIR file is indented, so no site found its
  location and every executions figure read 0. The tool now exits with an error when no site is weighed, and when a
  location names no metadata.
- `DILocation`s printed without a column (column 0) were not matched; the column is now optional.

## Reading

- The origins a canTrap change could prove safe (stack slot, global) stay under 0.2 % mapped in every benchmark;
  the largest is gcc at 0.136 %.
- Only xz exceeds 0.2 % (0.282 %), and only through operands that are arguments or loads, which no local proof
  covers.
- The bounds (gcc 0.661 %, xz 2.812 %) are loose by construction, because they count every same-family PC in a hot
  function that holds an unmapped site. They cannot show a gain, only fail to exclude one.
- Every figure is an upper bound on the instructions saved: hoisting saves the executions inside the loop less one
  per loop entry, and can add spills.

Against the runtime lane's go/no-go (build only if some benchmark's blocked sites reach 0.2 % of its instructions),
the recommendation is to drop C2+C3 as a canTrap change, with the tables above as the record. No canTrap change was
built.

What would reopen it: a hot loop where a provably safe increment sits in a conditional block. The census lists every
blocked site per function and can be rerun on a new benchmark in minutes.

## C-87, found here

With the measured flag set, adding `-g` changes the generated code: 9 of 69 perlbench objects, 24 of 387 gcc and 2 of
98 imagick get a different `.text`, while mcf, lbm, nab and xz do not. In perlbench's `av.o` the `-g` build has an
extra `mv` and a reordered compare. dev's compiler (4e3bae5d83fc) does the same on `av.o`, so it is not confined to
the measured compiler. Entry C-87 in ISSUES.md.
