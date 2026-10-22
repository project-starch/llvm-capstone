# va-alloc-csret as a bare JTAG arm: item 6 (generations in virtual node ids) read on silicon (pre-registered 2026-10-23, before any board run)

The batch bitstream `caplifive_vp2c_c721c1a2a.bit` (sha256 8f4a486f2c96…85ed6e) carries item 6: a virtual node id is
{generation[29:16], index[15:0]}, the generation is bumped on every pop of a recycled index, and every generation-aware
gate in the rev-node unit (the stale-retire refusal above all) compares it. Its first boot read the s23 re-arm vector
(`s23-bare-2026-10-21`); this arm reads item 6 itself: `va-alloc`'s VA_CSRET section, which runs at M under
csupctl bit 3 (the table namespace) before any context is entered, so it is bare-compatible as it stands.

Built by `build.sh` from the test's source at the batch tip c721c1a2a (copied into `tests/`, checked equal at build
time), with the bare M-mode wrapper of `sup-bare-2026-10-03-ladder` (env/) and the node-aware header of
`s23-bare-2026-10-21` (inc/, copied). Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (`SUP_IMG` = the image,
`SUP_REC_ADDR` from the `.rec`, `FPGA_BITSTREAM` = the resident name); the report is the UART's `SR <count>` /
`SV <value>` lines. The recorder's scratch registers are x8, x4 and x16, which the test never names; its CAPCREATEs carry
node 2 and allocate nothing, so the recorder does not move the allocator and the simulated vector transfers.

| image | sha256 | board_rec | SR |
|---|---|---|---|
| `images/va-alloc-csret.bin` | `9491707dfa1613e300d3c52bf21f4468babcb81dd61a0e4ceae75fe78849ed18` | 0x8000a800 | 20 |

## The two vectors, simulated through the Anvil prologue (each run directory carries its PROVENANCE)

Print order (the test's own comments name each): srevroot read-back minus the table (0); the CSNODE ancestor id; the
first CSRETIRE's status; the second CSNODE's ancestor id; the STALE retire's status; the load through the re-noded fresh
owner; the free-list head and the allocation count; then the record reads REC_READ(4), REC_GEN(4), REC_READ(5),
REC_GEN(5) (flags, parent, depth; generation) and the domain arms' readings; mcause last.

| tree | vector (SV, in order) |
|---|---|
| **BATCH c721c1a2a** (date=2026-10-22T07:10:23Z rtl_commit=c721c1a2a anvil_rev_node=99e92df7f3b4c900 ) | `0x0 0x10004 0x0 0x0 0x20004 0x1 0x1 0x0 0x0 0x0 0x4 0x3 0x0 0x0 0x2 0x3 0x4 0x1 0x2 0x0` |
| PRE-BATCH tree A 8522d72fe, the test transplanted (date=2026-10-22T06:55:27Z rtl_commit=8522d72fe anvil_rev_node=6d0f9699a247d012 ) | `0x0 0x4 0x0 0x0 0x7 0x1 0x1 0x0 0x0 0x0 0x4 0x2 0x0 0x0 0x0 0x2 0x4 0x1 0x0 0x0` |

The two vectors differ at exactly these positions (0-based), computed from the two simulations; the meaning of each
position is the test's own comment at that print, not restated here:

| position | batch | pre-batch tree A |
|---|---|---|
| 1 | `0x10004` | `0x4` |
| 4 | `0x20004` | `0x7` |
| 11 | `0x3` | `0x2` |
| 14 | `0x2` | `0x0` |
| 15 | `0x3` | `0x2` |
| 18 | `0x2` | `0x0` |

Positions 1 and 4 are the CSNODE ancestor ids (generation-carrying on the batch, bare on the pre-batch unit); 14 and 18
are the REC_GEN reads (2 on the batch, 0 where the pre-batch unit never writes the field); 11 and 15 are the record flag
reads of the same two records. Grading rule: VOID unless SR = 20 and position 0 = 0; PASS when all 20 equal the batch
vector; a pre-batch reading (all 20 equal the tree-A vector) says the bitstream in service is not the batch; any other
vector is a finding, graded value by value against the test's comments.
