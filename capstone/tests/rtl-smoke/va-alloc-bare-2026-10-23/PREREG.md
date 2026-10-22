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
| `images/va-alloc-csret.bin` | `9491707dfa1613e300d3c52bf21f4468babcb81dd61a0e4ceae75fe78849ed18` | 0x8000a800 | 17 |

## The two vectors, simulated through the Anvil prologue (each run directory carries its PROVENANCE)

**Correction after the first board run (2026-10-23 15:15, VOID by the first rule):** the simulation TRACE prints a CAPPRINT
that sits within a few instructions after a lifetime op TWICE (the op's commit flush re-issues the younger print, and the
trace records both retirements), while the recorder counts the architectural execution once: the board printed SR 17 for
this arm where the raw trace showed 20. The three doubled prints are the status print after each of the two CSRETIREs
and the load print after the second. The vectors below are the 17-value architectural ones (the trace vectors with those
three doubles removed); the first version of this file published the raw 20-value trace and was wrong by exactly those.

Print order (the test's own comments name each): srevroot read-back minus the table (0); the CSNODE ancestor id; the
first CSRETIRE's status; the second CSNODE's ancestor id; the stale retire's status; the load through the re-noded fresh
owner; the free-list word; the allocation count; REC_READ(4) (flags, parent, depth); REC_GEN(4); REC_READ(5); REC_GEN(5);
mcause last. SR = 17.

| tree | vector (SV, in order, 17 values) |
|---|---|
| **BATCH c721c1a2a** | `0x0 0x10004 0x0 0x20004 0x1 0x0 0x0 0x4 0x3 0x0 0x0 0x2 0x3 0x4 0x1 0x2 0x0` |
| PRE-BATCH tree A 8522d72fe (the test transplanted) | `0x0 0x4 0x0 0x7 0x1 0x0 0x0 0x4 0x2 0x0 0x0 0x0 0x2 0x4 0x1 0x0 0x0` |

The two vectors differ at exactly these positions (0-based):

| position | batch | pre-batch tree A |
|---|---|---|
| 1 | `0x10004` | `0x4` |
| 3 | `0x20004` | `0x7` |
| 8 | `0x3` | `0x2` |
| 11 | `0x2` | `0x0` |
| 12 | `0x3` | `0x2` |
| 15 | `0x2` | `0x0` |

Positions 1 and 3 are the CSNODE ancestor ids (generation-carrying on the batch, bare on the pre-batch unit); 11 and 15
are the REC_GEN reads (2 on the batch, 0 where the pre-batch unit never writes the field); 8 and 12 are the record flag
reads of the same two records. Grading rule: VOID unless SR = 17 and position 0 = 0; PASS when all 17 equal the batch
vector; the tree-A vector means the bitstream in service is not the batch; any other vector is a finding, graded value by
value against the test's comments.

## Bitstream 2 (5d365d9a6, the 18-bit index): the SAME frozen image, pre-registered 2026-10-24 before its build sealed

The image is unchanged (sha256 `9491707d…ed18`, board_rec 0x8000a800, SR 17): it carries the batch-era decodes
(`REC_GEN` = word 1 `>> 46 & 0x3fff`, `REC_READ`'s parent mask 0xffff), so on bitstream 2's layout (generation at
[59:48], parent at [47:30]) its generation read is `(gen << 2) | parent[47:46]`, and the ids carry {gen[29:18], idx[17:0]}.
Simulated from the kit's own source (`tests/va-alloc.S`, sha256 `49ad4968…6dae`, equal to `c721c1a2a`'s) on the
bitstream-2 RTL through the Anvil prologue (vp8 `.sup/kitb2`: rtl 0f94f2cff dirty=no, whose `core/` is 5d365d9a6's;
rev_node unit f532caf0220b7003, ex_stage c86402d729260989); the raw trace has the same 20 prints with the same three
same-register doubles (cycles 96976/96982, 97053/97165, 97621/97756), removed by the rule above.

| tree | vector (SV, in order, 17 values) |
|---|---|
| **BITSTREAM 2 5d365d9a6** | `0x0 0x40004 0x0 0x80004 0x1 0x0 0x0 0x4 0x3 0x0 0x0 0x8 0x3 0x4 0x1 0x8 0x0` |

Against the batch vector it differs at exactly four positions: 1 (`0x40004`: generation 1, index 4 in the new id
format), 3 (`0x80004`: generation 2), 11 and 15 (`0x8`: generation 2 at [59:48] read through the old decode, with
parent bits [47:46] = 0). Every other value, the flags, parents, depths, statuses, the load, the free-list word and the
count, is identical. Grading rule on bitstream 2: VOID unless SR = 17 and position 0 = 0; PASS when all 17 equal the
bitstream-2 vector; the batch vector means the bitstream in service is still the batch; the tree-A vector means a
pre-batch tree; any other vector is a finding, graded value by value.
