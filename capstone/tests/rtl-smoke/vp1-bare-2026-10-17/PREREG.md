# vp1's first session: item B's translation and Q-17's check order as bare JTAG arms (pre-registered 2026-10-17, before any board run)

The first thing to run on `caplifive_vp1_9ed9a2594.bit` after the physical regression (the S-16 bare set, the Sublet
arms and one corpus set, which are the d93a8d66b images as they are and must read identical). Built by `build.sh` from
the RTL tree's sources (capstone-ariane `virtual-profile` 7234b6d43, tests only on top of the RTL of 9ed9a2594; copied
into `tests/` and checked equal at build time) with the bare M-mode wrapper of `sup-bare-2026-10-03-ladder` (env/) and
the node-aware header of `sublet-bare-2026-10-15` (inc/, copied). Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py`
(`SUP_IMG` = the image, `SUP_REC_ADDR` from the `.rec`, `FPGA_BITSTREAM` = the resident name); the report is the UART's
`SR <count>` / `SV <value>` lines. No SDK, no monitor, no Linux: the test's own monitor builds Sv39 tables in its data,
programs one all-permitting PMP entry for the page-table walker (entry 0, NAPOT all ones, RWX), sets satp and
`csupctl.virt`, and arms supervised domains with `CSSUPERVISE`.

| image | sha256 | board_rec | SR | readings (SV, in order), from the simulation of the committed tree |
|---|---|---|---|---|
| `images/vb-trans.bin` | `a3fa54d0e6f54f16618c16b9680c30567cc2c07efa7fffa49c0cf320eaeac7f1` | 0x80011800 | 36 | 0x0 0x1 0x0 0x28 0x0 0x77 0x1234 0x0 0x1 0x0 0x5a5a 0x0 0x0 0x5 0xd 0x0 0x0 0x1 0x99 0x0 0x5 0xf 0x0 0x0 0x0 0x5 0xf 0x0 0x1 0x0 0x5 0x1c 0x0 0x0 0x1 0x0 |
| `images/vb-trans-off.bin` | `79e670cc77352ee4486ec31f8b4681cae0200e4c2fdd781f252b5d51c2d653d1` | 0x80011800 | 13 | 0x0 0x1 0x0 0x28 0x0 0x77 0x1234 0x0 0x1 0x0 0x0 0x5a5a 0x0 |
| `images/q17-order.bin` | `ee2adf9b373e502838bc9739b61b6c325c5ca64411d22137bb0fbcd20ba6c66a` | 0x80004000 | 3 | 0x19 0x19 0x2 |

The vectors are the monitor-side CAPPRINT operands in retirement order on `virtual-profile` (run `sub`, 2026-10-17,
RTL 9ed9a2594); the three in-domain CAPPRINTs of arms 4 and 6 compile out under the wrapper (`BOARD_TEST_CH`), so the
board vector of `vb-trans` is 36 values against the simulation's 39. What each reading means is in `tests/vb-trans.S`'s
header, arm by arm: arm 0 identity under translation (1, then 0x77 and 0x1234 read through the translated domain);
arm 1 the store through the remapped page landing at the PHYSICAL page (0x5a5a, then 0 at the identity location); arm 2
the load page fault (csupstatus 5, csupcause 0xd, csupepc at the load, csuptval at the page) resumed to read 0x99; arm 3
the store page fault (5, 0xf, 0, 0); arm 4 the consuming ldc refused on the read-only page (5, 0xf, 0) and the resumed
ldc returning after clearing its source (1); arm 5 the bounds fault under translation (5, 0x1c, 0); arm 6 the
non-consuming ldc from a read-only page completing (1); the trap handler's mcause 0 last. `vb-trans-off` is the same
source with the flag off (csupctl 0, satp bare): arms 0 and 1 only, arm 1's two readings inverted (0 then 0x5a5a).
`q17-order`: 0x19 (a cderive from a dead child with a window outside it: the liveness query first), 0x19, trap count 2.

## Positive controls on the resident bitstream (the same images on caplifive_sublet_d93a8d66b)

| image | SR | readings (SV, in order), from the simulation of capstone-sublet b7e3287e4 with the same sources |
|---|---|---|
| `images/vb-trans.bin` on caplifive_sublet_d93a8d66b (positive control) | 36 | 0x0 0x1 0x0 0x28 0x0 0x77 0x1234 0x0 0x1 0x0 0x0 0x5a5a 0x0 0x1 0x0 0x14 0xffffffff7fff9000 0x5 0x99 0x0 0x3 0x8000000000000010 0x4 0xffffffff7fff806f 0x0 0x1 0x0 0x18 0x5 0x0 0x5 0x1c 0x0 0x0 0x1 0x0 |
| `images/q17-order.bin` on caplifive_sublet_d93a8d66b (positive control) | 3 | 0x1d 0x19 0x2 |

On d93a8d66b `csupctl` has no bit 1 and nothing translates: arm 1 inverts (0 then 0x5a5a), arm 2's load reads the
physical page (csupstatus 1, no fault, then the resume CALL re-enters and the domain returns: 5 is the stale status of
the second CALL's escape, 0x99 the word it stored), arm 3's store lands (1), arm 4 reads the parked word's low half
(the stale-status readings 5/1/0/0x18), arms 5 and 6 as on vp1; `q17-order` reads 0x1d first (the window test before
the liveness query, Q-17's defect) against 0x19 on vp1. A bitstream that reads vp1's vector on arm 1 translates; one
that reads the control's does not. `vb-trans-off` reads its 13 values on both bitstreams (the flag off is the physical
profile, and the 95-test sweep is identical between the two trees).
