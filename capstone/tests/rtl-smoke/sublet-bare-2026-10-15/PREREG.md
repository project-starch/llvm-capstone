# The Sublet acceptance floor: three bare JTAG arms (pre-registered 2026-10-15, before any board run)

The first thing to run on `caplifive_sublet_d93a8d66b.bit` after the S-16 bare set: CDERIVE, CREVOKE, MREV deepening and
the LINEAR split's sibling placement on silicon, with no SDK, no monitor and no Linux between the test and the core.
Built by `build.sh` from the RTL tree's sources (capstone-sublet, copied into `tests/` and checked equal at build time)
with the bare M-mode wrapper of `sup-bare-2026-10-03-ladder` (env/) and this folder's node-aware header (inc/: the
wrapper's record capabilities can be attached to a node the test names, and CAP_PASS attaches its own record from
its UART capability). Runner: `sup-bare-2026-10-03-ladder/run_sup_bare.py` (SUP_IMG, SUP_REC_ADDR from the .rec,
FPGA_BITSTREAM = the resident name); the report is the UART's `SR <count>` / `SV <value>` lines.

| image | sha256 | board_rec | SR | readings (SV, in order), from the simulation of the committed tree |
|---|---|---|---|---|
| `images/sublet-derive-kill.bin` | `6b6eb589929a2d9003277582ae751fc40a985d1a92249a0f8304fc3ab718eea3` | 0x80004100 | 19 | 0xe 0x100 0x300 0x1 0x1234 0xe 0x1d 0x1b 0x0 0x19 0x19 0x1d 0x1d 0x0 0x19 0xe 0x5678 0x1a 0x18 |
| `images/sublet-split-shares-node.bin` | `029dff8faf069d6e68079a576c277415e5e1b60228835f57203176c422cb0a80` | 0x80004000 | 7 | 0x1111 0x2222 0x0 0x19 0x19 0x0 0x19 |
| `images/sublet-mrev-orphan.bin` | `22989ae01b40f4dca02044a613b0ab6114201bd144aa4f3832db6b629de3bf92` | 0x80005000 | 8 | 0x0 0x0 0x0 0x0 0x0 0x19 0x19 0x19 |
| `images/sup-sublet.bin` | `d7b3e3f811aa582d6d943b47498799c20d877ae2c67623d023e0a132bc72c7d0` | 0x80007c40 | 26 | 0x0 0x1 0x77 0xe 0x5a5a 0x0 0x5 0x19 0x0 0x4 0x0 0x5 0x1b 0x0 0x4 0x0 0x5 0x1d 0x0 0x4 0x0 0x5 0x1a 0x0 0x4 0x0 |

The vectors are the operands of the tests' CAPPRINTs in retirement order on the Sublet tip (snapshot sublet-renorm,
RTL d93a8d66b, the same on every step since each arm was added); the simulation's trace also shows the trap handler's
mcause reads, which the board wrapper does not record, so the SR counts above are the CAPPRINT counts. What each
reading means is in the tests' headers (sublet-derive-kill.S: perms, the child's values, the kill's causes 27 / 29 /
25 ...; sublet-split-shares-node.S: the shared node's escape reads 25 here and 0 on a tree without 52ad6e5ea;
sublet-mrev-orphan.S: the orphan reads 0 at #6 and #8 on a tree without 464abe2ce, 25 here).

Registers: derive-kill and split-shares-node leave x8 / x16 / x17 to the wrapper and never revoke the base's node, so
their record capabilities carry the base's node as the ladder's do. mrev-orphan uses a6, so its wrapper scratch is
x9 / x17 / x8, and its REVOKEs end the base's node, so its record capabilities take the node of the LINEAR split
sibling the test keeps in s11 (placed BEFORE the base's node, outside every walk the test makes); the test's board
exit is `CAP_PASS(s11)` under `#ifdef BOARD_REC_NODE`, the simulation path untouched (re-run after the edit: the
same 8 readings).

## Controls

- **Positive control, on the FP bitstream BEFORE the flash** (`caplifive_fpb_afcd19bea.bit`, resident): the same
  `sublet-derive-kill.bin`. Its first CDERIVE is an undefined encoding there and must raise cause 2 (illegal
  instruction); the test's trap handler records the cause and resumes, so the SV vector is NOT the one above and its
  first fault-code reading is 0x2. The exact FP vector is not pre-registered; the criterion is "differs from the
  Sublet vector, with 0x2 where the Sublet tree reads 0xe". A run that reads the Sublet vector on the FP bitstream
  means the image or the board was not what it was supposed to be.
- **Negative controls inside the Sublet run**: derive-kill's refusals (0x1d, 0x1b, 0x19 at the recorded positions),
  the dead-child load denied by R-45's flush (0x19 after the kill), and mrev-orphan's 0x19 at #6 and #8 (the orphan
  that a tree without deepening lets live).
- **Timing control**: the S-16 bare set first, as on every bitstream (98 readings identical to 0568f93a9's or the
  chain stops).

## What is NOT in this floor

sup-sublet is the second wave, built here already: the supervised-domain arm (CDERIVE and CREVOKE inside a domain
under CSSUPERVISE/CALL, the four refusals escaping to the monitor with CSUPCAUSE 25 / 27 / 29 / 26). Its three in-domain
CAPPRINTs are compiled out on the board (the wrapper's CAPPRINT mints a record capability, which a supervised domain
may not do; capstone-sublet carries the `#ifndef BOARD_TEST_CH`), so its board vector is the simulation's 29 values
minus the domain's 0xe, 0x77 and 0, 26 values, with the monitor's reads of 0x77 / 0xe / 0x5a5a through s1 carrying
the same information. It uses the monitor's CSSUPERVISE and CALL, so it runs after the three bare arms, and it is the
first supervised domain on the Sublet bitstream: a trap anywhere ends it with mcause as the last reading. On the FP
bitstream it is a second positive control: the domain's first CDERIVE escapes with CSUPCAUSE 2 instead of completing.
The SDK-level witness
(a physical image calling the runtime's derive and revoke) waits for the runtime's sublet_carve fix (R-55 family) and
the full-format SDK defaults. The s1-revoke-gap pair runs on this bitstream as is, for the FP-against-Sublet
comparison its PREREG records.
