# cap-type-word: the type number a STORED capability word carries on the RTL (Q-35)

One directed RTL test for ISSUES Q-35. It runs in capstone-ariane's verif tree: copy `cap-type-word.S` to
`verif/tests/custom/capstone/` and `testlist_captype.yaml` to `verif/tests/`, then run through the Anvil prologue
like any custom test (`asm_insn.h` from that tree).

What it does: builds a NONLIN and a LINEAR capability over the same 256-byte buffer, reads each one's type with
`LCC` selector 1, STCs it, and reads the stored second doubleword back with a plain `ld` from an integer address.
CAPPRINTs per arm: the hi word, `(hi >> 27) & 7`, the LCC type.

Reading, capstone-ariane fe4810dd1 (the local stage-B batch on C.5a 1fc2cd5aa: ariane_pkg.sv, the DYN unit and the
store path are C.5a's; sim-only SUP_TRACE detectors in cva6.sv; PROVENANCE anvil_dyn_unit 131818fb5bfda03c, ariane_pkg 124213bf126eb983), 8 Nov 2026,
SUCCESS after 1,629 cycles:

| Arm | hi word | bits [29:27] | LCC type |
|---|---|---|---|
| NONLIN | 0x9d0403000 | 2 | 1 |
| LINEAR | 0x9c8403000 | 1 | 0 |

The two words differ only at bit 27; perm [33:30] = 7, id [63:34] = 2 and bounds [26:0] decode at QEMU's
positions. The prediction written in the test header before the run was N 2 / L 1 and LCC N 1 / L 0. A hi word
of 0 would have made the run void (the plain load not returning metadata).
