# b5b2-r60r32-bare-2026-11-26

This kit reads ISSUES R-60 and R-32 on Line B's final Full core, b42c8c644, on silicon. Both fixes are in that core's
ancestry. P1's kits for them (silchk-b1-2026-11-21) cannot run on Line B, because each test executes CAPENTER, which
the Sublet format deletes. These are their Line B-native replacements, minted through CSMINT.

**Siblings:** `b5b2-misscost-bare-2026-11-26`, `b5b2-ldcfix-bare-2026-11-25` and the other B5b-2 kits run on the same
seals. The wrapper (`inc/`) is theirs, byte-identical, and so is `sublet-life`'s image.

**Images, in this order** (every one returns):
- `sublet-life`: the control.
- `b5b2-r60-alias` (R-60). A 0x1000-byte child at a 0x4000-aligned base is moved one window span out by CINCOFFSET,
  then stored through in protected U. A sentinel a span away is read back through an independent witness. With the
  fix the store traps 24, the sentinel stays intact, and the moved capability is untagged.
- `b5b2-r32-shrinkto` (R-32). SHRINKTO imm on fresh capabilities, with the bounds read back by LCC in M. With the fix
  each child holds imm bytes.

The tests are capstone-ariane b5b2-hangmon-c's `verif/tests/custom/capstone/b5b2-r60-alias.S` (1f1740a9c) and
`b5b2-r32-shrinkto.S` (ce6c77b73), unchanged. Their headers hold the simulation readings, including the negative
controls. `PREREG.md` holds the grading.
