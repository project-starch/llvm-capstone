# The +Spatial kernel: RISCV_CAPSTONE_SPATIAL_ONLY (runtime lane, an add-on to the trap-route series at 6d4504f8b073)

For the campaign's +Spatial row on Line B: rtl's spatial-only variant of b42c8c644 (CAPSTONE_SPATIAL_ONLY,
`docs/plans/2026-11-26-spatial-only-lineb.md` on land/rtl-crevoke4b 9143e3d3cac6) keeps no lifetime table.
CSMINT writes no record and answers id 3, every CDERIVE child carries its parent's id, CREVOKE has no effect.

Two places in the series read the table, and both refuse under that processor (sublet-table.c):
- `root_create` reads back the record CSMINT wrote and requires it live at depth 0;
- `sublet_ref_inspect`, used by every block call (munmap, mremap, refill: sublet-blocks.c `block_request`), feeds
  `request_operand`, which requires live && depth == 1, and `refill_rule`, which requires no descendants.

`sublet-spatial-only.patch` adds the option (default n). Under it, the root is accepted on type NONLIN and
CSMINT's answer == the slot reference's id, and a tagged block operand counts as live at depth 1 with no
descendants. Bounds checks (`request_operand`'s base/end) are unchanged. No kernel code compares the ids of two
references (searched for id comparisons and CV_ID_ uses), so a shared id 3 is not a problem.

**Checks:**
- Option off: `sublet-table.o` and `sublet-blocks.o` disassemble identically to the frozen twin tree's (749fed26).
  Built on the first version of the patch; the later fix changes a line inside the option's #ifdef only.
- Twin, capstone-qemu sublet/spatial-only 4e30cfa23864 (compiler's x-capstone-spatial-only; its off gates were
  still pending at this reading), firmware keep-tags, kernel Image-spatial b47cc3fd7d686b84:
  - with the property ON: the trap-route test reads `regs status 0`, and msgt native and protected both read
    rc 0 with `fails=0` and lines identical to Full's. No `sublet:` fault or refusal line.
  - the control, the Full kernel 749fed26 on the same QEMU with the property on: `sublet: R reads back as type 1,
    id 0x3 (no record), and CSMINT answered 0x3` and `regs status 91`. So the property is applied, and the option
    is what makes the kernel run on it.
  - the option on a Full twin (property off): `regs status 0`.
- One defect found on the way and fixed: the first version kept `!record` in the root check. On the spatial twin
  the record lookup returns NULL (the table is never populated), and the pair read 91 / 139 like the control.

**On silicon (boot 14, board 1, the spatial-only seal 14f2ef81a, 2026-11-26).** The final +Spatial kernel is
`build/build-kernel-final-spatial.sh`: the final NO_HZ_IDLE Full kernel plus this option, and its .config differs from
the Full kernel's only in CONFIG_RISCV_CAPSTONE_SPATIAL_ONLY=y. It read:
- the smoke control rc=7;
- mtrip in band;
- msgt native and protected rc 0 with fails=0 and the twin's five lines;
- no sublet fault, call or R-reads-back line.
The protected msgt run is the step that goes through the spatial-only root check. See RESULTS.md, "Boot 14".

The qualification SPEC images (spectest Amendment 12) use the same option on the pinned bring-up tree instead
(`trapk/fpga-v1-spatial`). That tree's .config differs from fpga-v1's only in this option.

**Not covered:** the Full kernel's refusal on silicon. It is read only on the spatial-only twin above.
