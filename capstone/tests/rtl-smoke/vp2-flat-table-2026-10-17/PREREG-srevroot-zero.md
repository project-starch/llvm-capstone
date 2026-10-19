# Module revision srevroot-zero (S-23's data face): pre-registration, 2026-10-19, before any run

Lane `module-srevroot-zero` on `module-resume-rd-x0` 8807df747411 (the board module's source, a8643d81d12971e6). The mechanism comes from the
runtime and RTL lanes and was re-read here at capstone-ariane 8522d72fe:
- the switcher's ENTRY loads srevroot (id 67, CSR 0x5c1) from frame+8, and its ESCAPE restores the caller's ids 0..66
  only (capstone_dom_switcher.anvil :154-162, :170-173, :479-481, :527-538), so S keeps the context's table after every
  escape;
- select_root/restore_root then rewrite the same value, and the trackers forget only on an edge (ex_stage.sv:1417).

**Change**, under the 0644 parameter `srevroot_zero`, default on:
- (a) `csrw 0x5c1, zero` in the same asm as each CSRUNV (run, run_x0);
- (b) restore_root writes 0.
Every other capability op in the module sits inside a select_root window (CSRETIRE :716 :947 :1596, CSMINT :858
:950), so none relies on srevroot after a CSRUNV. The diag tag is srevroot-zero-1.

**Builds** (build-module-srz.sh):
- QEMU 2fcef16d09e81269, board 607164b9c1407c7c.
- Control, the unmodified source with the same recipe: board 39d8492d5c18216b, byte-identical to the board's module.
  QEMU a3c788ff13d77cca, whose code is identical to a5dec6a32c49b3d3: they differ only in the `__FILE__` build path in
  .rodata.str1.8 and in the build-id.
- 0 RVC in all four. Control: 3 CSRUNV with no zeroing write. Revision: 6 CSRUNV (param on/off copies), of which
  exactly the 3 on-path ones are followed by `csrw 0x5c1, zero`.

**Costs:**
- Each 0x5c1 write raises flush_o at commit (csr_regfile.sv:2014): one full pipeline flush per CSRUNV return and one
  per window end, since restore_root's 0 is now always a change. Not expected to matter at the module's call rates;
  measured if the Tier 1 boot gives a cycle count.
- **Residual:** with srevroot 0 the caller sits in the PHYSICAL namespace, where a free-list pop writes the popped
  record with valid 0 and broadcasts the allocated index (capstone_rev_node.anvil:288-307; tap ex_stage.sv:1689). The
  module issues no lifetime op outside its windows, so the expected remaining source is the monitor's own region
  operations at M (csupctl.table clear, i.e. physical), followed by the mret's flush. The workaround moves that
  broadcast; it does not remove it.

**Predictions:**
- **QEMU** (which does not model the trackers): Tier 1 with srz equal to the control module run in the same session,
  i.e. the same PASS count, the same FALSE set, the same 22 exits in order, and the same kernel-line counts (banner 2,
  'collect skipped'). The 15-arm chain on a kit carrying srz equals gate6-t3-rev2 arm by arm. `srevroot_zero` reads
  back 1.
- **Board** (one variable, sql0 with the old module against srz): DRT1 17 down to 0. A nonzero count means the residual
  or the mechanism is wrong, and the RFSR readings on pb10-csr and sql0-csr say which.
