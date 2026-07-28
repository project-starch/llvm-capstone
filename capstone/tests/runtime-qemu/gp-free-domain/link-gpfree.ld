/* gp-free domain linker script (silicon).
 *
 * Identical to my_first_domain/link.ld EXCEPT it forces all globals
 * (.rodata/.data/.capstone_cap_init/.bss/.gct) to start at a FIXED image offset
 * GPFREE_GLOBALS_OFFSET = 0x1000, immediately after .text. This lets the monitor
 * derive the domain's `gp` on real silicon by SPLITting the code image at
 * base+0x1000 into an execute code cap (PCC = [base, base+0x1000)) and an R/W
 * globals cap (gp = [base+0x1000, base+code_size)) -- both via the real SPLIT
 * instruction, never the QEMU-only C_GEN_CAP. See
 * plans/gp-cap-derive-on-silicon-proposal.md and
 * history/22-07-2026_18-05-00_gp-free-silicon-smoke-*.md.
 *
 * The offset MUST match GPFREE_GLOBALS_OFFSET in the monitor create_domain.
 * .text must fit in [0, 0x1000) (true for the proof app; text ~0x310).
 */
ENTRY(_start)

PHDRS {
  domain PT_LOAD FLAGS(0x7);
}

SECTIONS {
  . = 0x10000;

  .text : ALIGN(16) {
    *(.text.entry)
    *(.text .text.*)
  } :domain

  /* Force globals to a fixed offset (0x1000) from the image base so the monitor
     can SPLIT the image into code|globals at a boundary it knows a priori. */
  . = 0x10000 + 0x1000;

  /* Anchor at the globals base (== base + GPFREE_GLOBALS_OFFSET). The cap-table
     glue uses `lla <sym> - __gpfree_globals_base` to compute a global's offset into
     the monitor-copied initializer blob at the front of dom_data (large-RO delivery
     path). Kept as an absolute address symbol so it is linkable from the glue TU. */
  __gpfree_globals_base = .;

  /* The blob-relative init descriptor MUST be the first thing in the globals
     region, for two reasons.
     (1) It has to be INSIDE the blob at all. The monitor copies
         [base+GPFREE_GLOBALS_OFFSET, base+code_size) into the front of dom_data;
         as an orphan section lld placed this at ~0x10cb0, i.e. BELOW
         __gpfree_globals_base, so it was never copied and the entry-glue
         interpreter could not have read it.
     (2) Placing it first makes its blob offset exactly 0, so the glue finds it
         with no `lla` arithmetic at all -- it is simply the front of dom_data.
     KEEP() because the SQLite build links with --gc-sections and nothing
     references this section from code; without it, it is collected. */
  .capstone_gp_initdesc : ALIGN(16) {
    __capstone_gp_initdesc_start = .;
    KEEP(*(.capstone_gp_initdesc))
    __capstone_gp_initdesc_end = .;
  } :domain

  .rodata : ALIGN(16) {
    *(.srodata .srodata.*)
    *(.rodata .rodata.*)
  } :domain

  .data : ALIGN(16) {
    *(.sdata .sdata.*)
    *(.data .data.*)
  } :domain

  .capstone_cap_init : ALIGN(8) {
    __capstone_cap_init_start = .;
    KEEP(*(.capstone_cap_init))
    __capstone_cap_init_end = .;
  } :domain

  .bss (NOLOAD) : ALIGN(16) {
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(16);
  } :domain

  .gct : ALIGN(16) {
    KEEP(*(.gct))
  }

  /DISCARD/ : {
    *(.comment)
    *(.note*)
    *(.eh_frame*)
    *(.llvm_addrsig)
    *(.riscv.attributes)
  }
}
