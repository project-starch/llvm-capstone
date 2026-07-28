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
  nobits PT_NULL;
}

SECTIONS {
  . = 0x10000;
  .text : ALIGN(16) {
    *(.text.entry)
    *(.text .text.*)
  } :domain

  /* Globals start at GPFREE_GLOBALS_OFFSET from the image base. The monitor no
     longer has to know this a priori -- it reads it from .capstone_domhdr above --
     so it is a build knob rather than a shared constant. build-ladder-domain.sh
     substitutes the token for DOMAIN_WINDOW; the default keeps every existing rung
     byte-identical. This is what replaces link-gpfree-32k.ld, whose separate copy of
     the constant was the thing the monitor silently disagreed with. */
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

  /* End of the LOADED globals template: rodata + data + capstone_cap_init. This is
     exactly the byte range the monitor copies into the front of dom_data. */
  . = ALIGN(16);
  __gpfree_globals_tmpl_end = .;

  /* .bss is deliberately NOT in the :domain PT_LOAD.
     Under gp-captable every global's storage is CARVED FROM dom_data by the entry
     glue, so the image's .bss is never touched -- it is pure padding that the loader
     nonetheless counts, because libcapstone derives code_len from p_memsz. Leaving it
     in costs dom_data the whole .bss size, and for SQLite (a 1 MiB memsys5 arena)
     that is the difference between fitting and not: dom_data comes out ~860 KB with
     .bss in the PT_LOAD versus ~1.92 MB with it out, against ~1.47 MB needed.
     PT_NULL keeps the addresses assigned (so symbols resolve and the glue's zero path
     still knows the sizes) while dropping it from the loaded image. */
  .bss (NOLOAD) : ALIGN(16) {
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(16);
  } :nobits

  . = ALIGN(16);
  __gpfree_globals_end = .;
  __gpfree_globals_tmpl_len   = __gpfree_globals_tmpl_end - __gpfree_globals_base;
  __gpfree_globals_region_len = __gpfree_globals_end - __gpfree_globals_base;

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
