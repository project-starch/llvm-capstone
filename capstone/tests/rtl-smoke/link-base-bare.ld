/* OpenSBI's fw_payload jumps to the payload at 0x80200000 (payload offset
   0x200000 from the 0x80000000 load base), which is exactly where the Linux Image
   used to sit. Link there so the same firmware prefix can be reused verbatim. */
ENTRY(_start)
SECTIONS {
  . = 0x80200000;
  .text : { *(.text.entry) *(.text .text.*) }
  .rodata : ALIGN(16) { *(.rodata .rodata.*) }
  .data : ALIGN(16) { *(.data .data.*) *(.sdata .sdata.*) }
  .bss : ALIGN(16) { __bss_start = .; *(.bss .bss.* .sbss .sbss.* COMMON) __bss_end = .; }
  . = ALIGN(16);
  . += 0x10000;              /* 64 KiB stack */
  __stack_top = .;
}
