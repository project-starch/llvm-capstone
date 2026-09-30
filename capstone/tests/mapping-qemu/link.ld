ENTRY(_start)
SECTIONS {
  . = 0x80000000;
  .text : { *(.text.start) *(.text*) . = ALIGN(16); _code_end = .; }
  . = ALIGN(4096);
  .data : { *(.data*) *(.rodata*) }
  .bss : { *(.bss*) }
}
