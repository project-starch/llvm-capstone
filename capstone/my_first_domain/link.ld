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

  .rodata : ALIGN(16) {
    *(.srodata .srodata.*)
    *(.rodata .rodata.*)
  } :domain

  .data : ALIGN(16) {
    *(.sdata .sdata.*)
    *(.data .data.*)
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




