ENTRY(_start)

PHDRS {
  text PT_LOAD FLAGS(5);
  data PT_LOAD FLAGS(6);
}

SECTIONS {
  . = 0x10000;

  .text : ALIGN(0x1000) {
    *(.text.entry)
    *(.text .text.*)
    *(.rodata .rodata.*)
  } :text

  .data : ALIGN(0x1000) {
    *(.data .data.*)
    *(.sdata .sdata.*)
    *(.gct)
  } :data

  .bss (NOLOAD) : ALIGN(0x1000) {
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
  } :data

  /DISCARD/ : {
    *(.comment)
    *(.note*)
    *(.eh_frame*)
    *(.riscv.attributes)
  }
}

