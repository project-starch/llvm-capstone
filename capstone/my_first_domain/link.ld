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

  /* Capability-global initializer table (one 8-byte PC-relative entry per
     module with capability globals): each entry = (initfn - entry). start.S
     iterates [start,end) before domain_main and calls each initializer at
     entry_runtime + offset. Position-independent (the domain processes no
     load-time relocations). Empty (start == end) for domains with none. */
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




