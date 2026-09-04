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

  /* THE INIT AND FINI ARRAYS, defined so their MARKERS EXIST.
   *
   * musl's exit path walks the fini array:
   *
   *     extern weak hidden void (*const __fini_array_start)(void), (*const
   *                                  __fini_array_end)(void);
   *     uintptr_t a = (uintptr_t)&__fini_array_end;
   *     for (; a > (uintptr_t)&__fini_array_start; a -= sizeof(void(*)()))
   *
   * Undefined weak symbols resolve to ADDRESS ZERO, so the loop correctly never
   * runs -- but forming the address at all does not come free on this target.
   * Address zero derived against the image capability is 0x10000 BELOW its base
   * (the image is linked at 0x10000), and clang's subobject bounds then shrink
   * that to the 16 bytes of a function pointer. The result is a SHRINK whose
   * requested range lies outside the capability it shrinks, which the spec makes
   * an `Illegal operand value (29)`:
   *
   *     SHRINK illegal operand value: requested [1023f0000, 1023f0010),
   *                                   current  [102400000, ...)
   *
   * Reached by mruby's gem test suite, where an unimplemented symlink() raises a
   * Ruby exception and the driver calls exit(). Any domain that calls exit() has
   * the same hole.
   *
   * Defining the sections gives both markers real, EQUAL addresses inside the
   * image, so the comparison is false without any capability being formed out of
   * bounds. They are empty today -- verified across all 150 objects of the
   * largest domain and all 1346 archive members, none has an .init_array,
   * .fini_array, .ctors or .dtors -- so this adds symbols and no bytes.
   *
   * KEEP, and placed HERE rather than left as orphans: an orphan section is
   * placed by the linker wherever it likes, and the .bss note below explains why
   * "wherever it likes" is not a safe answer in this script.
   *
   * NOTE FOR THE DAY ONE APPEARS: nothing in a domain calls the INIT array.
   * .capstone_cap_init is a different table with its own loop in start.S, and a
   * real .init_array would need the same treatment there. The linker will keep
   * it; the entry glue will not run it. */
  .init_array : ALIGN(16) {
    __init_array_start = .;
    KEEP(*(.init_array))
    __init_array_end = .;
  } :domain

  .fini_array : ALIGN(16) {
    __fini_array_start = .;
    KEEP(*(.fini_array))
    __fini_array_end = .;
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

  .gct : ALIGN(16) {
    KEEP(*(.gct))
  }

  /* .bss LAST, and that is load-bearing rather than tidy.
   *
   * It is NOBITS, so it contributes no bytes of its own -- but a PT_LOAD's
   * p_filesz spans from its first to its last LOADED section, so any loaded
   * section placed AFTER .bss forces the segment to carry .bss's whole extent
   * in the file as padding. With .gct after it, a domain with a 256 KiB static
   * heap shipped 262,288 bytes of zeros: 19.5% of the mruby image, measured.
   *
   * They cost real time. The guest loader mmaps the .dom and memcpys out of it
   * (libcapstone.c:160-186) after memsetting the destination to zero, so those
   * bytes are copied to make memory zero that is already zero. Reading straight
   * from the 9p share, each 4 KiB page of that is an RPC under TCG.
   *
   * Nothing about the domain's MEMORY changes: the loader sizes the image from
   * p_memsz (`seg_end = p_vaddr + p_memsz`), which is unaffected by the order.
   * Only p_filesz shrinks. .gct is reached through __capstone_gct_start, so it
   * does not care where it sits. */
  .bss (NOLOAD) : ALIGN(16) {
    *(.sbss .sbss.*)
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(16);
  } :domain

  /DISCARD/ : {
    *(.comment)
    *(.note*)
    *(.eh_frame*)
    *(.llvm_addrsig)
    *(.riscv.attributes)
  }
}




