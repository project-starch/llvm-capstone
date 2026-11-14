/* A minimal .dom image for the trap route's first program: two page-aligned PT_LOAD segments,
 * text (R X) and data (R W) right after it,. */
OUTPUT_ARCH(riscv)
ENTRY(_start)
PHDRS { text PT_LOAD FLAGS(5); data PT_LOAD FLAGS(6); }
SECTIONS {
    . = 0x10000;
    .text : { *(.text*) . = ALIGN(4096); } :text
    .data : { *(.data*) *(.rodata*) } :data
    /DISCARD/ : { *(.riscv.attributes) *(.comment) *(.note*) }
    __data_off = ADDR(.data) - ADDR(.text);
}
