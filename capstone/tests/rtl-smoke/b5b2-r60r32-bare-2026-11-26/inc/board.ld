OUTPUT_ARCH( "riscv" )
ENTRY(board_entry)
SECTIONS
{
  . = 0x80000000;
  .text.board : { *(.text.board) }
  .text.init : { *(.text.init) }
  . = ALIGN(0x1000);
  .tohost : { *(.tohost) }
  . = ALIGN(0x1000);
  .uvmif : { *(.uvmif) }
  . = ALIGN(0x1000);
  .text : { *(.text) }
  . = ALIGN(0x1000);
  _board_data_lo = .;
  .data : { *(.data) }
  . = ALIGN(8);
  _board_data_hi = .;
  _board_bss_lo = .;
  .bss : { *(.bss) }
  . = ALIGN(8);
  _board_bss_hi = .;
  _end = .;
}
