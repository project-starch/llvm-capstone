/* usage: clang ... -DBOARD_TEST_FILE='"sup-quantum.S"' -DBOARD_TEST_CH=81 board_wrap.S */
#include "asm_insn.h"
#include "board_prologue.S"
#include BOARD_TEST_FILE
.data
.align 6
.globl board_rec
board_rec:
  .zero (8 * (BOARD_REC_MAX + 1))
board_rec_end:
