/* usage: clang ... -DBOARD_TEST_FILE='"sup-quantum.S"' -DBOARD_TEST_CH=81 board_wrap.S */
/* b5b2-bare (2026-11-24): the board header is NOT included ahead of the test. Each test includes "asm_insn.h"
 * itself, which resolves to inc/asm_insn.h (first on the include path), so the macros are defined from the same
 * line as in the tree; included first, they expanded macro names in the tests' leading `#` comments. */
#define BOARD_UART      0x10000000
#include "board_prologue.S"
#include BOARD_TEST_FILE
#include "asm_insn.h"
.data
.align 6
.globl board_rec
board_rec:
  .zero (8 * (BOARD_REC_MAX + 1))
board_rec_end:
.align 3
board_save:
  .zero 24
