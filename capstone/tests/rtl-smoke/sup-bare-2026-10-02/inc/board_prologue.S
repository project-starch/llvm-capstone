/* Integer-mode prologue at 0x80000000: runs BEFORE the test's _start, before CAPENTER, so plain
 * M-mode stores reach the UART and memory. Initialises the 16550 (25 MHz, 57600 8N1, reg shift 2,
 * 32-bit registers -- platform/fpga/ariane/platform.c), prints "SUPTEST BEGIN <id>", rewrites every
 * dword of .data with its own value and zeroes .bss (core stores, so a stale shadow tag left in DDR by
 * an earlier boot cannot survive into the test; the JTAG load writes data but not tags), then jumps
 * to the unmodified test. */
#define IPUTI(ch) li t1, ch; 91: lw t2, 20(t0); andi t2, t2, 32; beqz t2, 91b; sw t1, 0(t0)
.section .text.board, "ax"
.globl board_entry
board_entry:
    li t0, BOARD_UART
    sw zero, 4(t0)                 /* IER = 0 */
    li t1, 0x80; sw t1, 12(t0)     /* LCR: DLAB */
    li t1, 27;   sw t1, 0(t0)      /* DLL: 25e6 / (16 * 57600) = 27.1 */
    sw zero, 4(t0)                 /* DLM */
    li t1, 3;    sw t1, 12(t0)     /* LCR: 8N1 */
    li t1, 7;    sw t1, 8(t0)      /* FCR: enable + clear FIFOs */
    IPUTI(13); IPUTI(10)
    IPUTI(83); IPUTI(85); IPUTI(80); IPUTI(84); IPUTI(69); IPUTI(83); IPUTI(84); IPUTI(32)   /* "SUPTEST " */
    IPUTI(66); IPUTI(69); IPUTI(71); IPUTI(73); IPUTI(78); IPUTI(32)                       /* "BEGIN " */
    IPUTI(BOARD_TEST_CH); IPUTI(13); IPUTI(10)
    lla t3, _board_data_lo; lla t4, _board_data_hi
1:  bgeu t3, t4, 2f; ld t5, 0(t3); sd t5, 0(t3); addi t3, t3, 8; j 1b
2:  lla t3, _board_bss_lo; lla t4, _board_bss_hi
3:  bgeu t3, t4, 4f; sd zero, 0(t3); addi t3, t3, 8; j 3b
4:
#ifdef BOARD_NODE_INIT
    BOARD_NODE_INIT
#endif
    fence
    IPUTI(83); IPUTI(66); IPUTI(49); IPUTI(13); IPUTI(10)                                /* "SB1" */
    j _start
