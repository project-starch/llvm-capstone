#include "coremark_hostcall.h"

/*
 * Shared regions received via REGION_SHARE invocations (first=metadata, second=payload).
 *
 * NOTE: domain_main is implemented in coremark_domain_entry.S rather than here.
 * The Capstone LLVM backend generates broken code for domain_main at every
 * optimization level:
 *   -O0: the prologue emits `cincoffsetimm s0, sp, N` (rd≠rs1) which consumes sp
 *        (LINEAR), then the epilogue's `ldc ra, N(sp)` crashes on sp.tag=0.
 *   -O1/-O2: the compiler merges the CALL and REGION_SHARE paths into a single
 *        sink where the same base register is used both as rs1 in cincoffsetimm
 *        (consuming it) and as the base for a subsequent sw, crashing on tag=0.
 * The hand-written assembly in coremark_domain_entry.S is capability-safe.
 */
volatile struct hostcall_v0 *hc_metadata = 0;
volatile char               *hc_payload  = 0;
unsigned                     g_region_count = 0;
