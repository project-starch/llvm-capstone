#include "coremark.h"

/*
 * core_init_state() replacement for Capstone PureCap.
 *
 * Three issues with the upstream version on Capstone PureCap:
 *
 * 1. Static pointer arrays (intpat[], floatpat[], etc.).
 *    Each element is a 16-byte capability that must be runtime-initialized
 *    before ldc can load it.
 *
 * 2. gp-derived capabilities are CAP_TYPE_LIN (linear).
 *    cincoffset rd, rs1, rs2 consumes (nullifies) rs1 when rd != rs1.  The
 *    compiler hoists the gp-relative address computations for the four flat
 *    arrays before the outer while loop.  Those LINEAR caps (t1..t4) are
 *    consumed on the first outer iteration; subsequent iterations see NULL
 *    and the cscincoffset assertion fires.
 *
 * 3. buf[j] in the inner copy loop generates cincoffset tmp, buf, j which
 *    also consumes buf on the first inner iteration.
 *
 * Fix for issue 2: each pattern-table pointer is obtained via a small
 * __attribute__((noinline)) helper.  Because the helper is not inlined, the
 * compiler cannot hoist its gp-relative GEP out of the outer loop; each call
 * recomputes the address via a fresh gp auto-recovery.  The helper then
 * applies the delin instruction (.insn r 0x5b, 0x1, 0x3, rd, x0, x0) to
 * convert the LINEAR result to CAP_TYPE_NONLIN before returning, making it
 * safe to use across multiple outer iterations and in the inner loop.
 *
 * Fix for issue 3: replace buf[j] with *buf followed by buf++ so the inner
 * loop uses lbu at offset 0 (safe for LINEAR) and the in-place
 * cincoffsetimm buf, buf, 1 (rs1==rd, never consumed).
 */

static const char capstone_intpat[4][5] = {
    "5012", "1234", "-874", "+122"
};
static const char capstone_floatpat[4][9] = {
    "35.54400", ".1234500", "-110.700", "+0.64400"
};
static const char capstone_scipat[4][9] = {
    "5.500e+3", "-.123e-2", "-87e+832", "+0.6e-12"
};
static const char capstone_errpat[4][9] = {
    "T0.3e-1F", "-T.T++Tq", "1T3.4e4z", "34.0e-T^"
};

/* Delinearise rd: convert a LINEAR capability to NONLIN in-place.
 * The delin instruction (opcode=0x5b, f3=0x1, f7=0x3) is only valid on a
 * CAP_TYPE_LIN register; the helpers below guarantee this precondition. */
#define CAPSTONE_DELIN(rd) \
    __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

/*
 * These helpers are noinline so the compiler cannot hoist the gp-relative
 * address computation for each flat array out of the caller's outer loop.
 * Every call site gets a fresh gp auto-recovery, then delin converts the
 * result to NONLIN so the caller can reuse the pointer safely.
 */
static __attribute__((noinline)) const char *
capstone_intpat_ptr(ee_u32 idx)
{
    const char *p = capstone_intpat[idx];
    CAPSTONE_DELIN(p);
    return p;
}

static __attribute__((noinline)) const char *
capstone_floatpat_ptr(ee_u32 idx)
{
    const char *p = capstone_floatpat[idx];
    CAPSTONE_DELIN(p);
    return p;
}

static __attribute__((noinline)) const char *
capstone_scipat_ptr(ee_u32 idx)
{
    const char *p = capstone_scipat[idx];
    CAPSTONE_DELIN(p);
    return p;
}

static __attribute__((noinline)) const char *
capstone_errpat_ptr(ee_u32 idx)
{
    const char *p = capstone_errpat[idx];
    CAPSTONE_DELIN(p);
    return p;
}

void core_init_state(ee_u32 size, ee_s16 seed, ee_u8 *p) {
    ee_u32 total = 0, next = 0;
    const char *buf = 0;
    ee_u32 sub_idx;

    size--;
    while ((total + next + 1) < size) {
        if (next > 0) {
            ee_u32 j;
            for (j = 0; j < next; j++) {
                *(p + total + j) = (ee_u8)*buf;
                buf++;
            }
            *(p + total + next) = ',';
            total += next + 1;
        }
        seed++;
        sub_idx = (ee_u32)((seed >> 3) & 0x3);
        switch (seed & 0x7) {
            case 0: case 1: case 2:
                buf = capstone_intpat_ptr(sub_idx);
                next = 4;
                break;
            case 3: case 4:
                buf = capstone_floatpat_ptr(sub_idx);
                next = 8;
                break;
            case 5: case 6:
                buf = capstone_scipat_ptr(sub_idx);
                next = 8;
                break;
            case 7:
                buf = capstone_errpat_ptr(sub_idx);
                next = 8;
                break;
            default:
                break;
        }
    }
    size++;
    while (total < size) {
        *(p + total) = 0;
        total++;
    }
}
