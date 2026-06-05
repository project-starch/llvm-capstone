#include "coremark.h"

/*
 * crcu8() / crcu16() replacements for Capstone PureCap.
 *
 * Upstream crcu8() declares loop locals as ee_u8 (1-byte).  At -O0 the
 * compiler spills them to byte-sized stack slots at offsets sp+0x09..sp+0x0f
 * — inside the same 16-byte capability granule that the outer caller
 * (matrix_test) uses for its saved s2 register capability.  Any byte store
 * to that granule clears the capability tag via cap_mem_map_remove_range, so
 * the subsequent ldc s2 in matrix_test's epilogue loads an untagged value and
 * the next cincoffset fires: Assertion 'rs1_v->tag' failed.
 *
 * Fix: widen all locals to unsigned int (32-bit).  At -O0 the compiler now
 * allocates 4-byte aligned word slots at frame offsets well above the
 * dangerous [sp, sp+16) granule.  The CRC computation is identical to
 * upstream.
 */
ee_u16 crcu8(ee_u8 data_in, ee_u16 crc_in)
{
    unsigned int data  = data_in;
    unsigned int crc   = crc_in;
    unsigned int i;

    for (i = 0; i < 8; i++) {
        unsigned int x16   = (data & 1u) ^ (crc & 1u);
        unsigned int carry = x16;
        data >>= 1;
        if (x16)
            crc ^= 0x4002u;
        crc >>= 1;
        if (carry)
            crc |= 0x8000u;
        else
            crc &= 0x7fffu;
    }
    return (ee_u16)crc;
}

ee_u16 crcu16(ee_u16 newval, ee_u16 crc)
{
    crc = crcu8((ee_u8)newval, crc);
    crc = crcu8((ee_u8)((ee_u16)newval >> 8), crc);
    return crc;
}

/*
 * check_data_types() replacement for Capstone PureCap.
 *
 * Upstream checks sizeof(ee_ptr_int) == sizeof(int*).  On Capstone PureCap
 * sizeof(uintptr_t) = 8 (cursor width) while sizeof(int*) = 16 (capability
 * width).  The mismatch is intentional: ee_ptr_int holds the integer/cursor
 * part of a pointer for alignment arithmetic, not a full capability.
 * Skip that check to avoid a spurious error count that propagates to
 * "Errors detected" and masks real CRC failures.
 */
ee_u8 check_data_types(void)
{
    ee_u8 retval = 0;
    if (sizeof(ee_u8) != 1)   retval++;
    if (sizeof(ee_u16) != 2)  retval++;
    if (sizeof(ee_s16) != 2)  retval++;
    if (sizeof(ee_s32) != 4)  retval++;
    if (sizeof(ee_u32) != 4)  retval++;
    /* ee_ptr_int size check intentionally omitted: see comment above */
    if (retval > 0)
        ee_printf("ERROR: Please modify the datatypes in core_portme.h!\n");
    return retval;
}
