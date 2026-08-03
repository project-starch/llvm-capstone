#ifndef BL8K_H
#define BL8K_H
/* R-16 BLOB axis -- the one thing that separates the entering and stalling SQLite images.
 * f10 (STATIC_BUILTINS=0) ENTERS with blob 75120; swa/strim (STATIC_BUILTINS=1) STALL with
 * blob 84336/82592. Carve count (181), storage (354320) and allocation class (order 9) are
 * IDENTICAL across them, so the blob is the only geometric difference.
 * The blob is the initialised-globals template COPIED AT DOMAIN ENTRY, before domain_main --
 * exactly where R-16 stalls. Earlier probes never grew it: .bss is uninitialised and .rodata
 * padding is never copied. This grows it with genuinely INITIALISED data.
 * The array is NOT static: the generator's large-RO copy path emits `lla <sym>`, so the
 * symbol must have external linkage and a size that is a multiple of 8. */
char bl8k_arr[8192] = { 1 };
static char bl8k_g[2] = { 9, 0 };
static unsigned bl8k_compute(void)
{
  bl8k_arr[0] = 1; bl8k_arr[8191] = 1;
  return (unsigned)(unsigned char)bl8k_g[0]
       + (unsigned)(unsigned char)bl8k_arr[0] - 1u;
}
#endif
