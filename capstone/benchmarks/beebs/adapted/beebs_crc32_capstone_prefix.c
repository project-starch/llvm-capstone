/* Capstone-adapted prefix for the BEEBS crc32 benchmark.
 *
 * The upstream source typedefs DWORD as unsigned long. On Capstone that is
 * 64 bits, but the benchmark's table-indexing code expects 32-bit CRC table
 * entries and computes offsets with a 4-byte stride. Keep DWORD explicitly
 * 32-bit for the domain build.
 */

#include "support.h"

#define SCALE_FACTOR (REPEAT_FACTOR >> 5)

typedef unsigned char BYTE;
typedef unsigned int DWORD;
typedef unsigned short WORD;

#define UPDC32(octet, crc) \
  (crc_32_tab[((crc) ^ ((BYTE)octet)) & 0xff] ^ ((crc) >> 8))

typedef DWORD UNS_32_BITS;

