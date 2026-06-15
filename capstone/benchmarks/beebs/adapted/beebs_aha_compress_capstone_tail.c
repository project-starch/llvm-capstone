/* Capstone-adapted tail for BEEBS aha-compress.
 *
 * The upstream compress_test.c defines a global symbol 'test' which
 * conflicts with the Capstone domain entry point of the same name in
 * start.S.  This tail replaces everything from the test[] array onwards,
 * making the array static to give it internal linkage.
 *
 * The four compression functions (compress1..4) are kept from the upstream
 * source; only the data array and benchmark wrapper are replaced here.
 *
 * Workaround: the Capstone backend treats cincoffset as commutative (like
 * ADD) but the ISA requires rs1=capability.  Multiple independent array
 * accesses via a variable index each regenerate the base-pointer with
 * cincoffset, and the second+ regenerations get the operands swapped.
 * Fix: compute row = test_data + i once (single correct cincoffset via the
 * gp+PCREL pattern), DELIN it so it is reusable, then load all three
 * elements via constant-offset loads (ld val, N(cap)) which emit no
 * cincoffset at all.
 */

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static const unsigned long test_data[] = {
/*     Data        Mask       Result */
    0xFFFFFFFF, 0x80000000, 0x00000001,
    0xFFFFFFFF, 0x0010084A, 0x0000001F,
    0xFFFFFFFF, 0x55555555, 0x0000FFFF,
    0xFFFFFFFF, 0x88E00F55, 0x00001FFF,
    0x01234567, 0x0000FFFF, 0x00004567,
    0x01234567, 0xFFFF0000, 0x00000123,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
    0,          0,          0,
    0,          0xFFFFFFFF, 0,
    0xFFFFFFFF, 0,          0,
    0x80000000, 0x80000000, 1,
    0x55555555, 0x55555555, 0x0000FFFF,
    0x55555555, 0xAAAAAAAA, 0,
    0x789ABCDE, 0x0F0F0F0F, 0x00008ACE,
    0x789ABCDE, 0xF0F0F0F0, 0x000079BD,
    0x92345678, 0x80000000, 0x00000001,
    0x12345678, 0xF0035555, 0x000004ec,
    0x80000000, 0xF0035555, 0x00002000,
};

void initialise_benchmark(void) {}

int benchmark(void) {
  int errors = 0, n, i;
  const unsigned long *row;
  unsigned long d, m, e;

  n = (int)(sizeof(test_data) / sizeof(test_data[0]));

  for (i = 0; i < n; i += 3) {
    row = test_data + i;
    CAPSTONE_DELIN(row);
    d = row[0]; m = row[1]; e = row[2];
    if (compress1((unsigned)d, (unsigned)m) != (unsigned int)e) errors = 1;
  }
  for (i = 0; i < n; i += 3) {
    row = test_data + i;
    CAPSTONE_DELIN(row);
    d = row[0]; m = row[1]; e = row[2];
    if (compress2((unsigned)d, (unsigned)m) != (unsigned int)e) errors = 1;
  }
  for (i = 0; i < n; i += 3) {
    row = test_data + i;
    CAPSTONE_DELIN(row);
    d = row[0]; m = row[1]; e = row[2];
    if (compress3((unsigned)d, (unsigned)m) != (unsigned int)e) errors = 1;
  }
  for (i = 0; i < n; i += 3) {
    row = test_data + i;
    CAPSTONE_DELIN(row);
    d = row[0]; m = row[1]; e = row[2];
    if (compress4((unsigned)d, (unsigned)m) != (unsigned int)e) errors = 1;
  }

  return errors;
}

int verify_benchmark(int r) {
  return (r == 0) ? 1 : 0;
}
