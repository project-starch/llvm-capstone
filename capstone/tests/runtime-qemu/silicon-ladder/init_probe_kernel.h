#ifndef INIT_PROBE_KERNEL_H
#define INIT_PROBE_KERNEL_H
/* Silicon-ladder generator probe: exercises the INITIALIZED-global path.
 *
 * LUT is a const (initialized) table -> the generator must materialize its values
 * into the carved cap-table storage as li/sd immediates (no image data-read). acc
 * is .bss. The checksum depends on the LUT values, so a wrong materialization is
 * caught. Shared by the domain and a native oracle. */
static const int LUT[8] = {101, 202, 303, 404, 505, 606, 707, 808};
static const unsigned MIX = 0x9E3779B1u; /* scalar initialized global too */
static int acc[8];

static unsigned ip_compute(void) {
  for (int i = 0; i < 8; i++) acc[i] = LUT[i] * 3 + i;
  unsigned h = 2166136261u;
  for (int i = 0; i < 8; i++) {
    unsigned v = (unsigned)acc[i] ^ MIX;
    for (int b = 0; b < 4; b++) { h ^= (v >> (8 * b)) & 0xffu; h *= 16777619u; }
  }
  return h;
}
#endif
