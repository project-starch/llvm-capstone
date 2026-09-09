/*
 * Reproduce SQLite gap 3: bytewise copying a runtime struct containing a
 * tagged pointer copies its address bits but not the out-of-band capability
 * tag. The dereference through the copied field is expected to fault today.
 */
struct holder {
  const char *text;
  unsigned value;
};

static __attribute__((noinline)) void copy_bytes(void *destination,
                                                 const void *source,
                                                 unsigned long size) {
  unsigned char *dst = destination;
  const unsigned char *src = source;
  for (unsigned long i = 0; i < size; ++i)
    dst[i] = src[i];
}

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  struct holder source;
  struct holder destination;
  source.text = "tagged";
  source.value = 7;
  copy_bytes(&destination, &source, sizeof(destination));
  *res = (unsigned)destination.text[0] + destination.value;
}
