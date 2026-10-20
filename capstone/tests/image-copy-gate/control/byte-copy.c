/* Planted control for check-image-copies.py: a freestanding image whose memcpy and memmove copy bytes, as a domain
 * that brings its own does. The gate must report NO-CAP-COPY for both. `copy` moves 32 bytes of unknown alignment,
 * which since #246 calls memcpy rather than being inlined as scalar loads and stores. */
typedef __SIZE_TYPE__ size_t;
void *memcpy(void *restrict d, const void *restrict s, size_t n) {
  unsigned char *dp = d; const unsigned char *sp = s;
  while (n--) { __asm__ volatile("" ::: "memory"); *dp++ = *sp++; }
  return d;
}
void *memmove(void *d, const void *s, size_t n) {
  unsigned char *dp = d; const unsigned char *sp = s;
  if (dp < sp) while (n--) { __asm__ volatile("" ::: "memory"); *dp++ = *sp++; }
  else { dp += n; sp += n; while (n--) { __asm__ volatile("" ::: "memory"); *--dp = *--sp; } }
  return d;
}
void copy(char *d, const char *s) { __builtin_memcpy(d, s, 32); }
void move(char *d, const char *s, size_t n) { memmove(d, s, n); }
void _start(void) { for (;;) {} }
