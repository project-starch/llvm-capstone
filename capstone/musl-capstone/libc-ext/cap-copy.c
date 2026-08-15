/* Tag-preserving block copy. See cap-copy.h for WHY this cannot be a byte loop.
 *
 * BUILT AT -O0, AND THAT IS NOT A PREFERENCE (see build-musl-capstone.sh, which
 * enforces it, and ISSUES.md C-28).
 *
 * Deciding whether a capability CAN be moved needs the low 4 bits of the two
 * addresses, because tags exist only at 16-byte-aligned addresses. The frontend
 * compiles `(uintptr_t)p & 15` correctly -- uintptr_t is 64-bit on this target,
 * and the IR at -O0 reads `ptrtoint ptr addrspace(200) to i64`, `and i64`. At
 * -O1 InstCombine RE-WIDENS that to the pointer's own width:
 *
 *     %0 = ptrtoint ptr addrspace(200) %dest to i128
 *     %3 = and i128 %2, 15
 *
 * and instruction selection cannot match a 128-bit `and`, so the compiler dies
 * with "Cannot select: i128 = and". Measured, not guessed: four spellings were
 * tried and all four crash -- xor of the two addresses, each address masked
 * separately, aligning one pointer and testing the other against zero, and
 * clang's own `__builtin_is_aligned`. The fold does not care how it is written.
 *
 * This is the real reason musl's nine word-at-a-time src/string files "do not
 * compile for this target"; the note in malloc.c blamed `(uintptr_t)s % ALIGN`,
 * which is the source line, not the mechanism.
 *
 * The proper fix is in the backend -- i128 and/or/xor want the same treatment
 * ISSUES.md C-25 gave i128 sub, truncate to XLen and extend back. Until that is
 * made and validated, one small file at -O0 buys the correctness now, and the
 * cost is bounded because the file contains nothing but these two loops.
 */
#include <stddef.h>

#include "cap-copy.h"

/* 64-bit on this target: pointers are 128-bit but uintptr_t is not, which is
   exactly the property the fold above destroys. The cast is ONE-WAY -- reading a
   capability's address as an integer is fine, it is the cast BACK that forges an
   untagged pointer (libc-ext/errno.c). */
typedef __UINTPTR_TYPE__ cap_copy_uptr;
#define LOW4(p) ((cap_copy_uptr)(p) & 15u)

void __capstone_cap_copy_fwd(void *dest, const void *src, size_t n) {
  unsigned char *d = dest;
  const unsigned char *s = src;

  /* Congruent mod 16 or no capability could survive the move at all, whatever
     shape the copy takes, and the byte loop below is already correct. */
  if (LOW4(d) == LOW4(s)) {
    size_t head = (size_t)((16u - LOW4(d)) & 15u);
    if (head > n)
      head = n;
    for (size_t i = 0; i < head; i++)
      d[i] = s[i];
    d += head;
    s += head;
    n -= head;

    /* `*(void **)` is what emits the capability-wide access, and it dispatches
       on the memory tag, so a scalar in the range still moves as a scalar. */
    for (; n >= 16; d += 16, s += 16, n -= 16)
      *(void **)d = *(void *const *)s;
  }

  for (size_t i = 0; i < n; i++)
    d[i] = s[i];
}

void __capstone_cap_copy_bwd(void *dest, const void *src, size_t n) {
  unsigned char *d = dest;
  const unsigned char *s = src;

  if (LOW4(d) == LOW4(s)) {
    /* Align the END of the range, since that is the side this starts from; the
       congruence above then puts s + n on a boundary too. */
    while (n && LOW4(d + n)) {
      n--;
      d[n] = s[n];
    }
    while (n >= 16) {
      n -= 16;
      *(void **)(d + n) = *(void *const *)(s + n);
    }
  }

  while (n) {
    n--;
    d[n] = s[n];
  }
}
