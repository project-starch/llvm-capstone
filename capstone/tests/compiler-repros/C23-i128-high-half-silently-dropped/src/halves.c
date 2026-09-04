/* C-23 reproducer.  No headers, freestanding, no libc. */
typedef unsigned long long u64;
typedef unsigned __int128 u128;

/* The value written must be a in bits 0..63 and b in bits 64..127. */
void store_assembled(u128 *out, u64 a, u64 b) {
  *out = (u128)a | ((u128)b << 64);
}

/* A full 128-bit compare.  Must read both halves of x. */
int eq_full(u128 x, u64 a, u64 b) {
  return x == ((u128)a | ((u128)b << 64));
}

/* POSITIVE CONTROL: b is returned, so a2 must appear in this function.
   If the detector cannot find a2 here, it cannot find it anywhere and a
   clean result above means nothing. */
u64 control_returns_b(u128 *out, u64 a, u64 b) {
  (void)out; (void)a;
  return b;
}
