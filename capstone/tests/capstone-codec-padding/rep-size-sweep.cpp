// The compiler's representable-size helper (llvm/lib/Target/Capstone/MCTargetDesc/CapstoneRepresentable.h) against
// QEMU's codec (cap_compress.c, compress then uncompress at the base). One line per case:
//   kind base length expect qemu
// expect/qemu are EXACT or INEXACT. Kinds:
//   pad    [B, B + length(S)) with B a multiple of alignment(S): must be EXACT (the helper's claim);
//   small  S < 4096 at an arbitrary (odd) base: must be EXACT (the threshold's positive control);
//   raw    [B, B + S) with S >= 4096 not a multiple of its granule: must be INEXACT (the padding is needed);
//   skew   [B + alignment/2, + length(S)) when alignment >= 16: must be INEXACT (the alignment is needed);
//   short  [B, B + length(S) - 8) when length(S) - 8 >= S and >= 4096: must be INEXACT (the length is minimal).
// rep-size-sweep.py re-checks every line against the RTL's bounds model.
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include "CapstoneRepresentable.h"
extern "C" {
#include "cap_compress.h"
}
using namespace llvm::CapstoneRepresentable;

static uint64_t rng = 0x243f6a8885a308d3ULL;
static uint64_t next() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static long bad = 0;

static bool exact(uint64_t base, uint64_t len) {
  capfat_t c = {};
  c.bounds.base = base; c.bounds.end = base + len; c.bounds.cursor = base; c.type = CAP_TYPE_LIN;
  uint64_t lo, hi;
  cap_compress(&c, &lo, &hi);
  capfat_t d = {};
  cap_uncompress(lo, hi, &d);
  return d.bounds.base == base && d.bounds.end == base + len;
}

static void emit(const char *kind, uint64_t base, uint64_t len, bool expect) {
  bool q = exact(base, len);
  if (q != expect) ++bad;
  std::printf("%s %" PRIx64 " %" PRIx64 " %s %s\n", kind, base, len, expect ? "EXACT" : "INEXACT", q ? "EXACT" : "INEXACT");
}

static void size(uint64_t S) {
  if (!S) return;
  uint64_t L = length(S), A = alignment(S);
  if (L < S || (A & (A - 1))) { std::printf("HELPER-BROKEN %" PRIx64 "\n", S); ++bad; return; }
  for (int k = 0; k < 3; ++k) {
    uint64_t B = (0x3f80000000ULL + (next() & ((1ULL << 36) - 1))) & ~(A - 1);
    emit("pad", B, L, true);
    if (S < 4096) emit("small", B | 1, S, true);
    if (S >= 4096 && S % granule(S)) emit("raw", B, S, false);
    if (A >= 16) emit("skew", B + A / 2, L, false);
    if (L >= 8 + 4096 && L - 8 >= S) emit("short", B, L - 8, false);
  }
}

int main(int argc, char **argv) {
  unsigned long n = argc > 1 ? std::strtoul(argv[1], nullptr, 0) : 20000;
  for (int k = 4; k <= 34; ++k)
    for (int d = -9; d <= 9; ++d) size((1ULL << k) + (uint64_t)(int64_t)d);
  for (uint64_t s : {0xfffULL, 0x1000ULL, 0x1001ULL, 0x1fffULL, 0x2000ULL, 0x10000ULL, 0x80f00ULL, 0x7ffe0ULL})
    size(s);
  for (unsigned long i = 0; i < n; ++i) size(1 + (next() & ((1ULL << (1 + next() % 32)) - 1)));
  std::fprintf(stderr, "rep-size-sweep: qemu disagreements with the expectation: %ld\n", bad);
  return bad ? 1 : 0;
}
