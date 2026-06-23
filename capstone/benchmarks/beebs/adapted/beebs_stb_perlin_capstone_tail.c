/*
 * Capstone adapted tail for BEEBS `stb_perlin`.
 *
 * Upstream `verify_benchmark` returns -1, but the benchmark already contains a
 * complete, self-contained oracle: `benchmark()` computes a 10x10 Perlin-noise
 * plane and compares every value against a `static const float expected[10][10]`
 * global (which lives in .rodata, so there is no Bug #9 stack-copy hazard),
 * returning 0 iff all 100 values match exactly and 1 on any mismatch.  The
 * comparison is exact float equality; we confirmed on the host
 * (gcc -O0 -ffp-contract=off) that the float results match the embedded table
 * bit-for-bit, and the soft-float build uses -ffp-contract=off so no FMA
 * contraction can diverge.  We therefore just check that the benchmark reported
 * a full match.  The build script macro-renames the upstream verify stub.
 */
#undef verify_benchmark

int verify_benchmark(int res) { return res == 0; }
