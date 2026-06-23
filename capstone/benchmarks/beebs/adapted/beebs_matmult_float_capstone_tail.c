/* Capstone-adapted tail for the BEEBS matmult-float benchmark.
 *
 * Same upstream source as `matmult`, built with -DMATMULT_FLOAT: a float 10x10
 * matrix multiply (MATMULT_FLOAT sets UPPERLIMIT 10) into the global
 * ResultArray. Upstream verify_benchmark uses a local `float exp[][]` (Bug #3
 * non-power-of-two i128 stride + Bug #9 local const array); we replace it with
 * an FNV-1a checksum of the global ResultArray read as a flat byte stream (byte
 * stride -> no i128 GEP, global -> no Bug #9).
 *
 * The float multiply-accumulate is deterministic IEEE single-precision; built
 * -ffp-contract=off, the soft-float result is bit-identical to a native host
 * reference (gcc -O0 -ffp-contract=off) which produced the constant below.
 *
 * `matrix`, `ResultArray` and `UPPERLIMIT` are already declared in the upstream
 * source concatenated above this tail, so we reference them directly.
 */

int verify_benchmark(int res) {
  (void)res;
  unsigned long h = 1469598103934665603UL;
  const unsigned char *p = (const unsigned char *)ResultArray;
  for (unsigned i = 0; i < sizeof(ResultArray); i++) {
    h ^= p[i];
    h *= 1099511628211UL;
  }
  return h == 0xbdbace3d315e67a4UL;
}
