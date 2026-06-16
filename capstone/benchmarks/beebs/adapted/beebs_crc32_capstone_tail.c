/* Capstone-adapted tail for the BEEBS crc32 benchmark.
 *
 * The upstream verify_benchmark checks for the CRC result after 32 iterations
 * of benchmark() (the value produced when BOARD_REPEAT_FACTOR causes SCALE_FACTOR
 * to equal 32).  Our domain wrapper calls benchmark() once, so the static seed
 * in rand_beebs has only advanced 1024 steps rather than 32*1024 steps.  The
 * single-call result is 1703161001 (0x65842CA9), not 1207487004.
 *
 * This tail replaces verify_benchmark with a version that checks the
 * single-call expected value.
 */

int verify_benchmark(int r)
{
  int expected = 1703161001;  /* crc32pseudo() after 1 call with seed starting at 0 */

  if (r != expected)
    return 0;
  return 1;
}
