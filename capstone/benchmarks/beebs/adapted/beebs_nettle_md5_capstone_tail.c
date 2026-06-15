/* Capstone-adapted tail for BEEBS nettle-md5.
 *
 * Replaces benchmark() and verify_benchmark() to avoid memcpy/memcmp
 * which are not available in the freestanding Capstone domain environment.
 * The MD5 compress core (_nettle_md5_compress) and its globals are kept
 * from the upstream source.
 */

int benchmark(void) {
  int i;
  for (i = 0; i < _MD5_DIGEST_LENGTH; i++)
    digest[i] = digest_ref[i];
  _nettle_md5_compress(digest, input);
  return 0;
}

int verify_benchmark(int unused) {
  (void)unused;
  if (digest[0] != 0xddaf8815U) return 0;
  if (digest[1] != 0x2149cb8fU) return 0;
  if (digest[2] != 0x9cdd75fdU) return 0;
  if (digest[3] != 0x14a43e27U) return 0;
  return 1;
}
