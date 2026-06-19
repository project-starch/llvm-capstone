/* Capstone-adapted benchmark tail for BEEBS slre.
 *
 * The upstream libslre.c uses a global array of char* pointers
 * (char *regexes[] = {...}) whose elements would be stored as untagged
 * integers in .data without caprelocs support.  This tail:
 *   - keeps the string data as a char array (address is a capability)
 *   - calls slre_match directly with string-literal arguments so the
 *     capability is derived from the string object (via LGA), not loaded
 *     from an untagged pointer slot in .data
 */

char text[] = "abbbababaabccababcacbcbcbabbabcbabcabcbbcbbac";

void initialise_benchmark(void) {}

int benchmark(void) {
  int len = strlen(text);
  struct slre_cap captures;
  volatile int ret = 0;
  ret += slre_match("(ab)+",           text, len, &captures, 1);
  ret += slre_match("(b.+)+",          text, len, &captures, 1);
  ret += slre_match("a[ab]*",          text, len, &captures, 1);
  ret += slre_match("([ab^c][ab^c])+", text, len, &captures, 1);
  return ret;
}

int verify_benchmark(int r) {
  return r == 102;
}
