/*
 * Capstone adapted tail for BEEBS `fasta`.
 *
 * Upstream `fasta` discards all output (the fwrite/putchar calls are commented
 * out) and `verify_benchmark` returns -1, so there is nothing externally
 * observable.  We keep the deterministic generator core from the upstream source
 * (`myrandom` LCG + `accumulate_probabilities`) and reimplement the two
 * consumers (`repeat_fasta`/`random_fasta`) to fold every generated character
 * into an FNV-1a checksum instead of writing it out.  `verify_benchmark` then
 * compares that checksum against a host reference computed from this same source
 * + the same soft-float math (gcc -O0 -ffp-contract=off): `myrandom`'s f32 ops
 * are correctly-rounded on both host and target soft-float, so the generated
 * character stream is bit-identical and the comparison is exact.
 *
 * `aminoacid_t`, `myrandom`, `accumulate_probabilities`, and the `WIDTH`/`MIN`/
 * `NELEMENTS` macros are kept from the upstream source concatenated above.
 * `memcpy`/`strlen` come from the shared adapted/beebs_freestanding_string.c.
 * `repeat_fasta` uses a fixed static buffer rather than `alloca`.
 */

void *memcpy(void *, const void *, size_t);
size_t strlen(const char *);

static unsigned long fasta_fnv = 1469598103934665603UL;

static void fasta_fold(unsigned char c) {
  fasta_fnv ^= c;
  fasta_fnv *= 1099511628211UL;
}

static void repeat_fasta(char const *s, size_t count) {
  size_t pos = 0;
  size_t len = strlen(s);
  static char s2[512]; /* >= strlen(alu) + WIDTH (~347) */
  memcpy(s2, s, len);
  memcpy(s2 + len, s, WIDTH);
  do {
    size_t line = MIN(WIDTH, count);
    for (size_t k = 0; k < line; k++)
      fasta_fold((unsigned char)s2[pos + k]);
    fasta_fold((unsigned char)'\n');
    pos += line;
    if (pos >= len)
      pos -= len;
    count -= line;
  } while (count);
}

static void random_fasta(aminoacid_t const *genelist, size_t count) {
  do {
    size_t line = MIN(WIDTH, count);
    size_t pos = 0;
    char buf[WIDTH + 1];
    do {
      float r = myrandom(1.0);
      size_t i = 0;
      while (genelist[i].p < r)
        ++i; /* weighted linear search */
      buf[pos++] = genelist[i].c;
    } while (pos < line);
    buf[line] = '\n';
    for (size_t k = 0; k <= line; k++)
      fasta_fold((unsigned char)buf[k]);
    count -= line;
  } while (count);
}

void initialise_benchmark(void) {}

int benchmark(void) {
  const int n = 1000;

  static aminoacid_t iub[] = {
      {0.27, 'a'}, {0.12, 'c'}, {0.12, 'g'}, {0.27, 't'}, {0.02, 'B'},
      {0.02, 'D'}, {0.02, 'H'}, {0.02, 'K'}, {0.02, 'M'}, {0.02, 'N'},
      {0.02, 'R'}, {0.02, 'S'}, {0.02, 'V'}, {0.02, 'W'}, {0.02, 'Y'}};

  static aminoacid_t homosapiens[] = {{0.3029549426680, 'a'},
                                      {0.1979883004921, 'c'},
                                      {0.1975473066391, 'g'},
                                      {0.3015094502008, 't'}};

  static char const *const alu =
      "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG"
      "GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA"
      "CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT"
      "ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA"
      "GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG"
      "AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC"
      "AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA";

  accumulate_probabilities(iub, NELEMENTS(iub));
  accumulate_probabilities(homosapiens, NELEMENTS(homosapiens));

  repeat_fasta(alu, 2 * n);
  random_fasta(iub, 3 * n);
  random_fasta(homosapiens, 5 * n);
  return 0;
}

#undef verify_benchmark

int verify_benchmark(int res) {
  (void)res;
  return fasta_fnv == 0x24d70971e2d6dc0fUL;
}
