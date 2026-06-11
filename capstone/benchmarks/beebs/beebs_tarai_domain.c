extern void initialise_benchmark(void);
extern int benchmark(void);
extern int verify_benchmark(int result);

#define BEEBS_RET_CORRECT 0xC171C0DEU
#define BEEBS_RET_WRONG   0xC171BAD0U

void domain_main(unsigned *res, unsigned func) {
  (void)func;

  initialise_benchmark();
  int result = benchmark();
  int correct = verify_benchmark(result);

  if (res)
    *res = correct ? BEEBS_RET_CORRECT : BEEBS_RET_WRONG;
}
