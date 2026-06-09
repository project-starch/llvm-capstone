extern void initialise_benchmark(void);
extern int benchmark(void);
extern int verify_benchmark(int result);

#define BEEBS_RET_CORRECT 0xF1BCA110U
#define BEEBS_RET_WRONG   0xF1BAD000U

void domain_main(unsigned *res, unsigned func) {
  (void)func;

  initialise_benchmark();
  int result = benchmark();
  int correct = verify_benchmark(result);

  if (res)
    *res = correct ? BEEBS_RET_CORRECT : BEEBS_RET_WRONG;
}
