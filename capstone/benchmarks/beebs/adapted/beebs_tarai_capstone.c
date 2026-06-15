int x, y, z;

#define CAPSTONE_DELIN(rd) \
  __asm__ volatile (".insn r 0x5b, 0x1, 0x3, %0, x0, x0" : "+r"(rd))

static __attribute__((noinline)) int *beebs_tarai_x_ptr(void) {
  int *p = &x;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_tarai_y_ptr(void) {
  int *p = &y;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int *beebs_tarai_z_ptr(void) {
  int *p = &z;
  CAPSTONE_DELIN(p);
  return p;
}

static __attribute__((noinline)) int beebs_tarai_x_get(void) {
  int *p = beebs_tarai_x_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_tarai_x_set(int value) {
  int *p = beebs_tarai_x_ptr();
  *p = value;
}

static __attribute__((noinline)) int beebs_tarai_y_get(void) {
  int *p = beebs_tarai_y_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_tarai_y_set(int value) {
  int *p = beebs_tarai_y_ptr();
  *p = value;
}

static __attribute__((noinline)) int beebs_tarai_z_get(void) {
  int *p = beebs_tarai_z_ptr();
  return *p;
}

static __attribute__((noinline)) void beebs_tarai_z_set(int value) {
  int *p = beebs_tarai_z_ptr();
  *p = value;
}

int tarai(int x_arg, int y_arg, int z_arg) {
  int ox = x_arg;
  int oy = y_arg;

  while (x_arg > y_arg) {
    ox = x_arg;
    oy = y_arg;

    x_arg = tarai(x_arg - 1, y_arg, z_arg);
    y_arg = tarai(y_arg - 1, z_arg, ox);

    if (x_arg <= y_arg)
      break;

    z_arg = tarai(z_arg - 1, ox, oy);
  }

  return y_arg;
}

int benchmark(void) {
  volatile int cnt = 0;
  cnt = tarai(beebs_tarai_x_get(), beebs_tarai_y_get(),
              beebs_tarai_z_get());
  return cnt;
}

void initialise_benchmark(void) {
  beebs_tarai_x_set(9);
  beebs_tarai_y_set(6);
  beebs_tarai_z_set(3);
}

int verify_benchmark(int r) {
  return r == 9;
}
