/*
 * Capstone adapted tail for BEEBS `qurt`.
 *
 * Upstream `verify_benchmark` returns -1. The adapted benchmark captures all
 * three quadratic solves and checks their known roots within a tolerance. qurt
 * uses its own approximate sqrt, so exact bit equality is not appropriate.
 */
#undef benchmark
#undef verify_benchmark

extern float x1[2], x2[2];
extern float *a;
extern int flag;
extern int qurt(void);

static int qurt_res[3];
static int qurt_flag[3];
static float qurt_x1[3][2];
static float qurt_x2[3][2];

static void qurt_capture(int idx) {
  qurt_res[idx] = qurt();
  qurt_flag[idx] = flag;
  qurt_x1[idx][0] = x1[0];
  qurt_x1[idx][1] = x1[1];
  qurt_x2[idx][0] = x2[0];
  qurt_x2[idx][1] = x2[1];
}

int benchmark(void) {
  a = in1;
  qurt_capture(0);
  a = in2;
  qurt_capture(1);
  a = in3;
  qurt_capture(2);
  return 0;
}

static int qurt_approx(float a, float b) {
  float d = a - b;
  if (d < 0)
    d = -d;
  return d < 1e-3f;
}

int verify_benchmark(int res) {
  (void)res;
  for (int i = 0; i < 3; i++)
    if (qurt_res[i] != 0)
      return 0;

  if (qurt_flag[0] != 1)
    return 0;
  if (!qurt_approx(qurt_x1[0][0], 2.0f) || !qurt_approx(qurt_x1[0][1], 0.0f))
    return 0;
  if (!qurt_approx(qurt_x2[0][0], 1.0f) || !qurt_approx(qurt_x2[0][1], 0.0f))
    return 0;

  if (qurt_flag[1] != 0)
    return 0;
  if (!qurt_approx(qurt_x1[1][0], 1.0f) || !qurt_approx(qurt_x1[1][1], 0.0f))
    return 0;
  if (!qurt_approx(qurt_x2[1][0], 1.0f) || !qurt_approx(qurt_x2[1][1], 0.0f))
    return 0;

  if (qurt_flag[2] != -1)
    return 0;
  if (!qurt_approx(qurt_x1[2][0], 2.0f) || !qurt_approx(qurt_x1[2][1], 2.0f))
    return 0;
  if (!qurt_approx(qurt_x2[2][0], 2.0f) || !qurt_approx(qurt_x2[2][1], -2.0f))
    return 0;
  return 1;
}
