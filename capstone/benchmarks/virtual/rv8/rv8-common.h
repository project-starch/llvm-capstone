/* Shared by the RV8 drivers. Each driver is one translation unit: the upstream program, copied into
 * the build directory with its main() stripped (build.py, asserted substitutions), is included
 * first through RV8_UPSTREAM, and the driver then replays that main() as the harness unit with a
 * build-time size RV8_N. Upstream rv8-bench has no verification of its own; each driver checks the
 * property its main() prints (round trip, sortedness, the largest prime, Dhrystone's end state). */
#ifndef RV8_COMMON_H
#define RV8_COMMON_H
#include <stdlib.h>
#include <string.h>
#include "bench-harness.h"
#ifndef RV8_N
#error "RV8_N (the program's size parameter) must be set"
#endif
#endif
