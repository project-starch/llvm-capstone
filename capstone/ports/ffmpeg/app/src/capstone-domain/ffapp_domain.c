/* Domain entry. musl-capstone's runtime (hostcall.c) calls capstone_main from domain_main;
 * its return value reaches the host as the DONE result.
 *
 * Two things the runtime does not do for a program, and FFmpeg needs:
 *  - __environ is never set, and log.c calls getenv(). A NULL environ makes getenv
 *    dereference NULL. libc-test's domain entry sets the same empty environment
 *    (musl-capstone/libc-test/libc_test_domain.c).
 *  - There is no argv, so the input path and the stop stage are compile-time. One image
 *    per milestone keeps every run returning a result (ffapp_decode.h). */
#include "ffapp_decode.h"

#ifndef FFAPP_INPUT
#define FFAPP_INPUT "/mnt/host/input.mkv"
#endif
#ifndef FFAPP_STOP_AT
#define FFAPP_STOP_AT FFAPP_M5_ALL
#endif

extern char **__environ;
static char *ffapp_empty_environ[1] = { 0 };

int capstone_main(void)
{
    __environ = ffapp_empty_environ;
    return ffapp_run(FFAPP_INPUT, FFAPP_STOP_AT);
}
