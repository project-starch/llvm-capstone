/* Complete decoder entry; argv and environment come from the application SDK. */
#include <stdio.h>

#include "ffapp_decode.h"

#ifndef FFAPP_INPUT
#define FFAPP_INPUT "/mnt/host/input.mkv"
#endif
#ifndef FFAPP_STOP_AT
#define FFAPP_STOP_AT FFAPP_M5_ALL
#endif

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    setvbuf(stdout, NULL, _IOLBF, 0);
    int status = ffapp_run(argc > 1 ? argv[1] : FFAPP_INPUT, FFAPP_STOP_AT);
    fflush(stdout);
    return status;
}
