/* Native entry: the same decode core as the domain, against the same minimal FFmpeg
 * configuration built for the host. It is the oracle's own check: its hash column must
 * equal the reference framemd5 before a domain run is compared against anything. */
#include <stdio.h>
#include <stdlib.h>

#include "ffapp_decode.h"

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "input.mkv";
    int stop_at = argc > 2 ? atoi(argv[2]) : FFAPP_M5_ALL;
    int status = ffapp_run(path, stop_at);
    printf("FFAPP status=%d\n", status);
    return status == stop_at ? 0 : 1;
}
