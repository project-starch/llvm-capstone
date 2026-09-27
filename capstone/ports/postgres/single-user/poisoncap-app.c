#include "poisoncap.h"
#include <stdlib.h>

int __real_main(int argc, char **argv);

int __wrap_main(int argc, char **argv) {
  const char *mode = getenv("PG_POISONCAP_MODE");
  if (!mode || (mode[0] != '0' && mode[0] != '1') || mode[1] != '\0')
    return 125;
  pg_poisoncap_init((unsigned)(mode[0] - '0'));
  if (atexit(pg_poisoncap_report) != 0)
    return 125;
  return __real_main(argc, argv);
}
