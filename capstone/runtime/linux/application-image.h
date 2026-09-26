#ifndef CAPSTONE_APPLICATION_IMAGE_H
#define CAPSTONE_APPLICATION_IMAGE_H
#include "capstone/launch.h"

/* Return a sealed, immutable memfd or -1 with errno. */
int capstone_application_image(const char *path,
    struct capstone_application_descriptor *descriptor);
#endif
