#ifndef CAPSTONE_APPLICATION_IMAGE_H
#define CAPSTONE_APPLICATION_IMAGE_H
#include "capstone/launch.h"

/* Digest a sealed image without changing its file position. */
int capstone_application_hash(int fd, char hex[65]);

/* Return a sealed, immutable memfd or -1 with errno. Accepts a 40-byte v1
 * descriptor (exchange_bytes reported 0) or a 48-byte v2 one. */
int capstone_application_image(const char *path,
    struct capstone_application_descriptor_v2 *descriptor);
#endif
