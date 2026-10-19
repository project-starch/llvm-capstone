#ifndef CAPSTONE_APPLICATION_IMAGE_H
#define CAPSTONE_APPLICATION_IMAGE_H
#include "capstone/launch.h"

/* Digest a sealed image without changing its file position. */
int capstone_application_hash(int fd, char hex[65]);

/* Open a regular file read-only as a candidate image; -1 with errno. */
int capstone_application_open(const char *path);

/* 0 when denying writers to fd's inode keeps its bytes fixed; -1 with errno
 * ETXTBSY otherwise. Linux counts the writers of a file with a name, so a
 * denial excludes them all. A memfd has no name and its own descriptor is
 * not counted, so it qualifies only sealed against writing, growing and
 * shrinking. */
int capstone_application_fixed(int fd);

/* Validate the image held by fd: every header, file span and descriptor
 * section. The bytes must not change afterwards: fd is a sealed memfd, or a
 * file whose writers are denied (the virtual launcher's module holds that).
 * 0, or -1 with errno (ENOEXEC for anything but a delegated image whose
 * descriptor has exactly the current layout). */
int capstone_application_inspect(int fd,
    struct capstone_application_descriptor_v2 *descriptor);

/* The same validation over an image already in memory (size bytes at data),
 * for a loader that maps the file itself and loads from the mapping it
 * checked, so the bytes validated are the bytes loaded. 0, or -1 with errno
 * as capstone_application_inspect. */
int capstone_application_validate(const void *data, size_t size,
    struct capstone_application_descriptor_v2 *descriptor);

/* Return a sealed, immutable memfd copy of the open file source (read with
 * pread from offset 0; source stays open), validated; or -1 with errno, as
 * capstone_application_inspect. */
int capstone_application_copy(int source,
    struct capstone_application_descriptor_v2 *descriptor);

/* capstone_application_copy of the file at path. */
int capstone_application_image(const char *path,
    struct capstone_application_descriptor_v2 *descriptor);
#endif
