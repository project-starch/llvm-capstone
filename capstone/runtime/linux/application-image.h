#ifndef CAPSTONE_APPLICATION_IMAGE_H
#define CAPSTONE_APPLICATION_IMAGE_H
#include "capstone/launch.h"

/* Digest a sealed image without changing its file position. */
int capstone_application_hash(int fd, char hex[65]);

/* Open a regular file read-only as a candidate image; -1 with errno. */
int capstone_application_open(const char *path);

/* 1 when fd's filesystem keeps one Linux inode per file, so the writers Linux
 * counts on that inode are all the file's writers and denying them keeps its
 * bytes fixed: tmpfs (and memfds), ext2/3/4, btrfs, xfs, erofs, squashfs. 0
 * otherwise -- 9p with cache=none or mmap makes a new inode at each lookup, so
 * a writer through another name (a hard link) is counted on another inode;
 * NFS, FUSE and overlayfs make no such promise either -- and the caller loads
 * a sealed copy (capstone_application_image) instead. */
int capstone_application_lockable(int fd);

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

/* Return a sealed, immutable memfd copy of the file at path, validated; or -1
 * with errno, as capstone_application_inspect. */
int capstone_application_image(const char *path,
    struct capstone_application_descriptor_v2 *descriptor);
#endif
