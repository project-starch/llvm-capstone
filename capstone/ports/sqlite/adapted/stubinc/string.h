/* Freestanding stand-in for <string.h>: the domain gets its libc from capstone_sqlite_libc.h,
 * force-included before every translation unit. Without this file the driver falls through to
 * the host's /usr/include, whose string.h pulls glibc's features.h and, on Debian-layout hosts,
 * a bits/wordsize.h that lives under the multiarch directory the target search path does not
 * have. The old build survived only where the host's glibc kept that file in /usr/include. */
