/* Freestanding stand-in for glibc's <sys/types.h>.
 *
 * The BSD-compat names matter: Zend/zend_hash.h:25 and Zend/zend_mm.h:25 include this
 * header specifically to get `uint` and `ulong`, which glibc exposes from its
 * __USE_MISC block. Without them zend_hash.h does not parse at all. These are the
 * same widths glibc uses on LP64 -- supplying them is what makes PHP compile
 * unmodified, not a change to PHP. */
#ifndef _SYS_TYPES_H
#define _SYS_TYPES_H 1
#include <stddef.h>

typedef unsigned char  u_char;
typedef unsigned short u_short;
typedef unsigned int   u_int;
typedef unsigned long  u_long;
typedef unsigned char  uchar;
typedef unsigned short ushort;
typedef unsigned int   uint;
typedef unsigned long  ulong;

typedef long           ssize_t;
typedef long           off_t;
typedef int            pid_t;
typedef unsigned int   uid_t;
typedef unsigned int   gid_t;
typedef unsigned int   mode_t;
typedef long           time_t;
typedef unsigned long  dev_t;
typedef unsigned long  ino_t;
typedef unsigned int   nlink_t;
#endif
