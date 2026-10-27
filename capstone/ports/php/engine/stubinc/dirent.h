/* Minimal stub: reached through php.h / TSRM. A domain has no filesystem or clock, so
 * nothing declared here is ever called; the declarations exist so php.h parses. */
#ifndef _DIRENT_H
#define _DIRENT_H 1
#include <sys/types.h>
struct dirent {
    unsigned long  d_ino;
    long           d_off;
    unsigned short d_reclen;
    unsigned char  d_type;
    char           d_name[256];
};
typedef struct __dirstream DIR;
DIR *opendir(const char *);
int  closedir(DIR *);
struct dirent *readdir(DIR *);
void rewinddir(DIR *);
int  readdir_r(DIR *, struct dirent *, struct dirent **);
#endif
