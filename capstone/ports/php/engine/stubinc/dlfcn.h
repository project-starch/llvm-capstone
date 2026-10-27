/* DECLARING THESE IS NOT OPTIONAL ON A CAPABILITY TARGET.
 *
 * PHP builds with -Wno-implicit-function-declaration (Makefile CFLAGS_CLEAN), so on a
 * normal target an undeclared dlopen() merely warns and returns int. Here that truncates
 * a 128-bit capability to 64 bits and the tag is gone: Zend/zend_extensions.c:34 assigns
 * the result straight into a void*, which clang reports as
 *   "incompatible integer to pointer conversion assigning to 'void *' from 'int'".
 * The domain never calls these -- dynamic loading is stubbed to failure -- but every one
 * that returns a pointer must still be declared so no capability is laundered through int. */
#ifndef _DLFCN_H
#define _DLFCN_H 1
#define RTLD_LAZY   1
#define RTLD_NOW    2
#define RTLD_GLOBAL 256
void *dlopen(const char *, int);
void *dlsym(void *, const char *);
int   dlclose(void *);
char *dlerror(void);
#endif
