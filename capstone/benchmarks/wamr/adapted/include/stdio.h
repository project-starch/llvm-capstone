/* Freestanding <stdio.h> for WAMR in a Capstone domain: declarations only.
   snprintf/vsnprintf are IMPLEMENTED by xlang/lua-cdp/capstone-lua/lua_libc.c,
   which already exists and has a native self-test; duplicating them here would
   collide at link. */
#ifndef CAPSTONE_WAMR_STDIO_H
#define CAPSTONE_WAMR_STDIO_H
#include <stddef.h>
#include <stdarg.h>
typedef struct _CAPSTONE_FILE FILE;
#ifndef EOF
#define EOF (-1)
#endif
int printf(const char *, ...);
int snprintf(char *, size_t, const char *, ...);
int vsnprintf(char *, size_t, const char *, va_list);
int vprintf(const char *, va_list);
int puts(const char *);
int putchar(int);
#endif
