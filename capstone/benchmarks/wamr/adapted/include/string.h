/* Freestanding <string.h> for WAMR: declarations only. The routines come from
   benchmarks/beebs/adapted/beebs_freestanding_string.c, which every domain in
   this tree already links; defining them here would collide. */
#ifndef CAPSTONE_WAMR_STRING_H
#define CAPSTONE_WAMR_STRING_H
#include <stddef.h>
void *memcpy(void *, const void *, size_t);
void *memmove(void *, const void *, size_t);
void *memset(void *, int, size_t);
int memcmp(const void *, const void *, size_t);
void *memchr(const void *, int, size_t);
size_t strlen(const char *);
int strcmp(const char *, const char *);
int strncmp(const char *, const char *, size_t);
char *strcpy(char *, const char *);
char *strncpy(char *, const char *, size_t);
char *strcat(char *, const char *);
char *strchr(const char *, int);
char *strrchr(const char *, int);
char *strstr(const char *, const char *);
char *strdup(const char *);
char *strtok_r(char *, const char *, char **);
#endif
