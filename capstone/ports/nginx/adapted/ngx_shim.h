/* The whole of nginx that ngx_palloc.c needs. Thirteen types and nine symbols, in place of the
   forty-seven headers ngx_core.h pulls in, which are the whole server. Same move the PostgreSQL
   port makes with its sixteen dummies. */
#ifndef NGX_SHIM_H
#define NGX_SHIM_H
#include <stddef.h>
#include <stdint.h>
void *memset(void *, int, size_t);

typedef unsigned char       u_char;
typedef intptr_t            ngx_int_t;
typedef uintptr_t           ngx_uint_t;
typedef int                 ngx_fd_t;
typedef int                 ngx_err_t;
typedef struct { int unused; } ngx_log_t;
typedef struct ngx_chain_s  ngx_chain_t;
struct ngx_chain_s { void *buf; ngx_chain_t *next; };
typedef struct ngx_pool_s   ngx_pool_t;

#define NGX_OK        0
#define NGX_ERROR    -1
#define NGX_DECLINED -5
#define NGX_ALIGNMENT   sizeof(unsigned long)
#define ngx_align_ptr(p, a) \
    (u_char *) (((uintptr_t) (p) + ((uintptr_t) (a) - 1)) & ~((uintptr_t) (a) - 1))
#define ngx_align(d, a)  (((d) + (a - 1)) & ~(a - 1))
#define ngx_memzero(buf, n)  (void) memset(buf, 0, n)
void ngx_free_wrap(void *);
#define ngx_free            ngx_free_wrap
#define NGX_LOG_ALERT 1
#define NGX_LOG_DEBUG_ALLOC 0
#define NGX_FILE_ERROR -1
#define NGX_ENOENT 2
#define ngx_log_debug1(l,g,c,f,a) ((void)0)
#define ngx_log_debug2(l,g,c,f,a,b) ((void)0)
#define ngx_log_error(level, log, err, ...) ((void) 0)
#define ngx_close_file(fd)  (0)
#define ngx_delete_file(n)  (0)
#define ngx_errno           0
#define ngx_close_file_n    "close()"
#define ngx_delete_file_n   "unlink()"
#define ngx_get_cached_block(size)  (NULL)

#define ngx_inline inline
extern ngx_uint_t ngx_pagesize;
void *ngx_alloc(size_t size, ngx_log_t *log);
void *ngx_memalign(size_t alignment, size_t size, ngx_log_t *log);
void free(void *);

#include "ngx_palloc.h"
#endif
