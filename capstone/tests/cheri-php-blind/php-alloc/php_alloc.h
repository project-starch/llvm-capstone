/* Faithful reimplementation of PHP 5.0.0's Zend allocator, non-ZEND_MM path.
 *
 * WHY A REIMPLEMENTATION AND NOT THE REAL FILE: Zend/zend_alloc.c cannot be
 * compiled standalone -- it pulls in zend.h, the TSRM layer, zend_globals and
 * the error machinery. What the CHERI measurement depends on is only the
 * *allocation shape*, and that is small enough to transcribe exactly:
 *
 *   - a zend_mem_header prefix carrying two list links and a 31-bit size;
 *   - emalloc(n) -> malloc(sizeof(header) + MEM_HEADER_PADDING + REAL_SIZE(n))
 *     returning (char*)p + sizeof(header) + MEM_HEADER_PADDING;
 *   - REAL_SIZE(n) = ((n+7) & ~0x7), so up to 7 bytes of slack past the object;
 *   - AG(cache)[MAX_CACHED_MEMORY][MAX_CACHED_ENTRIES], bucket = REAL_SIZE>>3;
 *   - efree(p) PUSHES ONTO THE CACHE and does not call free() at all when the
 *     bucket has room -- this is the entire reason ASAN cannot see 31 of the
 *     corpus defects, and the hypothesis under test for CHERI;
 *   - the intrusive doubly-linked live list, walked by
 *     shutdown_memory_manager(), which DOES drain the cache and free() each
 *     block. So an efree'd block stays malloc-live until shutdown.
 *
 * Transcribed from the corpus tree, PHP 5.0.0:
 *   Zend/zend_alloc.c:132  REAL_SIZE
 *   Zend/zend_alloc.c:134  CALCULATE_REAL_SIZE_AND_CACHE_INDEX
 *   Zend/zend_alloc.c:100  REMOVE_POINTER_FROM_LIST
 *   Zend/zend_alloc.c:113  ADD_POINTER_TO_LIST
 *   Zend/zend_alloc.c:139  _emalloc
 *   Zend/zend_alloc.c:247  _efree
 *   Zend/zend_alloc.c:468  shutdown_memory_manager
 *   Zend/zend_alloc.h:37   zend_mem_header
 *   Zend/zend_alloc.h:63   MAX_CACHED_MEMORY / MAX_CACHED_ENTRIES
 *
 * ZEND_DEBUG is 0 (the corpus builds --disable-debug), so there is no magic,
 * no end magic, no memset-on-free, and END_MAGIC_SIZE is 0.
 *
 * TWO BUILD ARMS, exactly the two sed patches the corpus applies in
 * scripts/build-php-5.0.0-variant.sh:61-65 --
 *   default           : the real allocator.          Expect CHERI to be blind.
 *   -DPHP_PATCHED_ALLOC: cache off + REAL_SIZE exact. POSITIVE CONTROL.
 */
#ifndef PHP_ALLOC_H
#define PHP_ALLOC_H

#include <stddef.h>

/* --- the two corpus patches, as one switch ------------------------------- */
#ifdef PHP_PATCHED_ALLOC
# define ZEND_DISABLE_MEMORY_CACHE 1
# define REAL_SIZE(size) (size)
#else
# define ZEND_DISABLE_MEMORY_CACHE 0
# define REAL_SIZE(size) (((size)+7) & ~0x7)
#endif

#define MAX_CACHED_MEMORY   11
#define MAX_CACHED_ENTRIES  256

typedef union _align_test {
    void   *ptr;
    double  dbl;
    long    lng;
} align_test;

#define PLATFORM_ALIGNMENT (__alignof__(align_test))

typedef struct _zend_mem_header {
    struct _zend_mem_header *pNext;
    struct _zend_mem_header *pLast;
    unsigned int size:31;
    unsigned int cached:1;
} zend_mem_header;

#define MEM_HEADER_PADDING \
    (((PLATFORM_ALIGNMENT - sizeof(zend_mem_header)) % PLATFORM_ALIGNMENT \
      + PLATFORM_ALIGNMENT) % PLATFORM_ALIGNMENT)

void  zend_alloc_startup(void);
void *_emalloc(size_t size);
void  _efree(void *ptr);
void *_erealloc(void *ptr, size_t size);
char *_estrndup(const char *s, unsigned int length);
void  shutdown_memory_manager(void);

#define emalloc(n)        _emalloc(n)
#define efree(p)          _efree(p)
#define erealloc(p, n)    _erealloc((p), (n))
#define estrndup(s, n)    _estrndup((s), (n))

/* Introspection for the rows: report the layout the measurement depends on. */
size_t php_alloc_real_size(size_t n);
size_t php_alloc_header_bytes(void);
int    php_alloc_cache_is_on(void);

#endif /* PHP_ALLOC_H */
