#include "ggml.h"
#include "port.h"
#include <stdbool.h>
#include <string.h>
#undef GGML_ASSERT
#undef GGML_ABORT
#define GGML_ASSERT(x) ((x) ? (void)0 : wg_fail(101))
#define GGML_ABORT(...) wg_fail(102)
#define GGML_MALLOC(n) wg_meta_alloc(n)
#define GGML_FREE(p) wg_meta_free(p)
#define ggml_aligned_malloc(n) wg_aligned_alloc(n)
#define ggml_aligned_free(p, n) wg_aligned_free(p, n)
#define GGML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
#define GGML_PRINT_DEBUG(...) ((void)0)
#define GGML_LOG_WARN(...) ((void)0)
/* This extraction executes serial allocator calls, not inference or threads. */
static inline void ggml_critical_section_start(void) {}
static inline void ggml_critical_section_end(void) {}
#define ggml_time_init() ((void)0)
