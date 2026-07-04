/*
 * umm_malloc_cfg.h -- Capstone PureCap freestanding port configuration.
 *
 * Self-contained (does NOT use upstream's UMM_MALLOC_CFGFILE / cfgport include
 * chain). Targets a single-threaded, no-OS Capstone domain.
 *
 * Block geometry (see VENDOR-NOTES in umm_malloc.c): the block header is padded
 * to 16 bytes so the returned data pointer (at block_base + 16) is 16-byte
 * (capability) aligned whenever the heap array itself is 16-aligned -- required
 * because user code stores capabilities into malloc'd memory, and a capability
 * loses its tag if stored to an under-aligned slot. With a 16-byte body the
 * block is 32 bytes (a 16-multiple), so every block base is 16-aligned.
 *
 * umm keeps all its free-list metadata as uint16_t block *indices*, never as
 * pointers, so no capability is ever stored in heap metadata -- the property
 * that makes umm PureCap-safe where stock dlmalloc (in-band capability links)
 * is not.
 */
#ifndef UMM_MALLOC_CFG_H
#define UMM_MALLOC_CFG_H

#include <stdint.h>
#include <stddef.h>

/* Total block size in bytes (must be a multiple of 16 for cap alignment, >= 32
 * so the 16-byte header leaves a non-empty body). Overridable per benchmark via
 * -DUMM_BLOCK_BODY_SIZE (e.g. larger blocks raise the 32767-block index ceiling
 * for bigger heaps). */
#ifndef UMM_BLOCK_BODY_SIZE
#define UMM_BLOCK_BODY_SIZE 32
#endif

/* Best-fit (upstream default) minimizes fragmentation. */
#define UMM_BEST_FIT

/* Single-threaded domain: no critical sections, no lazy-init check. */
#define UMM_CRITICAL_DECL(tag)
#define UMM_CRITICAL_ENTRY(tag)
#define UMM_CRITICAL_EXIT(tag)
#define UMM_CHECK_INITIALIZED()

/* No struct packing: rely on natural layout + a 16-aligned heap array. */
#define UMM_H_ATTPACKPRE
#define UMM_H_ATTPACKSUF

/* UMM_INFO / UMM_INTEGRITY_CHECK / UMM_POISON_CHECK intentionally left off:
 * the metric hooks compile to nothing. */
#define UMM_FRAGMENTATION_METRIC_INIT()
#define UMM_FRAGMENTATION_METRIC_ADD(c)
#define UMM_FRAGMENTATION_METRIC_REMOVE(c)

/* Debug logging disabled. */
#define DBGLOG_TRACE(...)
#define DBGLOG_DEBUG(...)
#define DBGLOG_CRITICAL(...)
#define DBGLOG_ERROR(...)
#define DBGLOG_WARNING(...)
#define DBGLOG_INFO(...)
#define DBGLOG_FORCE(...)

#endif /* UMM_MALLOC_CFG_H */
