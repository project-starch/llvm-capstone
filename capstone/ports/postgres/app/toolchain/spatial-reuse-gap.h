#ifndef PG_SPATIAL_REUSE_GAP_H
#define PG_SPATIAL_REUSE_GAP_H

#include <stddef.h>
#include <stdint.h>

void pg_spatial_gap_issue(uintptr_t context, uintptr_t start, size_t size);
void pg_spatial_gap_release(uintptr_t start);
void pg_spatial_gap_resize(uintptr_t start, size_t size);
void pg_spatial_gap_reset(uintptr_t context);
void pg_reuse_gap_report(void);

#endif
