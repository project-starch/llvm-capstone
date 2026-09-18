#ifndef WG_RECORD_H
#define WG_RECORD_H
#include <stddef.h>
void wg_record_init(void *ctx, void *buffer, size_t size, int owned,
                    size_t header_size);
void wg_record_object(void *ctx, unsigned type, size_t size);
void wg_record_reset(void *ctx);
void wg_record_free(void *ctx);
#endif
