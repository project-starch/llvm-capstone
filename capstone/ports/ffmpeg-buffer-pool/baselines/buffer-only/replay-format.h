#ifndef FFPOOL_REPLAY_FORMAT_H
#define FFPOOL_REPLAY_FORMAT_H
#include <stdint.h>
#define FFTRACE_MAGIC UINT64_C(0x4650465452433031)
#define FFTRACE_BYTES (8UL * 1024 * 1024)
#define FFREPORT_BYTES FFTRACE_BYTES
#define FFARENA_BYTES (32UL * 1024 * 1024)
enum { FF_CREATE = 1, FF_GET, FF_RETURN, FF_CLOSE, FF_END };
/* Little-endian, fixed-width wire format. No addresses cross architectures.
 * Only op/pool/lease/size/aux are replay inputs. The remaining fields are
 * observations, independently recomputed by replay and compared afterwards.
 * aux on CREATE selects av_buffer_alloc (0) or av_buffer_allocz (1).
 */
struct ff_event {
    uint64_t op, pool, lease, backing, size, aux;
    uint64_t allocations, gap, live_bytes, retained_bytes;
};
struct ff_header { uint64_t magic, count, status, arena_used; };
#endif
