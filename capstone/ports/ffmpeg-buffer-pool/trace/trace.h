#ifndef FF_COMBINED_TRACE_H
#define FF_COMBINED_TRACE_H
#include <stddef.h>
#include <stdint.h>

#define FF2_MAGIC UINT64_C(0x4650465452433032)
#define FF2_FILE_BYTES (128UL * 1024 * 1024)
#define FF2_PAYLOAD_BYTES (64UL * 1024 * 1024)
#define FF2_META_BYTES (16UL * 1024 * 1024)
#define FF2_POOLS 1024
#define FF2_BLOCKS 8192
enum { FF2_BUFFER = 1, FF2_REFSTRUCT = 2 };
enum { FF2_CREATE = 1, FF2_GET, FF2_RETURN, FF2_CLOSE,
       FF2_NEW, FF2_DROP, FF2_CALLBACK, FF2_DONE };
enum { FF2_INIT_CB = 1, FF2_RESET_CB, FF2_FREE_ENTRY_CB, FF2_FREE_POOL_CB };
#define FF2_FINISH 128U
#define FF2_HAS_INIT (UINT64_C(1) << 32)
#define FF2_HAS_RESET (UINT64_C(1) << 33)
#define FF2_HAS_FREE_ENTRY (UINT64_C(1) << 34)
#define FF2_HAS_FREE_POOL (UINT64_C(1) << 35)

/* Little-endian fixed-width records. BEGIN operations are commands; END,
 * NEW and DROP outcomes are measured. Callback bodies are represented by
 * their nested pool effects, not by replaying codec computation or raw bytes.
 * object is a lease identity; backing is a distinct backing-allocation identity.
 */
struct ff2_event {
    uint64_t op, kind, pool, call, parent, object, size, flags;
    uint64_t backing, allocations, gap, live, retained, reserved[3];
};
struct ff2_header {
    uint64_t magic, count, status, metadata_used, payload_used;
    uint64_t split, mrev, delin, revoke, init, init_bytes, mode;
    uint64_t reserved[4];
};

void ff2_lock(void);
void ff2_unlock(int *);
#define FF2_GUARD ff2_lock(); int ff2_guard __attribute__((cleanup(ff2_unlock))) = 0
uint64_t ff2_begin(unsigned op, unsigned kind, void *pool, size_t size,
                   uint64_t flags, void *object);
void ff2_end(uint64_t call, void *result);
void ff2_new(unsigned kind, void *pool, void *object);
void ff2_drop(unsigned kind, void *pool, void *object);
uint64_t ff2_callback(unsigned kind, void *pool, void *object, unsigned type);
void ff2_finish(void);
void ff2_sink(const struct ff2_event *);
_Noreturn void ff2_fail(unsigned);
void ff2_reset(void);

/* The port's lifetime hooks. Metadata and capability slots are outside the
 * payload region. No address is ever turned into a capability by this API.
 */
void *ff2_payload_alloc(size_t);
void ff2_payload_free(void *);
void ff2_payload_return(void *);
void *ff2_payload_issue(uintptr_t);
void *ff2_ref_alloc(size_t, size_t);
void *ff2_ref_data(void *);
void *ff2_ref_meta(const void *);
void ff2_ref_free(void *);
void ff2_ref_return(void *);
void *ff2_ref_issue(void *);
void ff2_memory_init(void *, size_t);
void ff2_payload_init(void *, size_t);
void ff2_memory_report(struct ff2_header *);
void ff2_set_mode(unsigned);
#endif
