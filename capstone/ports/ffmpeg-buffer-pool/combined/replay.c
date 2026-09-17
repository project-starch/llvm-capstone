/* Execute pool APIs and the allocator effects nested inside recorded callbacks.
 * Expected allocation choices and counters are absent from the command stream.
 * Refcount traffic before the last unref, codec work, and object bytes are not
 * replayed. Semantic tests cover shared references separately.
 */
#include <string.h>
#include "libavutil/buffer.h"
#include "libavutil/refstruct.h"
#include "trace.h"

static struct ff2_header *report;
static const struct ff2_header *trace;
static uint64_t cursor;
static struct replay_pool { void *p; unsigned kind, closed; } pools[FF2_POOLS];
static struct replay_lease { void *p; uint64_t id, pool; } leases[2048];
static void dispatch(void);
#ifdef FFPOOL_DOMAIN
static void node_snapshot(void)
{
    unsigned long mark = 0xff20000000000000UL | cursor;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x43, x0, %0, x0\n"
                     ".insn r 0x5b, 0x1, 0x46, x0, x0, x0\n" : : "r"(mark) : "memory");
}
#endif

void ff2_lock(void) {}
void ff2_unlock(int *p) { (void)p; }
static const struct ff2_event *peek(void)
{
    if (cursor >= trace->count) ff2_fail(201);
    return (const struct ff2_event *)(trace + 1) + cursor;
}
void ff2_sink(const struct ff2_event *actual)
{
#ifdef FF2_SECURITY
    (void)actual;
    report->reserved[0]++;
#else
    const struct ff2_event *command = peek();
    if (memcmp(actual, command, 8 * sizeof(uint64_t))) ff2_fail(202);
    const uint64_t *outcomes = &command->backing;
    for (unsigned i = 0; i < 8; i++) if (outcomes[i]) ff2_fail(203);
    ((struct ff2_event *)(report + 1))[cursor++] = *actual;
    report->count = cursor;
#ifdef FFPOOL_DOMAIN
    if (!(cursor % 32768)) node_snapshot();
#endif
#endif
}
static void callback_effects(void)
{
    while (peek()->op != (FF2_CALLBACK | FF2_FINISH)) dispatch();
}
static int init_cb(AVRefStructOpaque opaque, void *obj)
{ (void)opaque; (void)obj; callback_effects(); return 0; }
static void object_cb(AVRefStructOpaque opaque, void *obj)
{ (void)opaque; (void)obj; callback_effects(); }
static void pool_cb(AVRefStructOpaque opaque)
{ (void)opaque; callback_effects(); }

static void dispatch(void)
{
    struct ff2_event cmd = *peek();
    if (!cmd.pool || cmd.pool >= FF2_POOLS || cmd.kind < 1 || cmd.kind > 2)
        ff2_fail(204);
    struct replay_pool *p = &pools[cmd.pool];
    switch (cmd.op) {
    case FF2_CREATE:
        if (p->p || p->kind || !cmd.size || cmd.size > FF2_PAYLOAD_BYTES) ff2_fail(205);
        p->kind = cmd.kind;
        if (cmd.kind == FF2_BUFFER) {
            if (cmd.flags > 1) ff2_fail(206);
            p->p = av_buffer_pool_init(cmd.size, cmd.flags ? av_buffer_allocz : NULL);
        } else {
            uint64_t allowed = FF2_HAS_INIT | FF2_HAS_RESET | FF2_HAS_FREE_ENTRY | FF2_HAS_FREE_POOL | UINT64_C(0xffffffff);
            if (cmd.flags & ~allowed) ff2_fail(207);
            p->p = av_refstruct_pool_alloc_ext(cmd.size, (unsigned)cmd.flags, p,
                    cmd.flags & FF2_HAS_INIT ? init_cb : NULL,
                    cmd.flags & FF2_HAS_RESET ? object_cb : NULL,
                    cmd.flags & FF2_HAS_FREE_ENTRY ? object_cb : NULL,
                    cmd.flags & FF2_HAS_FREE_POOL ? pool_cb : NULL);
        }
        if (!p->p) ff2_fail(208);
        break;
    case FF2_GET: {
        if (!p->p || p->closed || p->kind != cmd.kind) ff2_fail(209);
        /* Reserve before entering callbacks, which may allocate more leases. */
        unsigned i;
        for (i = 0; i < 2048 && leases[i].id; i++) {}
        if (i == 2048) ff2_fail(210);
        leases[i].id = cmd.object; leases[i].pool = cmd.pool;
        leases[i].p = cmd.kind == FF2_BUFFER ? (void *)av_buffer_pool_get(p->p) : av_refstruct_pool_get(p->p);
        if (!leases[i].p) ff2_fail(211);
        /* Valid payload probes exercise access without pretending to decode. */
        unsigned char *data = cmd.kind == FF2_BUFFER ? ((AVBufferRef *)leases[i].p)->data : leases[i].p;
        data[0] = (unsigned char)cmd.object;
        data[cmd.size - 1] = (unsigned char)(cmd.object >> 8);
        break;
    }
    case FF2_RETURN: {
        unsigned i;
        for (i = 0; i < 2048; i++) if (leases[i].id == cmd.object) break;
        if (i == 2048 || !leases[i].p || leases[i].pool != cmd.pool || p->kind != cmd.kind) ff2_fail(212);
        void *obj = leases[i].p;
        unsigned char *data = cmd.kind == FF2_BUFFER ? ((AVBufferRef *)obj)->data : obj;
        if (cmd.size > 1 && (data[0] != (unsigned char)cmd.object || data[cmd.size - 1] != (unsigned char)(cmd.object >> 8))) ff2_fail(213);
        leases[i].id = 0; leases[i].p = NULL;
        if (cmd.kind == FF2_BUFFER) av_buffer_unref((AVBufferRef **)&obj);
        else av_refstruct_unref(&obj);
        break;
    }
    case FF2_CLOSE:
        if (!p->p || p->closed || p->kind != cmd.kind) ff2_fail(214);
        p->closed = 1;
        if (cmd.kind == FF2_BUFFER) av_buffer_pool_uninit((AVBufferPool **)&p->p);
        else av_refstruct_unref(&p->p);
        break;
    default: ff2_fail(215);
    }
}

static void replay(void *metadata)
{
    uint64_t mode = report->mode;
    *report = (struct ff2_header){.magic = FF2_MAGIC, .status = 1, .mode = mode};
    if (trace->magic != FF2_MAGIC || trace->status || !trace->count ||
        trace->count > (FF2_FILE_BYTES - sizeof *trace) / sizeof(struct ff2_event)) ff2_fail(216);
    ff2_memory_init(metadata, FF2_META_BYTES);
    ff2_set_mode(mode);
    ff2_reset();
#ifdef FF2_SECURITY
    void ff2_security_run(unsigned, unsigned long);
    ff2_security_run(trace->reserved[0], trace->reserved[1]);
    ff2_finish();
    report->count = trace->count;
#else
    while (peek()->op != FF2_DONE) dispatch();
    ff2_finish();
    if (cursor != trace->count) ff2_fail(217);
#endif
    ff2_memory_report(report);
#ifdef FFPOOL_DOMAIN
    node_snapshot();
#endif
    report->status = 0;
}

#ifdef FFPOOL_DOMAIN
static void *metadata;
static unsigned shares;
static unsigned long *domain_result;
unsigned char ff2_exit_frame[32] __attribute__((aligned(16), used));
_Noreturn void ff2_fail(unsigned code)
{
    if (report) report->status = code;
    if (domain_result) *domain_result = code;
    __asm__ volatile(
        "1: auipc t0, %%pcrel_hi(ff2_exit_frame)\n"
        "addi t0, t0, %%pcrel_lo(1b)\n"
        ".insn r 0x5b, 0x1, 0xc, t0, gp, t0\n"
        ".insn i 0x5b, 0x3, sp, 0(t0)\n"
        ".insn i 0x5b, 0x3, ra, 16(t0)\n"
        "ret\n" ::: "memory");
    __builtin_unreachable();
}
void ff2_entry(unsigned long *result, unsigned func)
{
    if (func == 1) {
        switch (shares++) {
        case 0: report = (struct ff2_header *)result; break;
        case 1: metadata = result; break;
        case 2: trace = (const struct ff2_header *)result; break;
        case 3: ff2_payload_init(result, FF2_PAYLOAD_BYTES); break;
        default: ff2_fail(218);
        }
        return;
    }
    domain_result = result;
    if (shares != 4) ff2_fail(219);
    replay(metadata);
    *result = 42044;
}
__asm__(".text\n.globl domain_main\ndomain_main:\n"
        "1: auipc t0, %pcrel_hi(ff2_exit_frame)\n"
        "addi t0, t0, %pcrel_lo(1b)\n"
        ".insn r 0x5b, 0x1, 0xc, t0, gp, t0\n"
        ".insn s 0x5b, 0x4, sp, 0(t0)\n"
        ".insn s 0x5b, 0x4, ra, 16(t0)\nj ff2_entry\n");
#else
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
static jmp_buf failure;
_Noreturn void ff2_fail(unsigned code)
{
    report->status = code;
    fprintf(stderr, "FF2 fail=%u event=%llu op=%llu\n", code,
            (unsigned long long)cursor, (unsigned long long)(cursor < trace->count ? peek()->op : 0));
    longjmp(failure, 1);
}
int main(int argc, char **argv)
{
    if (argc < 3 || argc > 4) return 2;
    struct ff2_header *input = calloc(1, FF2_FILE_BYTES);
    report = calloc(1, FF2_FILE_BYTES);
    void *meta = aligned_alloc(64, FF2_META_BYTES);
    void *payload = aligned_alloc(64, FF2_PAYLOAD_BYTES);
    FILE *f = fopen(argv[1], "rb");
    if (!input || !report || !meta || !payload || !f) return 2;
    size_t n = fread(input, 1, FF2_FILE_BYTES, f);
    if (ferror(f) || fgetc(f) != EOF || n < sizeof *input ||
        input->count > (FF2_FILE_BYTES - sizeof *input) / sizeof(struct ff2_event) ||
        n != sizeof *input + input->count * sizeof(struct ff2_event)) return 3;
    fclose(f); trace = input;
    report->mode = argc == 4 ? strtoul(argv[3], NULL, 0) : 0;
    if (!setjmp(failure)) { ff2_payload_init(payload, FF2_PAYLOAD_BYTES); replay(meta); }
    f = fopen(argv[2], "wb");
    size_t bytes = sizeof *report + report->count * sizeof(struct ff2_event);
    if (!f || fwrite(report, 1, bytes, f) != bytes || fclose(f)) return 4;
    printf("FF2 status=%llu events=%llu metadata=%llu payload=%llu\n",
           (unsigned long long)report->status, (unsigned long long)report->count,
           (unsigned long long)report->metadata_used, (unsigned long long)report->payload_used);
    return report->status ? 1 : 0;
}
#endif
