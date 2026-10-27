/* slotins: a DynamoRIO client that counts, in one run, the instructions each process executes and the heap
 * allocations and frees it makes: allocations per instruction from the same execution. A separate census run cannot
 * supply the numerator, because benchmarks that run for a fixed time (Octane, pyperf's calibrated loops) do less work
 * under instrumentation.
 *
 * usage: drrun -c libslotins.so OUT_FILE -- program [args]
 * Instructions: every basic block adds its application instruction count to its thread's own 64-bit counter with two
 * inline instructions (load the counter's address from a raw TLS slot, add; drreg saves the flags only where they
 * are live). A clean call per block, the inscount sample's way, made PARSEC native 66x slower than the census
 * (blackscholes 1,590 s against 24 s, measured), and a shared counter would make threads contend. A thread's count
 * joins the process's when the thread exits, and DynamoRIO runs every live thread's exit event before the process
 * exit event, so the total is exact.
 * Allocations: drwrap wraps malloc, calloc, realloc, free, posix_memalign, aligned_alloc, memalign and valloc in every
 * module that exports them, counted by the rules in census.h (without bytes or sizes: the post-call sees only the
 * result). Unlike the preloaded tracer, a wrapped entry point also sees libc's internal calls (stdio buffers and the
 * like), a few objects more.
 * drrun follows child processes; a fork child starts with the live set it inherits and zero counts, and an exec
 * starts a new process line. At exit each process appends one JSON line to OUT_FILE: pid, ppid, cmdline (as
 * slotcensus.c records it), forked, instructions, allocs, frees, realloc_moved, realloc_inplace, failed, peak_live,
 * end_live. The output path is a client option, so programs that clear their children's environment still report. */
#include <stdint.h>
#include <string.h>

#include "census.h"
#include "dr_api.h"
#include "drmgr.h"
#include "drreg.h"
#include "drwrap.h"

static uint64 count;   /* the exited threads' instructions */
static int tls;               /* drmgr TLS field: the thread's counter, for the exit events */
static reg_id_t tls_seg;       /* raw TLS slot: the same pointer, for the inline add */
static uint tls_offs;
static struct census_counts counts;
static int forked;
static char out_path[4096];

static void thread_init(void *drcontext)
{
    uint64 *mine = dr_thread_alloc(drcontext, sizeof *mine);
    *mine = 0;
    drmgr_set_tls_field(drcontext, tls, mine);
    *(uint64 **)(dr_get_dr_segment_base(tls_seg) + tls_offs) = mine;   /* runs on the new thread */
}

static void thread_exit(void *drcontext)
{
    uint64 *mine = drmgr_get_tls_field(drcontext, tls);
    if (mine == NULL)
        return;
    /* a fork child also runs the exit events of its parent's other threads: their counts are the parent's */
    if (dr_get_process_id() == dr_get_process_id_from_drcontext(drcontext))
        __atomic_add_fetch(&count, *mine, __ATOMIC_RELAXED);
    drmgr_set_tls_field(drcontext, tls, NULL);
    dr_thread_free(drcontext, mine, sizeof *mine);
}

static dr_emit_flags_t count_bb(void *drcontext, void *tag, instrlist_t *bb, instr_t *inst, bool for_trace,
                                bool translating, void *user_data)
{
    if (!drmgr_is_first_instr(drcontext, inst))
        return DR_EMIT_DEFAULT;
    uint n = 0;
    for (instr_t *i = instrlist_first_app(bb); i != NULL; i = instr_get_next_app(i))
        ++n;
    reg_id_t r;
    if (drreg_reserve_aflags(drcontext, bb, inst) != DRREG_SUCCESS ||
        drreg_reserve_register(drcontext, bb, inst, NULL, &r) != DRREG_SUCCESS)
        DR_ASSERT(false);
    dr_insert_read_raw_tls(drcontext, bb, inst, tls_seg, tls_offs, r);
    instrlist_meta_preinsert(bb, inst, INSTR_CREATE_add(drcontext, OPND_CREATE_MEM64(r, 0), OPND_CREATE_INT32((int)n)));
    if (drreg_unreserve_register(drcontext, bb, inst, r) != DRREG_SUCCESS ||
        drreg_unreserve_aflags(drcontext, bb, inst) != DRREG_SUCCESS)
        DR_ASSERT(false);
    return DR_EMIT_DEFAULT;
}

/* the allocating calls: the result decides */
static void post_alloc(void *wrapcxt, void *user_data)
{
    census_alloc(&counts, drwrap_get_retval(wrapcxt), 0, 0);
}

static void post_posix_memalign(void *wrapcxt, void *user_data)
{
    census_alloc(&counts, (int)(ptr_int_t)drwrap_get_retval(wrapcxt) == 0 ? (void *)1 : NULL, 0, 0);
}

static void pre_free(void *wrapcxt, void **user_data)
{
    if (drwrap_get_arg(wrapcxt, 0) != NULL)
        census_free(&counts, 0);
}

/* realloc: the old pointer goes to the post-call, its low bit set when the requested size is 0 (heap pointers are
 * 16-byte aligned). census_realloc needs only whether the size is 0: slotins records no sizes */
static void pre_realloc(void *wrapcxt, void **user_data)
{
    *user_data = (void *)((ptr_uint_t)drwrap_get_arg(wrapcxt, 0) | (drwrap_get_arg(wrapcxt, 1) == 0 ? 1 : 0));
}

static void post_realloc(void *wrapcxt, void *user_data)
{
    void *old = (void *)((ptr_uint_t)user_data & ~(ptr_uint_t)1);
    int to_zero = (int)((ptr_uint_t)user_data & 1);
    void *p = drwrap_get_retval(wrapcxt);
    if (old == NULL)
        census_alloc(&counts, p, 0, 0);
    else
        census_realloc(&counts, old, p, to_zero ? 0 : 1, 0, 0);
}

static void wrap(const module_data_t *mod, const char *name, void (*pre)(void *, void **),
                 void (*post)(void *, void *))
{
    app_pc f = (app_pc)dr_get_proc_address(mod->handle, name);
    if (f != NULL)
        drwrap_wrap(f, pre, post);
}

static void module_load(void *drcontext, const module_data_t *mod, bool loaded)
{
    wrap(mod, "malloc", NULL, post_alloc);
    wrap(mod, "calloc", NULL, post_alloc);
    wrap(mod, "memalign", NULL, post_alloc);
    wrap(mod, "aligned_alloc", NULL, post_alloc);
    wrap(mod, "valloc", NULL, post_alloc);
    wrap(mod, "posix_memalign", NULL, post_posix_memalign);
    wrap(mod, "realloc", pre_realloc, post_realloc);
    wrap(mod, "free", pre_free, NULL);
}

static void child_after_fork(void *drcontext)
{
    count = 0;
    *(uint64 *)drmgr_get_tls_field(drcontext, tls) = 0;   /* the forking thread: its count so far is the parent's */
    census_after_fork(&counts);
    forked = 1;
}

static void report(void)
{
    char cmd[1024] = "?";
    file_t f = dr_open_file("/proc/self/cmdline", DR_FILE_READ);
    if (f != INVALID_FILE) {
        census_cmdline(cmd, (long)dr_read_file(f, cmd, sizeof cmd - 1));
        dr_close_file(f);
    }
    char line[1700];
    int len = dr_snprintf(line, sizeof line,
                          "{\"pid\":%d,\"ppid\":%d,\"cmdline\":\"%s\",\"forked\":%d,\"instructions\":%llu,"
                          "\"allocs\":%lld,\"frees\":%lld,\"realloc_moved\":%lld,\"realloc_inplace\":%lld,"
                          "\"failed\":%lld,\"peak_live\":%lld,\"end_live\":%lld}\n",
                          (int)dr_get_process_id(), (int)dr_get_parent_id(), cmd, forked, (unsigned long long)count,
                          (long long)counts.allocs, (long long)counts.frees, (long long)counts.moved,
                          (long long)counts.inplace, (long long)counts.failed, (long long)counts.peak,
                          (long long)counts.live);
    file_t o = dr_open_file(out_path, DR_FILE_WRITE_APPEND);
    if (o != INVALID_FILE && len > 0) {
        dr_write_file(o, line, (size_t)(len < (int)sizeof line ? len : (int)sizeof line - 1));
        dr_close_file(o);
    }
}

static void event_exit(void)
{
    report();
    drmgr_unregister_tls_field(tls);
    dr_raw_tls_cfree(tls_offs, 1);
    drreg_exit();
    drwrap_exit();
    drmgr_exit();
}

DR_EXPORT void dr_client_main(client_id_t id, int argc, const char *argv[])
{
    if (argc < 2 || strlen(argv[1]) >= sizeof out_path) {
        dr_fprintf(STDERR, "slotins: usage: -c libslotins.so OUT_FILE\n");
        dr_abort();
    }
    strcpy(out_path, argv[1]);
    dr_set_client_name("slotins: per-process instructions and allocations", "");
    drreg_options_t ops = {sizeof(ops), 1, false};
    if (!drmgr_init() || !drwrap_init() || drreg_init(&ops) != DRREG_SUCCESS ||
        !dr_raw_tls_calloc(&tls_seg, &tls_offs, 1, 0))
        DR_ASSERT(false);
    tls = drmgr_register_tls_field();
    if (tls == -1 || !drmgr_register_thread_init_event(thread_init) || !drmgr_register_thread_exit_event(thread_exit))
        DR_ASSERT(false);
    dr_register_exit_event(event_exit);
    dr_register_fork_init_event(child_after_fork);
    if (!drmgr_register_module_load_event(module_load) ||
        !drmgr_register_bb_instrumentation_event(NULL, count_bb, NULL))
        DR_ASSERT(false);
}
