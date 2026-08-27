/* WAMR inside a Capstone domain: entry point and staged bring-up.
 *
 * WD_STAGE bisects startup. EVERY stage returns a marker, so a run always yields
 * a result rather than a wedge -- the rule this project learned the expensive way:
 * a wedged domain emits nothing, so a run that only ever fails tells you one bit
 * per boot. Stages are cumulative and ascending; build one .dom per stage and run
 * them in ONE boot, ascending, with the control first.
 *
 *   0  return at once      -- was the domain entered at all?
 *
 *   10 + os_malloc from the arena only          -- does the platform layer work?
 *   11 + the mutex and the ref-count guard      -- what full_init does before it
 *                                                  reaches anything of WAMR's
 *   12 + wasm_runtime_memory_init over the pool -- the allocator on our arena
 *   13 + wasm_runtime_set_default_running_mode
 *   14 + wasm_native_init  (the live half of wasm_runtime_env_init)
 *
 *   The 1x rungs open up stage 1, which faulted with cause 24 while stage 0
 *   returned. They call the pieces full_init calls, in its order, so the first one
 *   that does not return names the step rather than the function.
 *
 *   20 + read the module's magic (get_package_type)
 *   21 + a load that must FAIL EARLY: size 3, so it returns through
 *        set_error_buf without the parser ever running
 *   22 + the real load
 *
 *   The 2x rungs open up stage 2. Rung 21 is the useful one: it exercises the
 *   error path and the caller's buffer without touching the loader, so a fault
 *   there and a fault at 22 mean different things.
 *
 *   1  + runtime init over the arena
 *   2  + load the module
 *   3  + instantiate
 *   5  + look the function up and create an exec_env, but do NOT call
 *   4  + call "run" and return WHAT IT COMPUTED   (the default)
 *
 * Stage 4 returns the value, not a pass marker. 7 + 35 = 42 comes back as 42 or
 * the run failed; a separate "passed" marker would be one more thing that can be
 * right while the answer is wrong.
 */
#include "platform_internal.h"
#include "wasm_export.h"
#include "mem_alloc.h"
#include "wasm_native.h"   /* wasm_native_init, for rung 14 */
#include "wamr_test_module.h"

#ifndef WD_STAGE
#define WD_STAGE 4
#endif

/* 0x5741 is "WA". Stage in bits 8..15, code in the low byte, so a marker reads as
   5741ssnn and cannot be confused with MicroPython's 4D50 or JerryScript's 1E. */
#define WD_MARK(n) do { *res = 0x57410000u | ((unsigned)WD_STAGE << 8) | (unsigned)(n); return; } while (0)
#define WD_OK   0x01u
#define WD_FAIL 0xEEu

/* The runtime's heap. os_mmap's arena in capstone_platform.c backs the platform's
   own allocations; this is what WAMR sub-allocates every internal object from,
   and it is the object the corpus is about: one region, carved in software, whose
   frees never reach the hardware. Sized here rather than in the platform layer so
   the two are visibly separate. */
#ifndef WD_HEAP_BYTES
#define WD_HEAP_BYTES (192u * 1024u)
#endif
static char wd_heap[WD_HEAP_BYTES] __attribute__((aligned(16)));

#ifndef WD_STACK_BYTES
#define WD_STACK_BYTES (16u * 1024u)
#endif

void
domain_main(unsigned *res, unsigned func)
{
    (void)func;

#if WD_STAGE >= 10 && WD_STAGE <= 14
    MemAllocOption pool_option;
    memset(&pool_option, 0, sizeof(pool_option));
    pool_option.pool.heap_buf = wd_heap;
    pool_option.pool.heap_size = sizeof(wd_heap);
#endif

#if WD_STAGE == 0
    WD_MARK(WD_OK);   /* touches nothing: proves entry, cap-init and return */

#elif WD_STAGE == 40
    /* SPECIMEN: WAMR's EMS allocator, the missing pinuse bit (PR 2279).
     *
     * gc_realloc_vo_internal grows in place by absorbing part of the next free
     * block and re-adds the remainder with gci_add_fc(), but never sets pinuse on
     * the new hmu_next. A later free then coalesces BACKWARDS into the live object
     * and the allocator hands the same memory out twice.
     *
     * The verdict is NOT "does it fault". The forged header is written INSIDE p's
     * own allocation, a legal write no bounds check can object to. The detectable
     * event is one step later: a second pointer that ALIASES the first. This
     * measures whether the machine can see an allocator hand out overlapping
     * memory, which a pure bounds model cannot.
     *
     * THE GROWTH IS SWEPT, NOT GUESSED. Upstream's reproducer hard-codes a 1000-byte
     * arena and a 12-byte growth, tuned to an 8-byte-pointer layout. Patch 0002 put
     * this target's GC alignment on the POINTER, so headers and tree nodes are
     * wider. Measured here: a 12-byte growth reallocs IN PLACE and leaves no
     * remainder, so the vulnerable branch is never entered and both arms agree --
     * a clean result that says nothing. The bug needs a growth big enough to split
     * the next free block and small enough to leave a remainder, so the stage tries
     * every one and reports the first that aliases.
     *
     * Report: bit 15 set means an aliasing growth was FOUND, low bits carry it.
     * Bit 15 clear means none in the swept range, and the low bits are the count of
     * growths that completed the sequence -- zero there would mean the sweep never
     * ran and must not be read as "no bug".
     */
    {
        enum { EMS_ARENA = 4096, EMS_FIRST = 256, GROW_MAX = 1024 };
        static char ems_store[EMS_ARENA];
        unsigned g, ran = 0u;

        for (g = 4u; g <= GROW_MAX; g += 4u) {
            mem_allocator_t a;
            uint8_t *p, *p2;
            unsigned i;

            for (i = 0; i < (unsigned)sizeof(ems_store); i++)
                ems_store[i] = 0;
            a = mem_allocator_create(ems_store, (uint32_t)sizeof(ems_store));
            if (!a)
                continue;
            p = mem_allocator_malloc(a, EMS_FIRST);
            if (p)
                p = mem_allocator_realloc(a, p, EMS_FIRST + g);
            if (!p)
                continue;
            ran++;
            /* Legal writes inside p, shaped like a free HMU header. */
            *(uint32_t *)(p + EMS_FIRST) = (1u << 30) | 0x20u;
            if (g >= 8u)
                *(uint32_t *)(p + EMS_FIRST + g - 4) = g;

            p2 = mem_allocator_malloc(a, EMS_FIRST);
            if (p2) {
                uintptr_t up = (uintptr_t)p, up2 = (uintptr_t)p2;
                if (up2 >= up && up2 < up + EMS_FIRST + g) {
                    *res = 0x57410000u | 0x8000u | (g & 0x1FFFu);
                    return;
                }
                mem_allocator_free(a, p2);
            }
            mem_allocator_free(a, p);
        }
        *res = 0x57410000u | (ran & 0x1FFFu);
        return;
    }

#elif WD_STAGE == 30
    /* WHERE IS THIS IMAGE LOADED? Not a bring-up rung: an instrument.
     *
     * A trap reports its PC as a RUNTIME address, and mapping that back to a
     * function needs the load base, which nothing prints. Guessing it from the
     * region's alignment gave an offset outside .text, so it is measured instead.
     *
     * The loader calls a domain repeatedly and prints every return value, and the
     * entry glue does NOT re-run initialisers between calls, so a static counter
     * survives. Two calls give two anchors, and two anchors prove the mapping is a
     * constant offset rather than assuming it. */
    {
        static unsigned nth;
        const void *anchors[] = { (const void *)&domain_main,
                                  (const void *)&wasm_runtime_load };
        *res = (unsigned)(uintptr_t)anchors[nth & 1u];
        nth++;
        return;
    }

#elif WD_STAGE >= 20 && WD_STAGE <= 22
    {
        /* FIRST CALL REPORTS WHERE THIS IMAGE IS LOADED, and it must be in THIS
         * image rather than a separate one: a trap's PC is a runtime address, the
         * load base is not printed anywhere, and two different images are not
         * loaded at the same place. So call 1 hands back the runtime address of a
         * symbol whose ELF address we can read from the same file, and call 2 does
         * the work that faults. One boot, and the mapping is measured rather than
         * guessed -- guessing it from the region's alignment put the offset outside
         * .text, which is how we know guessing does not work here. */
        static unsigned nth;
        if (nth++ == 0) {
            *res = (unsigned)(uintptr_t)&domain_main;
            return;
        }

        MemAllocOption po;
        memset(&po, 0, sizeof(po));
        po.pool.heap_buf = wd_heap;
        po.pool.heap_size = sizeof(wd_heap);
        RuntimeInitArgs ia;
        memset(&ia, 0, sizeof(ia));
        ia.mem_alloc_type = Alloc_With_Pool;
        ia.mem_alloc_option = po;
        if (!wasm_runtime_full_init(&ia))
            WD_MARK(WD_FAIL);

        if (get_package_type(wamr_test_module, (uint32_t)sizeof(wamr_test_module))
            != Wasm_Module_Bytecode) {
            wasm_runtime_destroy();
            WD_MARK(WD_FAIL);
        }
#if WD_STAGE == 20
        wasm_runtime_destroy();
        WD_MARK(WD_OK);
#else
        char e[96];
        e[0] = 0;
        /* MUST fail: three bytes is below the four the header needs, so this
           returns through set_error_buf and writes into e. A null module here is
           the CORRECT outcome, and a non-null one would mean the size check did
           not run. */
        if (wasm_runtime_load(wamr_test_module, 3, e, (uint32_t)sizeof(e))) {
            wasm_runtime_destroy();
            WD_MARK(WD_FAIL);
        }
        if (e[0] == 0) {          /* the error path must have written something */
            wasm_runtime_destroy();
            WD_MARK(0xE1);
        }
#if WD_STAGE == 21
        wasm_runtime_destroy();
        WD_MARK(WD_OK);
#else
        wasm_module_t m = wasm_runtime_load(wamr_test_module,
                                            (uint32_t)sizeof(wamr_test_module),
                                            e, (uint32_t)sizeof(e));
        if (!m) {
            wasm_runtime_destroy();
            WD_MARK(WD_FAIL);
        }
        wasm_runtime_unload(m);
        wasm_runtime_destroy();
        WD_MARK(WD_OK);
#endif /* 21 */
#endif /* 20 */
    }

#elif WD_STAGE >= 10 && WD_STAGE <= 14
    /* The fine ladder inside full_init. Each rung does what the one below it did
       and one step more, so the first that fails to return IS the step. */
    {
        void *p = os_malloc(64);
        if (!p)
            WD_MARK(WD_FAIL);
        ((char *)p)[0] = 0x5A;
        if (((volatile char *)p)[0] != 0x5A)
            WD_MARK(WD_FAIL);
    }
#if WD_STAGE == 10
    WD_MARK(WD_OK);
#else
    {
        /* What full_init does before it touches anything of WAMR's: take its
           static mutex and read its static ref count. Both are file-scope globals,
           so this rung is really asking whether the gp-captable carve reached
           them. */
        static korp_mutex probe_lock = OS_THREAD_MUTEX_INITIALIZER;
        static int probe_count;
        os_mutex_lock(&probe_lock);
        probe_count++;
        os_mutex_unlock(&probe_lock);
        if (probe_count != 1)
            WD_MARK(WD_FAIL);
    }
#if WD_STAGE == 11
    WD_MARK(WD_OK);
#else
    if (!wasm_runtime_memory_init(Alloc_With_Pool, &pool_option))
        WD_MARK(WD_FAIL);
#if WD_STAGE == 12
    wasm_runtime_memory_destroy();
    WD_MARK(WD_OK);
#else
    if (!wasm_runtime_set_default_running_mode(Mode_Default))
        WD_MARK(WD_FAIL);
#if WD_STAGE == 13
    wasm_runtime_memory_destroy();
    WD_MARK(WD_OK);
#else
    /* The only step of wasm_runtime_env_init that is not compiled out in this
       configuration. env_init itself is static and cannot be called from here;
       bh_platform_init is this port's own and rung 10 already covered it.

       NOTE, and it is a correction to this ladder rather than a finding: rung 14
       used to call wasm_runtime_init(), which faulted. That was MY error, not
       WAMR's -- wasm_runtime_init is an ALTERNATIVE entry that initialises the
       memory subsystem with Alloc_With_System_Allocator, a malloc this domain
       does not have. The full_init path never calls it. A rung that tests a path
       the program does not take produces a clean, monotone, entirely void
       result. */
    if (!wasm_native_init())
        WD_MARK(WD_FAIL);
    wasm_runtime_memory_destroy();
    WD_MARK(WD_OK);
#endif /* 13 */
#endif /* 12 */
#endif /* 11 */
#endif /* 10 */

#else
    /* Same anchor as the 2x rungs: call 1 reports where this image is loaded so a
       trap pc can be mapped, call 2 does the work. */
    {
        static unsigned nth_main;
        if (nth_main++ == 0) {
            *res = (unsigned)(uintptr_t)&domain_main;
            return;
        }
    }

    RuntimeInitArgs init;
    memset(&init, 0, sizeof(init));
    init.mem_alloc_type = Alloc_With_Pool;
    init.mem_alloc_option.pool.heap_buf = wd_heap;
    init.mem_alloc_option.pool.heap_size = sizeof(wd_heap);

    if (!wasm_runtime_full_init(&init))
        WD_MARK(WD_FAIL);
#if WD_STAGE == 1
    wasm_runtime_destroy();
    WD_MARK(WD_OK);
#else
    char err[96];
    err[0] = 0;
    wasm_module_t mod = wasm_runtime_load(wamr_test_module,
                                          (uint32_t)sizeof(wamr_test_module),
                                          err, (uint32_t)sizeof(err));
    if (!mod)
        WD_MARK(WD_FAIL);
#if WD_STAGE == 2
    wasm_runtime_unload(mod);
    wasm_runtime_destroy();
    WD_MARK(WD_OK);
#else
    wasm_module_inst_t inst =
        wasm_runtime_instantiate(mod, WD_STACK_BYTES, 0, err, (uint32_t)sizeof(err));
    if (!inst)
        WD_MARK(WD_FAIL);
#if WD_STAGE == 3
    wasm_runtime_deinstantiate(inst);
    wasm_runtime_unload(mod);
    wasm_runtime_destroy();
    WD_MARK(WD_OK);
#else
    wasm_function_inst_t fn = wasm_runtime_lookup_function(inst, "run");
    wasm_exec_env_t env = wasm_runtime_create_exec_env(inst, WD_STACK_BYTES);
    if (!fn || !env)
        WD_MARK(WD_FAIL);
#if WD_STAGE == 5
    /* Everything the call needs, EXCEPT the call. Splits stage 4 in two: an
       exec_env is a fresh allocation carrying the interpreter stack, and looking a
       function up walks the instance's export table, so a fault here and a fault in
       the call itself mean different things. */
    wasm_runtime_destroy_exec_env(env);
    wasm_runtime_deinstantiate(inst);
    wasm_runtime_unload(mod);
    wasm_runtime_destroy();
    WD_MARK(WD_OK);
#endif

    uint32_t argv[1] = { 0 };
    bool ok = wasm_runtime_call_wasm(env, fn, 0, argv);
#if WD_STAGE == 6
    /* Was the memset that halts stage 4 handed an untagged destination, and how long
       was it? A length of zero would mean the call cannot be the one clearing this
       module's locals, since "run" has no parameters and no locals.
       Needs BEEBS_TAGCHECK=1; bit 23 carries the selftest, so a reading of all zeros
       is distinguishable from an instrument that was never compiled in. */
#if !defined(BEEBS_MEMCPY_TAGCHECK) || !BEEBS_MEMCPY_TAGCHECK
#error "stage 6 reads the tag-check counters; build it with BEEBS_TAGCHECK=1"
#endif
    {
        extern unsigned long capstone_mcp_hits, capstone_mcp_selftest_seen;
        extern unsigned long capstone_mcp_where, capstone_mcp_ety, capstone_mcp_n;
        unsigned h = capstone_mcp_hits > 7ul ? 7u : (unsigned)capstone_mcp_hits;
        *res = 0x6D000000u
             | ((capstone_mcp_selftest_seen ? 1u : 0u) << 23)
             | (h << 20)
             | (((unsigned)capstone_mcp_where & 0xFu) << 16)
             | (((unsigned)capstone_mcp_ety & 0xFu) << 12)
             | ((unsigned)capstone_mcp_n & 0xFFFu);
        return;
    }
#endif

    wasm_runtime_destroy_exec_env(env);
    wasm_runtime_deinstantiate(inst);
    wasm_runtime_unload(mod);
    wasm_runtime_destroy();

    /* The COMPUTED value, tagged so a zero from a failed call cannot be mistaken
       for a legitimate result. 0x5741_0000 | 42 is what a working interpreter
       returns; anything else is the failure it says it is. */
    /* MASKED to 16 bits. Unmasked, a result with high bits set overwrites the
       0x5741 tag and the answer reads as a crash: 11 + -40 came back as
       0xFFFFFFE3 and was taken for one. Sixteen bits is all the marker protocol
       has, so say so rather than let the tag be the thing that gives way. */
    *res = ok ? (0x57410000u | ((unsigned)argv[0] & 0xFFFFu))
              : (0x57410000u | 0x0400u | WD_FAIL);
#endif /* 3 */
#endif /* 2 */
#endif /* 1 */
#endif /* 0 */
}
