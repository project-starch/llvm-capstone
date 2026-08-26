/* WAMR inside a Capstone domain: entry point and staged bring-up.
 *
 * WD_STAGE bisects startup. EVERY stage returns a marker, so a run always yields
 * a result rather than a wedge -- the rule this project learned the expensive way:
 * a wedged domain emits nothing, so a run that only ever fails tells you one bit
 * per boot. Stages are cumulative and ascending; build one .dom per stage and run
 * them in ONE boot, ascending, with the control first.
 *
 *   0  return at once      -- was the domain entered at all?
 *   1  + runtime init over the arena
 *   2  + load the module
 *   3  + instantiate
 *   4  + call "run" and return WHAT IT COMPUTED   (the default)
 *
 * Stage 4 returns the value, not a pass marker. 7 + 35 = 42 comes back as 42 or
 * the run failed; a separate "passed" marker would be one more thing that can be
 * right while the answer is wrong.
 */
#include "platform_internal.h"
#include "wasm_export.h"
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

#if WD_STAGE == 0
    WD_MARK(WD_OK);   /* touches nothing: proves entry, cap-init and return */
#else
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

    uint32_t argv[1] = { 0 };
    bool ok = wasm_runtime_call_wasm(env, fn, 0, argv);

    wasm_runtime_destroy_exec_env(env);
    wasm_runtime_deinstantiate(inst);
    wasm_runtime_unload(mod);
    wasm_runtime_destroy();

    /* The COMPUTED value, tagged so a zero from a failed call cannot be mistaken
       for a legitimate result. 0x5741_0000 | 42 is what a working interpreter
       returns; anything else is the failure it says it is. */
    *res = ok ? (0x57410000u | (unsigned)argv[0]) : (0x57410000u | 0x0400u | WD_FAIL);
#endif /* 3 */
#endif /* 2 */
#endif /* 1 */
#endif /* 0 */
}
