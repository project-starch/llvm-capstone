#include <stdint.h>
#include <string.h>
#include "py/builtin.h"
#include "py/compile.h"
#include "py/runtime.h"
#include "py/gc.h"
#include "py/cstack.h"
#include "py/mphal.h"
#include "py/stackctrl.h"

/* ---- the GC heap. One global; under -capstone-gp-captable its storage is CARVED
   from dom_data at entry, so it is charged against the domain's stack budget.
   32-byte aligned and a multiple of 32 so gc_init's align-down is a no-op. */
#ifndef MPY_HEAP_SIZE
#define MPY_HEAP_SIZE (96U * 1024U)
#endif
static unsigned char mpy_heap[MPY_HEAP_SIZE] __attribute__((aligned(32)));

/* ---- output: the hostcall shared region, same shape as benchmarks/sqlite */
#define MPY_DPI_REGION_SHARE 1U
struct mpy_hostcall_v0 { unsigned long long phase, opcode, offset, length; long long result, error; };
#define MPY_HC_REGION_SIZE 4096UL
static volatile struct mpy_hostcall_v0 *hc_meta;
static volatile char *hc_payload;
static unsigned hc_share_count;

mp_uint_t mp_hal_stdout_tx_strn(const char *str, size_t len) {
    if (!hc_meta || !hc_payload) return 0;
    char *payload = (char *)hc_payload;
    unsigned long off = hc_meta->length;
    while (len-- && off + 1 < MPY_HC_REGION_SIZE) payload[off++] = *str++;
    hc_meta->length = off;
    return off;
}

/* ---- what py/ demands and the port must supply ---- */
void nlr_jump_fail(void *val) {
    (void)val;
    mp_hal_stdout_tx_strn("\nNLRFAIL\n", 9);
    for (;;) { }
}

void gc_collect(void) {
    jmp_buf regs;              /* spills ra,s0-s11,sp as tagged 16-byte slots */
    setjmp(regs);
    gc_collect_start();
    void **sp_now = (void **)&regs;
    char *top = MP_STATE_THREAD(stack_top);
    gc_collect_root(sp_now, ((size_t)(top - (char *)sp_now)) / sizeof(void *));
    gc_collect_end();
}

/* ---- one line of Python ---- */
static int do_str(const char *src, mp_parse_input_kind_t kind) {
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        mp_lexer_t *lex = mp_lexer_new_from_str_len(MP_QSTR__lt_stdin_gt_, src, strlen(src), 0);
        qstr source_name = lex->source_name;
        mp_parse_tree_t pt = mp_parse(lex, kind);
        mp_obj_t module_fun = mp_compile(&pt, source_name, false);
        mp_call_function_0(module_fun);
        nlr_pop();
        return 0;
    }
    mp_obj_print_exception(&mp_plat_print, MP_OBJ_FROM_PTR(nlr.ret_val));
    return 1;
}

/* how much stack the entry glue actually left: lcc(sp,2)-lcc(sp,3), the stkhr probe */
static size_t cap_stack_headroom(void) {
    unsigned long cur, base;
    __asm volatile(".insn r 0x5b, 0x1, 0x4, %0, sp, x2" : "=r"(cur));
    __asm volatile(".insn r 0x5b, 0x1, 0x4, %0, sp, x3" : "=r"(base));
    return (size_t)(cur - base);
}

void domain_main(unsigned *res, unsigned func) {
    if (func == MPY_DPI_REGION_SHARE) {
        if (hc_share_count == 0) hc_meta = (volatile struct mpy_hostcall_v0 *)res;
        else if (hc_share_count == 1) hc_payload = (volatile char *)res;
        ++hc_share_count;
        return;
    }
    if (hc_meta) hc_meta->length = 0;
    mp_cstack_init_with_sp_here(cap_stack_headroom());
    gc_init(mpy_heap, mpy_heap + sizeof(mpy_heap));
    mp_init();
    int rc = do_str("print(1+1)", MP_PARSE_FILE_INPUT);
    mp_deinit();
    *res = 0x4D500000u | (unsigned)rc;
}
