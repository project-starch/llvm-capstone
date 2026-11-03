/* Compare statically initialized capabilities with normal address lowering.
 * Volatile slots force loads after startup; querying all fields distinguishes
 * an untagged integer or a correctly addressed but over-broad capability.
 * Build both with and without -mllvm -capstone-cap-init-table=false.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static char bytes[64] = "table";
extern char bytes_alias[64] __attribute__((weak, alias("bytes")));
static int function(void) { return 37; }
extern int function_alias(void) __attribute__((weak, alias("function")));
struct nested { unsigned n; void *p[5]; };
static const struct nested pointers = {5, {bytes, bytes + 17, bytes + 64,
                                          bytes_alias, NULL}};
static int (*volatile functions[2])(void) = {function, function_alias};
static int constructor_ran;

static unsigned long kind(void *p) {
    unsigned long v;
    __asm__ volatile(".insn r 0x5b, 1, 4, %0, %1, x1" : "=r"(v) : "r"(p));
    return v;
}
static unsigned long permissions(void *p) {
    unsigned long v;
    __asm__ volatile(".insn r 0x5b, 1, 4, %0, %1, x5" : "=r"(v) : "r"(p));
    return v;
}
static void require(int ok, const char *name) {
    printf("CAP_INIT_CHECK:%s:%s\n", name, ok ? "PASS" : "FAIL");
    if (!ok) exit(1);
}
__attribute__((noinline)) static int equal(void *p, void *q) {
    return __builtin_capstone_cap_get_tag(p) &&
        __builtin_capstone_cap_get_tag(q) &&
        __builtin_capstone_cap_get_cursor(p) == __builtin_capstone_cap_get_cursor(q) &&
        __builtin_capstone_cap_get_base(p) == __builtin_capstone_cap_get_base(q) &&
        __builtin_capstone_cap_get_end(p) == __builtin_capstone_cap_get_end(q) &&
        kind(p) == kind(q) && kind(p) == 1 && permissions(p) == permissions(q);
}
__attribute__((constructor)) static void before_main(void) {
    void *p = ((const volatile struct nested *)&pointers)->p[0];
    constructor_ran = equal(p, bytes) && *(char *)p == 't';
}
__attribute__((noinline)) static int dispatch(unsigned i) {
    static void *volatile labels[] = {&&first, &&second};
    void *p = labels[i];
    if (!equal(p, i ? &&second : &&first)) return -1;
    goto *p;
first:
    return 11;
second:
    return 23;
}
int main(void) {
    const volatile struct nested *p = &pointers;
    require(constructor_ran, "before_constructor");
    require(equal(p->p[0], bytes), "data");
    require(equal(p->p[1], bytes + 17), "interior");
    require(equal(p->p[2], bytes + 64), "one_past");
    require(equal(p->p[3], bytes_alias), "weak_data_alias");
    require(!p->p[4] && !__builtin_capstone_cap_get_tag(p->p[4]), "null");
    require(equal((void *)functions[0], (void *)function), "function");
    require(equal((void *)functions[1], (void *)function_alias), "weak_function_alias");
    require(functions[0]() == 37 && functions[1]() == 37, "call");
    require(dispatch(0) == 11 && dispatch(1) == 23, "labels");
    puts("CAP_INIT_OK");
    return 0;
}
