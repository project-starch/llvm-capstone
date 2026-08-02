#include <stdint.h>
typedef void (*fn_t)(void);
typedef uintptr_t method_t;
static method_t pack(fn_t f)   { return (((uintptr_t)f) << 2) | 1; }
static fn_t     unpack(method_t m) { return (fn_t)((uintptr_t)m >> 2); }
void g(void);
int main(void){ method_t m = pack(g); unpack(m)(); return 0; }
