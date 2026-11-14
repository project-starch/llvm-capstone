/* Test-only wrappers. Zero models NOGROW without changing the old lifetime. */
#define capstone_cap_derive life_real_derive
#define capstone_cap_renew life_real_renew
#include_next <capstone/capability.h>
#undef capstone_cap_derive
#undef capstone_cap_renew
extern int life_fail, life_calls, life_renew_fail;
static inline void *capstone_cap_derive(void *p, unsigned long off, unsigned long n){
    if (++life_calls == life_fail) return 0;
    return life_real_derive(p, off, n);
}
static inline void *capstone_cap_renew(void *p){
    if (life_renew_fail) return 0;
    return life_real_renew(p);
}
