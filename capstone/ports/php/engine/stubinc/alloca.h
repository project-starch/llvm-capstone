/* alloca() for the PHP engine domain — NOT on the stack.
 *
 * Zend/zend_API.c:1300 does `lowercase_name = do_alloca(fname_len+1)` INSIDE
 * zend_register_functions' loop over the function table (Zend/zend_API.c:1221), and
 * Zend/zend.h:180 makes do_alloca() == alloca() whenever HAVE_ALLOCA is set. Stack
 * allocated in a loop is not reclaimed until the enclosing function RETURNS, so one call
 * that registers N functions holds N allocations live at once. The same shape is at
 * zend_execute_API.c:888, zend_builtin_functions.c:817, zend_compile.c:995 and
 * zend_object_handlers.c:668.
 *
 * On a domain the stack is a fixed capability with hard bounds, and overrunning it is a
 * capability fault in an arbitrary innocent callee -- the first observed symptom was a
 * spill inside memmove(), nowhere near the cause.
 *
 * Routing alloca off the stack is UPSTREAM-SANCTIONED, not an invention: Zend/zend.h:183
 * is `#define do_alloca(p) emalloc(p)` for platforms without alloca. This pool is the same
 * trade with the same lifetime (never individually freed) and, unlike emalloc, it cannot
 * perturb the capability-bounded heap the experiment measures.
 */
#ifndef _ALLOCA_H
#define _ALLOCA_H 1
void *php_capstone_alloca(unsigned long);
extern unsigned long php_capstone_alloca_total;  /* bytes handed out, cumulative */
extern unsigned long php_capstone_alloca_peak;   /* high-water mark */
#define alloca(n) php_capstone_alloca((unsigned long)(n))
#endif
