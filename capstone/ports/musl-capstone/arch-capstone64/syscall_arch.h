/* Pointer-typed syscall arguments retain tags across the C call boundary.
 * Integer arguments occupy the same registers without a tag. */
#define __SYSCALL_LL_E(x) (x)
#define __SYSCALL_LL_O(x) (x)

typedef void *syscall_arg_t;
#define __scc(X) ((syscall_arg_t)(X))

#ifdef CAPSTONE_RUNTIME_TRAP
#define __capstone_hostcall __capstone_linux_syscall
void *__capstone_linux_syscall_ref(long, void *, void *, void *, void *, void *, void *);
#endif

long __capstone_hostcall(long n, syscall_arg_t a, syscall_arg_t b,
                         syscall_arg_t c, syscall_arg_t d,
                         syscall_arg_t e, syscall_arg_t f);

static inline long __syscall0(long n)
{ return __capstone_hostcall(n, 0, 0, 0, 0, 0, 0); }

static inline long __syscall1(long n, syscall_arg_t a)
{ return __capstone_hostcall(n, a, 0, 0, 0, 0, 0); }

static inline long __syscall2(long n, syscall_arg_t a, syscall_arg_t b)
{ return __capstone_hostcall(n, a, b, 0, 0, 0, 0); }

static inline long __syscall3(long n, syscall_arg_t a, syscall_arg_t b,
                              syscall_arg_t c)
{ return __capstone_hostcall(n, a, b, c, 0, 0, 0); }

static inline long __syscall4(long n, syscall_arg_t a, syscall_arg_t b,
                              syscall_arg_t c, syscall_arg_t d)
{ return __capstone_hostcall(n, a, b, c, d, 0, 0); }

static inline long __syscall5(long n, syscall_arg_t a, syscall_arg_t b,
                              syscall_arg_t c, syscall_arg_t d,
                              syscall_arg_t e)
{ return __capstone_hostcall(n, a, b, c, d, e, 0); }

static inline long __syscall6(long n, syscall_arg_t a, syscall_arg_t b,
                              syscall_arg_t c, syscall_arg_t d,
                              syscall_arg_t e, syscall_arg_t f)
{ return __capstone_hostcall(n, a, b, c, d, e, f); }

/* No capability-aware vDSO is installed yet. */

#define IPC_64 0
