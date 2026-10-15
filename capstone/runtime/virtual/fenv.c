/* The floating-point environment of a virtual-C application is the hardware
 * fcsr: applications run F and D, and QEMU's supervisor keeps each context's
 * fcsr with its f registers (cmake/VirtualCodegen.cmake).
 *
 * musl's riscv64 implementation of these functions is assembly
 * (src/fenv/riscv64/fenv.S) selected by __riscv_flen. This target defines
 * __capstone_flen instead, and the port compiles only musl's C sources, so
 * musl's archive holds the generic no-op src/fenv/fenv.c. These seven
 * functions are exactly the ones that file defines; the runtime is linked
 * ahead of the archive, so the no-op object is never pulled in. musl's
 * fesetround, feholdexcept, feupdateenv and the rest are built on them.
 * Same semantics as musl's fenv.S. */
#include <fenv.h>

int feclearexcept(int mask)
{
    mask &= FE_ALL_EXCEPT;
    __asm__ volatile("csrc fflags, %0" : : "r"(mask));
    return 0;
}

int feraiseexcept(int mask)
{
    mask &= FE_ALL_EXCEPT;
    __asm__ volatile("csrs fflags, %0" : : "r"(mask));
    return 0;
}

int fetestexcept(int mask)
{
    int flags;
    __asm__ volatile("frflags %0" : "=r"(flags));
    return flags & mask & FE_ALL_EXCEPT;
}

int fegetround(void)
{
    int mode;
    __asm__ volatile("frrm %0" : "=r"(mode));
    return mode;
}

/* fesetround has checked the mode. */
int __fesetround(int mode)
{
    __asm__ volatile("fsrm %0" : : "r"(mode));
    return 0;
}

int fegetenv(fenv_t *env)
{
    unsigned int fcsr;
    __asm__ volatile("frcsr %0" : "=r"(fcsr));
    *env = fcsr;
    return 0;
}

int fesetenv(const fenv_t *env)
{
    unsigned int fcsr = env == FE_DFL_ENV ? 0 : *env;
    __asm__ volatile("fscsr %0" : : "r"(fcsr));
    return 0;
}

/* compiler-rt's soft-float builtins remain for long double (binary128, no Q
 * extension). Its addition and subtraction ask these two for the rounding
 * mode and to raise inexact; the generic fp_mode.c of the builtins list
 * answers "to nearest" and raises nothing, so long double sums (and rintl,
 * built on them) would ignore fesetround. These are compiler-rt's
 * riscv/fp_mode.c, which selects on __riscv_f; defined here, ahead of the
 * builtins archive, they replace the generic object. */
#include "../../../compiler-rt/lib/builtins/fp_mode.h"

CRT_FE_ROUND_MODE __fe_getround(void)
{
    switch (fegetround()) {
    case FE_TOWARDZERO: return CRT_FE_TOWARDZERO;
    case FE_DOWNWARD: return CRT_FE_DOWNWARD;
    case FE_UPWARD: return CRT_FE_UPWARD;
    default: return CRT_FE_TONEAREST;
    }
}

int __fe_raise_inexact(void)
{
    feraiseexcept(FE_INEXACT);
    return 0;
}
