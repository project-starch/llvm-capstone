#include <stdint.h>

// Start from "nothing optional is enabled" and add back only what the domain needs.
// Overridable via DOMAIN_EXTRA_DEFS so several feature levels can be built and run in ONE boot:
// the level decides which builtins exist at all, and a test that fails for want of a builtin says
// nothing about capabilities. Three knobs are guarded, not just the level, because raising the
// level alone leaves long integers and error text where MINIMAL put them.
#ifndef MICROPY_CONFIG_ROM_LEVEL
#define MICROPY_CONFIG_ROM_LEVEL (MICROPY_CONFIG_ROM_LEVEL_MINIMUM)
#endif

#define MICROPY_ENABLE_COMPILER           (1)
#define MICROPY_ENABLE_GC                 (1)
#define MICROPY_USE_INTERNAL_ERRNO         (1)

#define MICROPY_OBJ_REPR                  (MICROPY_OBJ_REPR_A)
#define MP_INT_TYPE                       (MP_INT_TYPE_INT64)
#define MP_SSIZE_MAX                      (0x7fffffffffffffffLL)
// Stream ioctl multiplexes integer and pointer arguments. uintptr_t retains only the address
// on Capstone, so use a capability-wide carrier and convert numeric requests at their edges.
#define MICROPY_STREAM_IOCTL_ARG_TYPE     void *
#ifndef MICROPY_LONGINT_IMPL
#define MICROPY_LONGINT_IMPL              (MICROPY_LONGINT_IMPL_NONE)
#endif
#define MICROPY_FLOAT_IMPL                (MICROPY_FLOAT_IMPL_NONE)

#define MICROPY_PERSISTENT_CODE_LOAD      (0)
#define MICROPY_MODULE_FROZEN_MPY         (0)
#define MICROPY_MODULE_FROZEN_STR         (0)

#define MICROPY_ENABLE_EXTERNAL_IMPORT    (0)
#define MICROPY_READER_POSIX              (0)
#define MICROPY_READER_VFS                (0)
#define MICROPY_VFS                       (0)
#define MICROPY_HELPER_REPL               (0)
#define MICROPY_PY_BUILTINS_INPUT         (0)
#define MICROPY_PY_BUILTINS_EXECFILE      (0)
#define MICROPY_PY_SYS_STDFILES           (0)
#define MICROPY_PY_UCTYPES                (0)
#define MICROPY_KBD_EXCEPTION             (0)
#define MICROPY_ENABLE_SCHEDULER          (0)
#define MICROPY_ENABLE_FINALISER          (0)
#define MICROPY_GC_SPLIT_HEAP             (0)
#define MICROPY_ENABLE_PYSTACK            (0)

#define MICROPY_PY_SYS_MODULES            (0)
#define MICROPY_PY_SYS_EXIT               (0)
#define MICROPY_PY_SYS_PATH               (0)
#define MICROPY_PY_SYS_ARGV               (0)

#define MICROPY_NLR_SETJMP                (1)

#define MICROPY_STACK_CHECK               (1)
#define MICROPY_STACK_CHECK_MARGIN        (4096)

// TERSE shortens exception text, so a test scored against an upstream .exp file fails on the
// MESSAGE while the behaviour is right -- assign_expr_syntaxerror.py is exactly that.
#ifndef MICROPY_ERROR_REPORTING
#define MICROPY_ERROR_REPORTING           (MICROPY_ERROR_REPORTING_TERSE)
#endif

#define MICROPY_ALLOC_PARSE_CHUNK_INIT    (16)

// declared by py/mphal.h; the macro expands only inside py/mpprint.c, which includes it
#define MP_PLAT_PRINT_STRN(str, len) mp_hal_stdout_tx_strn((str), (len))

typedef long mp_off_t;
#include <alloca.h>

#define MICROPY_HW_BOARD_NAME "capstone-domain"
#define MICROPY_HW_MCU_NAME "cva6-capstone"
#define MP_STATE_PORT MP_STATE_VM
