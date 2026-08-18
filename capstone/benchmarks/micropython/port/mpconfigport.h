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

// The EXTRA census exercises warnings and exceptions while the heap is locked. Keep the minimum
// profile small, but make the selected feature level internally consistent with those tests.
#ifndef MICROPY_WARNINGS
#define MICROPY_WARNINGS                   (MICROPY_CONFIG_ROM_LEVEL_AT_LEAST_EXTRA_FEATURES)
#endif
#ifndef MICROPY_ENABLE_EMERGENCY_EXCEPTION_BUF
#define MICROPY_ENABLE_EMERGENCY_EXCEPTION_BUF (MICROPY_CONFIG_ROM_LEVEL_AT_LEAST_EXTRA_FEATURES)
#endif
#ifndef MICROPY_MEM_STATS
#define MICROPY_MEM_STATS                   (MICROPY_CONFIG_ROM_LEVEL_AT_LEAST_EXTRA_FEATURES)
#endif
#ifndef MICROPY_MALLOC_USES_ALLOCATED_SIZE
#define MICROPY_MALLOC_USES_ALLOCATED_SIZE  (MICROPY_CONFIG_ROM_LEVEL_AT_LEAST_EXTRA_FEATURES)
#endif
#ifndef MICROPY_PY_TSTRINGS
#define MICROPY_PY_TSTRINGS                 (MICROPY_PY_FSTRINGS \
    && MICROPY_CONFIG_ROM_LEVEL_AT_LEAST_EXTRA_FEATURES)
#endif

#define MICROPY_OBJ_REPR                  (MICROPY_OBJ_REPR_A)
#define MP_INT_TYPE                       (MP_INT_TYPE_INT64)
#define MP_SSIZE_MAX                      (0x7fffffffffffffffLL)
// Stream ioctl multiplexes integer and pointer arguments. uintptr_t retains only the address
// on Capstone, so use a capability-wide carrier and convert numeric requests at their edges.
#define MICROPY_STREAM_IOCTL_ARG_TYPE     void *
#ifndef MICROPY_LONGINT_IMPL
#define MICROPY_LONGINT_IMPL              (MICROPY_LONGINT_IMPL_NONE)
#endif
#ifndef MICROPY_FLOAT_IMPL
#define MICROPY_FLOAT_IMPL                (MICROPY_FLOAT_IMPL_NONE)
#endif
// MicroPython's lib/libm_dbl has no log2; this is the switch that makes modmath.c supply its own.
#define MP_NEED_LOG2                      (1)

#define MICROPY_PERSISTENT_CODE_LOAD      (0)
#define MICROPY_MODULE_FROZEN_MPY         (0)
#define MICROPY_MODULE_FROZEN_STR         (0)

#define MICROPY_ENABLE_EXTERNAL_IMPORT    (0)
#define MICROPY_READER_POSIX              (0)
#ifndef MICROPY_READER_VFS
#define MICROPY_READER_VFS                (0)
#endif
// MPY_VFS=1 turns on the filesystem stack, for MPY-T14/MPY-T15 only. Guarded so the
// default image is byte-identical without it: the two rows need extmod/vfs*.c and
// lib/oofatfs, which cost ~70 KB of image and would otherwise be carried by every
// build that has no use for them.
#ifndef MICROPY_VFS
#define MICROPY_VFS                       (0)
#endif
#if MICROPY_VFS
#define MICROPY_VFS_FAT                   (1)
#define MICROPY_READER_VFS                (1)
// Demanded by extmod/vfs_fat.c:32, not chosen: a FAT file object needs a finaliser
// to close itself when collected. It adds a finaliser table to the heap, which is a
// real change to the collector's layout -- which is why it stays behind MPY_VFS
// rather than being turned on for every image.
#define MICROPY_ENABLE_FINALISER          (1)
// extmod/vfs_fat.c calls f_chdir and f_getcwd unconditionally, and ff.c only compiles
// them when FF_FS_RPATH is set. 2 is what every MicroPython port with a filesystem uses.
#define MICROPY_FATFS_RPATH               (2)
#endif
#define MICROPY_HELPER_REPL               (0)
#define MICROPY_PY_BUILTINS_INPUT         (0)
#define MICROPY_PY_BUILTINS_EXECFILE      (0)
#define MICROPY_PY_SYS_STDFILES           (0)
// uctypes is architecturally impossible here, measured rather than assumed: 15 of its 16 tests
// take a capability fault and the 16th fails. Its contract is uctypes.addressof() -> int and
// back, and a 64-bit integer cannot carry a 128-bit capability, so every reconstructed pointer
// arrives untagged. Enabling it also turns basics/struct_micropython.py from PASS into a fault,
// because that test's second half is guarded by `import uctypes`.
//
// It is nevertheless left ON, because turning it OFF does not boot. Three uctypes-off images --
// 200 tests at globals offset 0xa0000, the same 200 at 0xb0000, and 100 tests at 0xb0000 -- each
// hung on their FIRST domain call, while every uctypes-on image of the same chunks ran clean and
// a re-run of a known-good .dom confirmed the harness was healthy. One flag, three layouts, and
// the only code removed is a module nothing else calls. That is the image-perturbation hazard the
// handoff records for silicon (fpga-repros/S01-image-perturbation-hang), reproduced under QEMU
// where it costs minutes instead of a board session; the mechanism is NOT established, and
// py/parse.c:663 -- where MICROPY_PY_UCTYPES adds an entry to the const-folding module table --
// is a place to start, not a conclusion.
#ifndef MICROPY_PY_UCTYPES
#define MICROPY_PY_UCTYPES                (1)
#endif
#define MICROPY_KBD_EXCEPTION             (0)
#define MICROPY_ENABLE_SCHEDULER          (0)
// Set by the MPY_VFS block above when the filesystem stack is on, because
// extmod/vfs_fat.c refuses to compile without it. Left OFF by default: it adds a
// finaliser table to the heap, and the GC is where this port's capability fixes
// already live. See the MICROPY_PY_WEAKREF note below for the other half of that.
#ifndef MICROPY_ENABLE_FINALISER
#define MICROPY_ENABLE_FINALISER          (0)
#endif
#define MICROPY_GC_SPLIT_HEAP             (0)
#define MICROPY_ENABLE_PYSTACK            (0)

// Two features whose upstream default is above EXTRA, but which need nothing this domain lacks --
// no clock, no filesystem, no device. sys.exit only raises SystemExit, which the test runner
// already handles as the target-skip convention.
//
// MICROPY_PY_WEAKREF is NOT among them: it needs a second GC side table, and with
// MICROPY_ENABLE_FINALISER off upstream's gc_sweep_run_finalisers does not even compile (it takes
// BLOCKS_PER_FTB and the declaration of `block` from the finaliser branch). Enabling both is a GC
// change, and the GC is where this port's capability fixes already live.
#define MICROPY_PY_IO_BUFFEREDWRITER      (MICROPY_CONFIG_ROM_LEVEL_AT_LEAST_EXTRA_FEATURES)

#define MICROPY_PY_SYS_MODULES            (MICROPY_CONFIG_ROM_LEVEL_AT_LEAST_EXTRA_FEATURES)
#define MICROPY_PY_SYS_EXIT               (MICROPY_CONFIG_ROM_LEVEL_AT_LEAST_EXTRA_FEATURES)
#define MICROPY_PY_SYS_PATH               (MICROPY_CONFIG_ROM_LEVEL_AT_LEAST_EXTRA_FEATURES)
#define MICROPY_PY_SYS_ARGV               (MICROPY_CONFIG_ROM_LEVEL_AT_LEAST_EXTRA_FEATURES)

#define MICROPY_NLR_SETJMP                (1)

#define MICROPY_STACK_CHECK               (1)
#ifndef MICROPY_STACK_CHECK_MARGIN
#define MICROPY_STACK_CHECK_MARGIN        (4096)
#endif

// Detailed text is part of the EXTRA test contract; retain TERSE for the minimum profile.
#ifndef MICROPY_ERROR_REPORTING
#define MICROPY_ERROR_REPORTING           (MICROPY_CONFIG_ROM_LEVEL_AT_LEAST_EXTRA_FEATURES \
    ? MICROPY_ERROR_REPORTING_DETAILED : MICROPY_ERROR_REPORTING_TERSE)
#endif

#define MICROPY_ALLOC_PARSE_CHUNK_INIT    (16)

// declared by py/mphal.h; the macro expands only inside py/mpprint.c, which includes it
#define MP_PLAT_PRINT_STRN(str, len) mp_hal_stdout_tx_strn((str), (len))

typedef long mp_off_t;
#include <alloca.h>

#define MICROPY_HW_BOARD_NAME "capstone-domain"
#define MICROPY_HW_MCU_NAME "cva6-capstone"
#define MP_STATE_PORT MP_STATE_VM
