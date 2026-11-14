#ifndef CAPSTONE_LINUX_START_H
#define CAPSTONE_LINUX_START_H
#include <stddef.h>

/* Exec stack for the 16-byte pointer ABI. argc occupies the low 8 bytes of
 * the first 16-byte cell. argv and envp follow as tagged pointer arrays,
 * each terminated by a full zero cell. auxv follows envp's terminator.
 * Scalar aux values use the low word of value. Pointer values retain tags.
 * AT_NULL terminates auxv. The kernel supplies a tagged sp and image gp. */
struct capstone_linux_auxv {
    size_t type;
    union { size_t scalar; void *pointer; } value;
};
_Static_assert(sizeof(struct capstone_linux_auxv) == 32, "exec auxv layout");
_Static_assert(offsetof(struct capstone_linux_auxv, value) == 16, "auxv value alignment");
#endif
