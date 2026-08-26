/* No syscalls in a domain, so errno is a plain global nothing sets but WAMR's
   WASI-off paths still reference. */
#ifndef CAPSTONE_WAMR_ERRNO_H
#define CAPSTONE_WAMR_ERRNO_H
extern int errno;
#define EINVAL 22
#define ENOMEM 12
#define ENOSYS 38
#define EAGAIN 11
#define EBADF  9
#define EOVERFLOW 75
#endif
