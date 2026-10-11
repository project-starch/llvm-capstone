#ifndef VIRTUAL_RUNTIME_R0_WIRE_H
#define VIRTUAL_RUNTIME_R0_WIRE_H
#include <linux/types.h>
#include <linux/ioctl.h>
/* Experimental one-thread, resident-memory gate, not a general driver ABI. */
struct r0_request {
    __u64 code, data, stack, tls;
    __u64 code_bytes, data_bytes, data_perms, code_perms;
    __u64 c_entry;
    __u64 kind, cause, pc, address, result, preemptions, scattered;
};
#define R0_RUN _IOWR('V', 0, struct r0_request)
#endif
