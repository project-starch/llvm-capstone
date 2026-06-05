#ifndef COREMARK_HOSTCALL_H
#define COREMARK_HOSTCALL_H

typedef unsigned long long hostcall_u64_t;
typedef long long          hostcall_s64_t;

struct hostcall_v0 {
  hostcall_u64_t phase;
  hostcall_u64_t opcode;
  hostcall_u64_t offset;
  hostcall_u64_t length;
  hostcall_s64_t result;
  hostcall_s64_t error;
};

#define HC_V0_PHASE_REQ   1ULL
#define HC_V0_PHASE_RESP  2ULL
#define HC_V0_PHASE_DONE  3ULL

#define HC_V0_OP_WRITE_STDOUT 1ULL

#define HC_V0_RET_DONE    0UL
#define HC_V0_RET_PENDING 1UL

#define HC_REGION_SIZE    4096UL

/*
 * Annotation constants matching sbi_capstone.h and libcapstone.h.
 * PERM_INOUT = 0x1 (read+write), REV_SHARED = 0x2 (not revoked on return).
 */
#define HC_ANNOTATION_PERM_INOUT   0x1UL
#define HC_ANNOTATION_REV_SHARED   0x2UL

#endif
