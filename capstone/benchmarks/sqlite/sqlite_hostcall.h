#ifndef CAPSTONE_SQLITE_HOSTCALL_H
#define CAPSTONE_SQLITE_HOSTCALL_H

typedef unsigned long long sqlite_hostcall_u64_t;
typedef long long sqlite_hostcall_s64_t;

struct sqlite_hostcall_v0 {
  sqlite_hostcall_u64_t phase;
  sqlite_hostcall_u64_t opcode;
  sqlite_hostcall_u64_t offset;
  sqlite_hostcall_u64_t length;
  sqlite_hostcall_s64_t result;
  sqlite_hostcall_s64_t error;
};

#define SQLITE_HC_RET_DONE 0UL
#define SQLITE_HC_REGION_SIZE 4096UL

#define SQLITE_HC_ANNOTATION_PERM_INOUT 0x1UL
#define SQLITE_HC_ANNOTATION_REV_SHARED 0x2UL

#endif
