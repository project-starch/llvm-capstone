#ifndef CAPSTONE_TESTS_RUNTIME_QEMU_HOSTCALL_STDOUT_PROBE_H
#define CAPSTONE_TESTS_RUNTIME_QEMU_HOSTCALL_STDOUT_PROBE_H

/*
 * This file freezes the tiny HostCall v0 ABI used by the current runtime probe
 * family. The host helper(s) and the custom .smode payload(s) both include this
 * header so they agree on field widths, status values, and the fixed proof
 * payloads.
 *
 * Important shared-metadata rule: `metadata` stays live as `INOUT + SHARED`, so
 * helpers should snapshot the request fields they intend to trust immediately
 * after a `call_dom()` return and before performing host-side work. That keeps
 * each service round from depending on repeated reads of mutable shared state.
 */

#define HOSTCALL_STDOUT_PROBE_REGION_SIZE 4096UL
#define HOSTCALL_STDOUT_PROBE_MESSAGE "hostcall-v0 payload from domain\n"
#define HOSTCALL_STDOUT_PROBE_MESSAGE_LEN \
  (sizeof(HOSTCALL_STDOUT_PROBE_MESSAGE) - 1)

#define HOSTCALL_FILEWRITE_PROBE_MESSAGE "hostcall-v0 file payload"
#define HOSTCALL_FILEWRITE_PROBE_MESSAGE_LEN \
  (sizeof(HOSTCALL_FILEWRITE_PROBE_MESSAGE) - 1)
#define HOSTCALL_FILEWRITE_PROBE_OUTPUT_PATH "/tmp/hostcall_v0_filewrite.txt"

#define HOSTCALL_FILEREAD_PROBE_MESSAGE "hostcall-v0 input payload"
#define HOSTCALL_FILEREAD_PROBE_MESSAGE_LEN \
  (sizeof(HOSTCALL_FILEREAD_PROBE_MESSAGE) - 1)
#define HOSTCALL_FILEREAD_PROBE_INPUT_PATH "/tmp/hostcall_v0_read_source.txt"

#define HOSTCALL_FILE_OPEN_CLOSE_PROBE_INPUT_PATH \
  "/tmp/hostcall_v0_open_close_source.txt"
#define HOSTCALL_FILE_OPEN_CLOSE_PROBE_INPUT_PATH_LEN \
  (sizeof(HOSTCALL_FILE_OPEN_CLOSE_PROBE_INPUT_PATH) - 1)
#define HOSTCALL_FILE_OPEN_CLOSE_PROBE_MESSAGE \
  "hostcall-v0 open-close source"
#define HOSTCALL_FILE_OPEN_CLOSE_PROBE_MESSAGE_LEN \
  (sizeof(HOSTCALL_FILE_OPEN_CLOSE_PROBE_MESSAGE) - 1)

#define HOSTCALL_FILE_HANDLE_WRITE_PROBE_OUTPUT_PATH \
  "/tmp/hostcall_v0_handle_write.txt"
#define HOSTCALL_FILE_HANDLE_WRITE_PROBE_OUTPUT_PATH_LEN \
  (sizeof(HOSTCALL_FILE_HANDLE_WRITE_PROBE_OUTPUT_PATH) - 1)
#define HOSTCALL_FILE_HANDLE_WRITE_PROBE_MESSAGE \
  "hostcall-v0 handle write payload"
#define HOSTCALL_FILE_HANDLE_WRITE_PROBE_MESSAGE_LEN \
  (sizeof(HOSTCALL_FILE_HANDLE_WRITE_PROBE_MESSAGE) - 1)

#define HOSTCALL_FILE_HANDLE_READ_PROBE_INPUT_PATH \
  "/tmp/hostcall_v0_handle_read.txt"
#define HOSTCALL_FILE_HANDLE_READ_PROBE_INPUT_PATH_LEN \
  (sizeof(HOSTCALL_FILE_HANDLE_READ_PROBE_INPUT_PATH) - 1)
#define HOSTCALL_FILE_HANDLE_READ_PROBE_MESSAGE \
  "hostcall-v0 handle read payload"
#define HOSTCALL_FILE_HANDLE_READ_PROBE_MESSAGE_LEN \
  (sizeof(HOSTCALL_FILE_HANDLE_READ_PROBE_MESSAGE) - 1)

#define HOSTCALL_FILE_HANDLE_STAT_PROBE_INPUT_PATH \
  "/tmp/hostcall_v0_handle_stat.txt"
#define HOSTCALL_FILE_HANDLE_STAT_PROBE_INPUT_PATH_LEN \
  (sizeof(HOSTCALL_FILE_HANDLE_STAT_PROBE_INPUT_PATH) - 1)
#define HOSTCALL_FILE_HANDLE_STAT_PROBE_MESSAGE \
  "hostcall-v0 handle stat payload"
#define HOSTCALL_FILE_HANDLE_STAT_PROBE_MESSAGE_LEN \
  (sizeof(HOSTCALL_FILE_HANDLE_STAT_PROBE_MESSAGE) - 1)

#define HOSTCALL_FILE_HANDLE_TRUNCATE_PROBE_INPUT_PATH \
  "/tmp/hostcall_v0_handle_truncate.txt"
#define HOSTCALL_FILE_HANDLE_TRUNCATE_PROBE_INPUT_PATH_LEN \
  (sizeof(HOSTCALL_FILE_HANDLE_TRUNCATE_PROBE_INPUT_PATH) - 1)
#define HOSTCALL_FILE_HANDLE_TRUNCATE_PROBE_INITIAL_MESSAGE \
  "hostcall-v0 handle truncate payload"
#define HOSTCALL_FILE_HANDLE_TRUNCATE_PROBE_INITIAL_MESSAGE_LEN \
  (sizeof(HOSTCALL_FILE_HANDLE_TRUNCATE_PROBE_INITIAL_MESSAGE) - 1)
#define HOSTCALL_FILE_HANDLE_TRUNCATE_PROBE_TARGET_SIZE 9ULL

#define HOSTCALL_FILE_HANDLE_SYNC_PROBE_OUTPUT_PATH \
  "/tmp/hostcall_v0_handle_sync.txt"
#define HOSTCALL_FILE_HANDLE_SYNC_PROBE_OUTPUT_PATH_LEN \
  (sizeof(HOSTCALL_FILE_HANDLE_SYNC_PROBE_OUTPUT_PATH) - 1)
#define HOSTCALL_FILE_HANDLE_SYNC_PROBE_MESSAGE \
  "hostcall-v0 handle sync payload"
#define HOSTCALL_FILE_HANDLE_SYNC_PROBE_MESSAGE_LEN \
  (sizeof(HOSTCALL_FILE_HANDLE_SYNC_PROBE_MESSAGE) - 1)

#define HOSTCALL_PATH_ACCESS_PROBE_EXISTING_PATH \
  "/tmp/hostcall_v0_path_access_present.txt"
#define HOSTCALL_PATH_ACCESS_PROBE_EXISTING_PATH_LEN \
  (sizeof(HOSTCALL_PATH_ACCESS_PROBE_EXISTING_PATH) - 1)
#define HOSTCALL_PATH_ACCESS_PROBE_MISSING_PATH \
  "/tmp/hostcall_v0_path_access_missing.txt"
#define HOSTCALL_PATH_ACCESS_PROBE_MISSING_PATH_LEN \
  (sizeof(HOSTCALL_PATH_ACCESS_PROBE_MISSING_PATH) - 1)
#define HOSTCALL_PATH_ACCESS_PROBE_MESSAGE \
  "hostcall-v0 path access payload"
#define HOSTCALL_PATH_ACCESS_PROBE_MESSAGE_LEN \
  (sizeof(HOSTCALL_PATH_ACCESS_PROBE_MESSAGE) - 1)

#define HOSTCALL_COMBINED_FILE_OBJECT_PROBE_PATH \
  "/tmp/hostcall_v0_combined_file_object.txt"
#define HOSTCALL_COMBINED_FILE_OBJECT_PROBE_PATH_LEN \
  (sizeof(HOSTCALL_COMBINED_FILE_OBJECT_PROBE_PATH) - 1)
#define HOSTCALL_COMBINED_FILE_OBJECT_PROBE_MESSAGE \
  "hostcall-v0 combined file object payload"
#define HOSTCALL_COMBINED_FILE_OBJECT_PROBE_MESSAGE_LEN \
  (sizeof(HOSTCALL_COMBINED_FILE_OBJECT_PROBE_MESSAGE) - 1)

typedef unsigned long long hostcall_u64_t;
typedef long long hostcall_s64_t;

struct hostcall_v0 {
  /* Shared state machine: INIT -> REQ -> RESP -> DONE / ERROR. */
  hostcall_u64_t phase;
  /* Which host service is being requested. */
  hostcall_u64_t opcode;
  /* Byte range inside the payload region that the host should consume. */
  hostcall_u64_t offset;
  hostcall_u64_t length;
  /* Host-written service result and errno-like code. */
  hostcall_s64_t result;
  hostcall_s64_t error;
};

#define HC_V0_PHASE_INIT 0ULL
#define HC_V0_PHASE_REQ 1ULL
#define HC_V0_PHASE_RESP 2ULL
#define HC_V0_PHASE_DONE 3ULL
#define HC_V0_PHASE_ERROR 4ULL

#define HC_V0_OP_NONE 0ULL
#define HC_V0_OP_WRITE_STDOUT 1ULL
#define HC_V0_OP_WRITE_GUEST_TMPFILE 2ULL
#define HC_V0_OP_READ_GUEST_TMPFILE 3ULL
/* Diagnostic-only opcodes used by the minimal second-PENDING probe. */
#define HC_V0_OP_SECOND_PENDING_STAGE1 4ULL
#define HC_V0_OP_SECOND_PENDING_STAGE2 5ULL
/* Diagnostic-only opcodes used by the second-PENDING payload-reuse probe. */
#define HC_V0_OP_SECOND_PENDING_PAYLOAD_STAGE1 6ULL
#define HC_V0_OP_SECOND_PENDING_PAYLOAD_STAGE2 7ULL

/* First practical file-service opcodes. */
#define HC_V0_OP_FILE_OPEN 16ULL
#define HC_V0_OP_FILE_READ 17ULL
#define HC_V0_OP_FILE_WRITE 18ULL
#define HC_V0_OP_FILE_CLOSE 19ULL
#define HC_V0_OP_FILE_STAT_BASIC 20ULL
#define HC_V0_OP_FILE_SYNC 21ULL
#define HC_V0_OP_FILE_TRUNCATE 22ULL
/* First SQLite-facing path-service opcode. */
#define HC_V0_OP_PATH_ACCESS 23ULL

#define HC_V0_RET_DONE 0UL
#define HC_V0_RET_PENDING 1UL
#define HC_V0_RET_ERROR 2UL

/*
 * Local copies of the current runtime annotation values used by this probe.
 *
 * Why they live here instead of including the OpenSBI internal header directly:
 * - the userspace probe and the custom .smode payload both need one small shared
 *   header with no broader OpenSBI build-time dependencies,
 * - the public modcapstone userspace header exports the ioctl fields but does not
 *   currently export these symbolic names,
 * - the authoritative numeric values currently match
 *   components/opensbi/lib/sbi/capstone-sbi/sbi_capstone.h.
 */
#define HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_IN 0x0UL
#define HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT 0x1UL
#define HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT 0x2UL

#define HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED 0x1UL
#define HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED 0x2UL

struct hc_file_open_req_v0 {
  hostcall_u64_t flags;
  hostcall_u64_t mode;
  char path[];
};

struct hc_file_read_req_v0 {
  hostcall_u64_t handle;
  hostcall_u64_t file_offset;
  hostcall_u64_t flags;
  hostcall_u64_t reserved0;
  unsigned char data[];
};

struct hc_file_write_req_v0 {
  hostcall_u64_t handle;
  hostcall_u64_t file_offset;
  hostcall_u64_t flags;
  hostcall_u64_t reserved0;
  unsigned char data[];
};

struct hc_file_close_req_v0 {
  hostcall_u64_t handle;
};

struct hc_file_sync_req_v0 {
  hostcall_u64_t handle;
  hostcall_u64_t flags;
};

struct hc_file_stat_basic_req_v0 {
  hostcall_u64_t handle;
  hostcall_u64_t flags;
};

struct hc_file_stat_basic_resp_v0 {
  hostcall_u64_t file_size;
  hostcall_u64_t mode;
  hostcall_u64_t reserved0;
  hostcall_u64_t reserved1;
};

struct hc_file_truncate_req_v0 {
  hostcall_u64_t handle;
  hostcall_u64_t size;
  hostcall_u64_t flags;
  hostcall_u64_t reserved0;
};

struct hc_path_access_req_v0 {
  hostcall_u64_t flags;
  char path[];
};

#define HC_FILE_OPEN_REQ_V0_PATH_OFFSET 16ULL
#define HC_FILE_READ_REQ_V0_DATA_OFFSET 32ULL
#define HC_FILE_WRITE_REQ_V0_DATA_OFFSET 32ULL
#define HC_FILE_STAT_BASIC_RESP_V0_SIZE 32ULL
#define HC_PATH_ACCESS_REQ_V0_PATH_OFFSET 8ULL

#define HC_PATH_ACCESS_FLAG_EXISTS 0ULL

#endif


