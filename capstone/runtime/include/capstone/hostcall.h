#ifndef CAPSTONE_HOSTCALL_H
#define CAPSTONE_HOSTCALL_H

/* Shared HostCall v0 wire ABI. Snapshot mutable requests before servicing them. */
#define HC_V0_REGION_SIZE 4096UL

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
#define HC_V0_OP_PATH_DELETE 24ULL
/* Time. A domain has no clock of its own: rdtime is a counter whose frequency
 * lives in a device tree the domain cannot read, so wall-clock and monotonic
 * time both come from the helper. Request: clock_id at payload offset 0.
 * Response: seconds and nanoseconds at payload offset 0, length 16. First
 * consumer is musl's clock_gettime, and behind it mkstemp's __randname. */
#define HC_V0_OP_CLOCK_GETTIME 25ULL
/* Directories. Request: handle (from FILE_OPEN of a directory) and the listing
 * position, at payload offset 0. Response: linux_dirent64 records in the payload
 * from metadata.offset, result = their byte count, 0 at the end. The position is
 * a directory cookie (a record's d_off), not a byte offset; the domain keeps it.
 * First consumer is musl's readdir, and behind it CPython's os.listdir and
 * import's directory cache. */
#define HC_V0_OP_DIR_READ 26ULL
/* Rename. Request: PATH_ACCESS's layout with two paths behind the flags word,
 * "old NUL new", metadata.length covering both and the NUL. Response: result 0.
 * The helper's rename(2); a flag (renameat2's) is refused on the domain side.
 * First consumer is PostgreSQL's durable_rename. */
#define HC_V0_OP_PATH_RENAME 27ULL
/* Make a directory. Request: PATH_ACCESS's layout, the mode in the flags word.
 * Response: result 0. The helper's mkdir(2). The matching removal is
 * PATH_DELETE with HC_PATH_DELETE_FLAG_DIRECTORY, the helper's rmdir(2).
 * First consumer is PostgreSQL's CREATE DATABASE (base/<oid>). */
#define HC_V0_OP_PATH_MKDIR 28ULL
/* Read a symbolic link. Request: PATH_ACCESS's layout, flags 0. Response: the
 * link's target at payload offset 0, result = its length, no terminator, as
 * readlink(2). A name that is not a link answers EINVAL, which is what musl's
 * realpath() asks each path component and expects for the common case. */
#define HC_V0_OP_PATH_READLINK 29ULL

#define HC_V0_RET_DONE 0UL
#define HC_V0_RET_PENDING 1UL
#define HC_V0_RET_ERROR 2UL

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

struct hc_path_delete_req_v0 {
  hostcall_u64_t flags;
  char path[];
};

#define HC_FILE_OPEN_REQ_V0_PATH_OFFSET 16ULL
#define HC_FILE_READ_REQ_V0_DATA_OFFSET 32ULL
#define HC_FILE_WRITE_REQ_V0_DATA_OFFSET 32ULL
#define HC_FILE_STAT_BASIC_RESP_V0_SIZE 32ULL
struct hc_clock_gettime_req_v0 {
  hostcall_u64_t clock_id;
};

struct hc_clock_gettime_resp_v0 {
  hostcall_s64_t sec;
  hostcall_s64_t nsec;
};

#define HC_CLOCK_GETTIME_RESP_V0_SIZE 16ULL
#define HC_PATH_ACCESS_REQ_V0_PATH_OFFSET 8ULL
struct hc_dir_read_req_v0 {
  hostcall_u64_t handle;
  hostcall_u64_t cookie;
};
#define HC_DIR_READ_REQ_V0_DATA_OFFSET 16ULL
#define HC_PATH_DELETE_REQ_V0_PATH_OFFSET 8ULL
#define HC_PATH_RENAME_REQ_V0_PATH_OFFSET 8ULL
#define HC_PATH_MKDIR_REQ_V0_PATH_OFFSET 8ULL
#define HC_PATH_READLINK_REQ_V0_PATH_OFFSET 8ULL

#define HC_PATH_ACCESS_FLAG_EXISTS 0ULL
#define HC_PATH_DELETE_FLAG_NONE 0ULL
#define HC_PATH_DELETE_FLAG_DIRECTORY 1ULL /* rmdir, not unlink */

#endif
