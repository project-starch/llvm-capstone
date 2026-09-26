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

#include "../../../runtime/include/capstone/hostcall.h"

#define HOSTCALL_STDOUT_PROBE_REGION_SIZE HC_V0_REGION_SIZE
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

#define HOSTCALL_PATH_DELETE_PROBE_PATH \
  "/tmp/hostcall_v0_path_delete_target.txt"
#define HOSTCALL_PATH_DELETE_PROBE_PATH_LEN \
  (sizeof(HOSTCALL_PATH_DELETE_PROBE_PATH) - 1)
#define HOSTCALL_PATH_DELETE_PROBE_MESSAGE \
  "hostcall-v0 path delete payload"
#define HOSTCALL_PATH_DELETE_PROBE_MESSAGE_LEN \
  (sizeof(HOSTCALL_PATH_DELETE_PROBE_MESSAGE) - 1)

#define HOSTCALL_COMBINED_FILE_OBJECT_PROBE_PATH \
  "/tmp/hostcall_v0_combined_file_object.txt"
#define HOSTCALL_COMBINED_FILE_OBJECT_PROBE_PATH_LEN \
  (sizeof(HOSTCALL_COMBINED_FILE_OBJECT_PROBE_PATH) - 1)
#define HOSTCALL_COMBINED_FILE_OBJECT_PROBE_MESSAGE \
  "hostcall-v0 combined file object payload"
#define HOSTCALL_COMBINED_FILE_OBJECT_PROBE_MESSAGE_LEN \
  (sizeof(HOSTCALL_COMBINED_FILE_OBJECT_PROBE_MESSAGE) - 1)

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

#define HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_DEFAULT 0x0UL
#define HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED 0x1UL
#define HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED 0x2UL

#endif
