#include <fcntl.h>

#include "../hostcall-stdout-probe/hostcall_stdout_probe.h"

#define SBI_EXT_CAPSTONE 0x12345678
#define SBI_EXT_CAPSTONE_DOM_RETURN 0x5
#define SBI_EXT_CAPSTONE_REGION_QUERY 0x6
#define SBI_EXT_CAPSTONE_REGION_COUNT 0x8

#define CAPSTONE_REGION_FIELD_BASE 0x0

typedef unsigned long uintptr_t;
typedef unsigned long region_id_t;

struct sbiret {
  long error;
  long value;
};

static struct hostcall_v0 *metadata;
static char *payload;
static char read_snapshot[HOSTCALL_STDOUT_PROBE_REGION_SIZE];
static char stack[4096];

static struct sbiret sbi_ecall(int ext, int fid, unsigned long arg0,
                               unsigned long arg1, unsigned long arg2,
                               unsigned long arg3, unsigned long arg4,
                               unsigned long arg5) {
  struct sbiret ret;

  register uintptr_t a0 asm("a0") = (uintptr_t)(arg0);
  register uintptr_t a1 asm("a1") = (uintptr_t)(arg1);
  register uintptr_t a2 asm("a2") = (uintptr_t)(arg2);
  register uintptr_t a3 asm("a3") = (uintptr_t)(arg3);
  register uintptr_t a4 asm("a4") = (uintptr_t)(arg4);
  register uintptr_t a5 asm("a5") = (uintptr_t)(arg5);
  register uintptr_t a6 asm("a6") = (uintptr_t)(fid);
  register uintptr_t a7 asm("a7") = (uintptr_t)(ext);
  asm volatile("ecall"
               : "+r"(a0), "+r"(a1)
               : "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7)
               : "memory");
  ret.error = a0;
  ret.value = a1;
  return ret;
}

static region_id_t region_count(void) {
  return (region_id_t)sbi_ecall(SBI_EXT_CAPSTONE, SBI_EXT_CAPSTONE_REGION_COUNT,
                                0, 0, 0, 0, 0, 0)
      .value;
}

static void *region_base(region_id_t region_id) {
  return (void *)sbi_ecall(SBI_EXT_CAPSTONE, SBI_EXT_CAPSTONE_REGION_QUERY,
                           region_id, CAPSTONE_REGION_FIELD_BASE, 0, 0, 0, 0)
      .value;
}

static void dom_return(unsigned long value) {
  struct sbiret ignored =
      sbi_ecall(SBI_EXT_CAPSTONE, SBI_EXT_CAPSTONE_DOM_RETURN, value, 0, 0, 0,
                0, 0);
  (void)ignored;
}

static void copy_bytes(char *dest, const char *src, unsigned long len) {
  while (len > 0) {
    *dest = *src;
    dest++;
    src++;
    len--;
  }
}

static int bytes_match(const char *lhs, const char *rhs, unsigned long len) {
  while (len > 0) {
    if (*lhs != *rhs)
      return 0;
    lhs++;
    rhs++;
    len--;
  }
  return 1;
}

static void init_regions(void) {
  region_id_t region_n = region_count();
  region_id_t metadata_region_id = region_n - 2;
  region_id_t payload_region_id = region_n - 1;

  metadata = (struct hostcall_v0 *)region_base(metadata_region_id);
  payload = (char *)region_base(payload_region_id);
}

static void start_impl(void) {
  struct hc_file_open_req_v0 *open_req;
  struct hc_file_write_req_v0 *write_req;
  struct hc_file_close_req_v0 *close_req;
  struct hc_file_read_req_v0 *read_req;
  hostcall_u64_t write_handle_token;
  hostcall_u64_t read_handle_token;
  hostcall_u64_t i;

  init_regions();

  open_req = (struct hc_file_open_req_v0 *)payload;
  open_req->flags = O_CREAT | O_TRUNC | O_WRONLY;
  open_req->mode = 0644;
  copy_bytes(open_req->path, HOSTCALL_COMBINED_FILE_OBJECT_PROBE_PATH,
             HOSTCALL_COMBINED_FILE_OBJECT_PROBE_PATH_LEN);
  metadata->phase = HC_V0_PHASE_REQ;
  metadata->opcode = HC_V0_OP_FILE_OPEN;
  metadata->offset = HC_FILE_OPEN_REQ_V0_PATH_OFFSET;
  metadata->length = HOSTCALL_COMBINED_FILE_OBJECT_PROBE_PATH_LEN;
  metadata->result = 0;
  metadata->error = 0;
  dom_return(HC_V0_RET_PENDING);

  if (metadata->phase != HC_V0_PHASE_RESP || metadata->error != 0 ||
      metadata->result <= 0) {
    metadata->phase = HC_V0_PHASE_ERROR;
    dom_return(HC_V0_RET_ERROR);
  }

  write_handle_token = (hostcall_u64_t)metadata->result;
  write_req = (struct hc_file_write_req_v0 *)payload;
  write_req->handle = write_handle_token;
  write_req->file_offset = 0;
  write_req->flags = 0;
  write_req->reserved0 = 0;
  copy_bytes((char *)write_req->data, HOSTCALL_COMBINED_FILE_OBJECT_PROBE_MESSAGE,
             HOSTCALL_COMBINED_FILE_OBJECT_PROBE_MESSAGE_LEN);
  metadata->phase = HC_V0_PHASE_REQ;
  metadata->opcode = HC_V0_OP_FILE_WRITE;
  metadata->offset = HC_FILE_WRITE_REQ_V0_DATA_OFFSET;
  metadata->length = HOSTCALL_COMBINED_FILE_OBJECT_PROBE_MESSAGE_LEN;
  metadata->result = 0;
  metadata->error = 0;
  dom_return(HC_V0_RET_PENDING);

  if (metadata->phase != HC_V0_PHASE_RESP || metadata->error != 0 ||
      metadata->result != HOSTCALL_COMBINED_FILE_OBJECT_PROBE_MESSAGE_LEN) {
    metadata->phase = HC_V0_PHASE_ERROR;
    dom_return(HC_V0_RET_ERROR);
  }

  close_req = (struct hc_file_close_req_v0 *)payload;
  close_req->handle = write_handle_token;
  metadata->phase = HC_V0_PHASE_REQ;
  metadata->opcode = HC_V0_OP_FILE_CLOSE;
  metadata->offset = 0;
  metadata->length = 0;
  metadata->result = 0;
  metadata->error = 0;
  dom_return(HC_V0_RET_PENDING);

  if (metadata->phase != HC_V0_PHASE_RESP || metadata->error != 0 ||
      metadata->result != 0) {
    metadata->phase = HC_V0_PHASE_ERROR;
    dom_return(HC_V0_RET_ERROR);
  }

  open_req = (struct hc_file_open_req_v0 *)payload;
  open_req->flags = O_RDONLY;
  open_req->mode = 0;
  copy_bytes(open_req->path, HOSTCALL_COMBINED_FILE_OBJECT_PROBE_PATH,
             HOSTCALL_COMBINED_FILE_OBJECT_PROBE_PATH_LEN);
  metadata->phase = HC_V0_PHASE_REQ;
  metadata->opcode = HC_V0_OP_FILE_OPEN;
  metadata->offset = HC_FILE_OPEN_REQ_V0_PATH_OFFSET;
  metadata->length = HOSTCALL_COMBINED_FILE_OBJECT_PROBE_PATH_LEN;
  metadata->result = 0;
  metadata->error = 0;
  dom_return(HC_V0_RET_PENDING);

  if (metadata->phase != HC_V0_PHASE_RESP || metadata->error != 0 ||
      metadata->result <= 0) {
    metadata->phase = HC_V0_PHASE_ERROR;
    dom_return(HC_V0_RET_ERROR);
  }

  read_handle_token = (hostcall_u64_t)metadata->result;
  read_req = (struct hc_file_read_req_v0 *)payload;
  read_req->handle = read_handle_token;
  read_req->file_offset = 0;
  read_req->flags = 0;
  read_req->reserved0 = 0;
  metadata->phase = HC_V0_PHASE_REQ;
  metadata->opcode = HC_V0_OP_FILE_READ;
  metadata->offset = HC_FILE_READ_REQ_V0_DATA_OFFSET;
  metadata->length = HOSTCALL_COMBINED_FILE_OBJECT_PROBE_MESSAGE_LEN;
  metadata->result = 0;
  metadata->error = 0;
  dom_return(HC_V0_RET_PENDING);

  if (metadata->phase != HC_V0_PHASE_RESP || metadata->error != 0 ||
      metadata->result != HOSTCALL_COMBINED_FILE_OBJECT_PROBE_MESSAGE_LEN) {
    metadata->phase = HC_V0_PHASE_ERROR;
    dom_return(HC_V0_RET_ERROR);
  }

  for (i = 0; i < metadata->result; ++i)
    read_snapshot[i] = payload[metadata->offset + i];
  if (!bytes_match(read_snapshot, HOSTCALL_COMBINED_FILE_OBJECT_PROBE_MESSAGE,
                   HOSTCALL_COMBINED_FILE_OBJECT_PROBE_MESSAGE_LEN)) {
    metadata->phase = HC_V0_PHASE_ERROR;
    dom_return(HC_V0_RET_ERROR);
  }

  close_req = (struct hc_file_close_req_v0 *)payload;
  close_req->handle = read_handle_token;
  metadata->phase = HC_V0_PHASE_REQ;
  metadata->opcode = HC_V0_OP_FILE_CLOSE;
  metadata->offset = 0;
  metadata->length = 0;
  metadata->result = 0;
  metadata->error = 0;
  dom_return(HC_V0_RET_PENDING);

  if (metadata->phase != HC_V0_PHASE_RESP || metadata->error != 0 ||
      metadata->result != 0) {
    metadata->phase = HC_V0_PHASE_ERROR;
    dom_return(HC_V0_RET_ERROR);
  }

  metadata->phase = HC_V0_PHASE_DONE;
  metadata->result = 0;
  metadata->error = 0;
  dom_return(HC_V0_RET_DONE);

  metadata->phase = HC_V0_PHASE_ERROR;
  dom_return(HC_V0_RET_ERROR);

  while (1) {
  }
}

__attribute__((naked)) void _start(void) {
  __asm__ volatile("mv sp, %0\n"
                   "j start_impl\n"
                   :
                   : "r"(stack + sizeof(stack))
                   : "memory");
}

