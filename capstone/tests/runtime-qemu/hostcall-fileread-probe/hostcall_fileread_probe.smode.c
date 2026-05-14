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

/* Metadata is shared from the start; the response payload is shared later. */
static struct hostcall_v0 *metadata;
static char *payload;
static char stack[4096];

/* Minimal SBI ecall helper used from the custom .smode payload. */
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

/* Return control to the helper with a small scalar status code. */
static void dom_return(unsigned long value) {
  struct sbiret ignored =
	  sbi_ecall(SBI_EXT_CAPSTONE, SBI_EXT_CAPSTONE_DOM_RETURN, value, 0, 0, 0,
				0, 0);
  (void)ignored;
}

static int compare_bytes(const char *lhs, const char *rhs, unsigned long len) {
  while (len > 0) {
	if (*lhs != *rhs)
	  return 0;
	lhs++;
	rhs++;
	len--;
  }
  return 1;
}

static void init_metadata_region(void) {
  /* Only metadata is shared before the first round. */
  region_id_t metadata_region_id = region_count() - 1;
  metadata = (struct hostcall_v0 *)region_base(metadata_region_id);
}

static void refresh_payload_region(void) {
  /* The helper shares the response payload immediately before round 2. */
  region_id_t payload_region_id = region_count() - 1;
  payload = (char *)region_base(payload_region_id);
}

static void start_impl(void) {
  init_metadata_region();

  /*
   * Round 1: the domain asks for a tiny read-like host service. There is no
   * request payload; only the fixed-width metadata request is populated here.
   */
  metadata->phase = HC_V0_PHASE_REQ;
  metadata->opcode = HC_V0_OP_READ_GUEST_TMPFILE;
  metadata->offset = 0;
  metadata->length = HOSTCALL_FILEREAD_PROBE_MESSAGE_LEN;
  metadata->result = 0;
  metadata->error = 0;

  dom_return(HC_V0_RET_PENDING);

  /* Round 2: the helper should have published a borrowed response payload. */
  refresh_payload_region();
  if (metadata->phase != HC_V0_PHASE_RESP || metadata->error != 0 ||
	  metadata->result != HOSTCALL_FILEREAD_PROBE_MESSAGE_LEN) {
	metadata->phase = HC_V0_PHASE_ERROR;
	dom_return(HC_V0_RET_ERROR);
  }

  if (!compare_bytes(payload, HOSTCALL_FILEREAD_PROBE_MESSAGE,
					 HOSTCALL_FILEREAD_PROBE_MESSAGE_LEN)) {
	metadata->phase = HC_V0_PHASE_ERROR;
	dom_return(HC_V0_RET_ERROR);
  }

  metadata->phase = HC_V0_PHASE_DONE;
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

