#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "../hostcall-stdout-probe/hostcall_stdout_probe.h"

/* Print immediately so QEMU serial logs preserve the exact protocol sequence. */
#define print_nobuf(...)         \
  do {                           \
	printf(__VA_ARGS__);         \
	fflush(stdout);              \
  } while (0)

/*
 * Shared failure path used after Capstone initialization succeeds.
 * Dumping metadata in the serial log makes reverse-direction ABI mismatches easy
 * to diagnose.
 */
static int fail_cleanup(const char *message, unsigned long observed,
						struct hostcall_v0 *metadata) {
  fprintf(stderr, "hostcall-fileread-probe: %s (observed=%lu)\n", message,
		  observed);
  if (metadata) {
	fprintf(stderr,
			"hostcall-fileread-probe: metadata{phase=%llu opcode=%llu "
			"offset=%llu length=%llu result=%lld error=%lld}\n",
			metadata->phase, metadata->opcode, metadata->offset,
			metadata->length, metadata->result, metadata->error);
  }
  capstone_cleanup();
  return 1;
}

/*
 * Read exactly up to the requested number of bytes from the fixed guest-side input
 * file into the helper-mapped payload buffer.
 */
static ssize_t read_into_buffer(const char *path, char *buffer, size_t len) {
  int fd = open(path, O_RDONLY);
  if (fd < 0)
	return -1;

  ssize_t rc = read(fd, buffer, len);
  if (rc < 0) {
	int saved_errno = errno;
	close(fd);
	errno = saved_errno;
	return -1;
  }

  if (close(fd) < 0)
	return -1;

  return rc;
}

int main(int argc, char **argv) {
  if (argc != 2) {
	fprintf(stderr, "usage: %s <smode-domain-path>\n", argv[0]);
	return 2;
  }

  if (capstone_init()) {
	fprintf(stderr, "hostcall-fileread-probe: failed to initialize Capstone\n");
	return 1;
  }

  /*
   * Reuse the existing /test-domains/sbi.dom substrate and swap in a custom
   * .smode payload that requests the first read-like HostCall v0 opcode.
   */
  dom_id_t dom_id = create_dom("/test-domains/sbi.dom", argv[1]);
  if ((long)dom_id < 0)
	return fail_cleanup("create_dom failed", (unsigned long)dom_id, NULL);

  /*
   * Metadata is shared across both rounds. The payload is not shared yet because
   * this proof validates the opposite data-flow direction: the helper will fill it
   * only after the first return and then share it as borrowed input for round 2.
   */
  region_id_t metadata_region_id =
	  create_region(HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  region_id_t payload_region_id = create_region(HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  struct hostcall_v0 *metadata = (struct hostcall_v0 *)map_region(
	  metadata_region_id, HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  char *payload =
	  (char *)map_region(payload_region_id, HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  if (!metadata || !payload)
	return fail_cleanup("map_region failed", 0, metadata);

  memset(metadata, 0, HOSTCALL_STDOUT_PROBE_REGION_SIZE);
  memset(payload, 0, HOSTCALL_STDOUT_PROBE_REGION_SIZE);

  print_nobuf("hostcall-fileread-probe: created domain ID = %lu\n", dom_id);
  print_nobuf("hostcall-fileread-probe: metadata region ID = %lu\n",
			  metadata_region_id);
  print_nobuf("hostcall-fileread-probe: payload region ID = %lu\n",
			  payload_region_id);

  shared_region_annotated(dom_id, metadata_region_id,
						  HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
						  HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
  print_nobuf("hostcall-fileread-probe: metadata shared\n");

  /*
   * Round 1: the domain publishes a read-like request through metadata only.
   * There is no response payload yet.
   */
  unsigned long ret1 = call_dom(dom_id);
  print_nobuf("hostcall-fileread-probe: first call retval = %lu\n", ret1);
  print_nobuf(
	  "hostcall-fileread-probe: metadata phase after first call = %llu\n",
	  metadata->phase);
  if (ret1 != HC_V0_RET_PENDING)
	return fail_cleanup("unexpected first call retval", ret1, metadata);
  if (metadata->phase != HC_V0_PHASE_REQ)
	return fail_cleanup("unexpected phase after first call",
						(unsigned long)metadata->phase, metadata);
  if (metadata->opcode != HC_V0_OP_READ_GUEST_TMPFILE)
	return fail_cleanup("unexpected opcode after first call",
						(unsigned long)metadata->opcode, metadata);
  if (metadata->offset != 0)
	return fail_cleanup("unexpected response offset request",
						(unsigned long)metadata->offset, metadata);
  if (metadata->length != HOSTCALL_FILEREAD_PROBE_MESSAGE_LEN)
	return fail_cleanup("unexpected requested response length",
						(unsigned long)metadata->length, metadata);
  if (metadata->length > HOSTCALL_STDOUT_PROBE_REGION_SIZE)
	return fail_cleanup("requested response exceeds payload region",
						(unsigned long)metadata->length, metadata);

  /*
   * Service the read-like request: read bytes from the fixed guest-side input file
   * into the helper mapping, then share the payload as borrowed input for round 2.
   */
  print_nobuf(
	  "hostcall-fileread-probe: servicing HC_V0_OP_READ_GUEST_TMPFILE\n");
  ssize_t read_result = read_into_buffer(HOSTCALL_FILEREAD_PROBE_INPUT_PATH,
										 payload, (size_t)metadata->length);
  if (read_result < 0) {
	metadata->result = -1;
	metadata->error = errno;
	metadata->phase = HC_V0_PHASE_ERROR;
	return fail_cleanup("read tmp file failed", (unsigned long)errno, metadata);
  }

  shared_region_annotated(dom_id, payload_region_id,
						  HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_IN,
						  HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
	  "hostcall-fileread-probe: payload shared as borrowed-in response\n");

  metadata->result = read_result;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;

  /*
   * Round 2: the domain consumes the helper-produced payload and validates the
   * metadata response.
   */
  unsigned long ret2 = call_dom(dom_id);
  print_nobuf("hostcall-fileread-probe: second call retval = %lu\n", ret2);
  print_nobuf(
	  "hostcall-fileread-probe: metadata phase after second call = %llu\n",
	  metadata->phase);
  if (ret2 != HC_V0_RET_DONE)
	return fail_cleanup("unexpected second call retval", ret2, metadata);
  if (metadata->phase != HC_V0_PHASE_DONE)
	return fail_cleanup("unexpected phase after second call",
						(unsigned long)metadata->phase, metadata);
  if (metadata->result != HOSTCALL_FILEREAD_PROBE_MESSAGE_LEN)
	return fail_cleanup("unexpected result after second call",
						(unsigned long)metadata->result, metadata);
  if (metadata->error != 0)
	return fail_cleanup("unexpected error after second call",
						(unsigned long)metadata->error, metadata);

  print_nobuf("hostcall-fileread-probe: success\n");

  if (capstone_cleanup()) {
	fprintf(stderr, "hostcall-fileread-probe: failed to clean up Capstone\n");
	return 1;
  }

  return 0;
}

