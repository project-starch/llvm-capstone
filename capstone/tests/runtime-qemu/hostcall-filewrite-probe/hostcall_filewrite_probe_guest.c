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
 * Dumping metadata in the serial log makes ABI mismatches much easier to diagnose.
 */
static int fail_cleanup(const char *message, unsigned long observed,
						struct hostcall_v0 *metadata) {
  fprintf(stderr, "hostcall-filewrite-probe: %s (observed=%lu)\n", message,
		  observed);
  if (metadata) {
	fprintf(stderr,
			"hostcall-filewrite-probe: metadata{phase=%llu opcode=%llu "
			"offset=%llu length=%llu result=%lld error=%lld}\n",
			metadata->phase, metadata->opcode, metadata->offset,
			metadata->length, metadata->result, metadata->error);
  }
  capstone_cleanup();
  return 1;
}

/*
 * Write the entire payload into the fixed guest-side proof file.
 * The hostcall result reports the total bytes committed on success.
 */
static ssize_t write_full_file(const char *path, const char *buffer, size_t len) {
  int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
  if (fd < 0)
	return -1;

  size_t written = 0;
  while (written < len) {
	ssize_t rc = write(fd, buffer + written, len - written);
	if (rc < 0) {
	  int saved_errno = errno;
	  close(fd);
	  errno = saved_errno;
	  return -1;
	}
	if (rc == 0) {
	  close(fd);
	  errno = EIO;
	  return -1;
	}
	written += (size_t)rc;
  }

  if (close(fd) < 0)
	return -1;

  return (ssize_t)written;
}

int main(int argc, char **argv) {
  if (argc != 2) {
	fprintf(stderr, "usage: %s <smode-domain-path>\n", argv[0]);
	return 2;
  }

  if (capstone_init()) {
	fprintf(stderr, "hostcall-filewrite-probe: failed to initialize Capstone\n");
	return 1;
  }

  /*
   * Reuse the existing /test-domains/sbi.dom substrate and swap in a custom
   * .smode payload that requests the second HostCall v0 opcode.
   */
  dom_id_t dom_id = create_dom("/test-domains/sbi.dom", argv[1]);
  if ((long)dom_id < 0)
	return fail_cleanup("create_dom failed", (unsigned long)dom_id, NULL);

  /*
   * Metadata stays shared because both sides mutate the state machine. The
   * payload stays on the same validated borrowed-out discipline used by the
   * tightened stdout proof.
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

  print_nobuf("hostcall-filewrite-probe: created domain ID = %lu\n", dom_id);
  print_nobuf("hostcall-filewrite-probe: metadata region ID = %lu\n",
			  metadata_region_id);
  print_nobuf("hostcall-filewrite-probe: payload region ID = %lu\n",
			  payload_region_id);

  shared_region_annotated(dom_id, metadata_region_id,
						  HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_INOUT,
						  HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_SHARED);
  shared_region_annotated(dom_id, payload_region_id,
						  HOSTCALL_STDOUT_PROBE_ANNOTATION_PERM_OUT,
						  HOSTCALL_STDOUT_PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf(
	  "hostcall-filewrite-probe: metadata shared, payload borrowed-out\n");

  /*
   * Round 1: the domain publishes a filewrite request using the same metadata ABI
   * and the same borrowed payload direction as the stdout proof.
   */
  unsigned long ret1 = call_dom(dom_id);
  print_nobuf("hostcall-filewrite-probe: first call retval = %lu\n", ret1);
  print_nobuf(
	  "hostcall-filewrite-probe: metadata phase after first call = %llu\n",
	  metadata->phase);
  if (ret1 != HC_V0_RET_PENDING)
	return fail_cleanup("unexpected first call retval", ret1, metadata);
  if (metadata->phase != HC_V0_PHASE_REQ)
	return fail_cleanup("unexpected phase after first call",
						(unsigned long)metadata->phase, metadata);
  if (metadata->opcode != HC_V0_OP_WRITE_GUEST_TMPFILE)
	return fail_cleanup("unexpected opcode after first call",
						(unsigned long)metadata->opcode, metadata);
  if (metadata->offset != 0)
	return fail_cleanup("unexpected payload offset",
						(unsigned long)metadata->offset, metadata);
  if (metadata->length != HOSTCALL_FILEWRITE_PROBE_MESSAGE_LEN)
	return fail_cleanup("unexpected payload length",
						(unsigned long)metadata->length, metadata);
  if (metadata->offset + metadata->length > HOSTCALL_STDOUT_PROBE_REGION_SIZE)
	return fail_cleanup("payload range exceeds shared region",
						(unsigned long)(metadata->offset + metadata->length),
						metadata);

  /*
   * Service the second coarse host operation: consume the borrowed payload and
   * write it into a fixed guest-side tmp file using ordinary Linux file I/O.
   */
  print_nobuf(
	  "hostcall-filewrite-probe: servicing HC_V0_OP_WRITE_GUEST_TMPFILE\n");
  ssize_t write_result = write_full_file(HOSTCALL_FILEWRITE_PROBE_OUTPUT_PATH,
										 payload + metadata->offset,
										 (size_t)metadata->length);
  if (write_result < 0) {
	metadata->result = -1;
	metadata->error = errno;
	metadata->phase = HC_V0_PHASE_ERROR;
	return fail_cleanup("write tmp file failed", (unsigned long)errno,
						metadata);
  }

  metadata->result = write_result;
  metadata->error = 0;
  metadata->phase = HC_V0_PHASE_RESP;
  print_nobuf("hostcall-filewrite-probe: wrote %s\n",
			  HOSTCALL_FILEWRITE_PROBE_OUTPUT_PATH);

  /* Round 2: the domain validates the metadata response and finishes. */
  unsigned long ret2 = call_dom(dom_id);
  print_nobuf("hostcall-filewrite-probe: second call retval = %lu\n", ret2);
  print_nobuf(
	  "hostcall-filewrite-probe: metadata phase after second call = %llu\n",
	  metadata->phase);
  if (ret2 != HC_V0_RET_DONE)
	return fail_cleanup("unexpected second call retval", ret2, metadata);
  if (metadata->phase != HC_V0_PHASE_DONE)
	return fail_cleanup("unexpected phase after second call",
						(unsigned long)metadata->phase, metadata);
  if (metadata->result != HOSTCALL_FILEWRITE_PROBE_MESSAGE_LEN)
	return fail_cleanup("unexpected result after second call",
						(unsigned long)metadata->result, metadata);
  if (metadata->error != 0)
	return fail_cleanup("unexpected error after second call",
						(unsigned long)metadata->error, metadata);

  print_nobuf("hostcall-filewrite-probe: success\n");

  if (capstone_cleanup()) {
	fprintf(stderr, "hostcall-filewrite-probe: failed to clean up Capstone\n");
	return 1;
  }

  return 0;
}

