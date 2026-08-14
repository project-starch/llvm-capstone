/* Real file I/O from a pure-capability domain, through ordinary POSIX.
 *
 * open/write/close/open/read/close on a guest file, with the bytes compared on
 * return. Nothing here is Capstone-aware: the calls are musl's, they reach
 * __capstone_hostcall, and the host performs the real syscalls. `open` is our
 * own wrapper only because musl's open.c does not compile (see
 * runtime/musl_gaps.c), not because the service is missing.
 *
 * WHY READ-BACK AND COMPARE, rather than "the write returned the right count".
 * A write that returns 13 proves the request reached the host, nothing more --
 * the host could have written to the wrong file, at the wrong offset, or the
 * domain's copy into the payload region could be wrong in a way the count does
 * not show. Reading the bytes back and comparing them is the cheapest check
 * that fails if any of those is true.
 */
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

extern long __capstone_hc_write(long fd, const char *buf, unsigned long count);
#define SAY(s) __capstone_hc_write(1, (s), sizeof(s) - 1)

#ifndef FILE_PROBE_PAD_WORDS
#define FILE_PROBE_PAD_WORDS 48
#endif
__attribute__((used, retain)) volatile const unsigned long __pad[FILE_PROBE_PAD_WORDS] = {1};

#define PROBE_PATH "/tmp/capstone_file_probe.txt"
static const char PAYLOAD[] = "hostcall file I/O from a domain\n";

int capstone_main(void) {
  char back[sizeof(PAYLOAD)];
  int fd;
  long n;

  if (__pad[0] != 1)
    return 1;
  SAY("FILE S1: entered\n");

  fd = open(PROBE_PATH, O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) {
    SAY("FILE FAIL: open for write\n");
    return 2;
  }
  SAY("FILE S2: opened for write\n");

  n = write(fd, PAYLOAD, sizeof(PAYLOAD) - 1);
  if (n != (long)(sizeof(PAYLOAD) - 1)) {
    SAY("FILE FAIL: short write\n");
    return 3;
  }
  SAY("FILE S3: wrote payload\n");

  if (fsync(fd) != 0) {
    SAY("FILE FAIL: fsync\n");
    return 4;
  }
  if (close(fd) != 0) {
    SAY("FILE FAIL: close after write\n");
    return 5;
  }
  SAY("FILE S4: synced and closed\n");

  fd = open(PROBE_PATH, O_RDONLY, 0);
  if (fd < 0) {
    SAY("FILE FAIL: open for read\n");
    return 6;
  }
  SAY("FILE S5: reopened for read\n");

  memset(back, 0, sizeof(back));
  n = read(fd, back, sizeof(PAYLOAD) - 1);
  if (n != (long)(sizeof(PAYLOAD) - 1)) {
    SAY("FILE FAIL: short read\n");
    return 7;
  }
  close(fd);

  if (memcmp(back, PAYLOAD, sizeof(PAYLOAD) - 1) != 0) {
    SAY("FILE FAIL: bytes differ\n");
    return 8;
  }

  SAY("FILE S6: read back and compared equal\n");
  SAY("__CAPSTONE_FILE_PROBE_PASSED__\n");
  return 0;
}
