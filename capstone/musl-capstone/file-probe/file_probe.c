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
#include <sys/stat.h>
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

  /* fstat and lseek. Checked by VALUE, not by return code: a size that is merely
     non-negative, or a seek that merely returns the offset it was given, would
     pass while reading the wrong bytes. So the size is compared against what was
     written, and the seek is proved by reading from the middle of the file and
     comparing against the middle of the payload. */
  fd = open(PROBE_PATH, O_RDONLY, 0);
  if (fd < 0) {
    SAY("FILE FAIL: open for stat/seek\n");
    return 9;
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    SAY("FILE FAIL: fstat\n");
    close(fd);
    return 10;
  }
  if (st.st_size != (off_t)(sizeof(PAYLOAD) - 1)) {
    SAY("FILE FAIL: fstat reported the wrong size\n");
    close(fd);
    return 11;
  }
  SAY("FILE S7: fstat size matches what was written\n");

  /* SEEK_END with a negative offset lands SEEK_BACK bytes from the end; the
     bytes there must be the payload's tail. */
#define SEEK_BACK 8
  if (lseek(fd, -(off_t)SEEK_BACK, SEEK_END) != (off_t)(sizeof(PAYLOAD) - 1 - SEEK_BACK)) {
    SAY("FILE FAIL: lseek SEEK_END returned the wrong position\n");
    close(fd);
    return 12;
  }
  char tail[SEEK_BACK];
  if (read(fd, tail, SEEK_BACK) != SEEK_BACK) {
    SAY("FILE FAIL: short read after seek\n");
    close(fd);
    return 13;
  }
  if (memcmp(tail, PAYLOAD + sizeof(PAYLOAD) - 1 - SEEK_BACK, SEEK_BACK) != 0) {
    SAY("FILE FAIL: bytes after SEEK_END differ\n");
    close(fd);
    return 14;
  }
  SAY("FILE S8: SEEK_END positioned correctly and the bytes match\n");

  if (lseek(fd, 5, SEEK_SET) != 5 || lseek(fd, 3, SEEK_CUR) != 8) {
    SAY("FILE FAIL: SEEK_SET/SEEK_CUR arithmetic\n");
    close(fd);
    return 15;
  }
  if (lseek(fd, -100, SEEK_SET) != -1) {
    SAY("FILE FAIL: a negative position was accepted\n");
    close(fd);
    return 16;
  }
  SAY("FILE S9: SEEK_SET/CUR arithmetic, and a negative seek refused\n");
  close(fd);

  SAY("__CAPSTONE_FILE_PROBE_PASSED__\n");
  return 0;
}
