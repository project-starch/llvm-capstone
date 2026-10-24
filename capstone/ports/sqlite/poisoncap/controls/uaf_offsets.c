/* Evidence for case 15 (not a gate): which offsets of a freed memsys5 block does the published
 * mechanism report? One fresh child process per offset: configure memsys5, allocate a 128-byte block
 * (a 100-byte request), fill it, free it, read one byte at the offset through a pointer derived before
 * the free. The parent prints one line per offset with how the child ended (exit status or signal). */
#include "repro322_common.h"
#include <sys/wait.h>

static const int OFFSETS[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 55, 127};

static int run_case(void) {
  for (unsigned k = 0; k < sizeof OFFSETS / sizeof OFFSETS[0]; k++) {
    int off = OFFSETS[k];
    pid_t pid = fork();
    if (pid == 0) {
      if (repro_init()) _exit(70);
      unsigned char *blk = sqlite3_malloc(100);
      if (!blk) _exit(71);
      for (int i = 0; i < 100; i++) blk[i] = (unsigned char)(0x40 + i);
      volatile unsigned char *a = blk + off;
      sqlite3_free(blk);
      unsigned char v = a[0];
      printf("ctl child off=%d read value=%u\n", off, v);
      _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status))
      printf("OFFSET %d SIGNAL %d\n", off, WTERMSIG(status));
    else
      printf("OFFSET %d EXIT %d\n", off, WEXITSTATUS(status));
  }
  return 0;
}
REPRO322_MAIN("ctloffsets")
