/* Run a command and print the instructions the guest executed meanwhile.

   Under QEMU's -icount, rdinstret counts executed instructions, so the
   difference around the child is the cost of the whole job on a guest with one
   hart: the program, the kernel and whatever else ran. Build it for the guest:

       riscv64-linux-gnu-gcc -static -O2 icount.c -o icount  */
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

static inline unsigned long
instructions (void)
{
  unsigned long value;
  __asm__ volatile ("rdinstret %0" : "=r" (value));
  return value;
}

int
main (int argc, char **argv)
{
  unsigned long before, after;
  int status;
  pid_t child;

  if (argc < 2)
    {
      fprintf (stderr, "usage: icount COMMAND [ARGUMENT...]\n");
      return 2;
    }

  before = instructions ();
  child = fork ();
  if (child < 0)
    return 2;
  if (child == 0)
    {
      execv (argv[1], argv + 1);
      _exit (127);
    }
  waitpid (child, &status, 0);
  after = instructions ();

  printf ("ICOUNT instructions=%lu status=%d\n", after - before, status);
  return 0;
}
