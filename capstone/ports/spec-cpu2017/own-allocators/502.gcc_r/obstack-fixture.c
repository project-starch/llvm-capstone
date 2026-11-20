/* fixture.c MODE: a freed obstack object's reference must fault (nested lane, 2026-11-20). Built against a ported
 * obstack.c and obstack.h of SPEC's 502.gcc_r. Each mode prints FIXTURE lines, then reads through the reference p.
 *   0  control: p is never freed; the read must succeed (exit 0)
 *   1  free to a mark inside the current chunk: m, then p, then obstack_free(m); the read must fault
 *   2  free that drops whole chunks: m, then enough objects for new chunks, p the last; obstack_free(m); must fault
 *   3  control for 2: the same objects, no free; the read must succeed (exit 0)
 *   4  a, then b, in the first chunk; objects for new chunks; obstack_free(b) drops those chunks and cuts b in the
 *      first: a must still read (its reference survived the dropped chunks' forgetting), then b must fault
 *   5  3,000 objects of 16 bytes (more than one 1,024-reference segment, several chunks); obstack_free(x[1500]):
 *      x[10] and x[1499] must still read, then x[1500] must fault  */
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <strings.h>
#include "obstack.h"
#define obstack_chunk_alloc malloc
#define obstack_chunk_free free

int main(int argc, char **argv)
{
  struct obstack ob;
  int mode = argc > 1 ? atoi(argv[1]) : -1;
  char *m, *p = 0;
  volatile char c;
  void *first_chunk;

  if (mode < 0 || mode > 5)
    return 2;
  obstack_init (&ob);
  first_chunk = ob.chunk;
  m = (char *) obstack_alloc (&ob, 16);
  if (mode == 5)
    {
      static char *x[3000];
      for (int i = 0; i < 3000; i++)
        {
          x[i] = (char *) obstack_alloc (&ob, 16);
          x[i][0] = 'a' + i % 26;
        }
      printf ("FIXTURE mode=5 x10=%p x1500=%p new_chunk=%d\n", (void *) x[10], (void *) x[1500], (void *) ob.chunk != first_chunk);
      fflush (stdout);
      obstack_free (&ob, x[1500]);
      c = x[10][0];
      printf ("FIXTURE mode=5 read_x10=%c\n", c);
      c = x[1499][0];
      printf ("FIXTURE mode=5 read_x1499=%c\n", c);
      fflush (stdout);
      c = x[1500][0];
      printf ("FIXTURE mode=5 read_x1500=%c after_free=1\n", c);
      return 0;
    }
  if (mode == 4)
    {
      char *a = (char *) obstack_copy0 (&ob, "first", 5);
      char *b = (char *) obstack_copy0 (&ob, "second", 6);
      for (int i = 0; i < 64; i++)
        ((char *) obstack_alloc (&ob, 1000))[0] = 'L';
      printf ("FIXTURE mode=4 a=%p b=%p new_chunk=%d\n", (void *) a, (void *) b, (void *) ob.chunk != first_chunk);
      fflush (stdout);
      obstack_free (&ob, b);
      c = a[0];
      printf ("FIXTURE mode=4 read_a=%c\n", c);
      fflush (stdout);
      c = b[0];
      printf ("FIXTURE mode=4 read_b=%c after_free=1\n", c);
      return 0;
    }
  if (mode <= 1)
    p = (char *) obstack_copy0 (&ob, "lifetime", 8);
  else
    for (int i = 0; i < 64; i++)
      {
        p = (char *) obstack_alloc (&ob, 1000);
        p[0] = 'L';
      }
  printf ("FIXTURE mode=%d m=%p p=%p new_chunk=%d\n", mode, (void *) m, (void *) p, (void *) ob.chunk != first_chunk);
  fflush (stdout);
  if (mode == 1 || mode == 2)
    obstack_free (&ob, m);
  c = p[0];
  printf ("FIXTURE mode=%d read=%c after_free=%d\n", mode, c, mode == 1 || mode == 2);
  return 0;
}
