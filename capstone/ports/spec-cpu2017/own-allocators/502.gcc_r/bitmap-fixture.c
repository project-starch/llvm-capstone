/* bitmap-fixture.c MODE: a freed bitmap element's reference must fault (nested lane, 2026-11-20). Linked against a
 * gcc arm's bitmap.o and obstack.o; ggc_alloc_stat, xmalloc and fancy_abort are stubbed (GC bitmaps are not used).
 *   0  control: an element of a live bitmap reads
 *   1  bitmap_clear_bit empties the element (bitmap_element_free, one element): its old reference must fault
 *   2  bitmap_clear frees a chain of three elements (bitmap_elt_clear_from): the old reference of the middle one must
 *      fault, and the bitmap must work again afterwards (the next set_bit reuses freed elements)
 *   3  control for 2: the chain stays; the middle element reads  */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct bitmap_obstack bitmap_obstack;
typedef struct bitmap_element_def { struct bitmap_element_def *next, *prev; unsigned int indx; unsigned long bits[2]; } bitmap_element;
typedef struct bitmap_head_def { bitmap_element *first, *current; unsigned int indx; bitmap_obstack *obstack; } *bitmap;
extern void bitmap_obstack_initialize (bitmap_obstack *);
extern bitmap bitmap_obstack_alloc_stat (bitmap_obstack *);
extern int bitmap_set_bit (bitmap, int);
extern int bitmap_clear_bit (bitmap, int);
extern void bitmap_clear (bitmap);
extern int bitmap_bit_p (bitmap, int);

void *ggc_alloc_stat (size_t n) { return calloc (1, n); }
void *xmalloc (size_t n) { void *p = malloc (n); if (!p) abort (); return p; }
void fancy_abort (const char *f, int l, const char *fn) { fprintf (stderr, "fancy_abort %s:%d %s\n", f, l, fn); abort (); }

int main (int argc, char **argv)
{
  static char obstack_space[512] __attribute__ ((aligned (16)));
  bitmap_obstack *ob = (bitmap_obstack *) obstack_space;
  int mode = argc > 1 ? atoi (argv[1]) : -1;
  volatile unsigned long v;
  bitmap b;
  bitmap_element *e;

  if (mode < 0 || mode > 3)
    return 2;
  bitmap_obstack_initialize (ob);
  b = bitmap_obstack_alloc_stat (ob);
  bitmap_set_bit (b, 5);
  if (mode >= 2)
    {
      bitmap_set_bit (b, 1000);
      bitmap_set_bit (b, 5000);
    }
  e = mode >= 2 ? b->first->next : b->first;   /* the element of bit 5, or the middle of three */
  printf ("FIXTURE mode=%d element=%p indx=%u\n", mode, (void *) e, e->indx);
  fflush (stdout);
  if (mode == 1)
    bitmap_clear_bit (b, 5);
  else if (mode == 2)
    {
      bitmap_clear (b);
      bitmap_set_bit (b, 77);
      printf ("FIXTURE mode=2 reused bit77=%d\n", bitmap_bit_p (b, 77));
      fflush (stdout);
    }
  v = e->bits[0] | e->indx;
  printf ("FIXTURE mode=%d read=%lu after_free=%d\n", mode, (unsigned long) v, mode == 1 || mode == 2);
  return 0;
}
