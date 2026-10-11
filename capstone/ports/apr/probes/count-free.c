/* Interpose free() so "nothing reaches malloc" is counted rather than quoted. */
#include <stdlib.h>
extern unsigned long freed_to_malloc;
void free(void *p) {
  if (p) freed_to_malloc++;
  /* deliberately leaked: this probe is short-lived and counting is the point */
}
