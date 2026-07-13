/* Row 3, faithful to the *real* diesel defect (RUSTSEC-2021-0037): a column
 * pointer cached across sqlite3_step, where SQLite reuses the column buffer
 * IN PLACE on the next step (the memory is never freed — the handle is only
 * logically stale). This is the headline "stale-but-allocated" case: the
 * capability is still tagged and in-bounds, so CHERI — spatial OR temporal —
 * cannot catch it. (The corpus before.c models row 3 as a finalize/free UAF;
 * this variant models the reuse-not-free essence.) Returns the stale byte. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(void) {
  /* one long-lived buffer that a query engine would reuse across rows */
  char *colbuf = malloc(64);
  if (!colbuf) return 2;

  strcpy(colbuf, "row0_value");
  const char *cached = colbuf;      /* diesel caches this across step() */

  /* next step(): engine overwrites the SAME buffer in place, no free */
  strcpy(colbuf, "row1_value");

  /* use the cached (now stale) pointer: valid cap, in bounds, stale data */
  volatile char c = cached[0];
  printf("stale byte=%c\n", c);
  free(colbuf);
  return (int)c;
}
