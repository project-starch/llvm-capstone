/* xlang Phase-2 seam for the mruby rows (4-15).
 *
 * WHY THIS EXISTS
 * ---------------
 * Every mruby row reproduces by running the stock `bin/mruby` binary on its
 * `trigger.rb`. That leaves no place to substitute a capability allocator: the
 * allocate -> free -> use sequence under test happens entirely inside the VM,
 * behind an allocator the row cannot reach.
 *
 * mruby already has the seam we need -- `mrb_open_allocf()` takes a custom
 * allocator -- it is only the stock `main()` that never uses it. This file is
 * that `main()`: it opens the VM through `xlang_allocf` and then runs the same
 * trigger script, so a row driven by this host behaves exactly as it does under
 * `bin/mruby` today while routing every VM allocation through three functions
 * the capability phase can replace.
 *
 * PHASE 1 (now): the three functions below forward to the C allocator. This is
 * byte-for-byte what `mrb_default_allocf` (mruby/src/state.c:61) does, so the
 * reproduction is unchanged -- same ASan verdict, same crash site, same exit
 * code.
 *
 * PHASE 2: replace only the three bodies. Nothing else in the corpus moves --
 * not the triggers, not the build configs, not the run scripts.
 *
 * They are deliberately three and not one. A capability allocator needs to
 * distinguish the cases mruby's single realloc-shaped hook conflates:
 *   xlang_alloc   -- mint a fresh capability bounded to the new allocation
 *   xlang_realloc -- derive a capability for the new block; the old block is
 *                    what the temporal rows go on to use after it is gone, so
 *                    this is where revocation belongs
 *   xlang_free    -- revoke
 * Row 4 is the worked example: its UAF is a write through a register-stack
 * pointer cached across exactly the `xlang_realloc` call that frees the old
 * stack.
 */

#include <mruby.h>
#include <mruby/compile.h>
#include <stdio.h>
#include <stdlib.h>

/* ---------------------------------------------------------------- THE SEAM */

/* Counters exist for one reason: to prove the seam is actually on the
 * allocation path. At -O1 these three functions inline into the allocator
 * callback, so they do not appear in an ASan backtrace and a trace alone cannot
 * distinguish "routed through the seam" from "mruby used its default
 * allocator". Run any script with XLANG_SEAM_STATS=1 and non-zero counts are
 * that proof. Phase 2 should keep this: it is the check that fails loudly if a
 * capability allocator is silently bypassed. */
static unsigned long xlang_n_alloc, xlang_n_realloc, xlang_n_free;

void *xlang_alloc(size_t size)            { xlang_n_alloc++;   return malloc(size); }
void *xlang_realloc(void *p, size_t size) { xlang_n_realloc++; return realloc(p, size); }
void  xlang_free(void *p)                 { xlang_n_free++;    free(p); }

/* -------------------------------------------------------------------------- */

static void *
xlang_allocf(mrb_state *mrb, void *p, size_t size, void *ud)
{
  (void)mrb; (void)ud;

  /* mruby's contract, from mrb_default_allocf: size == 0 means free, and a
   * NULL p means fresh allocation. The stock version routes both through
   * realloc(); splitting them is what gives the capability phase its three
   * distinct cases, and realloc(NULL, n) == malloc(n) so behaviour is identical. */
  if (size == 0) {
    xlang_free(p);
    return NULL;
  }
  if (p == NULL) {
    return xlang_alloc(size);
  }
  return xlang_realloc(p, size);
}

int
main(int argc, char **argv)
{
  if (argc != 2) {
    fprintf(stderr, "usage: %s <script.rb>\n", argv[0]);
    return 2;
  }

  mrb_state *mrb = mrb_open_allocf(xlang_allocf, NULL);
  if (mrb == NULL) {
    fprintf(stderr, "xlang-host: mrb_open_allocf failed\n");
    return 1;
  }

  FILE *fp = fopen(argv[1], "r");
  if (fp == NULL) {
    perror(argv[1]);
    mrb_close(mrb);
    return 2;
  }

  mrb_load_file(mrb, fp);
  fclose(fp);

  /* Match stock `bin/mruby`: an uncaught Ruby exception prints and exits 1.
   * Row 12 depends on this -- it reproduces by exiting 1 with a caught
   * IOError rather than by faulting. */
  int rc = 0;
  if (mrb->exc) {
    mrb_print_error(mrb);
    rc = 1;
  }

  mrb_close(mrb);

  /* Opt-in, so normal runs are byte-identical to bin/mruby on stdout/stderr.
   * Note this cannot print on a row that aborts under ASan -- the process dies
   * inside the VM. Verify the seam on any non-crashing script instead; whether
   * it is live is a property of the host, not of the trigger. */
  if (getenv("XLANG_SEAM_STATS") != NULL) {
    fprintf(stderr, "xlang-seam: alloc=%lu realloc=%lu free=%lu\n",
            xlang_n_alloc, xlang_n_realloc, xlang_n_free);
  }

  return rc;
}
