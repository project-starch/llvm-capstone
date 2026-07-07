#include <stdio.h>
#include <stdlib.h>

#include "../../../caplifive-buildroot/package/modcapstone/userspace/lib/libcapstone.h"
#include "sqlite_sealed_callback_revoke_probe.h"

/* Host binding (lender/controller): owns the callback context (pApp), registers it
 * with the engine as a revocable borrow, drives the engine's sealed invocations of
 * the callback, and unregisters (revoke) mid-stream. See the header for the mapping
 * to the SEALED-CALLBACK rows (1/2/6/16). */

#define print_nobuf(...)  \
  do {                    \
    printf(__VA_ARGS__);  \
    fflush(stdout);       \
  } while (0)

/* PERM_IN (0x0): the host produces the context (pApp) and the engine's callback
 * body reads it (E->H borrow of the context). REV_BORROWED (0x1) makes it
 * revocable so unregister/replace/close can end it. */
#define PROBE_ANNOTATION_PERM_IN 0x0
#define PROBE_ANNOTATION_REV_BORROWED 0x1

static int fail_cleanup(const char *message, unsigned long observed) {
  fprintf(stderr, "sqlite-sealed-cb: %s (observed=0x%016lx)\n", message, observed);
  capstone_cleanup();
  return 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <smode-domain-path>\n", argv[0]);
    return 2;
  }
  if (capstone_init()) {
    fprintf(stderr, "sqlite-sealed-cb: failed to initialize Capstone\n");
    return 1;
  }

  dom_id_t dom_id = create_dom("/test-domains/sbi.dom", argv[1]);
  if ((long)dom_id < 0)
    return fail_cleanup("create_dom failed", (unsigned long)dom_id);

  /* The host binding's callback context (pApp). */
  region_id_t region_id = create_region(SQLITE_SEALED_CB_REGION_SIZE);
  unsigned long *context =
      (unsigned long *)map_region(region_id, SQLITE_SEALED_CB_REGION_SIZE);
  if (!context)
    return fail_cleanup("map_region failed", 0);
  context[0] = SQLITE_SEALED_CB_CONTEXT_VALUE; /* the callback's pApp payload */

  /* Register the callback: lend its context to the engine as a revocable borrow.
   * (sqlite3_progress_handler / sqlite3_create_function / sqlite3_set_authorizer.) */
  shared_region_annotated(dom_id, region_id, PROBE_ANNOTATION_PERM_IN,
                          PROBE_ANNOTATION_REV_BORROWED);
  print_nobuf("sqlite-sealed-cb: callback registered (context borrowed to engine)\n");

  /* Round 1: the engine invokes the sealed callback while it is registered; the
   * callback body caches pApp and reads it, returning what it read. */
  unsigned long r1 = call_dom(dom_id);
  print_nobuf("sqlite-sealed-cb: round 1 (invoke while registered) retval = 0x%016lx\n",
              r1);
  if (r1 != SQLITE_SEALED_CB_CONTEXT_VALUE)
    return fail_cleanup("round 1 did not read the context (borrow not live?)", r1);
  print_nobuf("sqlite-sealed-cb: engine read callback context OK while registered\n");

  /* Unregister / replace / close: free the callback context (pApp). */
  revoke_region(region_id);
  print_nobuf("sqlite-sealed-cb: callback unregistered (context revoked)\n");

  /* Round 2: the engine invokes the callback again and re-reads its CACHED pApp =
   * the use-after-free. */
  print_nobuf("sqlite-sealed-cb: entering round 2 (invoke after unregister)\n");
  unsigned long r2 = call_dom(dom_id);
  print_nobuf("sqlite-sealed-cb: round 2 returned 0x%016lx\n", r2);

  if (r2 == SQLITE_SEALED_CB_FAULT_SENTINEL) {
    print_nobuf("sqlite-sealed-cb: callback use-after-free TRAPPED "
                "(sealed invocation faulted on revoked context, ret=0x%lx)\n", r2);
  } else if (r2 == SQLITE_SEALED_CB_CONTEXT_VALUE) {
    print_nobuf("sqlite-sealed-cb: NO-TRAP-GAP callback re-invocation read stale "
                "context (needs a sealed-callback op -> Step 2)\n");
  } else {
    print_nobuf("sqlite-sealed-cb: round 2 unexpected retval 0x%016lx\n", r2);
  }

  capstone_cleanup();
  return 0;
}
