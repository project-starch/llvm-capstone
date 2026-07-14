/* tree_workload.h -- a bounded binary-search-tree build / lookup / destroy
 * workload, shared by the CHERI (tests/cheri-perf/) and Capstone
 * (tests/runtime-qemu/revoke-cost-probe/) perf arms so both measure the SAME
 * real allocation pattern rather than a synthetic tight malloc/free loop.
 *
 * Why a BST: it is the canonical pointer-chasing allocation workload -- each
 * round builds a live-set of TW_KEYS nodes (a real capability graph), does
 * TW_KEYS lookups that chase pointers through it (real work that dereferences
 * the live capabilities), then tears the whole set down. During lookup + until
 * teardown all TW_KEYS nodes are LIVE, so a CHERI revocation sweep has a genuine
 * live capability graph to scan -- unlike the microbench, whose heap is empty at
 * every free. Keys are a deterministic LCG so the shape reproduces run to run.
 *
 * The includer supplies TW_MALLOC / TW_FREE (and optionally TW_KEYS / TW_ROUNDS);
 * everything else is identical, so the only difference between two runs is the
 * allocator + revocation policy. Total allocation is TW_KEYS*TW_ROUNDS, kept
 * bounded so it fits the Capstone Phase-0 non-reclaiming arena.
 */
#ifndef TREE_WORKLOAD_H
#define TREE_WORKLOAD_H

#ifndef TW_KEYS
#define TW_KEYS 2000u        /* live-set per round */
#endif
#ifndef TW_ROUNDS
#define TW_ROUNDS 25u        /* build+lookup+destroy cycles */
#endif
#ifndef TW_MALLOC
#error "tree_workload.h: define TW_MALLOC(n) before including"
#endif
#ifndef TW_FREE
#error "tree_workload.h: define TW_FREE(p) before including"
#endif

struct tw_node {
  unsigned long key;
  struct tw_node *l, *r;
};

/* Node handles for this round's live-set, so teardown is a flat free (no
 * recursion -> no stack-depth risk on a skewed tree, and a realistic
 * "destroy all" pass). Sized for TW_KEYS capabilities. */
static struct tw_node *tw_nodes[TW_KEYS];

static inline unsigned long tw_lcg(unsigned long *s) {
  *s = *s * 6364136223846793005UL + 1442695040888963407UL;
  return (*s >> 11);
}

/* One round. Returns a checksum of looked-up keys so nothing is optimized away.
 * seed picks the (deterministic) key stream. Stops early only on allocation
 * failure (returns what it built so the caller can tell). */
static unsigned long tw_round(unsigned long seed, unsigned *built_out) {
  struct tw_node *root = 0;
  unsigned built = 0;
  unsigned long s = seed;

  /* build */
  for (unsigned i = 0; i < TW_KEYS; i++) {
    unsigned long key = tw_lcg(&s);
    struct tw_node *n = (struct tw_node *)TW_MALLOC(sizeof *n);
    if (!n)
      break;
    n->key = key;
    n->l = 0;
    n->r = 0;
    tw_nodes[built++] = n;
    if (!root) {
      root = n;
      continue;
    }
    struct tw_node *cur = root, *parent = 0;
    int right = 0;
    while (cur) {
      parent = cur;
      if (key < cur->key) { cur = cur->l; right = 0; }
      else                { cur = cur->r; right = 1; }
    }
    /* Select the child-slot ADDRESS, then a single capability store -- avoids
     * the conditional-capability-value store (`if (r) a=n; else b=n;`) that ICEs
     * the -O2 Capstone backend (see COORDINATION.md; rtl-smoke history note). */
    struct tw_node **slot = right ? &parent->r : &parent->l;
    *slot = n;
  }

  /* lookup: chase the same key stream through the tree (pointer-heavy work that
   * touches the live capabilities). */
  unsigned long checksum = 0;
  s = seed;
  for (unsigned i = 0; i < built; i++) {
    unsigned long key = tw_lcg(&s);
    struct tw_node *cur = root;
    while (cur) {
      if (key == cur->key) { checksum ^= cur->key; break; }
      cur = (key < cur->key) ? cur->l : cur->r;
    }
  }

  /* destroy: flat teardown of the live-set (each free is a revocation point). */
  for (unsigned i = 0; i < built; i++)
    TW_FREE(tw_nodes[i]);

  if (built_out) *built_out = built;
  return checksum;
}

/* The full workload: TW_ROUNDS rounds. Returns total nodes freed (== total
 * allocations completed), the natural per-op normalizer, and folds the checksum
 * into *sink so the compiler cannot elide the work. */
static unsigned long tw_run(volatile unsigned long *sink) {
  unsigned long total_ops = 0, cs = 0;
  for (unsigned r = 0; r < TW_ROUNDS; r++) {
    unsigned built = 0;
    cs ^= tw_round(0x9E3779B97F4A7C15UL + r, &built);
    total_ops += built;
  }
  *sink ^= cs;
  return total_ops;
}

#endif /* TREE_WORKLOAD_H */
