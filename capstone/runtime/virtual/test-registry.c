/* The launcher's mapping registry is an AVL tree keyed by address, searched with map_floor. This
 * test drives the real map_reserve / map_insert / map_remove / map_find / map_floor (exec.c is
 * included, so the code under test cannot drift from a copy) through 200,000 random insertions,
 * removals, moves and lookups, and after every step compares the registry with a direct scan of a
 * model: the floor of a random address, the exact find, the count, the strict address order of an
 * in-order walk, every node's height and balance, and every entry's fields. A pointer to a
 * registered mapping must stay valid (same node, same fields) until that mapping is removed.
 * Further phases replay the address patterns a launcher sees: registrations arriving top-down
 * (each below all earlier ones) and removed oldest first (the virtual heap's group mappings, the
 * pattern that made the old sorted array move its whole tail on every unmap), removals with a
 * surviving suffix, bottom-up, and a growth phase of 20,000 registrations whose tree height must
 * stay within the AVL bound. Host test: it needs no module and no capability hardware.
 * Built against the exec.c named by EXEC_C, so a deliberately broken copy (the floor comparison
 * flipped, a rotation or a rebalance dropped) can be run through the same cases: it must fail. */
#define _GNU_SOURCE
#define main vexec_main
#include EXEC_C
#undef main
#include <math.h>

enum { SLOTS = 2048, RANDOM_OPS = 200000, SWEEP = 4000, GROW = 20000 };
static int used[SLOTS];
static struct mapping *held[SLOTS];   /* the pointer map_find returned when the slot was added */
static uint64_t rng = 0x2545F4914F6CDD1DULL;
static unsigned next_random(void)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (unsigned)(rng >> 32);
}
/* The slots straddle a 4 GiB boundary at a typical mmap height, so a comparison that drops the high
 * half of an address reorders them. */
static uintptr_t slot_address(int i)
{
    return (uintptr_t)0x7f3affffe000ULL - (uintptr_t)(SLOTS / 2) * 8192 + (uintptr_t)i * 8192;
}
static long failures;
#define CHECK(cond, ...) do { if (!(cond)) { if (failures++ < 5) { fprintf(stderr, "FAIL line %d: ", __LINE__); \
                                fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } } while (0)

/* Every field of a registration is a function of its slot, so a registry that mixes entries up or
 * rewrites a field cannot pass. */
static struct mapping registration(int i)
{
    return (struct mapping){.address = (void *)slot_address(i), .bytes = i % 2 ? 8192 : 4096,
                            .visible = i % 3 ? 4096 : 0, .id = (uint64_t)i + 1, .heap = i & 1, .kind = i % 5,
                            .holder = (uint64_t)i * 7};
}
static struct map_node *held_node(struct mapping *m)
{ return m ? (struct map_node *)((char *)m - offsetof(struct map_node, m)) : NULL; }
static void add(int i)
{
    if (map_reserve()) { fprintf(stderr, "out of memory\n"); exit(2); }
    CHECK(map_free != NULL, "map_reserve left no node");
    struct map_node *next = map_free;
    map_insert(registration(i));
    CHECK(held_node(map_find(slot_address(i))) == next, "map_insert did not take the free list's head");
    used[i] = 1;
    held[i] = map_find(slot_address(i));
    CHECK(held[i] != NULL, "find(%d) right after insert", i);
}
static void drop(int i)
{
    struct mapping *m = map_find(slot_address(i));
    CHECK(m && m->id == (uint64_t)i + 1, "find(%d) before removal", i);
    CHECK(m == held[i], "entry %d moved while registered", i);
    if (m) map_remove(m);
    used[i] = 0; held[i] = NULL;
}
/* In-order walk: strictly ascending addresses, correct heights, |balance| <= 1, the count. */
static uintptr_t walk_last;
static size_t walk_count;
static int walk(const struct map_node *n)
{
    if (!n) return 0;
    int low = walk(n->child[0]);
    CHECK(!walk_count || (uintptr_t)n->m.address > walk_last, "not strictly ascending at %#lx after %#lx",
          (unsigned long)(uintptr_t)n->m.address, (unsigned long)walk_last);
    walk_last = (uintptr_t)n->m.address; ++walk_count;
    int high = walk(n->child[1]);
    int height = 1 + (low > high ? low : high);
    CHECK(n->height == height, "node %#lx records height %d, has %d", (unsigned long)(uintptr_t)n->m.address,
          n->height, height);
    CHECK(low - high <= 1 && high - low <= 1, "node %#lx unbalanced: %d vs %d",
          (unsigned long)(uintptr_t)n->m.address, low, high);
    return height;
}
static int check_shape(void)
{
    walk_last = 0; walk_count = 0;
    int height = walk(map_root);
    CHECK(walk_count == map_count, "walk found %zu nodes, count is %zu", walk_count, map_count);
    return height;
}
static void check_payloads(void)
{
    for (int j = 0; j < SLOTS; j++) {
        if (!used[j]) continue;
        struct mapping want = registration(j), *m = held[j];
        CHECK(m && m->address == want.address && m->id == want.id && m->bytes == want.bytes &&
              m->visible == want.visible && m->heap == want.heap && m->kind == want.kind &&
              m->holder == want.holder, "entry %d does not carry its registration", j);
    }
}
static void check_registry(void)
{
    uintptr_t p = slot_address(0) - 4096 + (uintptr_t)((((uint64_t)next_random() << 32) | next_random()) % ((uint64_t)SLOTS * 8192 + 8192));
    uintptr_t expected = 0;
    size_t count = 0;
    for (int j = 0; j < SLOTS; j++) {
        if (!used[j]) continue;
        ++count;
        if (slot_address(j) <= p && slot_address(j) > expected) expected = slot_address(j);
    }
    struct mapping *f = map_floor(p);
    CHECK((f ? (uintptr_t)f->address : 0) == expected, "floor(%#lx) is %#lx, expected %#lx",
          (unsigned long)p, (unsigned long)(f ? (uintptr_t)f->address : 0), (unsigned long)expected);
    CHECK(count == map_count, "count %zu, expected %zu", map_count, count);
    check_shape();
}
static void check_full(void) { check_registry(); check_payloads(); }
static void clear_all(void)
{
    for (int j = 0; j < SLOTS; j++) if (used[j]) drop(j);
    CHECK(map_root == NULL && map_count == 0, "registry not empty after clearing");
}

int main(void)
{
    for (long step = 0; step < RANDOM_OPS; step++) {
        int i = (int)(next_random() % SLOTS), op = (int)(next_random() % 3);
        if (op == 0 && !used[i]) add(i);
        else if (op == 1 && used[i]) drop(i);
        else if (op == 2 && used[i]) {
            /* A move, as remap_mapping does it: remove, then insert without a map_reserve. */
            int j = (int)(next_random() % SLOTS);
            if (!used[j]) {
                struct mapping *m = map_find(slot_address(i));
                CHECK(m != NULL && m == held[i], "find(%d) before move", i);
                if (!m) continue;
                map_remove(m);
                CHECK(map_free != NULL && &map_free->m == m, "map_remove did not keep the node for the move");
                used[i] = 0; held[i] = NULL;
                map_insert(registration(j));
                used[j] = 1; held[j] = map_find(slot_address(j));
                CHECK(held[j] != NULL, "find(%d) after move", j);
            }
        }
        check_registry();
        if (step % 1024 == 0) check_payloads();
        for (int probe = 0; probe < 2; probe++) {
            int k = (int)(next_random() % SLOTS);
            struct mapping *m = map_find(slot_address(k));
            CHECK((m != NULL) == (used[k] != 0), "find(%d) is %s, model says %s", k,
                  m ? "present" : "absent", used[k] ? "present" : "absent");
            CHECK(m == held[k], "find(%d) returned another pointer than at its insertion", k);
            /* find is an exact match on the base: an address inside a mapping, or one byte below
             * it, is no registration; the floor of an address inside a mapping is that mapping. */
            CHECK(map_find(slot_address(k) + 8) == NULL, "find(%d + 8) matched", k);
            CHECK(map_find(slot_address(k) - 1) == NULL, "find(%d - 1) matched", k);
            if (used[k]) CHECK(map_floor(slot_address(k) + 4095) == held[k], "floor inside %d", k);
        }
    }
    check_full();
    clear_all();
    /* Top-down: each registration below all earlier ones; recent ones come back first. */
    for (int i = SLOTS - 1; i >= SLOTS - SWEEP / 4; i--) { add(i); check_registry(); }
    for (int i = SLOTS - SWEEP / 4; i < SLOTS - SWEEP / 8; i++) { drop(i); check_registry(); }
    check_full();
    clear_all();
    /* Top-down, then the oldest (highest) removed first, as the virtual heap gives back the group
     * mappings of a large structure in allocation order. */
    for (int i = SLOTS - 1; i >= 0; i--) add(i);
    check_full();
    for (int i = SLOTS - 1; i >= 0; i--) { drop(i); if (i % 16 == 0) check_full(); else check_registry(); }
    /* Top-down, then remove the second to sixth from the lowest end and one from the middle. */
    for (int i = SLOTS - 1; i >= SLOTS - 64; i--) add(i);
    for (int back = 5; back >= 1; back--) { drop(SLOTS - 64 + back); check_full(); }
    drop(SLOTS - 32); check_full();
    clear_all();
    /* Bottom-up: each registration above all earlier ones. */
    for (int i = 0; i < SWEEP / 4; i++) { add(i); check_registry(); }
    check_full();
    clear_all();
    /* Growth: far more registrations than the model's slots, arriving top-down as the kernel gives
     * them. Every node is found, the height stays within the AVL bound 1.44 log2(n + 2), and all
     * come back oldest first. */
    {
        static struct mapping *grown[GROW];
        uintptr_t top = slot_address(SLOTS) + (uintptr_t)GROW * 8192;
        for (int i = 0; i < GROW; i++) {
            if (map_reserve()) { fprintf(stderr, "out of memory\n"); exit(2); }
            map_insert((struct mapping){.address = (void *)(top - (uintptr_t)i * 8192), .bytes = 4096, .visible = 4096,
                                        .id = (uint64_t)i + 1});
            grown[i] = map_find(top - (uintptr_t)i * 8192);
        }
        CHECK(map_count == GROW, "after growth: count %zu", map_count);
        int height = check_shape();
        CHECK(height <= (int)(1.4405 * log2((double)GROW + 2)), "height %d for %d nodes", height, GROW);
        for (int i = 0; i < GROW; i++) {
            struct mapping *m = map_find(top - (uintptr_t)i * 8192);
            CHECK(m && m == grown[i] && m->id == (uint64_t)i + 1, "grown entry %d lost", i);
            CHECK(map_floor(top - (uintptr_t)i * 8192 + 100) == m, "floor inside grown entry %d", i);
        }
        for (int i = 0; i < GROW; i++) {
            map_remove(grown[i]);
            if (i % 997 == 0) check_shape();
        }
        CHECK(map_root == NULL && map_count == 0, "registry not empty after the growth phase");
    }
    if (failures) { fprintf(stderr, "registry: %ld failed checks\n", failures); return 1; }
    printf("registry: %d random operations, four address-order sweeps and a growth phase of %d passed\n",
           RANDOM_OPS, GROW);
    return 0;
}
