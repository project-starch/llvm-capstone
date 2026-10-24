/* The launcher's mapping registry is an array sorted by address, highest first, searched with
 * map_floor. This test drives the real map_reserve / map_insert / map_remove / map_find /
 * map_floor (exec.c is included, so the code under test cannot drift from a copy) through
 * 200,000 random insertions, removals, moves and lookups, and after every step compares
 * the registry with a direct scan of a model: the floor of a random address, the exact find,
 * the count and the strict descending order. Three further phases replay the address patterns
 * the registry is built for: registrations arriving top-down (each below all earlier ones,
 * so each is appended), removals with a surviving suffix, and bottom-up. Host test: it needs no module and no capability hardware.
 * Built against the exec.c named by EXEC_C, so a deliberately broken copy (the floor
 * comparison flipped) can be run through the same cases: it must fail. */
#define _GNU_SOURCE
#define main vexec_main
#include EXEC_C
#undef main

enum { SLOTS = 2048, RANDOM_OPS = 200000, SWEEP = 4000 };
static int used[SLOTS];
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
                            .visible = i % 3 ? 4096 : 0, .id = (uint64_t)i + 1, .heap = i & 1, .kind = i % 5};
}
static void add(int i)
{
    if (map_reserve()) { fprintf(stderr, "out of memory\n"); exit(2); }
    map_insert(registration(i));
    CHECK(map_count <= map_capacity, "count %zu exceeds capacity %zu", map_count, map_capacity);
    used[i] = 1;
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
    for (size_t k = 1; k < map_count; k++)
        CHECK((uintptr_t)maps[k - 1].address > (uintptr_t)maps[k].address, "not strictly descending at %zu", k);
    for (size_t k = 0; k < map_count; k++) {
        int slot = (int)(((uintptr_t)maps[k].address - slot_address(0)) / 8192);
        struct mapping want = registration(slot < 0 || slot >= SLOTS ? 0 : slot);
        CHECK(slot >= 0 && slot < SLOTS && used[slot] && maps[k].id == want.id && maps[k].bytes == want.bytes &&
              maps[k].visible == want.visible && maps[k].heap == want.heap && maps[k].kind == want.kind,
              "entry %zu (%#lx) does not carry its registration", k, (unsigned long)(uintptr_t)maps[k].address);
    }
}
static void clear_all(void)
{
    while (map_count) map_remove(&maps[map_count - 1]);
    memset(used, 0, sizeof used);
}

int main(void)
{
    for (long step = 0; step < RANDOM_OPS; step++) {
        int i = (int)(next_random() % SLOTS), op = (int)(next_random() % 3);
        if (op == 0 && !used[i]) add(i);
        else if (op == 1 && used[i]) {
            struct mapping *m = map_find(slot_address(i));
            CHECK(m && m->id == (uint64_t)i + 1, "find(%d) before removal", i);
            if (m) map_remove(m);
            used[i] = 0;
        } else if (op == 2 && used[i]) {
            int j = (int)(next_random() % SLOTS);
            if (!used[j]) {
                struct mapping *m = map_find(slot_address(i));
                CHECK(m != NULL, "find(%d) before move", i);
                if (!m) continue;
                struct mapping before = registration(i);
                CHECK(m->bytes == before.bytes && m->visible == before.visible && m->heap == before.heap &&
                      m->kind == before.kind, "entry %d changed before its move", i);
                map_remove(m);
                map_insert(registration(j));
                used[i] = 0; used[j] = 1;
            }
        }
        check_registry();
        for (int probe = 0; probe < 2; probe++) {
            int k = (int)(next_random() % SLOTS);
            struct mapping *m = map_find(slot_address(k));
            CHECK((m != NULL) == (used[k] != 0), "find(%d) is %s, model says %s", k,
                  m ? "present" : "absent", used[k] ? "present" : "absent");
            /* find is an exact match on the base: an address inside a mapping, or one byte below
             * it, is no registration. */
            CHECK(map_find(slot_address(k) + 8) == NULL, "find(%d + 8) matched", k);
            CHECK(map_find(slot_address(k) - 1) == NULL, "find(%d - 1) matched", k);
        }
    }
    clear_all();
    /* Top-down: each registration below all earlier ones; recent ones come back first. */
    for (int i = SLOTS - 1; i >= SLOTS - SWEEP / 4; i--) { add(i); check_registry(); }
    for (int i = SLOTS - SWEEP / 4; i < SLOTS - SWEEP / 8; i++) {
        struct mapping *m = map_find(slot_address(i)); CHECK(m != NULL, "top-down find %d", i);
        if (m) map_remove(m); used[i] = 0; check_registry();
    }
    clear_all();
    /* Top-down, then remove the second to sixth from the end and one from the middle: a suffix
     * survives each removal, so the tail really moves. */
    for (int i = SLOTS - 1; i >= SLOTS - 64; i--) add(i);
    for (int back = 5; back >= 1; back--) {
        size_t at = map_count - 1 - (size_t)back;
        int slot = (int)(((uintptr_t)maps[at].address - slot_address(0)) / 8192);
        map_remove(&maps[at]); used[slot] = 0; check_registry();
    }
    {
        size_t at = map_count / 2;
        int slot = (int)(((uintptr_t)maps[at].address - slot_address(0)) / 8192);
        map_remove(&maps[at]); used[slot] = 0; check_registry();
    }
    clear_all();
    /* Bottom-up: each registration above all earlier ones, so each goes to the front. */
    for (int i = 0; i < SWEEP / 4; i++) { add(i); check_registry(); }
    clear_all();
    /* Growth: far more registrations than the model's slots, arriving top-down as the kernel gives
     * them, past every doubling of the capacity. */
    {
        enum { GROW = 6000 };
        uintptr_t top = slot_address(SLOTS) + (uintptr_t)GROW * 8192;
        for (int i = 0; i < GROW; i++) {
            if (map_reserve()) { fprintf(stderr, "out of memory\n"); exit(2); }
            map_insert((struct mapping){.address = (void *)(top - (uintptr_t)i * 8192), .bytes = 4096, .visible = 4096,
                                        .id = (uint64_t)i + 1});
            CHECK(map_count <= map_capacity, "count %zu exceeds capacity %zu", map_count, map_capacity);
        }
        CHECK(map_count == GROW && map_capacity >= GROW, "after growth: count %zu, capacity %zu", map_count, map_capacity);
        for (int i = 0; i < GROW; i++) {
            struct mapping *m = map_find(top - (uintptr_t)i * 8192);
            CHECK(m && m->id == (uint64_t)i + 1, "grown entry %d lost", i);
            CHECK(m == &maps[i], "grown entry %d is not at index %d", i, i);
        }
        while (map_count) map_remove(&maps[map_count - 1]);
    }
    free(maps); maps = NULL; map_count = map_capacity = 0;
    if (failures) { fprintf(stderr, "registry: %ld failed checks\n", failures); return 1; }
    printf("registry: %d random operations and three address-order sweeps and a growth phase passed\n", RANDOM_OPS);
    return 0;
}
