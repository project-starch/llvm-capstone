/* The kernel's Sublet format and paged table against QEMU's, on the host.
 *
 * Compiles the kernel's sublet-format.h and sublet-table.h unchanged, with host hooks, and
 * QEMU's target/riscv/cap.c, cap_compress.c, cap_rev_tree.c and cap_rev_table.c. The kernel
 * side builds and grows the table, QEMU's table code validates it and allocates, derives and
 * revokes records in it, and the kernel's record macros must read what QEMU wrote. QEMU's
 * capability encoder writes capabilities, and the kernel's capability macros must read their
 * address, slot, generation and bounds. Every check prints PASS or FAIL. run.sh also builds
 * mutants of sublet-format.h, each of which must fail. */
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cap_rev_table.h"
#include "cap_compress.h"

#define READ_ONCE(x) (*(const volatile __typeof__(x) *)&(x))
#define WRITE_ONCE(x, v) (*(volatile __typeof__(x) *)&(x) = (v))

#define PAGES_MAX 4096
static void *pages[PAGES_MAX];
static unsigned pages_used, pages_live;

static unsigned long *page_new(void)
{
    void *p;

    if (pages_used == PAGES_MAX || posix_memalign(&p, 4096, 4096)) return NULL;
    memset(p, 0, 4096);
    pages[pages_used++] = p;
    ++pages_live;
    return p;
}
static void page_free(void *p)
{
    for (unsigned i = 0; i < pages_used; ++i)
        if (pages[i] == p) { pages[i] = NULL; --pages_live; free(p); return; }
    abort();
}
#define SUBLET_PAGE_NEW() page_new()
#define SUBLET_PAGE_FREE(p) page_free(p)
#define SUBLET_PA(p) ((unsigned long)(uintptr_t)(p))
#define SUBLET_VA(pa) ((unsigned long *)(uintptr_t)(pa))
#define SUBLET_WMB() __sync_synchronize()
#include "sublet-table.h"

static int failures;
static void expect(const char *name, bool ok)
{
    printf("%s %s\n", ok ? "PASS" : "FAIL", name);
    failures += !ok;
}

/* ---- QEMU's view of the kernel's pages: physical address = host address ---- */
static bool range(void *ctx, uint64_t pa, uint64_t len)
{
    (void)ctx;
    for (unsigned i = 0; i < pages_used; ++i) {
        uint64_t p = (uintptr_t)pages[i];
        if (pages[i] && pa >= p && len <= 4096 && pa - p <= 4096 - len) return true;
    }
    return false;
}
static uint64_t rd(void *ctx, uint64_t pa) { (void)ctx; return *(uint64_t *)(uintptr_t)pa; }
static void wr(void *ctx, uint64_t pa, uint64_t v) { (void)ctx; *(uint64_t *)(uintptr_t)pa = v; }
static CapRevStore open_table(CapRevTable *t, unsigned long *root)
{
    CapRevTableMem m = {.read64 = rd, .write64 = wr, .range_ok = range};
    CapRevStore s;

    cap_rev_table_open(t, &m, (uintptr_t)root);
    cap_rev_table_store(t, &s);
    return s;
}

/* The kernel's record of an id, or NULL. */
static unsigned long *krec(unsigned long *root, uint64_t id)
{
    return sublet_table_record(root, CV_ID_SLOT(id));
}

static void check_table(void)
{
    unsigned long allocated = 0, *root = sublet_table_create(), *r_rec, *a_rec, *b_rec;
    uint64_t r = 0, a = 0, b = 0, c = 0, pa = 0;
    CapRevTable t;
    CapRevStore s;

    /* The kernel's empty table, then its first batch. */
    expect("sublet_kernel_table_created", root && sublet_table_grow(root, 1, CV_TABLE_BATCH_PAGES,
                                                                    CV_ID_PAGES_MAX, &allocated) == 0);
    s = open_table(&t, root);
    expect("sublet_qemu_accepts_kernel_table", t.ok && t.paged && t.recycle && !t.nogrow &&
           t.capacity == CV_TABLE_BATCH_PAGES * 256 && t.next == 2 && cap_rev_table_scan_ok(&t) &&
           cap_rev_table_free_list_ok(&t));
    expect("sublet_kernel_counts_available", sublet_table_available(root) == t.capacity - t.next &&
           cap_rev_table_avail(&t) == sublet_table_available(root));
    /* 16 record pages, one middle and one leaf directory page. */
    expect("sublet_kernel_batch_pages", allocated == CV_TABLE_BATCH_PAGES + 2);

    /* R, a child A and a grandchild B, written by QEMU, read by the kernel. */
    expect("sublet_qemu_create_derive", cap_rev_create(&s, &r) == CAP_REV_OK &&
           cap_rev_derive(&s, r, &a) == CAP_REV_OK && cap_rev_derive(&s, a, &b) == CAP_REV_OK);
    r_rec = krec(root, r); a_rec = krec(root, a); b_rec = krec(root, b);
    expect("sublet_kernel_walk_matches_qemu", r_rec && a_rec && b_rec &&
           cap_rev_table_record_pa(&t, r, &pa) && pa == (uintptr_t)r_rec &&
           cap_rev_table_record_pa(&t, b, &pa) && pa == (uintptr_t)b_rec);
    expect("sublet_kernel_reads_depth", r_rec && CV_REC_DEPTH(r_rec) == 0 && CV_REC_DEPTH(a_rec) == 1 &&
           CV_REC_DEPTH(b_rec) == 2);
    expect("sublet_kernel_reads_links", r_rec && CV_REC_NEXT(r_rec) == CV_ID_SLOT(a) &&
           CV_REC_NEXT(a_rec) == CV_ID_SLOT(b) && CV_REC_NEXT(b_rec) == 0 &&
           CV_REC_PREV(a_rec) == CV_ID_SLOT(r) && CV_REC_PREV(b_rec) == CV_ID_SLOT(a));
    expect("sublet_kernel_reads_live", r_rec && CV_REC_LIVE(r_rec, r) && CV_REC_LIVE(a_rec, a) &&
           CV_REC_LIVE(b_rec, b) && !CV_REC_LIVE(a_rec, a + (1UL << 31)) &&
           !CV_REC_LIVE(a_rec, CV_ID(CAP_REV_SUBLET_GEN_RETIRED, CV_ID_SLOT(a))));

    /* CREVOKE of A ends A and B: the kernel sees both stale, R live with no next, two free. */
    expect("sublet_qemu_revoke", cap_rev_kill(&s, a) == CAP_REV_OK);
    expect("sublet_kernel_sees_revoke", !CV_REC_LIVE(a_rec, a) && !CV_REC_LIVE(b_rec, b) &&
           CV_REC_LIVE(r_rec, r) && CV_REC_NEXT(r_rec) == 0 && cap_rev_live(&s, r) &&
           !cap_rev_live(&s, a) && sublet_table_free_count(root) == 2);
    /* The next derive takes a freed slot at its next generation. */
    expect("sublet_qemu_reuses_slot", cap_rev_derive(&s, r, &c) == CAP_REV_OK &&
           (CV_ID_SLOT(c) == CV_ID_SLOT(a) || CV_ID_SLOT(c) == CV_ID_SLOT(b)));
    expect("sublet_kernel_reads_generation", CV_ID_GENERATION(c) == 1 && krec(root, c) &&
           CV_REC_GENERATION(krec(root, c)) == 1 && CV_REC_LIVE(krec(root, c), c) &&
           !CV_REC_LIVE(krec(root, c), CV_ID(0, CV_ID_SLOT(c))));

    /* Growth: fill the table until QEMU asks for growth (cause 30), grow it, go on. */
    while (cap_rev_table_alloc_decision(&t) == CAP_REV_ALLOC_PROCEED && cap_rev_derive(&s, r, &c) == CAP_REV_OK)
        ;
    expect("sublet_qemu_asks_growth", cap_rev_table_alloc_decision(&t) == CAP_REV_ALLOC_GROW &&
           sublet_table_available(root) == 0);
    expect("sublet_kernel_grows", sublet_table_grow(root, 1, CV_TABLE_BATCH_PAGES, CV_ID_PAGES_MAX, &allocated) == 0 &&
           sublet_table_capacity(root) == 2 * CV_TABLE_BATCH_PAGES * 256);
    s = open_table(&t, root);
    expect("sublet_qemu_accepts_grown_table", t.ok && cap_rev_table_scan_ok(&t) &&
           cap_rev_table_alloc_decision(&t) == CAP_REV_ALLOC_PROCEED && cap_rev_derive(&s, r, &c) == CAP_REV_OK &&
           krec(root, c) && CV_REC_LIVE(krec(root, c), c) && CV_REC_DEPTH(krec(root, c)) == 1);

    /* NOGROW: once the table is full again, the allocation fails without a fault. */
    sublet_table_set_nogrow(root, true);
    while (cap_rev_derive(&s, r, &c) == CAP_REV_OK)
        ;
    s = open_table(&t, root);
    expect("sublet_qemu_honours_nogrow", t.ok && t.nogrow && sublet_table_nogrow(root) &&
           cap_rev_table_alloc_decision(&t) == CAP_REV_ALLOC_FAIL);
    sublet_table_set_nogrow(root, false);
    s = open_table(&t, root);
    expect("sublet_qemu_sees_nogrow_cleared", t.ok && !t.nogrow &&
           cap_rev_table_alloc_decision(&t) == CAP_REV_ALLOC_GROW);

    sublet_table_free(root);
    expect("sublet_kernel_frees_every_page", pages_live == 0);
}

/* Slots under root entry 31, the last slot of the table among them: the kernel's add_page and
 * record walk put each record where QEMU's walk finds it. */
static void check_walk(void)
{
    static const uint64_t slots[] = {3, (1ULL << 30) + 5, 31ULL << 26, (31ULL << 26) + 7, (1ULL << 31) - 1};
    unsigned long allocated = 0, *root = sublet_table_create();
    bool found = root != NULL;
    uint64_t pa = 0;
    CapRevTable t;

    for (unsigned i = 0; found && i < sizeof(slots) / sizeof(slots[0]); ++i)
        found = sublet_table_add_page(root, slots[i] & ~255ULL, &allocated) == 0;
    /* A header that names every slot: capacity 2^31, next 2^31, so each slot is issued. */
    if (found) {
        root[CV_HDR_CAPACITY] = CV_TABLE_FLAGS | CV_ID_RECORDS_MAX;
        root[CV_HDR_NEXT] = CV_ID_RECORDS_MAX;
    }
    open_table(&t, root);
    for (unsigned i = 0; found && i < sizeof(slots) / sizeof(slots[0]); ++i)
        found = cap_rev_table_record_pa(&t, slots[i], &pa) && pa == (uintptr_t)sublet_table_record(root, slots[i]);
    expect("sublet_kernel_walk_5_9_9_8", found);
    expect("sublet_kernel_walk_refuses_reserved", root && !sublet_table_record(root, 0) &&
           !sublet_table_record(root, 1) && !sublet_table_record(root, CV_ID_RECORDS_MAX) &&
           !sublet_table_record(root, 4096));
    sublet_table_free(root);
    expect("sublet_kernel_frees_sparse_table", pages_live == 0);
}

/* QEMU's encoder writes the capability, the kernel's macros read it. */
static void check_capability(void)
{
    static const struct { uint64_t base, end, cursor, gen, slot; } cases[] = {
        {0, 1ULL << 38, 0, 0, 2},
        {0x10000, 0x20000, 0x18008, 1, 3},
        {0x3fffff0000ULL, 0x3fffff1000ULL, 0x3fffff0ff0ULL, 0xffffffffULL - 1, (1ULL << 31) - 1},
        {0x123450, 0x123460, 0x123450, 0x5a5a5a5aULL, 0x2aaaaaaaULL},
    };
    bool fields = true, bounds_only = true;

    for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        capfat_t cap = {0};
        uint64_t lo, hi, lo2, hi2, id = cases[i].gen << 31 | cases[i].slot;

        cap.bounds.base = cases[i].base;
        cap.bounds.end = cases[i].end;
        cap.bounds.cursor = cases[i].cursor;
        cap.rev_node_id = id;
        cap_compress(&cap, &lo, &hi);
        fields &= CV_CAP_ADDRESS(lo) == cases[i].cursor && CV_CAP_SLOT(hi) == cases[i].slot &&
                  CV_CAP_GENERATION(hi) == cases[i].gen && CV_CAP_ID(hi) == id &&
                  CV_ID_SLOT(id) == cases[i].slot && CV_ID_GENERATION(id) == cases[i].gen &&
                  CV_CAP_BOUNDS(lo, hi) < (1UL << CV_CAP_BOUNDS_BITS);
        /* Other bounds, same cursor and id: only the kernel's bounds field may differ. */
        cap.bounds.base = cases[i].cursor & ~0xfffULL;
        cap.bounds.end = cap.bounds.base + 0x2000;
        cap_compress(&cap, &lo2, &hi2);
        bounds_only &= CV_CAP_BOUNDS(lo, hi) != CV_CAP_BOUNDS(lo2, hi2) &&
                       CV_CAP_ADDRESS(lo2) == cases[i].cursor && CV_CAP_ID(hi2) == id &&
                       (lo & CV_CAP_ADDR_MASK) == (lo2 & CV_CAP_ADDR_MASK) && (hi >> 1) == (hi2 >> 1);
    }
    expect("sublet_kernel_reads_capability_fields", fields);
    expect("sublet_kernel_bounds_field_position", bounds_only);
    expect("sublet_descriptor_and_address_end", CV_DESC_ROOT == 0x10f && CV_DESC_WORDS == 6 &&
           CV_USER_ADDRESS_END == 1ULL << CAP_SUBLET_ADDR_BITS);
}

int main(void)
{
    check_table();
    check_walk();
    check_capability();
    printf("%d failures\n", failures);
    return failures != 0;
}
