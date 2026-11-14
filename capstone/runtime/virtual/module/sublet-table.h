/* SPDX-License-Identifier: GPL-2.0-only */
/* The paged lifetime table of one address space: create, grow, walk and free.
 *
 * The kernel allocates the table's pages and writes its header, its directory entries and
 * its growth word. The processor writes the records and the allocation words (next, free,
 * allocations, retired). The layout is sublet-format.h's.
 *
 * Plain C over five hooks, so the host test compiles this file unchanged against QEMU's table
 * code. The includer defines them first:
 *   SUBLET_PAGE_NEW()      a zeroed 4 KiB page as unsigned long *, or NULL
 *   SUBLET_PAGE_FREE(p)    give the page back
 *   SUBLET_PA(p)           the physical address of a table page
 *   SUBLET_VA(pa)          the table page at a physical address
 *   SUBLET_WMB()           order the stores before it ahead of the stores after it
 * and READ_ONCE and WRITE_ONCE. Every function returns 0 or -ENOMEM, from <errno.h> on the
 * host and the kernel's errno otherwise. */
#ifndef SUBLET_TABLE_H
#define SUBLET_TABLE_H
#include "sublet-format.h"

static inline unsigned long sublet_table_capacity(const unsigned long *t)
{ return READ_ONCE(t[CV_HDR_CAPACITY]) & ~CV_TABLE_FLAGS; }
static inline unsigned long sublet_table_next(const unsigned long *t)
{ return READ_ONCE(t[CV_HDR_NEXT]); }
static inline unsigned long sublet_table_free_count(const unsigned long *t)
{ return CV_FREE_COUNT(READ_ONCE(t[CV_HDR_FREE])); }
/* Slots an allocation can still take without growth: never-used ones and free ones. A new
 * table has capacity 0 below the first slot, and no never-used slot. */
static inline unsigned long sublet_table_available(const unsigned long *t)
{
    unsigned long capacity = sublet_table_capacity(t), next = sublet_table_next(t);

    return (capacity > next ? capacity - next : 0) + sublet_table_free_count(t);
}
static inline bool sublet_table_nogrow(const unsigned long *t)
{ return READ_ONCE(t[CV_HDR_GROWTH]) & CV_NOGROW; }

/* A root page with the header of an empty table: no record page, capacity 0, next at the
 * first slot. The caller grows it before the processor sees it. */
static unsigned long *sublet_table_create(void)
{
    unsigned long *t = SUBLET_PAGE_NEW();

    if (!t) return NULL;
    t[CV_HDR_CAPACITY] = CV_TABLE_FLAGS;
    t[CV_HDR_NEXT] = CV_TABLE_FIRST_SLOT;
    t[CV_HDR_MAGIC] = CV_TABLE_MAGIC;
    return t;
}

/* The record page of the slots [slot, slot + 256) and the directory pages on its walk, zeroed
 * and entered into the directory. The capacity that names them follows later
 * (sublet_table_publish). A zeroed record page is initialized: its slots lie at or above the
 * never-used slot, which no lifetime names. *pages counts the pages this call allocated. */
static int sublet_table_add_page(unsigned long *t, unsigned long slot, unsigned long *pages)
{
    unsigned long *entry = t + CV_TABLE_DIRECTORY + (slot >> CV_TABLE_ROOT_SHIFT);

    for (unsigned level = 0; level < 3; ++level) {
        unsigned long *page;

        if (!*entry) {
            page = SUBLET_PAGE_NEW();
            if (!page) return -ENOMEM;
            ++*pages;
            WRITE_ONCE(*entry, SUBLET_PA(page));
        } else {
            page = SUBLET_VA(*entry);
        }
        if (level != 2)
            entry = page + ((slot >> (level == 0 ? CV_TABLE_MID_SHIFT : CV_TABLE_LEAF_SHIFT)) &
                            CV_TABLE_DIR_MASK);
    }
    return 0;
}

/* Publish a capacity of record_pages pages, after the pages and directory entries that back
 * them (cap_rev_table_abi.h: pages before capacity). */
static void sublet_table_publish(unsigned long *t, unsigned long record_pages)
{
    SUBLET_WMB();
    WRITE_ONCE(t[CV_HDR_CAPACITY], CV_TABLE_FLAGS | (record_pages * CV_TABLE_PAGE_RECORDS));
}

/* Make at least required slots available. Adds record pages, at least batch of them, never
 * past limit pages, and publishes the capacity once after them. 0 if required slots are
 * available afterwards, else -ENOMEM. *pages counts the pages allocated. Writes no NOGROW:
 * the caller decides it from the result. */
static int sublet_table_grow(unsigned long *t, unsigned long required, unsigned long batch,
                             unsigned long limit, unsigned long *pages)
{
    unsigned long record_pages = sublet_table_capacity(t) / CV_TABLE_PAGE_RECORDS;
    unsigned long before = record_pages, need, want;

    if (sublet_table_available(t) >= required) return 0;
    if (limit > CV_ID_PAGES_MAX) limit = CV_ID_PAGES_MAX;
    need = (required - sublet_table_available(t) + CV_TABLE_PAGE_RECORDS - 1) / CV_TABLE_PAGE_RECORDS;
    want = before + (batch > need ? batch : need);
    if (want > limit) want = limit;
    for (; record_pages < want; ++record_pages)
        if (sublet_table_add_page(t, record_pages * CV_TABLE_PAGE_RECORDS, pages)) break;
    if (record_pages != before)
        sublet_table_publish(t, record_pages);
    return sublet_table_available(t) >= required ? 0 : -ENOMEM;
}

/* Write the growth word: NOGROW tells the processor that the kernel adds no record pages, so
 * an allocation without a slot writes its failure result instead of raising cause 30. The
 * kernel writes only 0 or NOGROW there. */
static void sublet_table_set_nogrow(unsigned long *t, bool nogrow)
{
    WRITE_ONCE(t[CV_HDR_GROWTH], nogrow ? CV_NOGROW : 0UL);
}

/* The record of a slot, or NULL where the table has none: a reserved slot, a never-used one,
 * or one past the published capacity. */
static unsigned long *sublet_table_record(const unsigned long *t, unsigned long slot)
{
    const unsigned long *entry;

    if (slot < CV_TABLE_FIRST_SLOT || slot >= sublet_table_next(t) || slot >= sublet_table_capacity(t))
        return NULL;
    entry = t + CV_TABLE_DIRECTORY + (slot >> CV_TABLE_ROOT_SHIFT);
    for (unsigned level = 0; level < 2; ++level) {
        if (!*entry) return NULL;
        entry = (const unsigned long *)SUBLET_VA(*entry) +
                ((slot >> (level == 0 ? CV_TABLE_MID_SHIFT : CV_TABLE_LEAF_SHIFT)) & CV_TABLE_DIR_MASK);
    }
    if (!*entry) return NULL;
    return (unsigned long *)SUBLET_VA(*entry) + (slot & (CV_TABLE_PAGE_RECORDS - 1)) * CV_TABLE_RECORD_WORDS;
}

/* Free every page of the table. The processor must no longer name it. */
static void sublet_table_free(unsigned long *t)
{
    if (!t) return;
    for (unsigned long i = 0; i < CV_TABLE_ROOT_ENTRIES; ++i) {
        unsigned long *middle;

        if (!t[CV_TABLE_DIRECTORY + i]) continue;
        middle = SUBLET_VA(t[CV_TABLE_DIRECTORY + i]);
        for (unsigned long j = 0; j < CV_TABLE_DIR_ENTRIES; ++j) {
            unsigned long *leaf;

            if (!middle[j]) continue;
            leaf = SUBLET_VA(middle[j]);
            for (unsigned long k = 0; k < CV_TABLE_DIR_ENTRIES; ++k)
                if (leaf[k]) SUBLET_PAGE_FREE(SUBLET_VA(leaf[k]));
            SUBLET_PAGE_FREE(leaf);
        }
        SUBLET_PAGE_FREE(middle);
    }
    SUBLET_PAGE_FREE(t);
}
#endif
