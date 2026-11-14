/* SPDX-License-Identifier: GPL-2.0-only */
/* The Sublet capability, lifetime id, record and table formats, in one place.
 *
 * One format exists: the 63-bit Sublet format of the paper's Section 5, which QEMU
 * implements as x-capstone-ids=63 (capstone-qemu target/riscv/cap_rev_table_abi.h and
 * cap_compress.c). The kernel names every field of a capability, an id, a record and the
 * table header through the macros below. The macro names match the 63-bit section of the
 * adapter's cv-format.h (llvm-capstone-address-space-root, tag pre-trap-route-2026-11-14),
 * so the mapping code that plan v5 step 6 copies from there compiles against this file.
 *
 * The values come from cap_rev_table_abi.h, which the kernel build takes from the QEMU tree
 * it runs on. The static assertions below refuse a header of another format.
 *
 * The file is plain C with no kernel include, so the host test
 * (capstone/runtime/virtual/tests/sublet-format) compiles it unchanged. An includer that is
 * not the kernel defines READ_ONCE first. */
#ifndef SUBLET_FORMAT_H
#define SUBLET_FORMAT_H
#include "cap_rev_table_abi.h"

#if !defined(CAP_REV_SUBLET_ID_BITS) || CAP_REV_SUBLET_ID_BITS != 63
#error "sublet-format.h needs the 63-bit cap_rev_table_abi.h of capstone-qemu c1985bb529 or later"
#endif

/* ---- The capability, 128 bits: generation 127:96, slot 95:65, bounds 64:38, address 37:0.
 * As the two words a capability store writes:
 *   lo = bounds[25:0] << 38 | address[37:0]
 *   hi = generation[31:0] << 32 | slot[30:0] << 1 | bounds[26]
 * The cursor is the zero extension of the 38-bit address. The format stores no type and no
 * rights: Sublet checks bounds and lifetime, and the page table decides read, write and
 * execute. The lifetime id a capability names is generation << 31 | slot. */
#define CV_CAP_ADDR_BITS 38
#define CV_CAP_ADDR_MASK ((1UL << CV_CAP_ADDR_BITS) - 1)
#define CV_CAP_BOUNDS_BITS 27
#define CV_CAP_BOUNDS_LO_BITS 26
#define CV_CAP_SLOT_SHIFT 1
#define CV_CAP_GEN_SHIFT 32
#define CV_CAP_ADDRESS(lo) ((lo) & CV_CAP_ADDR_MASK)
#define CV_CAP_BOUNDS(lo, hi) (((lo) >> CV_CAP_ADDR_BITS) | (((hi) & 1UL) << CV_CAP_BOUNDS_LO_BITS))
#define CV_CAP_SLOT(hi) (((hi) >> CV_CAP_SLOT_SHIFT) & CAP_REV_SUBLET_SLOT_MASK)
#define CV_CAP_GENERATION(hi) ((hi) >> CV_CAP_GEN_SHIFT)
#define CV_CAP_ID(hi) (CV_CAP_GENERATION(hi) << CAP_REV_SUBLET_SLOT_BITS | CV_CAP_SLOT(hi))

/* ---- The lifetime id: generation[62:31] | slot[30:0]. */
#define CV_ID_RECORDS_MAX CAP_REV_SUBLET_SLOTS
#define CV_ID_SLOT(id) ((id) & CAP_REV_SUBLET_SLOT_MASK)
#define CV_ID_GENERATION(id) ((id) >> CAP_REV_SUBLET_SLOT_BITS)
#define CV_ID(generation, slot) ((unsigned long)(generation) << CAP_REV_SUBLET_SLOT_BITS | (slot))

/* ---- The record, 128 bits: generation 127:96, previous 95:65, next 64:34, depth 33:3,
 * reserved zero 2:0. As words, next straddles the two:
 *   word 0 = next[29:0] << 34 | depth << 3
 *   word 1 = generation << 32 | previous << 1 | next[30]
 * A record has no parent and no VALID, LINEAR or FREE flag. It holds exactly the id {its
 * generation, its slot}. A free record holds its next generation, which no reference names.
 * A retired slot holds generation 2^32 - 1, which the processor never issues, so no id with
 * it is live. */
#define CV_REC_NEXT(r) ((READ_ONCE((r)[0]) >> CAP_REV_SUBLET_NEXT_LO_SHIFT) | \
                        ((READ_ONCE((r)[1]) & 1UL) << CAP_REV_SUBLET_NEXT_LO_BITS))
#define CV_REC_DEPTH(r) ((READ_ONCE((r)[0]) >> CAP_REV_SUBLET_DEPTH_SHIFT) & CAP_REV_SUBLET_FIELD_MASK)
#define CV_REC_PREV(r) ((READ_ONCE((r)[1]) >> CAP_REV_SUBLET_PREV_SHIFT) & CAP_REV_SUBLET_FIELD_MASK)
#define CV_REC_GENERATION(r) ((READ_ONCE((r)[1]) >> CAP_REV_SUBLET_GEN_SHIFT) & CAP_REV_SUBLET_GEN_MASK)
#define CV_REC_LIVE(r, id) (CV_ID_GENERATION(id) != CAP_REV_SUBLET_GEN_RETIRED && \
                            CV_REC_GENERATION(r) == CV_ID_GENERATION(id))
#define CV_REC_HOLDS(r, id) CV_REC_LIVE(r, id)

/* ---- The paged table: a root page with the header and 32 root entries, two directory
 * levels of 512 entries and record pages of 256 records of 16 bytes. A slot splits 5:9:9:8
 * into root entry (slot >> 26), directory indices (slot >> 17 and slot >> 8, 9 bits each)
 * and record (slot & 255). Slots 0 and 1 are reserved, so the first slot an allocation takes
 * is 2. Header words (u64, word index = byte offset / 8): capacity | flags, next never-used
 * slot, free head | free count << 32, allocations, magic, retired, growth (NOGROW). */
#define CV_TABLE_DIRECTORY (CAP_REV_TABLE_DIRECTORY / 8)   /* the first root entry, as a word index */
#define CV_TABLE_ROOT_ENTRIES CAP_REV_SUBLET_ROOT_ENTRIES
#define CV_TABLE_ROOT_SHIFT 26
#define CV_TABLE_MID_SHIFT 17
#define CV_TABLE_LEAF_SHIFT 8
#define CV_TABLE_DIR_MASK 511UL
#define CV_TABLE_DIR_ENTRIES 512UL
#define CV_TABLE_PAGE_RECORDS 256UL                        /* CAP_REV_TABLE_PAGE_NODES */
#define CV_TABLE_RECORD_WORDS (CAP_REV_TABLE_RECORD / 8)
#define CV_TABLE_FLAGS CAP_REV_TABLE_FLAGS                 /* RECYCLE | PAGED, in the capacity word */
#define CV_TABLE_MAGIC CAP_REV_TABLE_MAGIC
#define CV_TABLE_FIRST_SLOT 2UL
#define CV_TABLE_BATCH_PAGES ((unsigned long)CAP_REV_SUBLET_BATCH_PAGES)
#define CV_NOGROW CAP_REV_SUBLET_NOGROW
#define CV_HDR_CAPACITY 0
#define CV_HDR_NEXT 1
#define CV_HDR_FREE 2                                      /* head[31:0] | count[63:32] */
#define CV_HDR_ALLOCATIONS 3
#define CV_HDR_MAGIC 4
#define CV_HDR_RETIRED 5
#define CV_HDR_GROWTH 6
#define CV_FREE_COUNT(word) ((word) >> 32)
#define CV_ID_PAGES_MAX (CV_ID_RECORDS_MAX / CV_TABLE_PAGE_RECORDS)

/* ---- CSMINT rd, rs1 = slot, rs2 = descriptor: six aligned words. w0 base, w1 end, w2 the
 * only value the format accepts (0x10f: every right and NONLIN, which the 63-bit capability
 * does not store), w3 cursor (base <= cursor <= end), w4 a second slot or 0, w5 reserved 0.
 * CSMINT makes one record at depth 0 and writes its id to rd. The kernel mints exactly one
 * reference with it per address space: R, over the user range. */
#define CV_DESC_WORDS 6
#define CV_DESC_ROOT 0x10fUL

/* Sv39: the user half ends at 2^38, and a cursor is the zero extension of 38 bits, so 2^38 is
 * the first user address no reference can carry. */
#define CV_USER_ADDRESS_END (1UL << CV_CAP_ADDR_BITS)

_Static_assert(CAP_REV_TABLE_HDR_CAPACITY == CV_HDR_CAPACITY * 8 && CAP_REV_TABLE_HDR_NEXT == CV_HDR_NEXT * 8 &&
               CAP_REV_TABLE_HDR_FREE == CV_HDR_FREE * 8 && CAP_REV_TABLE_HDR_MAGIC == CV_HDR_MAGIC * 8 &&
               CAP_REV_TABLE_HDR_ALLOCATIONS == CV_HDR_ALLOCATIONS * 8 &&
               CAP_REV_TABLE_HDR_RETIRED == CV_HDR_RETIRED * 8 && CAP_REV_TABLE_HDR_GROWTH == CV_HDR_GROWTH * 8,
               "the Sublet root header");
_Static_assert(CAP_REV_SUBLET_ROOT_ENTRIES << CV_TABLE_ROOT_SHIFT == CAP_REV_SUBLET_SLOTS &&
               CAP_REV_TABLE_PAGE_NODES == CV_TABLE_PAGE_RECORDS && CAP_REV_TABLE_RECORD == 16 &&
               CAP_REV_TABLE_PAGE_SIZE == CV_TABLE_PAGE_RECORDS * CAP_REV_TABLE_RECORD &&
               CAP_REV_TABLE_DIRECTORY + 8 * CAP_REV_SUBLET_ROOT_ENTRIES <= CAP_REV_TABLE_PAGE_SIZE,
               "the Sublet walk: 5:9:9:8, 32 root entries in the root page");
_Static_assert(CAP_REV_SUBLET_NEXT_LO_SHIFT + CAP_REV_SUBLET_NEXT_LO_BITS == 64 &&
               CAP_REV_SUBLET_DEPTH_SHIFT + CAP_REV_SUBLET_SLOT_BITS == CAP_REV_SUBLET_NEXT_LO_SHIFT &&
               CAP_REV_SUBLET_GEN_SHIFT + 32 == 64 && CAP_REV_SUBLET_PREV_SHIFT == 1 &&
               CAP_REV_SUBLET_GEN_RETIRED == CAP_REV_SUBLET_GEN_MASK,
               "the Sublet record: next straddling the words, depth at 33:3");
_Static_assert(CV_CAP_BOUNDS_LO_BITS + CV_CAP_ADDR_BITS == 64 &&
               CV_CAP_BOUNDS_BITS - CV_CAP_BOUNDS_LO_BITS == CV_CAP_SLOT_SHIFT &&
               CV_CAP_SLOT_SHIFT + CAP_REV_SUBLET_SLOT_BITS == CV_CAP_GEN_SHIFT,
               "the Sublet capability: bounds 64:38, slot 95:65, generation 127:96");
#endif
