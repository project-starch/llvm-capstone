/* The lifetime-id and table formats that the module reads and writes, in one place (block A, step F4).
 *
 * Every constant of an id or of a record, every field position, the root entries, the walk, the header
 * offsets and the growth protocol live here, selected by the BUILD defines CAPSTONE_VM_IDS (the
 * processor's lifetime-id width), CAPSTONE_VM_FLAT_TABLE and CAPSTONE_VM_INDEX_BITS. capstone_vm.c
 * names records and ids only through the macros below, so a new layout is a change of one section of
 * this file. The values come from cap_rev_table_abi.h, which build-adapter.sh copies from the
 * capstone-qemu tree beside this file; the SUBLET constants exist only in a header of the 63-bit
 * format (capstone-qemu c1985bb529 or later), so only the CAPSTONE_VM_IDS=63 section names them.
 *
 * What every section defines:
 *   CV_ID_RECORDS_MAX       slots a table may publish (the namespace)
 *   CV_ID_SLOT(id)          the slot (record index) an id names
 *   CV_ID_GENERATION(id)    the generation an id names (0 in a format without generations)
 *   CV_REF_FORMAT           1 if the module can read this format's records (root_grants needs it)
 *   CV_REC_NEXT(r)          the slot of the next record in the preorder list, 0 for none
 *   CV_REC_DEPTH(r)         the record's depth in the allocation tree
 *   CV_REC_LIVE(r, id)      the record holds exactly the live lifetime id: no other id, no free or
 *                           retired slot, no linear record (a linear record is never an operand)
 *   CV_REC_HOLDS(r, id)     the record holds the lifetime id, linear or not: R's own record, which
 *                           the older formats make linear (a root of CSMINT); root_create asks it
 *   CV_MINT_ANSWERS_ROOT    1 if CSMINT answers R's own id and makes R alone (at depth 0); 0 if it
 *                           makes a private parent and R below it and answers the parent's id, whose
 *                           record's parent field CV_REC_PARENT(r) reads
 *   CV_TABLE_*              the paged table: root entries, walk, header words, flags, magic
 *   CV_TABLE_GROWTH         1 if the table has the growth word (NOGROW) and the growth protocol of
 *                           cause 30 and the pressure escape (cap_rev_table_abi.h, SUBLET tables)
 *   CV_TABLE_COLLECTS       1 if the processor reclaims retired records (CSRUNV action 3)
 *   CV_LEGACY_GRANTS        1 if the format still has CSMINT grants with rights and CSRETIRE, the
 *                           path without root grants (root_grants=0)
 *   CV_USER_ADDRESS_END     the first user address that no reference of the program can carry as its
 *                           cursor (2^38 in every format on Sv39); R ends there, and no mapping block
 *                           comes closer to it than a page (capstone_vm.c, CV_BLOCK_END)
 */
#ifndef CAPSTONE_VM_FORMAT_H
#define CAPSTONE_VM_FORMAT_H
#include "cap_rev_table_abi.h"

/* The processor's lifetime-id width: 30 (every bitstream so far, the namespace's 2^30), 55 (QEMU's
 * x-capstone-ids=55: a 25-bit slot, so at most 2^25 records, and a table that publishes more is refused
 * at CSRUNV) or 63 (QEMU's x-capstone-ids=63, the Sublet format of the paper's Section 5). A BUILD
 * define, like CAPSTONE_VM_FLAT_TABLE: the kit picks it with the processor (make CAPSTONE_VM_IDS=63,
 * module/Makefile). The 55 and 63 limits come from the ABI header the build copies beside this file, so
 * a build against a header without them does not compile. The 30 build compiles to the same object as
 * before this header existed. */
#ifndef CAPSTONE_VM_IDS
#define CAPSTONE_VM_IDS 30
#endif
#if CAPSTONE_VM_IDS != 30 && CAPSTONE_VM_IDS != 55 && CAPSTONE_VM_IDS != 63
#error "CAPSTONE_VM_IDS: 30, 55 or 63"
#endif
#if CAPSTONE_VM_IDS == 55 && !defined(CAP_REV_WIDE_RECORDS)
#error "CAPSTONE_VM_IDS=55 needs the ids-55 cap_rev_table_abi.h beside the module"
#endif
#if CAPSTONE_VM_IDS == 63 && (!defined(CAP_REV_SUBLET_ID_BITS) || CAP_REV_SUBLET_ID_BITS != 63)
#error "CAPSTONE_VM_IDS=63 needs the Sublet cap_rev_table_abi.h of capstone-qemu c1985bb529 or later beside the module"
#endif

/* vp2 implements only the FLAT table format. CAP_REV_TABLE_FLAGS is RECYCLE|PAGED, so the default
 * header publishes a PAGED table, which the rev-node unit refuses (status 3, read as 29 at the context's
 * first allocation). In flat mode the table is ONE physically contiguous CV_FLAT_RECORDS*16 allocation,
 * records indexed straight off the root as QEMU lays them out (cap_rev_table.c: pa = root + id*16),
 * capacity fixed at CV_FLAT_RECORDS and never grown -- the hardware saturates capacity and next-unused
 * at 65,536 regardless, and a fixed capacity makes the pressure rule (capacity - next + free_count <=
 * RESERVE) read identically on both kits from the first allocation. Default off, so the paged kit is
 * unchanged and the two are told apart by the recorded define rather than by inspection.
 *
 * A COMPILE-TIME define, not a module parameter, and the reason is specific: vp2 has neither scapctl
 * (0x5c0) nor urevavail (0xcc0), and a csrr of either is an ILLEGAL INSTRUCTION in every mode. With a
 * runtime parameter both encodings sit in the one binary and "the branch is not taken" is the only
 * guarantee available -- which cannot be checked by inspecting the module and depends on the optimiser
 * not keeping an `asm volatile` under `if (0)`. With a define the flat build does not CONTAIN them,
 * which an objdump can verify and the paged build positively controls. For a trap that kills a context
 * at creation, absence beats an untaken branch. */
#ifdef CAPSTONE_VM_FLAT_TABLE
#define node_table_flat 1
#else
#define node_table_flat 0
#endif
#if CAPSTONE_VM_IDS == 63 && node_table_flat
#error "CAPSTONE_VM_IDS=63: the module keeps a paged table (a flat SUBLET table never grows)"
#endif
/* The flat table's index width, which is the BITSTREAM's (isa.md: a virtual id is {generation, index}
 * inside 30 bits). 16 for vp2 and the batch bitstream c721c1a2a; 18 for the bitstream after it (the RTL
 * lane's option A, docs/plans/2026-10-23-next-bitstream-scope.md: {generation[29:18], index[17:0]},
 * 262,144 records, a 4 MiB order-10 table per context). A BUILD define, not a parameter: a module whose
 * width disagrees with the silicon's either refuses ids the hardware hands out or publishes a capacity
 * the hardware saturates below, so the kit picks it with the bitstream. The 16-bit build compiles to the
 * same object as before this define existed; any other width exposes it read-only as index_bits, so a
 * kit can be checked by reading it. */
#ifndef CAPSTONE_VM_INDEX_BITS
#define CAPSTONE_VM_INDEX_BITS 16
#endif
/* 25 is option B's RTL (capstone-ariane rtl-ids55): a 55-bit id {generation[54:25], slot[24:0]}, so a
 * flat table indexes records by the 25-bit slot, in QEMU's wide record format (cap_rev_table_abi.h,
 * CAP_REV_WIDE_*). It exists only with CAPSTONE_VM_IDS=55, and a flat 55 build has no other width: the
 * RTL walks a flat table by slot. */
#if CAPSTONE_VM_INDEX_BITS != 16 && CAPSTONE_VM_INDEX_BITS != 18 && CAPSTONE_VM_INDEX_BITS != 20 && \
    CAPSTONE_VM_INDEX_BITS != 25
#error "CAPSTONE_VM_INDEX_BITS: 16 (vp2, batch), 18 (bitstreams 2/3), 20 (ids20, {generation[29:20], index[19:0]}) or 25 (ids55)"
#endif
#if (CAPSTONE_VM_INDEX_BITS == 25) != (CAPSTONE_VM_IDS == 55 && defined(CAPSTONE_VM_FLAT_TABLE))
#error "CAPSTONE_VM_INDEX_BITS=25 is the flat CAPSTONE_VM_IDS=55 build's width, and that build's only one"
#endif

/* ---- The paged table, every format: a root page with the header and the root entries, two directory
 * levels of 512 entries and record pages of 256 records of 16 bytes. A slot splits into root entry
 * (slot >> 26), directory indices (slot >> 17, slot >> 8, 9 bits each) and record (slot & 255): 4:9:9:8
 * for 30-bit ids (16 root entries), 5:9:9:8 for 31-bit SUBLET slots (32 root entries). Header words
 * (u64, word index = byte offset / 8): capacity | flags, next never-used slot, free head | free count
 * << 32, allocations, magic, retired, and in a SUBLET table the growth word (NOGROW). */
#define CV_TABLE_DIRECTORY (CAP_REV_TABLE_DIRECTORY / 8)   /* the first root entry, as a word index */
#define CV_TABLE_ROOT_SHIFT 26
#define CV_TABLE_MID_SHIFT 17
#define CV_TABLE_LEAF_SHIFT 8
#define CV_TABLE_DIR_MASK 511UL
#define CV_TABLE_PAGE_RECORDS 256UL                        /* CAP_REV_TABLE_PAGE_NODES */
#define CV_TABLE_RECORD_WORDS (CAP_REV_TABLE_RECORD / 8)
#define CV_TABLE_FLAGS CAP_REV_TABLE_FLAGS                 /* RECYCLE | PAGED, in the capacity word */
#define CV_TABLE_MAGIC CAP_REV_TABLE_MAGIC
#define CV_HDR_CAPACITY 0
#define CV_HDR_NEXT 1
#define CV_HDR_FREE 2                                      /* head[31:0] | count[63:32] */
#define CV_HDR_ALLOCATIONS 3
#define CV_HDR_MAGIC 4
#define CV_HDR_RETIRED 5
#define CV_HDR_GROWTH 6                                    /* SUBLET tables only */
#define CV_FREE_COUNT(word) ((word) >> 32)

#if CAPSTONE_VM_IDS == 63
/* ---- x-capstone-ids=63, the Sublet format of the paper's Section 5 (capstone-qemu c1985bb529):
 * id = generation[62:31] | slot[30:0]. The record, 128 bits: generation 127:96, previous 95:65, next
 * 64:34, depth 33:3, reserved zero 2:0, so next straddles the two words:
 *   word 0 = next[29:0] << 34 | depth << 3
 *   word 1 = generation << 32 | previous << 1 | next[30]
 * No parent, no VALID, LINEAR or FREE flag: a record holds exactly the id {its generation, its slot}. A
 * free record holds its next generation, which no reference names, and a retired slot the generation
 * 2^32 - 1, which is never issued (so no id with it is live, CV_REC_LIVE). CSMINT makes R alone, at
 * depth 0, and answers R's id. 32 root entries (5:9:9:8), at most 2^31 slots (0 and 1 reserved). A
 * reference's cursor is the zero extension of 38 bits, so R = [0, 2^38) is the largest range R can
 * cover. */
#define CV_ID_RECORDS_MAX CAP_REV_SUBLET_SLOTS
#define CV_TABLE_ROOT_ENTRIES CAP_REV_SUBLET_ROOT_ENTRIES
#define CV_REF_FORMAT 1
#define CV_ID_SLOT(id) ((id) & CAP_REV_SUBLET_SLOT_MASK)
#define CV_ID_GENERATION(id) ((id) >> CAP_REV_SUBLET_SLOT_BITS)
#define CV_REC_NEXT(r) ((READ_ONCE((r)[0]) >> CAP_REV_SUBLET_NEXT_LO_SHIFT) | \
                        ((READ_ONCE((r)[1]) & 1UL) << CAP_REV_SUBLET_NEXT_LO_BITS))
#define CV_REC_DEPTH(r) ((READ_ONCE((r)[0]) >> CAP_REV_SUBLET_DEPTH_SHIFT) & CAP_REV_SUBLET_FIELD_MASK)
#define CV_REC_GENERATION(r) ((READ_ONCE((r)[1]) >> CAP_REV_SUBLET_GEN_SHIFT) & CAP_REV_SUBLET_GEN_MASK)
#define CV_REC_LIVE(r, id) (CV_ID_GENERATION(id) != CAP_REV_SUBLET_GEN_RETIRED && \
                            CV_REC_GENERATION(r) == CV_ID_GENERATION(id))
#define CV_REC_HOLDS(r, id) CV_REC_LIVE(r, id)
#define CV_MINT_ANSWERS_ROOT 1
#define CV_TABLE_GROWTH 1
#define CV_NOGROW CAP_REV_SUBLET_NOGROW
#define CV_TABLE_BATCH_PAGES CAP_REV_SUBLET_BATCH_PAGES   /* T starts with one batch */
#define CV_TABLE_COLLECTS 0
#define CV_LEGACY_GRANTS 0
_Static_assert(CAP_REV_TABLE_HDR_CAPACITY == CV_HDR_CAPACITY * 8 && CAP_REV_TABLE_HDR_NEXT == CV_HDR_NEXT * 8 &&
               CAP_REV_TABLE_HDR_FREE == CV_HDR_FREE * 8 && CAP_REV_TABLE_HDR_MAGIC == CV_HDR_MAGIC * 8 &&
               CAP_REV_TABLE_HDR_RETIRED == CV_HDR_RETIRED * 8 && CAP_REV_TABLE_HDR_GROWTH == CV_HDR_GROWTH * 8,
               "the SUBLET root header");
_Static_assert(CAP_REV_SUBLET_ROOT_ENTRIES << CV_TABLE_ROOT_SHIFT == CAP_REV_SUBLET_SLOTS &&
               CAP_REV_SUBLET_NEXT_LO_SHIFT + CAP_REV_SUBLET_NEXT_LO_BITS == 64 &&
               CAP_REV_SUBLET_DEPTH_SHIFT + CAP_REV_SUBLET_SLOT_BITS == CAP_REV_SUBLET_NEXT_LO_SHIFT &&
               CAP_REV_SUBLET_GEN_SHIFT + 32 == 64 && CAP_REV_SUBLET_PREV_SHIFT == 1 &&
               CAP_REV_SUBLET_GEN_RETIRED == CAP_REV_SUBLET_GEN_MASK,
               "the SUBLET record: 5:9:9:8 walk, next straddling the words, depth at 33:3");
#else
/* ---- The 30-bit and 55-bit formats: a record carries VALID, LINEAR and FREE flags in word 1 at 62:60
 * and its parent's index; CSMINT makes a private parent and R below it and answers the parent's id. */
#define CV_TABLE_ROOT_ENTRIES CAP_REV_TABLE_ROOT_ENTRIES
#define CV_REC_STATE(r) ((READ_ONCE((r)[1]) >> CAP_REV_TABLE_FLAGS_SHIFT) & \
                         (CAP_REV_TABLE_VALID | CAP_REV_TABLE_LINEAR | CAP_REV_TABLE_FREE))
#define CV_MINT_ANSWERS_ROOT 0
#define CV_TABLE_GROWTH 0
#define CV_NOGROW 0UL
#define CV_TABLE_BATCH_PAGES 16UL
#define CV_TABLE_COLLECTS 1
#define CV_LEGACY_GRANTS 1
#if CAPSTONE_VM_IDS == 55
/* x-capstone-ids=55: id = generation[54:25] | slot[24:0]; word 0 = depth[13:0] at 63:50 | next[49:25] |
 * prev[24:0]; word 1 = flags[62:60] | generation[59:30] | parent[29:5] | depth[18:14] at 4:0. */
#define CV_ID_RECORDS_MAX CAP_REV_WIDE_RECORDS
#define CV_REF_FORMAT 1
#define CV_ID_SLOT(id) ((id) & CAP_REV_WIDE_INDEX_MASK)
#define CV_ID_GENERATION(id) ((id) >> CAP_REV_WIDE_INDEX_BITS)
#define CV_REC_NEXT(r) ((READ_ONCE((r)[0]) >> CAP_REV_WIDE_INDEX_BITS) & CAP_REV_WIDE_INDEX_MASK)
#define CV_REC_DEPTH(r) (((READ_ONCE((r)[0]) >> CAP_REV_WIDE_DEPTH_LOW_SHIFT) & \
                          ((1UL << CAP_REV_WIDE_DEPTH_LOW_BITS) - 1)) | \
                         ((READ_ONCE((r)[1]) & CAP_REV_WIDE_DEPTH_HIGH_MASK) << CAP_REV_WIDE_DEPTH_LOW_BITS))
#define CV_REC_GENERATION(r) ((READ_ONCE((r)[1]) >> CAP_REV_WIDE_GEN_SHIFT) & CAP_REV_WIDE_GEN_MASK)
#define CV_REC_PARENT(r) ((READ_ONCE((r)[1]) >> CAP_REV_WIDE_PARENT_SHIFT) & CAP_REV_WIDE_INDEX_MASK)
#elif node_table_flat && CAPSTONE_VM_INDEX_BITS == 16
/* The flat 16-bit table: id = generation[29:16] | index[15:0]; word 0 = next[63:32] | prev[31:0]; word 1
 * = flags[62:60] | generation[59:46] | parent[45:30] | depth[29:0]. */
#define CV_ID_RECORDS_MAX CAP_REV_TABLE_NAMESPACE_SIZE
#define CV_REF_FORMAT 1
#define CV_ID_SLOT(id) ((id) & CAP_REV_ID_INDEX_MASK)
#define CV_ID_GENERATION(id) ((id) >> CAP_REV_ID_GEN_SHIFT)
#define CV_REC_NEXT(r) (READ_ONCE((r)[0]) >> 32)
#define CV_REC_DEPTH(r) (READ_ONCE((r)[1]) & CAP_REV_TABLE_FIELD_MASK)
#define CV_REC_GENERATION(r) ((READ_ONCE((r)[1]) >> CAP_REV_TABLE_GEN_SHIFT) & CAP_REV_TABLE_GEN_MASK)
#define CV_REC_PARENT(r) ((READ_ONCE((r)[1]) >> CAP_REV_TABLE_PARENT_SHIFT) & CAP_REV_TABLE_FLAT_PARENT_MASK)
#elif !node_table_flat
/* The paged table at 30 bits: an id is its record's index, and no record has a generation. word 0 =
 * next[63:32] | prev[31:0]; word 1 = flags[62:60] | parent[59:30] | depth[29:0]. */
#define CV_ID_RECORDS_MAX CAP_REV_TABLE_NAMESPACE_SIZE
#define CV_REF_FORMAT 1
#define CV_ID_SLOT(id) (id)
#define CV_ID_GENERATION(id) 0UL
#define CV_REC_NEXT(r) (READ_ONCE((r)[0]) >> 32)
#define CV_REC_DEPTH(r) (READ_ONCE((r)[1]) & CAP_REV_TABLE_FIELD_MASK)
#define CV_REC_GENERATION(r) 0UL
#define CV_REC_PARENT(r) ((READ_ONCE((r)[1]) >> CAP_REV_TABLE_PARENT_SHIFT) & CAP_REV_TABLE_FIELD_MASK)
#else
/* The flat formats of 18 and 20 index bits exist only on bitstreams, which have no supervisor references:
 * the module reads none of their records, and init() refuses root_grants there. */
#define CV_ID_RECORDS_MAX CAP_REV_TABLE_NAMESPACE_SIZE
#define CV_REF_FORMAT 0
#define CV_ID_SLOT(id) (id)
#define CV_ID_GENERATION(id) 0UL
#define CV_REC_NEXT(r) 0UL
#define CV_REC_DEPTH(r) 0UL
#define CV_REC_GENERATION(r) 0UL
#define CV_REC_PARENT(r) 0UL
#endif
#define CV_REC_LIVE(r, id) (CV_REC_STATE(r) == CAP_REV_TABLE_VALID && \
                            CV_REC_GENERATION(r) == CV_ID_GENERATION(id))
#define CV_REC_HOLDS(r, id) ((CV_REC_STATE(r) & (CAP_REV_TABLE_VALID | CAP_REV_TABLE_FREE)) == \
                             CAP_REV_TABLE_VALID && CV_REC_GENERATION(r) == CV_ID_GENERATION(id))
#endif

#define CV_ID_PAGES_MAX (CV_ID_RECORDS_MAX / CV_TABLE_PAGE_RECORDS)
/* Sv39: the user half ends at 2^38. The older formats keep a cursor as the sign extension of 39 bits,
 * which 2^38 is not; the 63-bit format keeps it as the zero extension of 38 bits (capstone-qemu cap.h,
 * CAP_SUBLET_ADDR_BITS), so 2^38 is not representable there either. */
#define CV_USER_ADDRESS_END (1UL << 38)
#define CV_FLAT_RECORDS_MAX (1UL << CAPSTONE_VM_INDEX_BITS)
#if CV_FLAT_RECORDS_MAX > CV_ID_RECORDS_MAX
#error "the flat table's index width exceeds the lifetime id's"
#endif
#endif
