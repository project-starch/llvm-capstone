// SPDX-License-Identifier: GPL-2.0-only
/* The kernel's Sublet code: one lifetime table and one root reference per protected address
 * space, the blocks derived from it, and the .dom loader that builds a protected program.
 *
 * Built in with CONFIG_RISCV_CAPSTONE_TASK, never a module. Nothing protected runs yet: the
 * trap entry (plan v5 step 3) and the loader's entry into the program (step 4) do not exist,
 * so the loader builds a program's address space and refuses before it would return to it. */
#include <linux/module.h>
#include <linux/cpu.h>
#if !defined(CONFIG_RISCV_CAPSTONE_TASK) || defined(MODULE)
#error "The kernel's Sublet code requires built-in CONFIG_RISCV_CAPSTONE_TASK"
#endif
#include <asm/capstone_task.h>
#include <linux/capstone_native_helpers.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/sched/signal.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/list.h>
#include <linux/rbtree.h>
#include <linux/file.h>
#include <asm/csr.h>
#include <asm/io.h>
#include <asm/pgtable.h>
#include "vm-abi.h"
/* The 63-bit capability, id, record and table formats (sublet-format.h) and the paged table
 * (sublet-table.h), over the kernel's page allocator. Table pages are charged to the process's
 * memory cgroup like the block records. */
#define SUBLET_PAGE_NEW() ((unsigned long *)get_zeroed_page(GFP_KERNEL_ACCOUNT))
#define SUBLET_PAGE_FREE(p) free_page((unsigned long)(p))
#define SUBLET_PA(p) virt_to_phys(p)
#define SUBLET_VA(pa) ((unsigned long *)phys_to_virt(pa))
#define SUBLET_WMB() smp_wmb()
#include "sublet-table.h"
/* One rule, shared with the allocator. No #include of its own and only
 * __builtin_clzl and unsigned long, so it is valid kernel C. */
#include "representable.h"

/* Whether the processor keeps bounds exact (QEMU's exact-bounds), from the platform's
 * declaration at boot. */
static bool copyable_exact_bounds;

/* One block: a range of the program's address space with its own lifetime, one level below
 * R. The kernel keeps the program's reference P to it in slot until the program's entry
 * puts it into start register reg (plan v5 step 4). Blocks never overlap. */
struct arena {
    struct rb_node node;
    u64 id;
    unsigned long address, bytes;
    /* The image's read and execute prefix, zero for every other block: a private mapping of
     * the image file that is never writable. */
    unsigned long rx_bytes;
    unsigned long *slot;
    unsigned reg;
};
/* What the loader asks for: a block over [address, address + bytes) whose reference has
 * its cursor at cursor and goes into start register reg. */
struct sublet_grant {
    unsigned long address, bytes, cursor, rx_bytes;
    unsigned reg;
};
struct dom_delegate;
struct context {
    struct list_head link;
    struct mm_struct *mm;
    unsigned long *table;
    phys_addr_t table_pa;   /* the table's physical root, which srevroot names */
    /* Blocks by address, under vm_lock. registered_bytes is the sum of their sizes. */
    struct rb_root arenas;
    unsigned long registered_bytes;
    u64 next_id;
    bool exact_bounds;
    unsigned long node_bytes, node_growths;
    /* R, the one reference over the user range, and the slot root_derive builds a grant in,
     * both in pages of kernel slots that the context owns until its release. */
    struct list_head ref_pages;
    unsigned long *ref_free, *root_slot, *grant_slot;
    /* The image file, writers denied until release. */
    struct file *image;
    /* The loader's metadata and exchange regions, and the resource capsule. */
    unsigned long dom_meta, dom_exchange, dom_exchange_bytes;
    struct dom_delegate *dom_delegate;
};
static DEFINE_MUTEX(vm_lock);
static LIST_HEAD(contexts);

/* Quotas in record pages; zero means the format's limit of 2^31 slots. Read-only after boot,
 * so every context sees one policy. A table starts with one batch and grows by batches
 * (cap_rev_table_abi.h). */
static unsigned long node_initial_pages = CV_TABLE_BATCH_PAGES, node_batch_pages = CV_TABLE_BATCH_PAGES;
static unsigned long node_max_pages;
module_param(node_initial_pages, ulong, 0444);
module_param(node_batch_pages, ulong, 0444);
module_param(node_max_pages, ulong, 0444);

static unsigned long node_capacity(const struct context *c)
{ return sublet_table_capacity(c->table); }
static unsigned long node_available(const struct context *c)
{ return sublet_table_available(c->table); }
static void node_free_table(struct context *c)
{
    sublet_table_free(c->table);
    c->table = NULL;
}

/* ---- The growth protocol (cap_rev_table_abi.h) ----
 *
 * The processor raises cause 30 when an allocation finds no slot and the kernel can still grow
 * the table. Every growth attempt is one call of ensure_nodes(c, required), and only
 * ensure_nodes writes NOGROW. It adds record pages when fewer than required slots are available,
 * at least one batch, published before the larger capacity. Then it sets NOGROW exactly when
 * the table still has fewer than required slots, so an answer to cause 30 always makes
 * progress: afterwards the table has a slot or NOGROW is set, and the resumed instruction writes
 * its failure result instead of trapping again. A call whose requirement is met clears NOGROW.
 * The table never reclaims a slot: a release puts it on the free list at its next generation.
 * Counted in nogrow_sets, nogrow_clears and table_growths, for diagnosis. */
static unsigned long nogrow_sets, nogrow_clears, table_growths;
module_param(nogrow_sets, ulong, 0444);
module_param(nogrow_clears, ulong, 0444);
module_param(table_growths, ulong, 0444);

/* Caller holds vm_lock. 0 if required slots are available afterwards. */
static int ensure_nodes(struct context *c, unsigned long required)
{
    unsigned long pages = 0, before = node_capacity(c);
    bool nogrow;
    int error;

    if (!required || required > CV_ID_RECORDS_MAX) return -ENOMEM;
    error = sublet_table_grow(c->table, required, node_batch_pages, node_max_pages ?: CV_ID_PAGES_MAX, &pages);
    c->node_bytes += pages * PAGE_SIZE;
    if (node_capacity(c) != before) {
        ++c->node_growths;
        ++table_growths;
    }
    nogrow = node_available(c) < required;
    if (nogrow != sublet_table_nogrow(c->table)) {
        sublet_table_set_nogrow(c->table, nogrow);
        if (nogrow) ++nogrow_sets; else ++nogrow_clears;
    }
    return error;
}

/* Name the context's table in srevroot for one window of capability instructions. */
static unsigned long select_root(struct context *c)
{
    unsigned long old;
    asm volatile("csrr %0, 0x5c1\ncsrw 0x5c1, %1"
                 : "=&r"(old) : "r"(c->table_pa) : "memory");
    return old;
}
/* Outside a window S names no table. Plan v5 step 3 makes this write back the value that
 * select_root found, once a protected program's own table can be the one in srevroot. */
static void restore_root(unsigned long root)
{
    (void)root;
    asm volatile("csrw 0x5c1, zero" : : : "memory");
}

/* ---- R and the grants derived from it ----
 *
 * The kernel is the outermost allocator of an address space. When it creates a context's table
 * it mints one reference, R, over the user range with CSMINT, the one instruction that makes a
 * reference from numbers. Each grant is a block of depth 1: CDERIVE makes the program's
 * reference P from R over the whole arena. No grant is minted on its own.
 *
 * The kernel computes with plain addresses, so a reference lives either in a kernel slot or, for
 * the length of one asm block, in t0 and t1. The compiler never sees a reference: every block
 * loads what it needs from a slot, lists its registers as clobbered and clears them before it
 * ends, because a tagged register that the compiler reuses as scratch would carry its tag into
 * the next plain store. Every block runs inside a select_root window with interrupts and
 * preemption off.
 *
 * A trap to M from S, such as the machine timer on a processor without Sstc, saves and restores
 * the S registers in the firmware, and a firmware that does that with plain loads and stores
 * returns a reference in a register as its address, without its tag. So every block is one RUN:
 * it loads each reference with LDC and consumes it completely in straight-line code, with no
 * branch between the LDC and the last use, inside one 64-byte block (CV_RUN). QEMU takes
 * interrupts only between translation blocks, which end at a branch and at a page boundary, so
 * no trap falls inside a run there. Choices between variants are made on integers before a run,
 * and checks of the result after it, by loading the stored reference again.
 *
 * Encodings as in capstone-qemu tests/address-space-root/root.S. The selector of LCC is a
 * register NUMBER in the rs2 field. */
#define CV_RUN ".balign 64\n"
#define CV_LDC(rd, base) ".insn i 0x5b, 3, " rd ", " base ", 0\n"
#define CV_STC(src, base) ".insn s 0x5b, 4, " src ", 0(" base ")\n"
#define CV_INC(rd, src, offset) ".insn r 0x5b, 1, 0x0c, " rd ", " src ", " offset "\n"
#define CV_LCC(rd, src, selector) ".insn r 0x5b, 1, 4, " rd ", " src ", " selector "\n"
#define CV_LCC_TYPE "x1"
#define CV_LCC_ID "x8"
#define CV_DERIVE(rd, parent, size) ".insn r 0x5b, 1, 0x51, " rd ", " parent ", " size "\n"
#define CV_KILL(object) ".insn r 0x5b, 1, 0x52, zero, " object ", zero\n"
#define CV_MINT(rd, slot, desc) ".insn r 0x5b, 1, 0x50, " rd ", " slot ", " desc "\n"
/* What LCC selector 1 answers for a tagged value of the 63-bit format; an untagged value
 * answers 7. A check of this answer is a tag test. */
#define CV_TYPE_NONLIN 1UL

/* R covers [0, TASK_SIZE): exactly the addresses add() can register, since it admits a range
 * only if access_ok() does. A base of 0 is a multiple of every granule and TASK_SIZE is a power
 * of two (2^38 under Sv39), so the bounds hold the range exactly. */
#define CV_ROOT_BASE 0UL
#define CV_ROOT_END ((unsigned long)TASK_SIZE)

/* A page of 16-byte kernel slots for references, owned by the context until its release. */
struct ref_page {
    struct list_head link;
    unsigned long *slots;
};
#define CV_REF_SLOT_WORDS (16 / sizeof(unsigned long))
#define CV_REF_PAGE_SLOTS (PAGE_SIZE / 16)

/* A free 16-byte slot of the context, untagged and zero, or NULL. */
static unsigned long *ref_slot_get(struct context *c)
{
    unsigned long *slot;

    if (!c->ref_free) {
        struct ref_page *page = kmalloc(sizeof(*page), GFP_KERNEL_ACCOUNT);
        if (!page) return NULL;
        page->slots = (void *)get_zeroed_page(GFP_KERNEL_ACCOUNT);
        if (!page->slots) { kfree(page); return NULL; }
        list_add_tail(&page->link, &c->ref_pages);

        for (unsigned i = CV_REF_PAGE_SLOTS; i--;) {
            slot = page->slots + i * CV_REF_SLOT_WORDS;
            slot[0] = (unsigned long)c->ref_free;
            c->ref_free = slot;
        }
    }

    slot = c->ref_free;
    c->ref_free = (unsigned long *)slot[0];
    slot[0] = 0;
    return slot;
}

/* Every slot page goes back to Linux. A plain store to a granule clears its tag, so R leaves
 * no tag in a freed page. */
static void ref_pages_free(struct context *c)
{
    struct ref_page *page, *next;

    list_for_each_entry_safe(page, next, &c->ref_pages, link) {
        for (unsigned i = 0; i < CV_REF_PAGE_SLOTS; ++i)
            WRITE_ONCE(page->slots[i * CV_REF_SLOT_WORDS], 0UL);
        list_del(&page->link);
        free_page((unsigned long)page->slots);
        kfree(page);
    }
    c->ref_free = c->root_slot = c->grant_slot = NULL;
}

/* What a slot holds: the type LCC answers (7 for anything untagged) and, for a reference, the
 * id of its lifetime. Inside a select_root window. LCC selector 8 traps on an untagged value,
 * so the id is read in a second run, only after the first found a reference. */
static void ref_read(unsigned long *slot, unsigned long *type, unsigned long *id)
{
    unsigned long slot_type, slot_id = 0;

    asm volatile(CV_RUN
                 CV_LDC("t0", "%[slot]")
                 CV_LCC("%[type]", "t0", CV_LCC_TYPE)
                 "li t0, 0\n"
                 : [type] "=r"(slot_type)
                 : [slot] "r"(slot)
                 : "t0", "memory");
    if (slot_type == CV_TYPE_NONLIN)
        asm volatile(CV_RUN
                     CV_LDC("t0", "%[slot]")
                     CV_LCC("%[id]", "t0", CV_LCC_ID)
                     "li t0, 0\n"
                     : [id] "=r"(slot_id)
                     : [slot] "r"(slot)
                     : "t0", "memory");

    *type = slot_type;
    *id = slot_id;
}

/* The type LCC answers for the content of a slot. One run. */
static unsigned long ref_type(unsigned long *slot)
{
    unsigned long type;

    asm volatile(CV_RUN
                 CV_LDC("t0", "%[slot]")
                 CV_LCC("%[type]", "t0", CV_LCC_TYPE)
                 "li t0, 0\n"
                 : [type] "=r"(type)
                 : [slot] "r"(slot)
                 : "t0", "memory");
    return type;
}

/* Grants whose CDERIVE gave its failure result (root_derive: ENOMEM to the program). */
static unsigned long derive_failures;
module_param(derive_failures, ulong, 0444);

/* Room for one grant's record before its CDERIVE runs, so that the CDERIVE does not trap in
 * S. A growth attempt that falls short sets NOGROW (ensure_nodes), and the CDERIVE then writes
 * its failure result, which root_derive turns into ENOMEM. */
static int grant_nodes(struct context *c)
{
    return ensure_nodes(c, 1);
}

/* End the block whose fresh reference the grant slot holds, if it holds one: CREVOKE(P).
 * Inside a select_root window. The rollback of root_derive, while the kernel still holds the
 * only reference. */
static void grant_unwind(unsigned long *fresh)
{
    if (ref_type(fresh) == CV_TYPE_NONLIN)
        asm volatile(CV_RUN
                     CV_LDC("t0", "%[fresh]")
                     CV_KILL("t0")
                     "li t0, 0\n"
                     : : [fresh] "r"(fresh) : "t0", "memory");
}

/* The two runs that make a block. Run P: t0 is R, moved to the grant's base, and t1 is P, which
 * CDERIVE makes from t0 over the whole arena, one level below R, into the grant slot. Run
 * cursor: P moved to the requested cursor, into the target slot and slot2. Separate runs, so
 * that a CDERIVE that gives its failure result (an untagged 0 under NOGROW) is seen by an
 * integer check before anything uses it: a CINCOFFSET from an untagged value would trap in S. */
#define CV_DERIVE_P_RUN()                                                                        \
    asm volatile(CV_RUN                                                                          \
                 CV_LDC("t0", "%[root]")                                                         \
                 CV_INC("t0", "t0", "%[base]")                                                   \
                 CV_DERIVE("t1", "t0", "%[bytes]")                                               \
                 CV_STC("t1", "%[fresh]")                                                        \
                 "li t0, 0\n"                                                                    \
                 "li t1, 0\n"                                                                    \
                 : : [root] "r"(c->root_slot), [base] "r"(base), [bytes] "r"(bytes),             \
                     [fresh] "r"(fresh)                                                          \
                 : "t0", "t1", "memory")
#define CV_CURSOR_RUN()                                                                          \
    asm volatile(CV_RUN                                                                          \
                 CV_LDC("t0", "%[fresh]")                                                        \
                 CV_INC("t0", "t0", "%[cursor]")                                                 \
                 CV_STC("t0", "%[slot]")                                                         \
                 CV_STC("t0", "%[slot2]")                                                        \
                 "li t0, 0\n"                                                                    \
                 : : [fresh] "r"(fresh), [cursor] "r"(cursor), [slot] "r"(slot),                 \
                     [slot2] "r"(slot2)                                                          \
                 : "t0", "memory")

/* Make the block for the grant r: P, a child of R over the whole arena (depth 1), into slot
 * (and slot2 where the same reference goes twice, or NULL), with the cursor r names. 0 on
 * success. -ENOMEM if the CDERIVE gave its failure result (no record changed). -EFAULT if the
 * format cannot hold the cursor for this range; the rollback then ends P with CREVOKE. On a
 * failure every slot is cleared. */
static int root_derive(struct context *c, unsigned long *slot, unsigned long *slot2,
                       const struct sublet_grant *r)
{
    unsigned long flags, old, *fresh = c->grant_slot;
    unsigned long base = r->address - CV_ROOT_BASE, bytes = r->bytes;
    unsigned long cursor = r->cursor - r->address;
    int status = 0;

    if (!slot2) slot2 = slot;
    preempt_disable(); local_irq_save(flags);
    old = select_root(c);

    CV_DERIVE_P_RUN();
    if (ref_type(fresh) != CV_TYPE_NONLIN) {
        status = -ENOMEM;
    } else {
        CV_CURSOR_RUN();
        if (ref_type(slot) != CV_TYPE_NONLIN) {
            status = -EFAULT;
            grant_unwind(fresh);
        }
    }

    restore_root(old);
    local_irq_restore(flags); preempt_enable();
    WRITE_ONCE(fresh[0], 0UL); WRITE_ONCE(fresh[1], 0UL);
    if (status) {
        if (status == -ENOMEM)
            ++derive_failures;
        WRITE_ONCE(slot[0], 0UL); WRITE_ONCE(slot[1], 0UL);
        WRITE_ONCE(slot2[0], 0UL); WRITE_ONCE(slot2[1], 0UL);
    }
    return status;
}

/* Create R once, when the context's table exists. CSMINT with the descriptor CV_DESC_ROOT
 * makes R alone, a root at depth 0, writes it into its slot and answers its id. The kernel
 * then reads R back the way every later use does and checks the answer against the table: the
 * id LCC reads must be the one CSMINT answered, and its record must hold it at depth 0. A
 * processor without supervisor references, or a table this code misreads, is refused here.
 * The grant slot, which root_derive uses, is taken here too. */
static int root_create(struct context *c)
{
    unsigned long flags, old, desc[CV_DESC_WORDS], answer, type, id, *record;
    unsigned long *slot;

    if (!c->exact_bounds && !capstone_repr_ok(CV_ROOT_BASE, CV_ROOT_END - CV_ROOT_BASE))
        return -EOPNOTSUPP;
    /* The format holds cursors below 2^38 only: a larger user range (Sv48, Sv57) has
     * addresses that no reference can carry. */
    if (CV_ROOT_END > CV_USER_ADDRESS_END)
        return -EOPNOTSUPP;
    slot = ref_slot_get(c);
    if (!slot) return -ENOMEM;
    c->grant_slot = ref_slot_get(c);
    if (!c->grant_slot) return -ENOMEM;

    desc[0] = CV_ROOT_BASE;
    desc[1] = CV_ROOT_END;
    desc[2] = CV_DESC_ROOT;
    desc[3] = CV_ROOT_BASE;
    desc[4] = 0;
    desc[5] = 0;
    preempt_disable(); local_irq_save(flags);
    old = select_root(c);
    asm volatile(CV_MINT("%0", "%1", "%2") : "=&r"(answer) : "r"(slot), "r"(desc) : "memory");
    ref_read(slot, &type, &id);
    restore_root(old);
    local_irq_restore(flags); preempt_enable();

    record = type == CV_TYPE_NONLIN ? sublet_table_record(c->table, CV_ID_SLOT(id)) : NULL;
    if (!record || !answer || answer != id || !CV_REC_LIVE(record, id) || CV_REC_DEPTH(record) != 0) {
        pr_emerg("capstone_vm: R reads back as type %lu, id 0x%lx (%s), and CSMINT answered 0x%lx: "
                 "no supervisor references, or an unknown table format\n",
                 type, id, record ? "recorded" : "no record", answer);
        WRITE_ONCE(slot[0], 0UL); WRITE_ONCE(slot[1], 0UL);
        return -EOPNOTSUPP;
    }
    c->root_slot = slot;
    return 0;
}

/* Give a slot back to the context's free list, cleared: a plain store clears its tag. */
static void ref_slot_put(struct context *c, unsigned long *slot)
{
    WRITE_ONCE(slot[1], 0UL);
    WRITE_ONCE(slot[0], (unsigned long)c->ref_free);
    c->ref_free = slot;
}

/* No block reaches the topmost page below R's end. A cursor cannot be 2^38, so an object that
 * ended there could not hold a tagged pointer one past its end, which C allows every object. */
#define CV_BLOCK_END (CV_ROOT_END - PAGE_SIZE)

static struct arena *arena_below(struct context *c, unsigned long end)
{
    struct rb_node *n = c->arenas.rb_node;
    struct arena *best = NULL;
    while (n) {
        struct arena *a = rb_entry(n, struct arena, node);
        if (a->address < end) { best = a; n = n->rb_right; }
        else n = n->rb_left;
    }
    return best;
}
/* Does [address, address + bytes) overlap a block? */
static bool arena_overlaps(struct context *c, unsigned long address, unsigned long bytes)
{
    struct arena *a = arena_below(c, address + bytes);
    return a && a->address + a->bytes > address;
}
static void arena_insert(struct context *c, struct arena *a)
{
    struct rb_node **link = &c->arenas.rb_node, *parent = NULL;
    while (*link) {
        parent = *link;
        link = a->address < rb_entry(parent, struct arena, node)->address ?
               &parent->rb_left : &parent->rb_right;
    }
    rb_link_node(&a->node, parent, link);
    rb_insert_color(&a->node, &c->arenas);
}

/* Is every page of the grant mapped the way its block requires? The image's text is a
 * private mapping of the image file whose writers the loader denied, readable and executable
 * and never writable, so its bytes stay the bytes validated. The image's data is private,
 * readable and writable and never executable. Every other block is private anonymous memory.
 * Read, write and execute are the page table's: Sublet checks bounds and lifetime only. */
static int grant_mapped(struct context *c, const struct sublet_grant *g)
{
    unsigned long end = g->address + g->bytes, rx_end = g->address + g->rx_bytes;
    int error = 0;

    mmap_read_lock(c->mm);
    for (unsigned long at = g->address; at < end && !error;) {
        struct vm_area_struct *v = find_vma(c->mm, at);
        bool rx = at < rx_end;

        if (!v || v->vm_start > at || (v->vm_flags & VM_SHARED))
            error = -EACCES;
        else if (rx)
            error = !v->vm_file || !c->image || file_inode(v->vm_file) != file_inode(c->image) ||
                    (v->vm_flags & VM_WRITE) ||
                    (v->vm_flags & (VM_READ | VM_EXEC)) != (VM_READ | VM_EXEC) ? -EACCES : 0;
        else if (g->rx_bytes)
            error = (v->vm_flags & (VM_READ | VM_WRITE)) != (VM_READ | VM_WRITE) ||
                    (v->vm_flags & VM_EXEC) ? -EACCES : 0;
        else
            error = v->vm_file ? -EACCES : 0;
        at = min(min(end, v ? v->vm_end : end), rx ? rx_end : ULONG_MAX);
    }
    mmap_read_unlock(c->mm);
    return error;
}

/* Make the block for the grant g: check its range and its mappings, derive P from R and record
 * the block. Caller holds vm_lock. */
static int add(struct context *c, const struct sublet_grant *g)
{
    struct arena *a;
    int error;

    if (!g->bytes || g->bytes > CV_MAX_EAGER_BYTES || ((g->address | g->bytes) & ~PAGE_MASK) ||
        (!c->exact_bounds && !capstone_repr_ok(g->address, g->bytes)) ||
        g->address > CV_BLOCK_END || g->bytes > CV_BLOCK_END - g->address ||
        !access_ok((void __user *)g->address, g->bytes) || g->reg > 31 ||
        g->cursor < g->address || g->cursor > g->address + g->bytes ||
        (g->rx_bytes && ((g->rx_bytes & ~PAGE_MASK) || g->rx_bytes >= g->bytes)))
        return -EINVAL;
    if (arena_overlaps(c, g->address, g->bytes)) return -EINVAL;
    if (c->registered_bytes + g->bytes > CV_MAX_BYTES) return -ENOMEM;
    error = grant_mapped(c, g);
    if (error) return error;
    /* A block id names one block for the context's whole life. */
    if (c->next_id == U64_MAX) return -ENOSPC;
    if (grant_nodes(c)) return -ENOMEM;

    a = kzalloc(sizeof(*a), GFP_KERNEL_ACCOUNT);
    if (!a) return -ENOMEM;
    a->slot = ref_slot_get(c);
    if (!a->slot) { kfree(a); return -ENOMEM; }
    error = root_derive(c, a->slot, NULL, g);
    if (error) {
        ref_slot_put(c, a->slot);
        kfree(a);
        return error;
    }
    a->id = ++c->next_id;
    a->address = g->address; a->bytes = g->bytes; a->rx_bytes = g->rx_bytes; a->reg = g->reg;
    arena_insert(c, a);
    c->registered_bytes += a->bytes;
    return 0;
}

/* Does the image's filesystem keep one inode per file, so the writers Linux counts on it
 * (i_writecount) are all the file's writers? Named by the superblock's type, which the
 * filesystem cannot misreport. */
static bool one_inode_per_file(struct file *file)
{
    static const char *const fs[] = {"tmpfs", "ext2", "ext3", "ext4", "btrfs",
                                     "xfs", "erofs", "squashfs"};
    const char *name = file_inode(file)->i_sb->s_type->name;
    for (unsigned i = 0; i < ARRAY_SIZE(fs); ++i)
        if (!strcmp(name, fs[i])) return true;
    return false;
}

/* A context for the current address space: its table with node_initial_pages record pages,
 * published before the processor sees it, and R. One context per mm. */
static int context_create(struct context **out)
{
    struct context *c, *existing;
    unsigned long pages = 0;
    int error;

    if (num_online_cpus() != 1 || !current->mm || PAGE_SIZE != 4096)
        return -EOPNOTSUPP;
    if (!node_initial_pages || node_initial_pages > CV_ID_PAGES_MAX ||
        !node_batch_pages || node_batch_pages > CV_ID_PAGES_MAX ||
        node_max_pages > CV_ID_PAGES_MAX ||
        (node_max_pages && node_max_pages < node_initial_pages)) return -EINVAL;
    c = kvzalloc(sizeof(*c), GFP_KERNEL);
    if (!c) return -ENOMEM;
    INIT_LIST_HEAD(&c->ref_pages);
    c->arenas = RB_ROOT;
    c->exact_bounds = copyable_exact_bounds;

    error = -ENOMEM;
    c->table = sublet_table_create();
    if (!c->table) goto fail;
    c->table_pa = virt_to_phys(c->table);
    if (sublet_table_grow(c->table, 1, node_initial_pages, node_initial_pages, &pages)) goto fail;
    c->node_bytes = (pages + 1) * PAGE_SIZE;
    error = -EOPNOTSUPP;
    if (node_available(c) != node_capacity(c) - CV_TABLE_FIRST_SLOT) goto fail;
    error = root_create(c);
    if (error) goto fail;

    c->mm = current->mm; mmgrab(c->mm);
    mutex_lock(&vm_lock);
    list_for_each_entry(existing, &contexts, link)
        if (existing->mm == c->mm) {
            mutex_unlock(&vm_lock);
            mmdrop(c->mm);
            error = -EBUSY;
            goto fail;
        }
    list_add(&c->link, &contexts);
    mutex_unlock(&vm_lock);
    *out = c;
    return 0;
fail:
    ref_pages_free(c);
    node_free_table(c);
    kvfree(c);
    return error;
}

/* Release a context: its blocks, its slots (R among them), its table and its image. Freeing
 * the table ends every lifetime the context's processor table held. */
static void context_destroy(struct context *c)
{
    mutex_lock(&vm_lock);
    list_del(&c->link);
    for (struct rb_node *n = rb_first(&c->arenas), *next; n; n = next) {
        struct arena *a = rb_entry(n, struct arena, node);
        next = rb_next(n);
        rb_erase(&a->node, &c->arenas);
        kfree(a);
    }
    c->registered_bytes = 0;
    ref_pages_free(c);
    node_free_table(c);
    mutex_unlock(&vm_lock);
    if (c->image) { allow_write_access(c->image); fput(c->image); }
    mmdrop(c->mm);
    kvfree(c);
}
#include "binfmt-dom.h"

/* The platform declares the capability format and the tag-aware memory contract in its
 * device tree (capstone,unpinned-v1), and the page-tag code accepts only id-bits 63. The
 * kernel's Sublet code needs both, one possible hart, and no transparent huge pages. */
static int __init init(void)
{
    bool exact;
    int error;

    if (num_possible_cpus() != 1 || IS_ENABLED(CONFIG_TRANSPARENT_HUGEPAGE))
        return -EOPNOTSUPP;
    error = capstone_unpinned_config(CAP_REV_SUBLET_ID_BITS, &exact);
    if (error) {
        pr_info("capstone_vm: no 63-bit Sublet platform declared: disabled\n");
        return error;
    }
    copyable_exact_bounds = exact;
    pr_info("capstone_vm: 63-bit Sublet format, exact bounds %d, dom_binfmt %d\n", exact, dom_binfmt);
    if (dom_binfmt) insert_binfmt(&capstone_dom);
    return 0;
}
module_init(init);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("The kernel's Sublet code: lifetime tables, root references and the .dom loader");
