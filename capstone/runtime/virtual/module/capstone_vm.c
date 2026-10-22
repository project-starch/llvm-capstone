// SPDX-License-Identifier: GPL-2.0-only
/* Trusted OS adapter: one shared-mm namespace and explicit C contexts. */
#include <linux/module.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/sched/signal.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/list.h>
#include <linux/wait.h>
#include <linux/ktime.h>
#include <linux/log2.h>
#include <asm/csr.h>
#include <asm/io.h>
#include <asm/pgtable.h>
#include "wire.h"
#include "cap_rev_table_abi.h"
/* One rule, shared with the launcher and the allocator. No #include of its own and only
 * __builtin_clzl and unsigned long, so it is valid kernel C. */
#include "representable.h"
_Static_assert(CV_NODE_RESERVE == CAP_REV_TABLE_RESERVE, "processor reserve ABI");

struct arena {
    unsigned long address, bytes, ancestor;
    struct page **pages;
    unsigned count;
    bool lazy;
    /* S-21: the ancestor was already CSRETIREd at CV_THREAD_CREATE, to make a child's frame
     * unreachable by the domain before its first entry. A second retire is not an error, but the
     * non-destroying path reports one, so it must be skipped rather than tolerated. */
    bool ancestor_retired;
};
struct context;
struct cv_thread {
    struct list_head link;
    struct context *context;
    u64 id;
    unsigned long *frame;
    unsigned long frame_pa;
    unsigned long frame_user;
    struct page *frame_page;
    bool user_frame;
    bool started, reply_cap, terminal;
    unsigned long event;
    /* S-24: the same kind-4 fault resolved repeatedly without changing anything. The RTL's walker
     * faults on a clear A bit and never sets it, so a PTE this module leaves untouched can re-fault
     * for ever while resolve() keeps reporting success. Bounded and reported rather than spun. */
    unsigned long last_fault_addr, last_fault_cause;
    unsigned fault_repeat;
    /* A fault was resolved and the context has not been resumed since: the resume flushes the
     * resolved address on the resuming hart first (resume_sfences). */
    bool sfence_pending;
};
struct context {
    struct list_head link;
    struct mm_struct *mm;
    unsigned long *frame, *table;
    struct cv_thread *main_thread;
    struct list_head threads;
    u64 next_thread;
    struct arena arenas[CV_MAX_ARENAS];
    struct cv_stats stats;
    u64 next_id, ids[CV_MAX_ARENAS];
    bool exact_bounds;
    /* The flat format has no retired word: QEMU reads root+40 only under t->paged, and in a flat
     * table that offset is record id 2. CSRETIRE answers 0 on success and 1 on rejection and never
     * faults, so counting the zeros here is the same quantity the paged word held, not an estimate.
     * Reset by the collection that reclaims the retired records. */
    unsigned long retired_flat;
    /* Batch item 6 (generations in virtual ids, isa.md 2026-10-20): on a bitstream that recycles under
     * the virtual flag, a successful CSRETIRE also PUSHES the record onto the table's free list, so it
     * is counted in free_count at once and must not also count as retired-awaiting-collection. Counted
     * per retire from free_count before and after, so one module reads correctly on both bitstreams:
     * on one without the item free_count never moves and pushed_flat stays 0. */
    unsigned long pushed_flat;
    wait_queue_head_t remap_wait;
    struct arena *remapping;
    struct page **remap_pages;
    u64 remap_thread;
    unsigned long remap_bytes;
};
static DEFINE_MUTEX(vm_lock);
/* Quotas are in base pages; zero means the 30-bit ISA limit. Parameters are
 * read-only after module load so concurrent contexts see a stable policy. */
static unsigned long node_initial_pages = 4, node_batch_pages = 16;
static unsigned long node_max_pages;
module_param(node_initial_pages, ulong, 0444);
module_param(node_batch_pages, ulong, 0444);
module_param(node_max_pages, ulong, 0444);
/* vp2: the first virtual bitstream with a lifetime table keeps the RTL's 16-bit node index, so a
 * FLAT table holds at most 65534 usable ids (0 and 1 are reserved). Setting this to 65536 makes
 * the module refuse to allocate any id above 16 bits, through the ordinary ENOMEM/resource path
 * rather than by truncating one. 0 means no cap beyond the namespace size, which is the
 * pre-vp2 behaviour and the default, so a build that does not set it is unchanged.
 * The 30-bit widening arrives with the paged table; this is the flat table's ceiling.
 * Every measured workload is far below it: the runtime gates peak at 6372 live nodes and the
 * ports' high-water mark is 13054. */
static unsigned long node_max_records;
module_param(node_max_records, ulong, 0444);
MODULE_PARM_DESC(node_max_records,
    "vp2: cap the flat lifetime table at this many records (65536 for a 16-bit node index); 0 = no cap");
/* vp2 implements only the FLAT table format. CAP_REV_TABLE_FLAGS is RECYCLE|PAGED, so the
 * default header publishes a PAGED table, which the rev-node unit refuses (status 3, read as 29 at
 * the context's first allocation). In flat mode the table is ONE physically contiguous
 * CV_FLAT_RECORDS*16 allocation, records indexed straight off the root as QEMU lays them out
 * (cap_rev_table.c: pa = root + id*16), capacity fixed at CV_FLAT_RECORDS and never grown -- the
 * hardware saturates capacity and next-unused at 65,536 regardless, and a fixed capacity makes the
 * pressure rule (capacity - next + free_count <= RESERVE) read identically on both kits from the
 * first allocation. Default off, so the paged kit is unchanged and the two are told apart by the
 * recorded define rather than by inspection.
 *
 * A COMPILE-TIME define, not a module parameter, and the reason is specific: vp2 has neither
 * scapctl (0x5c0) nor urevavail (0xcc0), and a csrr of either is an ILLEGAL INSTRUCTION in every
 * mode. With a runtime parameter both encodings sit in the one binary and "the branch is not
 * taken" is the only guarantee available -- which cannot be checked by inspecting the module and
 * depends on the optimiser not keeping an `asm volatile` under `if (0)`. With a define the flat
 * build does not CONTAIN them, which an objdump can verify and the paged build positively
 * controls. For a trap that kills a context at creation, absence beats an untaken branch. */
#ifdef CAPSTONE_VM_FLAT_TABLE
#define node_table_flat 1
#else
#define node_table_flat 0
#endif
/* S-24: how many times the SAME (address, cause) may be "resolved" without changing anything
 * before the run is failed instead of retried. Generous, because a resolve followed by a quantum
 * stop is legitimately repeated by the launcher's idempotent CV_RESOLVE; a hardware fault the
 * module cannot clear repeats without bound, which is the case this catches. */
#define CV_FAULT_REPEAT_MAX 64U
/* The flat table's index width, which is the BITSTREAM's (isa.md: a virtual id is
 * {generation, index} inside 30 bits). 16 for vp2 and the batch bitstream c721c1a2a; 18 for the
 * bitstream after it (the RTL lane's option A, docs/plans/2026-10-23-next-bitstream-scope.md:
 * {generation[29:18], index[17:0]}, 262,144 records, a 4 MiB order-10 table per context). A BUILD
 * define, not a parameter: a module whose width disagrees with the silicon's either refuses ids the
 * hardware hands out or publishes a capacity the hardware saturates below, so the kit picks it
 * with the bitstream. The 16-bit build compiles to the same object as before this define existed;
 * any other width exposes it read-only as index_bits, so a kit can be checked by reading it. */
#ifndef CAPSTONE_VM_INDEX_BITS
#define CAPSTONE_VM_INDEX_BITS 16
#endif
#if CAPSTONE_VM_INDEX_BITS != 16 && CAPSTONE_VM_INDEX_BITS != 18
#error "CAPSTONE_VM_INDEX_BITS: 16 (vp2, batch) or 18 (option A)"
#endif
#define CV_FLAT_RECORDS_MAX (1UL << CAPSTONE_VM_INDEX_BITS)
#if CAPSTONE_VM_INDEX_BITS != 16
static unsigned long index_bits = CAPSTONE_VM_INDEX_BITS;
module_param(index_bits, ulong, 0444);
#endif
#ifdef CAPSTONE_VM_FLAT_TABLE
/* The flat table's record count. The ruling's 65,536 is the MAXIMUM, not merely a default:
 * silicon saturates capacity and next-unused there, so a larger value is refused -- at INSMOD,
 * in init() below, not at first open: the parameter is read-only after load, so there is nothing
 * a later check could catch that load time cannot, and a module that loads and then fails every
 * open reports the error as whatever the opener makes of -EINVAL. Lower values exist for testing
 * -- a collection row needs both arms to collect, and at the default that means driving them to
 * ~65,278 identities, which would exercise the ceiling in a row meant to be independent of it.
 * Read-only after load, so the allocation order computed from it is stable for the module's
 * lifetime. The board kit and the prediction kit load with this UNSET; the recorded value comes
 * from the module's own published capacity (table[0] below, via node_capacity() into
 * stats.node_capacity), never from the invocation. */
static unsigned long node_flat_records = CV_FLAT_RECORDS_MAX;
module_param(node_flat_records, ulong, 0444);
#if CAPSTONE_VM_INDEX_BITS == 16
MODULE_PARM_DESC(node_flat_records,
    "vp2 flat table records (default and maximum 65536; a larger value is refused at insmod)");
#else
MODULE_PARM_DESC(node_flat_records,
    "flat table records (default and maximum 2^index_bits; a larger value is refused at insmod)");
#endif
#define CV_FLAT_RECORDS node_flat_records
#else
/* No such parameter in the paged build: it never allocates a flat table, so the knob would be a
 * knob that does nothing, and `insmod node_flat_records=N` should fail as an unknown parameter
 * rather than be accepted and ignored. The constant is kept because the flat-only branches at
 * :465, :1203 and :1227 still have to compile here. */
#define CV_FLAT_RECORDS CV_FLAT_RECORDS_MAX
#endif
#define CV_FLAT_BYTES (CV_FLAT_RECORDS * CAP_REV_TABLE_RECORD)
/* S-23 (2026-10-20): the first id a context's table allocates. On vp2 a CPMP entry tracks its
 * capability's revnode id, and a node id carries no NAMESPACE identity: a context's table is a
 * separate namespace whose ids (generation 0 in both) cannot be told from the caller's physical
 * ones, so a broadcast from one clears a tracker of the other (the operative defect, corrected at
 * dev 735205b29f79; the CPMP's index-only compare is a second, separate collapse). A CSRETIRE in
 * retire()'s window broadcasts the retiring ancestor's VIRTUAL index; when that index equals the
 * PHYSICAL node of a caller entry -- on sql0, slot 3's region 14, the kernel's RAM, a LINEAR piece
 * the monitor carved at boot from a pool whose ids start at 3 (board lane sql0-frs, 2026-10-20) --
 * the entry is cleared and the very next fetch through it is refused: the FRTY right after the
 * CSRETIRE, once per sql0 session, since every fresh table repeats the same allocation order. No
 * placement of an instruction can heal it, because the refused fetch IS the instruction after the
 * CSRETIRE. So the module never hands out a virtual index a physical one can share: allocation
 * starts at a floor above the physical namespace's ids (bounded from source in the low tens; nothing
 * on vp2 exports the count, so the floor is generous rather than asserted). The records below the floor stay zero: an all-zero
 * record is invalid, depth 0, and terminates any walk that reaches it, and neither the bump
 * pointer nor CSNODE allocates below next. The paged (QEMU) build keeps 2; QEMU tracks each
 * namespace separately and has no such collision. A test table smaller than four floors gets a
 * quarter of its records as the floor, so exhaustion rows still have room. */
#define CV_NODE_FLOOR_MAX 1024UL
#if node_table_flat
#define CV_NODE_FIRST (CV_FLAT_RECORDS / 4 < CV_NODE_FLOOR_MAX ? \
                       (CV_FLAT_RECORDS / 4 < 2 ? 2UL : CV_FLAT_RECORDS / 4) : CV_NODE_FLOOR_MAX)
#else
#define CV_NODE_FIRST 2UL
#endif

/* S-25's narrow pin: on a fault, take the REST of the arena rather than pin_ahead_pages.
 *
 * Deliberately a one-line widening of pin_ahead's existing limit rather than a new helper. That
 * loop is already batched, already deadline-aware via cv_step_stop, already validates anon and
 * mapcount per page and NULLs what it could not take, and already stops at the VMA edge and at the
 * first already-pinned page -- so it is idempotent across repeated faults and needs no "done" flag
 * on struct arena. Avoiding that field also avoids touching the POSITIONAL struct literal in add(),
 * where a new member silently shifts `lazy`.
 *
 * Why this shape is enough: an arena's first access is enframe's plain byte read of the slot
 * prefix, near the arena's start, so pinning FORWARD from the faulting page covers the pages a
 * later memcpy stc could reach. It does not pin backward, so it is "the rest of the arena from
 * here", not literally the whole arena -- stated because the difference matters if a first access
 * ever lands late in an arena.
 *
 * Default OFF. The last module change that altered behaviour by default regressed every virtual
 * context on silicon while QEMU passed it 34/41, so this ships inert: one boot proves the code is
 * harmless when off, and the arm that sets it tests the pin. */
static bool arena_pin;
module_param(arena_pin, bool, 0644);

/* Pages a lazy-arena fault pins, from the faulting page on, clipped to the
 * arena, the VMA and the first page already pinned. 1 pins only the page. */
static unsigned long pin_ahead_pages = 1;
module_param(pin_ahead_pages, ulong, 0444);

/* The retired count, from wherever this format keeps it. */
/* The CSMINT descriptor, v7 (Q-23's ruling). Six words, 8-aligned:
 *   w0 base
 *   w1 end
 *   w2 perms in bits 0-3, bit 8 set = NONLIN, clear = LINEAR
 *   w3 cursor, base <= cursor <= end or the mint refuses with 29
 *   w4 a SECOND slot address, 0 for none; same checks as the first, and non-zero
 *      together with LINEAR refuses with 29
 *   w5 reserved, must be 0
 * One rev node however many slots are written, which is why this replaces the old
 * load/delin/scc/store round trip rather than a second mint: two mints would make two
 * nodes and revoking one would leave the other live. Trusted S performs no capability
 * memory access, so the module holds no ldc/stc at all. */
#define CV_DESC_WORDS 6
#define CV_DESC_NONLIN 0x100UL

static unsigned long node_retired(const struct context *c)
{ return node_table_flat ? c->retired_flat - c->pushed_flat : c->table[5]; }

static unsigned long node_capacity(const struct context *c)
{ return c->table[0] & ~CAP_REV_TABLE_FLAGS; }
/* Slot 1 word 0 is {2'b0, next[29:0], 2'b0, prev[29:0]} = {free_count, free_head}. */
static unsigned long node_free_count(const struct context *c)
{ return READ_ONCE(c->table[2]) >> 32; }
static unsigned long node_available(const struct context *c)
{ return node_capacity(c) - c->table[1] + node_free_count(c); }

/* The flat table is one physically contiguous allocation: select_root() publishes
 * virt_to_phys(c->table) into CSR 0x5c1, and QEMU's flat preflight requires the whole
 * capacity*16 extent to be range_ok from that physical root, so vmalloc will not do.
 * __GFP_RETRY_MAYFAIL tries hard without inviting the OOM killer; failing here is a clean
 * ENOMEM at context creation, which is strictly better than the paged path's failure
 * part-way through a running domain. */
static unsigned long *node_new_flat_table(struct context *c)
{
    unsigned long *p = (void *)__get_free_pages(GFP_KERNEL | __GFP_ZERO | __GFP_RETRY_MAYFAIL,
                                                get_order(CV_FLAT_BYTES));
    if (p) c->stats.node_bytes += CV_FLAT_BYTES;
    return p;
}

static unsigned long *node_new_page(struct context *c)
{
    unsigned long *p = (void *)get_zeroed_page(GFP_KERNEL);
    if (p) c->stats.node_bytes += PAGE_SIZE;
    return p;
}

/* These are metadata pages owned by the adapter, not application arenas.
 * vm_lock excludes every C execution while a published root grows. */
static int node_add_page(struct context *c, unsigned long id)
{
    unsigned long *entry = c->table + CAP_REV_TABLE_DIRECTORY / 8 + (id >> 26);
    for (unsigned level = 0; level < 3; ++level) {
        unsigned long *page;
        if (!*entry) {
            page = node_new_page(c);
            if (!page) return -ENOMEM;
            WRITE_ONCE(*entry, virt_to_phys(page));
        } else page = phys_to_virt(*entry);
        if (level != 2)
            entry = page + ((id >> (level == 0 ? 17 : 8)) & 511);
    }
    smp_wmb();
    WRITE_ONCE(c->table[0], CAP_REV_TABLE_FLAGS | (id + 256));
    return 0;
}

static int node_grow(struct context *c, unsigned long required)
{
    unsigned long limit = node_max_pages ?: CAP_REV_TABLE_MAX_PAGES;
    unsigned long pages = node_capacity(c) / 256, before = pages;
    unsigned long need, end;
    /* Flat capacity is fixed at creation; node_add_page writes the paged directory, which in a
     * flat table is record space. Exhaustion is the ordinary ENOMEM/resource path. */
    if (node_table_flat) return node_available(c) >= required ? 0 : -ENOMEM;
    if (node_available(c) >= required) return 0;
    need = DIV_ROUND_UP(required - node_available(c), 256UL);
    end = min(limit, pages + max(node_batch_pages, need));
    for (; pages < end; ++pages)
        if (node_add_page(c, pages * 256)) break;
    if (pages != before) ++c->stats.node_growths;
    return node_available(c) >= required ? 0 : -ENOMEM;
}

static void node_free_table(struct context *c)
{
    if (!c->table) return;
    if (node_table_flat) {
        free_pages((unsigned long)c->table, get_order(CV_FLAT_BYTES));
        c->table = NULL;
        return;
    }
    for (unsigned i = 0; i < CAP_REV_TABLE_ROOT_ENTRIES; ++i) {
        unsigned long pa = c->table[CAP_REV_TABLE_DIRECTORY / 8 + i];
        unsigned long *middle;
        if (!pa) continue;
        middle = phys_to_virt(pa);
        for (unsigned j = 0; j < 512; ++j) {
            unsigned long *leaf;
            if (!middle[j]) continue;
            leaf = phys_to_virt(middle[j]);
            for (unsigned k = 0; k < 512; ++k)
                if (leaf[k]) free_page((unsigned long)phys_to_virt(leaf[k]));
            free_page((unsigned long)leaf);
        }
        free_page((unsigned long)middle);
    }
    free_page((unsigned long)c->table);
    c->table = NULL;
}
static LIST_HEAD(contexts);
/* Q-26: THE SUPERVISOR SLOT. vp2 has ONE saved-context slot; the processor refuses a CSRUNV action
 * 0 whenever the slot is used and the request is not a resume of the SAME frame, and that refusal
 * is a cause-29 trap in S -- which took the kernel with it on vp2's boot 2. So the module must
 * never issue an entry the hardware would refuse, which means knowing the limit rather than
 * discovering it.
 *
 * A frame OCCUPIES the slot from its first entry until it terminates, not merely while an ioctl
 * runs: vm_lock already serialises ioctls, but a started frame that escaped on its quantum still
 * holds the slot between calls. `started && !terminal` is exactly that state, and it is the same
 * predicate ensure_nodes() already uses at :479 to find a collection owner.
 *
 * Read-only after load, default 1 (the silicon value, and QEMU's default since 3b8040d5ff6e). No
 * kit sets it; a value above 1 is for a machine that really has more slots. */
static unsigned long supervisor_slots = 1;
module_param(supervisor_slots, ulong, 0444);
MODULE_PARM_DESC(supervisor_slots,
    "saved-context slots the processor has (default 1, as vp2); the module never issues an entry "
    "beyond this many live frames");
static DECLARE_WAIT_QUEUE_HEAD(slot_wait);
/* The wait condition, evaluated WITHOUT vm_lock: it takes the lock itself, because the sleeper
 * released it. A false wake just re-checks. */
static bool slot_free_now(void);

/* Live frames across EVERY context, and one of them if asked. vm_lock is held by every caller. */
static unsigned slot_live(struct cv_thread **holder)
{
    struct context *c;
    struct cv_thread *t;
    unsigned n = 0;
    list_for_each_entry(c, &contexts, link)
        list_for_each_entry(t, &c->threads, link)
            if (t->started && !t->terminal) {
                if (holder && !n) *holder = t;
                ++n;
            }
    return n;
}

/* Evaluated by wait_event_interruptible WITHOUT vm_lock held -- the sleeper released it -- so this
 * takes the lock itself. A spurious wake simply re-checks. */
static bool slot_free_now(void)
{
    bool free;
    if (mutex_lock_interruptible(&vm_lock)) return true;   /* let the sleeper return -EINTR */
    free = slot_live(NULL) < supervisor_slots;
    mutex_unlock(&vm_lock);
    return free;
}
static u64 allocated_total, high_water, collections_total, reclaimed_total;

static unsigned long run(unsigned long frame, unsigned long action);

static struct cv_thread *find_thread(struct context *c, u64 id)
{
    struct cv_thread *t;
    list_for_each_entry(t, &c->threads, link)
        if (t->id == id) return t;
    return NULL;
}

static unsigned thread_count(struct context *c)
{
    unsigned count = 0;
    struct cv_thread *t;
    list_for_each_entry(t, &c->threads, link) ++count;
    return count;
}

static struct arena *find_arena(struct context *c, unsigned long address,
                                unsigned long bytes)
{
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
        if (c->arenas[i].pages && c->arenas[i].address == address &&
            c->arenas[i].bytes == bytes)
            return &c->arenas[i];
    return NULL;
}

static bool arena_has_thread_frame(struct context *c, struct arena *a)
{
    struct cv_thread *t;
    list_for_each_entry(t, &c->threads, link)
        if (t->user_frame && t->frame_user == a->address)
            return true;
    return false;
}

/* A child start frame stays in the owning mm. Pin exactly one page so the
 * QEMU supervisor sees a stable physical frame and collection can include it.
 * Requiring page alignment also prevents a frame from straddling an unpinned
 * page and keeps the frame ABI independently inspectable by the trusted
 * kernel. */
static int pin_thread_frame(struct context *c, unsigned long address,
                            struct cv_thread *t)
{
    struct page *page;
    struct vm_area_struct *v;
    if (!address || (address & (PAGE_SIZE - 1)) ||
        !access_ok((void __user *)address, PAGE_SIZE))
        return -EINVAL;
    mmap_read_lock(c->mm);
    v = find_vma(c->mm, address);
    if (!v || v->vm_start > address || address > v->vm_end - PAGE_SIZE ||
        v->vm_file || (v->vm_flags & VM_SHARED) ||
        !(v->vm_flags & VM_READ) || !(v->vm_flags & VM_WRITE)) {
        mmap_read_unlock(c->mm);
        return -EACCES;
    }
    mmap_read_unlock(c->mm);
    if (pin_user_pages_fast(address, 1, FOLL_WRITE, &page) != 1)
        return -EFAULT;
    if (!page_address(page)) {
        unpin_user_page(page);
        return -EOPNOTSUPP;
    }
    t->frame_page = page;
    t->frame = page_address(page);
    t->frame_pa = page_to_phys(page);
    t->frame_user = address;
    t->user_frame = true;
    return 0;
}

static void drop_thread_frame(struct cv_thread *t)
{
    if (t->user_frame && t->frame_page)
        unpin_user_page(t->frame_page);
    t->frame_page = NULL;
    t->frame = NULL;
    t->frame_pa = 0;
}

static void clear_thread_frame(struct cv_thread *t)
{
    if (!t->frame) return;
    /* The frame contains tagged PCC/GPR slots. Scalar clearing removes the
     * tags before the page is returned to the application's allocator. */
    memset(t->frame, 0, 656);
}

/* Called with mmap_read_lock held. Missing PTEs contain no application tags;
 * swapped/huge mappings are outside this profile and fail closed. */
static int resident_page(struct mm_struct *mm, unsigned long address)
{
    pgd_t *pgd = pgd_offset(mm, address);
    p4d_t *p4d;
    pud_t *pud;
    pmd_t *pmd;
    pte_t *pte, value;
    spinlock_t *lock;
    if (pgd_none(*pgd)) return 0;
    if (pgd_bad(*pgd)) return -EFAULT;
    p4d = p4d_offset(pgd, address);
    if (p4d_none(*p4d)) return 0;
    if (p4d_bad(*p4d)) return -EFAULT;
    pud = pud_offset(p4d, address);
    if (pud_none(*pud)) return 0;
    if (pud_leaf(*pud)) return -EOPNOTSUPP;
    if (pud_bad(*pud)) return -EFAULT;
    pmd = pmd_offset(pud, address);
    if (pmd_none(*pmd)) return 0;
    if (pmd_leaf(*pmd)) return -EOPNOTSUPP;
    if (pmd_bad(*pmd)) return -EFAULT;
    pte = pte_offset_map_lock(mm, pmd, address, &lock);
    if (!pte) return -EFAULT;
    value = READ_ONCE(*pte);
    pte_unmap_unlock(pte, lock);
    if (pte_none(value)) return 0;
    return pte_present(value) ? 1 : -EOPNOTSUPP;
}
static int pin_remaining(struct context *c, struct arena *a)
{
    int error = 0;
    mmap_read_lock(c->mm);
    for (unsigned j = 0; j < a->count; ++j) {
        struct page *page;
        unsigned long address = a->address + j * PAGE_SIZE;
        int resident;
        if (a->pages[j]) continue;
        resident = resident_page(c->mm, address);
        if (!resident) continue;
        if (resident < 0) { error = resident; break; }
        /* Trusted inspection may read PROT_NONE/RO pages, but it never faults
         * in unused pages, changes PTE rights or manufactures fresh backing. */
        if (pin_user_pages_remote(c->mm, address, 1, FOLL_FORCE | FOLL_NOFAULT,
                                  &page, NULL, NULL) != 1) {
            error = -EFAULT; break;
        }
        if (is_zero_pfn(page_to_pfn(page))) {
            unpin_user_page(page); continue;
        }
        if (!PageAnon(page) || page_mapcount(page) != 1) {
            unpin_user_page(page); error = -EACCES; break;
        }
        a->pages[j] = page;
        ++c->stats.pinned_pages;
        c->stats.peak_pages = max(c->stats.peak_pages, c->stats.pinned_pages);
    }
    mmap_read_unlock(c->mm);
    return error;
}

struct collect_list {
    unsigned long *head, *tail, count;
};

static int collect_page(struct collect_list *list, unsigned long pa)
{
    if (!list->tail || list->tail[1] == CAP_REV_COLLECT_ENTRIES) {
        unsigned long *page = (void *)get_zeroed_page(GFP_KERNEL);
        if (!page) return -ENOMEM;
        if (list->tail) list->tail[0] = virt_to_phys(page);
        else list->head = page;
        list->tail = page;
    }
    list->tail[2 + list->tail[1]++] = pa;
    ++list->count;
    return 0;
}

static void collect_free(struct collect_list *list)
{
    unsigned long *page = list->head;
    while (page) {
        unsigned long next = page[0];
        free_page((unsigned long)page);
        page = next ? phys_to_virt(next) : NULL;
    }
}

#if node_table_flat
/* Q-27: vp2's RTL refuses CSRUNV action 3 at execute as an illegal instruction (rs2 > 2), so on
 * silicon a collection is an S-mode trap; QEMU implements it. Nor can the module collect in
 * software on bf3db3175 (ruled 2026-10-18 from the RTL lane's record, not read here): under the
 * virtual-context flag a node id carries no generation, so
 * re-listing a retired index would make every stale copy naming it live again at the next CSMINT,
 * and trusted S has no capability memory access to sweep those copies' tags (Q-23). QEMU's
 * collect is safe only because it clears them first (virtual_collect's page sweep). The flat
 * build therefore never collects: exhaustion takes the ordinary -ENOMEM path node_grow() already
 * names, and both kits agree by construction. The line is pr_emerg because the runners keep only
 * KERN_EMERG on the console, and its budget is a count, never a ratelimit; `skips` keeps counting
 * past the budget. */
#define CV_COLLECT_SKIP_BUDGET 4U
static unsigned long collect_skips;
#endif

static int collect(struct context *c, struct cv_thread *owner)
{
    unsigned long flags, result;
    struct collect_list list = {0};
    int error = -ENOMEM;
    struct cv_thread *t;
#if node_table_flat
    if (++collect_skips <= CV_COLLECT_SKIP_BUDGET)
        pr_emerg("capstone_vm: collect skipped: collection is not available on bf3db3175 "
                 "(Q-27, S-entry); retired %lu available %lu, skips %lu\n",
                 node_retired(c), node_available(c), collect_skips);
    return -ENOMEM;
#endif
    /* Include resident pages populated by Linux/the trusted launcher without
     * a C fault. Missing pages are not materialized; any inspection failure
     * prevents identity reuse. Saved contexts share this mm's namespace. */
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i) {
        int error = pin_remaining(c, &c->arenas[i]);
        if (error) return error;
    }
    list_for_each_entry(t, &c->threads, link)
        if (collect_page(&list, t->frame_pa)) goto out;
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
        for (unsigned j = 0; j < c->arenas[i].count; ++j)
            if (c->arenas[i].pages[j] &&
                collect_page(&list, page_to_phys(c->arenas[i].pages[j]))) goto out;
    owner->frame[80] = list.count | CAP_REV_COLLECT_CHAIN;
    owner->frame[81] = virt_to_phys(list.head);
    preempt_disable(); local_irq_save(flags);
    result = run(owner->frame_pa, 3);
    local_irq_restore(flags); preempt_enable();
    owner->frame[80] = owner->frame[81] = 0;
    if (result == ULONG_MAX) { error = -EIO; goto out; }
    ++c->stats.collections; c->stats.reclaimed += result;
    error = 0;
out:
    collect_free(&list);
    return error;
}

/* Caller holds vm_lock, with interrupts/preemption enabled. The same path
 * supplies user instructions, privileged mint and the allocator's preflight.
 * A count is available capacity, not a reservation against other C threads. */
static int ensure_nodes(struct context *c, unsigned long required)
{
    struct cv_thread *owner = NULL, *t;
    unsigned long available = node_available(c);
    unsigned long sweep_pages = thread_count(c), collect_min;
    bool collected = false;
    int error;
    if (required > CAP_REV_TABLE_NAMESPACE_SIZE) return -ENOMEM;
    /* vp2's 16-bit node index: refuse growth past the flat table's ceiling. */
    if (node_max_records && required > node_max_records) return -ENOMEM;
    if (node_table_flat && required > CV_FLAT_RECORDS) return -ENOMEM;
    if (available >= required) return 0;
    list_for_each_entry(t, &c->threads, link)
        if (t->started && !t->terminal) { owner = t; break; }
    /* pin_remaining inspects every registered virtual page, including holes.
     * Amortize that scan over at least as many retired identities and one
     * growth batch; a tiny table beside a large sparse VMA must not cause a
     * full scan every few hundred allocations. Existing free IDs win above. */
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
        sweep_pages += c->arenas[i].bytes / PAGE_SIZE;
    collect_min = max(sweep_pages, node_batch_pages * 256);
    collect_min = max(collect_min, node_capacity(c) / 4);
    if (owner && node_retired(c) >= max(required - available, collect_min)) {
        error = collect(c, owner);
        if (error) return error;
        c->retired_flat = c->pushed_flat = 0;
        collected = true;
        if (node_available(c) >= required) return 0;
    }
    error = node_grow(c, required);
    if (!error) return 0;
    /* At a quota or memory limit, even a small successful sweep can help. */
    if (!collected && owner && node_retired(c)) {
        error = collect(c, owner);
        if (error) return error;
    }
    return node_available(c) >= required ? 0 : -ENOMEM;
}

/* Q-?? / sparse wedge: an OPTIONAL hardware fence immediately before the domain entry.
 *
 * Measured in the built object, not assumed: run() inlines to FOUR CSRUNV sites and not one
 * of them has a `fence` within the 500 instructions preceding it; the whole module contains
 * two fence instructions in 4323, neither on this path. The "memory" clobber below is a
 * COMPILER barrier and emits no instruction, and the local_irq_save ahead of each entry is a
 * `csrrc sstatus,2`, which does not order memory on RISC-V. On the ENTRY path the last frame
 * store sits TWO instructions ahead of the entry:
 *     sd a5,8(a0)     frame[1] = virt_to_phys(table)
 *     sd s11,56(a5)   frame[7] = 3
 *     ld a4,40(s2)
 *     CSRUNV
 * So if the switcher's RESUME walk can observe the frame before the kernel's stores drain,
 * nothing here prevents it.
 *
 * A module parameter rather than a build flag, deliberately: it makes the hypothesis
 * two-sided inside ONE boot, which is what the batching rule asks for, instead of spending
 * two board sessions on two hashes. Default off, so the production path keeps its current
 * behaviour and the only cost when off is a predictable load-and-branch in front of an entry
 * that already costs thousands of cycles. */
static bool fence_before_entry;
/* 0644, not 0444: writable through
 * /sys/module/capstone_vm/parameters/fence_before_entry, because run() reads it on every
 * entry, so a write takes effect at the next one. That is what makes the arm two-sided in a
 * SINGLE boot -- run the suite with it on, then clear it and repeat the wedge-expected case
 * as the last step -- instead of spending a second boot on the second arm, which the board
 * kernel would otherwise require since it cannot unload this module. A lone bool read by a
 * plain load needs no locking against the sysfs write; the write lands between cases. */
module_param(fence_before_entry, bool, 0644);

/* R-58, S-23's data face: the caller runs in its own lifetime namespace after every CSRUNV.
 * The switcher loads srevroot (CSR 0x5c1, switcher id 67) from frame+8 at CSRUNV entry, and saves
 * it at escape, but the caller's snapshot holds ids 0..66 only, so the caller's srevroot is never
 * restored (capstone-ariane 8522d72fe capstone_dom_switcher.anvil entry :170-173, escape :479-481).
 * After every escape S kept the CONTEXT's table. select_root() read it as `old`, and for the same
 * context wrote it back unchanged, so neither end of a window was an edge of the namespace flag
 * that the trackers flush on (ex_stage.sv:1417). A window's CSRETIRE or CSMINT walk broadcast the
 * context's indices, and a CPMP tracker whose index collided stayed invalid until the kernel's next
 * access through it was refused. On the board: sql0-csr read srevroot 0x82B00000 and 0x81E00000 at
 * every refusal; with this change sql0-srz read DRT1 0 against 16 (tree B, N=1).
 *
 * So every CSRUNV is followed IMMEDIATELY, in the same asm, by csrw 0x5c1 = 0, restoring the
 * caller's (physical) namespace, and restore_root() writes 0. Every window is then a real
 * 0 -> table -> 0 pair of edges, and the flush at its end re-arms whatever its walk cleared. Every
 * other capability op in this file sits inside a select_root/restore_root window, so none relies on
 * srevroot after a CSRUNV returns. A CSR write flushes at commit, so nothing after it runs early.
 * Unconditional (the runtime lane's ruling, 2026-10-20): it is correct everywhere, a no-op on QEMU,
 * whose switcher already restores, and on a bitstream carrying the R-58 fix; its cost is a pipeline
 * flush per write. First version, as a module parameter: lane module-srevroot-zero ce25d6de7280. */
static unsigned long run(unsigned long frame, unsigned long action)
{
    unsigned long result;
    if (fence_before_entry)
        asm volatile("fence rw,rw" ::: "memory");
    asm volatile(".insn r 0x5b, 1, 0x24, %0, %1, %2\n\tcsrw 0x5c1, zero"
                 : "=&r"(result) : "r"(frame), "r"(action) : "memory");
    return result;
}
/* The same CSRUNV with rd = x0. On vp2 a RESUME writes the literal 2 into the RESUMED CONTEXT's
 * x(rd) (commit_stage.sv:610-624, set_ra = rd / ra_data = 2, applied by finish() to every request
 * that is not a good entry) and re-latches rd, through which the next escape writes its kind
 * (:627, :839). With rd = x0 nothing is written into the context and the kind goes nowhere, so a
 * caller of this reads the event from the frame (run_event). To move another site here, change
 * its run(...) to run_x0(...); the collect site (action 3) cannot, its freed count IS rd. */
static void run_x0(unsigned long frame, unsigned long action)
{
    asm volatile(".insn r 0x5b, 1, 0x24, x0, %0, %1\n\tcsrw 0x5c1, zero"
                 : : "r"(frame), "r"(action) : "memory");
}
/* The event record every escape writes at frame+16: kind, cause, pc, address
 * (capstone-qemu target/riscv/capstone_supervisor.c event_write, call->event = frame + 16 :963).
 * The kinds written are 0 (return) to 5, so ~0 is never one: it is planted before the CSRUNV
 * and, read back unchanged, means the CSRUNV wrote no record -- REFUSED (a bad frame, or any
 * switcher refusal). A refusal is not an escape, so without the sentinel +16 would still hold
 * the previous event's kind (4) and the loop would resume for ever. */
#define CV_EVENT_UNWRITTEN (~0UL)
/* The cause reported to the launcher for a refused CSRUNV: outside every RISC-V cause, so a judge
 * reading the fault line can tell it from any real one. */
#define CV_CAUSE_REFUSED 0xcafe0001UL
static unsigned long run_event(struct cv_thread *t, unsigned long action)
{
    WRITE_ONCE(t->frame[2], CV_EVENT_UNWRITTEN);
    /* The sentinel must reach memory before the switcher can write the record over it; a kernel
     * store still draining after the escape's write would turn a real event into a refusal. */
    asm volatile("fence rw,rw" ::: "memory");
    run_x0(t->frame_pa, action);
    return READ_ONCE(t->frame[2]);
}
static unsigned long select_root(struct context *c)
{
    unsigned long old;
    asm volatile("csrr %0, 0x5c1\ncsrw 0x5c1, %1"
                 : "=&r"(old) : "r"(virt_to_phys(c->table)) : "memory");
    return old;
}
/* 0, not the `root` select_root returned: outside a window S is in the physical namespace, so the
 * window's end is an edge and its flush fires (R-58; see run()). */
static void restore_root(unsigned long root)
{
    (void)root;
    asm volatile("csrw 0x5c1, zero" : : : "memory");
}
/* Clear every capability tag in a page, keeping its bytes. A scalar store
 * clears the tag of the 16-byte granule it writes into, whatever its width
 * (capstone-qemu's store hook rounds both ends to granules; on the RTL a
 * plain store writes ctag 0 for the granule), so rewriting one 8-byte word
 * of each granule is enough: half the loads and stores of rewriting every
 * word, which was most of 519.lbm_r's module instructions. */
static void clear_page_tags(struct page *page)
{
    unsigned long *p = page_address(page);
    for (unsigned j = 0; j < PAGE_SIZE / sizeof(*p); j += 16 / sizeof(*p))
        WRITE_ONCE(p[j], READ_ONCE(p[j]));
}

static int retire(struct context *c, struct arena *a, bool destroying)
{
    unsigned long flags, old, status;
    unsigned i;
    if (!destroying && arena_has_thread_frame(c, a))
        return -EBUSY;
    if (!destroying) {
        int error = pin_remaining(c, a);
        if (error) return error;
    }
    /* S-21: skip a retire already done at CV_THREAD_CREATE. Without this, the domain unmapping
     * its child's frame page takes the NON-destroying path here, cap_rev_retire returns
     * CAP_REV_DEAD for the invalid node, status is 1, and this function reports -EIO -- which the
     * launcher surfaces as "service: Input/output error" and a 125 exit. Measured: the child thread
     * ran and printed VIRTUAL_THREAD_CHILD, and the step then failed on the unmap. An earlier
     * version of this change reasoned that no flag was needed because retire(destroying=true)
     * ignores the status; that enumerated only one of the two paths. */
    if (a->ancestor_retired) {
        status = 0;
    } else {
        unsigned long free_before = node_free_count(c);
        preempt_disable(); local_irq_save(flags);
        old = select_root(c);
        asm volatile(".insn r 0x5b, 1, 0x53, %0, %1, zero"
                     : "=&r"(status) : "r"(a->ancestor) : "memory");
        restore_root(old);
        local_irq_restore(flags); preempt_enable();
        if (status && !destroying) return -EIO;
        if (!status) {                    /* read only in flat mode; inert for the paged path */
            ++c->retired_flat;
            if (node_free_count(c) > free_before) ++c->pushed_flat;
        }
    }
    /* Clear every tag before returning frames to Linux. Later collection can
     * reuse dead IDs only after sweeping all remaining namespace storage. */
    for (i = 0; i < a->count; ++i) {
        if (!a->pages[i]) continue;
        clear_page_tags(a->pages[i]);
        set_page_dirty_lock(a->pages[i]);
        unpin_user_page(a->pages[i]);
        --c->stats.pinned_pages;
    }
    --c->stats.arenas;
    kvfree(a->pages);
    memset(a, 0, sizeof(*a));
    return 0;
}
/* Slots of the duplicate check's table for n frames: a power of two of at
 * least 2n, so linear probing stays short. */
static unsigned long frame_slots(long n)
{
    return roundup_pow_of_two(2 * (unsigned long)n);
}

/* No frame twice among the n pinned pages: each frame number is inserted
 * into an open-addressing table (frame_slots(n) zeroed entries holding pfn+1);
 * finding it already there is a duplicate. Expected O(n) (the table is at
 * most half full); pages keeps the address order the arena records. */
static bool frames_distinct(struct page **pages, long n, unsigned long *table)
{
    unsigned long mask = frame_slots(n) - 1;
    for (long i = 0; i < n; ++i) {
        unsigned long key = page_to_pfn(pages[i]) + 1;
        unsigned long at = (key * 0x9e3779b97f4a7c15UL) >> 32 & mask;
        while (table[at]) {
            if (table[at] == key) return false;
            at = (at + 1) & mask;
        }
        table[at] = key;
    }
    return true;
}

static int add(struct context *c, struct cv_thread *t, struct cv_map *r)
{
    unsigned i, index = 0;
    unsigned long *pfns = NULL;
    struct arena *a = NULL;
    unsigned long flags, old, desc[CV_DESC_WORDS], ancestor, *slot, mapped = 0;
    struct page **pages;
    long n;
    if (!r->bytes || r->bytes > CV_MAX_REGION_BYTES ||
        (!r->linear && r->bytes > CV_MAX_EAGER_BYTES) ||
        ((r->address | r->bytes) & (PAGE_SIZE - 1)) ||
        /* With exact bounds off the encoder quantises both edges, so an arena must be exactly
         * REPRESENTABLE. This used to demand a naturally aligned power of two, which is a much
         * smaller subset: it refused 371,780 of 400,000 of the allocator's real mapping shapes
         * (admitting 7.1%), and for the power-of-two ones the outcome depended on where mmap
         * landed, so part of the failure was nondeterministic across boots.
         *
         * The predicate is now the encoder's own condition, audited against QEMU's encoder
         * directly (cap_compress.c:40-57: E = msb(len) - 12 floored at 12, both edges quantised
         * by 8 << E) with 0 disagreements over 300,004 lengths, and cross-checked against the RTL
         * model by tests/exact-bounds-iii/repr-cross-check.py.
         *
         * The REFUSAL stays. QEMU's encoder has no refusal path -- it truncates the base down and
         * rounds the end up, silently widening the capability -- and that silent widening is the
         * hole exact bounds exists to close. So the model is followed for WHAT is representable,
         * not for what to do when a range is not. */
        (!c->exact_bounds && !capstone_repr_ok(r->address, r->bytes)) ||
        r->address > ULONG_MAX - r->bytes ||
        !access_ok((void __user *)r->address, r->bytes) ||
        r->permissions > 7 || r->linear > 1 || r->reg > 31 ||
        r->cursor < r->address || r->cursor > r->address + r->bytes ||
        (r->linear && r->cursor != r->address) ||
        (t->started && (t->event != 3 || r->reg != 10 || t->reply_cap)) ||
        (!t->started && r->reg != 2 && r->reg != 3 &&
         r->reg != 10 && r->reg != 11 && r->reg != 12 && r->reg != 13) ||
        (r->reg == 3 && (!(r->permissions & 1) || r->linear)))
        return -EINVAL;
    for (i = 0; i < CV_MAX_ARENAS; ++i) {
        struct arena *b = &c->arenas[i];
        if (!b->pages) { if (!a) { a = b; index = i; } continue; }
        mapped += b->bytes;
        if (r->address < b->address + b->bytes && b->address < r->address + r->bytes)
            return -EINVAL;
    }
    if (!a) return -ENOSPC;
    if (mapped + r->bytes > CV_MAX_BYTES) return -ENOMEM;
    if (ensure_nodes(c, 2 + CAP_REV_TABLE_RESERVE)) return -ENOMEM;
    mmap_read_lock(c->mm);
    for (unsigned long at = r->address; at < r->address + r->bytes;) {
        struct vm_area_struct *v = find_vma(c->mm, at);
        if (!v || v->vm_start > at || v->vm_file || (v->vm_flags & VM_SHARED) ||
            ((r->permissions & 4) && !(v->vm_flags & (r->linear ? VM_MAYREAD : VM_READ))) ||
            ((r->permissions & 2) && !(v->vm_flags & (r->linear ? VM_MAYWRITE : VM_WRITE))) ||
            ((r->permissions & 1) && !(v->vm_flags & (r->linear ? VM_MAYEXEC : VM_EXEC)))) {
            mmap_read_unlock(c->mm); return -EACCES;
        }
#ifdef CONFIG_TRANSPARENT_HUGEPAGE
        if (!(v->vm_flags & VM_NOHUGEPAGE)) {
            mmap_read_unlock(c->mm); return -EOPNOTSUPP;
        }
#endif
        at = min((unsigned long)(r->address + r->bytes), v->vm_end);
    }
    mmap_read_unlock(c->mm);
    pages = kvcalloc(r->bytes / PAGE_SIZE, sizeof(*pages), GFP_KERNEL);
    if (!pages) return -ENOMEM;
    if (!r->linear) {
        pfns = kvcalloc(frame_slots(r->bytes / PAGE_SIZE), sizeof(*pfns), GFP_KERNEL);
        if (!pfns) { kvfree(pages); return -ENOMEM; }
    }
    n = 0;
    /* Growing RW arenas start with absent PTEs. Linux resolves each first
     * touch through GUP, then the adapter pins that backing until retirement. */
    if (!r->linear) {
        n = pin_user_pages_fast(r->address, r->bytes / PAGE_SIZE, FOLL_WRITE, pages);
        if (n != r->bytes / PAGE_SIZE) goto unpin;
    }
    for (i = 0; i < n; ++i)
        if (!PageAnon(pages[i]) || page_mapcount(pages[i]) != 1) goto unpin;
    if (n > 1 && !frames_distinct(pages, n, pfns)) goto unpin;
    kvfree(pfns);
    pfns = NULL;
    slot = t->frame + 8 + r->reg * 2;
    /* Q-30: THE FRAME'S REGISTER SLOTS ARE THE CONTEXT'S LIVE STATE, NOT SPARE SPACE.
     *
     * This used to refuse a non-empty slot with -EFAULT, which only ever passed on QEMU. On an
     * escape the RTL writes x1..x31 WITH THEIR TAGS into the frame's own slots and restores the
     * register file over them (capstone_dom_switcher.anvil:155-161, :513-526 at bf3db3175), so
     * after a kind-3 service escape +224 holds the context's own a0 -- the size argument of the
     * MAP it is asking for. QEMU keeps the callee in save_state and leaves the slots at ENTRY's
     * zero (capstone_supervisor.c:419, :951-958), so the emptiness check was vacuous there.
     *
     * vp2's boot 3-T1 is the reading: CAPSTONE_VM_ADD_REFUSED errno=14 on the first reply-slot
     * LINEAR grant, then cause 24 downstream in mmap's errno store through tp=0. The RTL's frame
     * is the reference, so the reply OVERWRITES the slot and the RTL zeroes +224 again at the reply
     * resume (:297, :338-340).
     *
     * FALSIFIABLE, and the board lane's line: with the old module no
     * `ADD_OK ... reg=10 ... linear=1` can follow a kind-3 pause on silicon, because a MAP service
     * always carries a0 = size != 0. One such line in a transcript refutes this reasoning.
     *
     * The emptiness check is KEPT for a pre-start grant, where the slots really are ENTRY's zero
     * and a non-empty one would mean the launcher wrote somewhere it should not. The guard above
     * has already restricted a started context to event 3, reg 10 and !reply_cap, so the cleared
     * branch is only ever the reply. The stores are plain, which untags the granule. */
    if (t->started) {
        slot[0] = 0;
        slot[1] = 0;
    } else if (slot[0] || slot[1]) goto unpin;
    desc[0] = r->address;
    desc[1] = r->address + r->bytes;
    /* C pointers are copyable, so a non-linear grant is minted NONLIN directly. A linear
     * mapping grant stays move-only until the allocator derives a non-linear object. */
    desc[2] = r->permissions | (r->linear ? 0 : CV_DESC_NONLIN);
    /* The cursor the caller asked for. For a LINEAR grant this equals the base in every
     * caller (exec.c:292 and :844 both pass the region's own address), so naming it here
     * is what the old path produced by letting the mint default to base. */
    desc[3] = r->cursor;
    /* The image also has to appear in register slot 0: run() refuses to start a thread
     * unless frame[8] and frame[12] are both filled, and slot 0 is never granted. One
     * mint, two slots, one node. t->frame is unsigned long *, so + 8 is byte frame+64. */
    desc[4] = r->reg == 3 ? (unsigned long)(t->frame + 8) : 0;
    desc[5] = 0;
    preempt_disable(); local_irq_save(flags);
    old = select_root(c);
    asm volatile(".insn r 0x5b, 1, 0x50, %0, %1, %2"
                 : "=&r"(ancestor) : "r"(slot), "r"(desc) : "memory");
    restore_root(old);
    local_irq_restore(flags); preempt_enable();
    *a = (struct arena){r->address, r->bytes, ancestor, pages,
                       r->bytes / PAGE_SIZE, r->linear};
    r->id = c->ids[index] = ++c->next_id;
    ++c->stats.arenas; c->stats.pinned_pages += n;
    c->stats.peak_pages = max(c->stats.peak_pages, c->stats.pinned_pages);
    if (t->started) t->reply_cap = true;
    return 0;
unpin:
    for (i = 0; i < max(n, 0L); ++i) unpin_user_page(pages[i]);
    kvfree(pfns);
    kvfree(pages);
    return -EFAULT;
}
/* Linux performs mremap; the adapter only updates lifetime and physical-tag
 * ownership. Pinned frames retained by mremap keep their tags. Truncated frames
 * are cleared before their pins are released. The syscall may fail without
 * changing any capability lifetime. */
static int remap_begin(struct context *c, const struct cv_remap *r)
{
    struct cv_thread *t = find_thread(c, r->thread);
    struct arena *a = NULL;
    unsigned long mapped = 0;
    int error;
    if (!c->exact_bounds || c->remapping || !t || t->terminal ||
        !t->started || t->event != 3 || t->reply_cap ||
        !r->bytes || r->bytes > CV_MAX_REGION_BYTES || (r->bytes & ~PAGE_MASK))
        return -EINVAL;
    for (unsigned i=0; i<CV_MAX_ARENAS; ++i) {
        mapped += c->arenas[i].bytes;
        if (c->ids[i] == r->id && c->arenas[i].pages) a = &c->arenas[i];
    }
    if (!a || !a->lazy || arena_has_thread_frame(c, a)) return -EINVAL;
    if (mapped - a->bytes + r->bytes > CV_MAX_BYTES) return -ENOMEM;
    error = pin_remaining(c, a);
    if (error) return error;
    error = ensure_nodes(c, 2 + CAP_REV_TABLE_RESERVE);
    if (error) return error;
    c->remap_pages = kvcalloc(r->bytes / PAGE_SIZE, sizeof(struct page *), GFP_KERNEL);
    if (!c->remap_pages) return -ENOMEM;
    c->remapping = a; c->remap_thread = r->thread; c->remap_bytes = r->bytes;
    return 0;
}
static int remap_end(struct context *c, const struct cv_remap *r)
{
    struct arena *a = c->remapping;
    struct cv_thread *t = find_thread(c, r->thread);
    unsigned long flags, root, result, desc[CV_DESC_WORDS], *slot;
    unsigned count;
    struct vm_area_struct *v;
    bool valid;
    if (!a || !t || c->remap_thread != r->thread || c->remap_bytes != r->bytes)
        return -EINVAL;
    if (!r->address) {
        kvfree(c->remap_pages); c->remap_pages = NULL; c->remapping = NULL; wake_up_all(&c->remap_wait);
        return 0;
    }
    if ((r->address & ~PAGE_MASK) || r->address > ULONG_MAX-r->bytes ||
        !access_ok((void __user *)r->address, r->bytes)) return -EINVAL;
    /* The launcher is trusted, but reject overlap with another registered
     * arena and malformed VM state before retiring the old ancestor. */
    for (unsigned i=0; i<CV_MAX_ARENAS; ++i) {
        struct arena *b = &c->arenas[i];
        if (b != a && b->pages && r->address < b->address+b->bytes &&
            b->address < r->address+r->bytes) return -EINVAL;
    }
    mmap_read_lock(c->mm);
    v = find_vma(c->mm, r->address);
    valid = v && v->vm_start <= r->address && v->vm_end >= r->address+r->bytes &&
                 !v->vm_file && !(v->vm_flags & VM_SHARED) &&
                 (v->vm_flags & (VM_READ|VM_WRITE)) == (VM_READ|VM_WRITE);
    mmap_read_unlock(c->mm);
    if (!valid) return -EACCES;
    slot = t->frame + 28;
    /* Q-30, the same shape as add(). Unreachable in the FLAT build -- remap_begin returns -EINVAL
     * when exact_bounds is false -- so this is for the paged build and is written to match rather
     * than to be exercised here. remap_end only runs on a started context at its heap-grow reply,
     * so there is no pre-start case to preserve. */
    slot[0] = 0;
    slot[1] = 0;
    count = r->bytes / PAGE_SIZE;
    for (unsigned i=0; i<min(count, a->count); ++i) c->remap_pages[i] = a->pages[i];
    /* v7, behaviour-preserving: this slot is the heap's, which is LINEAR, and the old
     * path let the mint default the cursor to the base. No second slot. */
    desc[0] = r->address; desc[1] = r->address+r->bytes; desc[2] = 6;
    desc[3] = r->address; desc[4] = 0; desc[5] = 0;
    preempt_disable(); local_irq_save(flags); root = select_root(c);
    asm volatile(".insn r 0x5b, 1, 0x53, %0, %1, zero"
                 : "=&r"(result) : "r"(a->ancestor) : "memory");
    if (!result) {
        asm volatile(".insn r 0x5b, 1, 0x50, %0, %1, %2"
                     : "=&r"(result) : "r"(slot), "r"(desc) : "memory");
        a->ancestor = result;
    } else {
        restore_root(root); local_irq_restore(flags); preempt_enable();
        return -EIO;
    }
    restore_root(root); local_irq_restore(flags); preempt_enable();
    for (unsigned i=count; i<a->count; ++i) {
        if (!a->pages[i]) continue;
        clear_page_tags(a->pages[i]);
        set_page_dirty_lock(a->pages[i]); unpin_user_page(a->pages[i]);
        --c->stats.pinned_pages;
    }
    kvfree(a->pages); a->pages = c->remap_pages; c->remap_pages = NULL;
    a->address = r->address; a->bytes = r->bytes; a->count = count;
    c->remapping = NULL; wake_up_all(&c->remap_wait); t->reply_cap = true;
    return 0;
}

/* CV_STEP resolves faults and resumes for at most one supervisor quantum;
 * CV_RESOLVE gives its pin-ahead window the same bound. */
#define CV_STEP_BUDGET_NS (5 * NSEC_PER_MSEC)

#ifdef CV_TEST_HOOKS
/* Test build only (make CV_TEST_HOOKS=1; run-step-hooks.py). The production
 * module compiles none of this. A hook injects the condition cv_step_stop()
 * must catch, at a chosen point of CV_STEP; counters say whether CSRUNV was
 * entered while an injection was pending, and what GUP was asked for. */
static unsigned long test_inject;       /* 1 resched, 2 signal, 3 deadline, 4 SIGKILL */
static unsigned long test_inject_at;    /* 1 after a demand GUP, 2 before an entry, 3 before an ioctl's first entry,
                                           4 after CSRUNV returns a fault, before resolve() */
static unsigned long test_inject_every = 1;
static unsigned long test_short_gup;    /* pin-ahead GUP pins only half of a batch */
module_param(test_inject, ulong, 0444);
module_param(test_inject_at, ulong, 0444);
module_param(test_inject_every, ulong, 0444);
module_param(test_short_gup, ulong, 0444);
static u64 test_events, test_injections, test_stops, test_initial_stops, test_violations;
static u64 test_consumed, test_checks, test_checks_irqs_on, test_entries;
static u64 test_gup_requests, test_gup_max, test_short_results, test_gup_pending;
static bool test_pending, test_expired, test_in_step;
/* Faults whose first resolution came from the launcher's CV_RESOLVE: a stop
 * fell between the fault and its resolution in CV_STEP, so it got the full
 * pin-ahead window. Real quantum ends and signals cause these too. */
static u64 test_launcher_first;

static void test_hook(unsigned long at)
{
    if (!test_inject || !test_in_step || test_inject_at != at || test_pending ||
        ++test_events % test_inject_every)
        return;
    ++test_injections;
    test_pending = true;
    if (test_inject == 1) set_tsk_need_resched(current);
    else if (test_inject == 2) set_tsk_thread_flag(current, TIF_NOTIFY_SIGNAL);
    else if (test_inject == 3) test_expired = true;
    else if (test_inject == 4) send_sig(SIGKILL, current, 1);
}
/* Is the injected condition still there? A reschedule inside GUP consumes
 * an injected need_resched: entering CSRUNV after it is no violation. */
static bool test_condition(void)
{
    return test_inject == 1 ? need_resched() : test_inject == 2 ? signal_pending(current) :
           test_inject == 3 ? test_expired : fatal_signal_pending(current);
}
static void test_entry(void)
{
    if (!test_pending) return;
    if (test_condition()) ++test_violations;
    else ++test_consumed;
    test_pending = false;
}
static void test_gup(unsigned long pages)
{
    /* GUP may sleep and fault in memory: no stop condition may be pending. */
    if (test_pending && test_condition()) ++test_gup_pending;
    ++test_gup_requests;
    test_gup_max = max(test_gup_max, (u64)pages);
}
static void test_report(const char *why)
{
    pr_info("capstone_vm test %s: injections=%llu stops=%llu initial_stops=%llu "
            "violations=%llu consumed=%llu gup_requests=%llu gup_max=%llu short_results=%llu "
            "gup_pending=%llu checks=%llu checks_irqs_on=%llu entries=%llu "
            "launcher_first=%llu\n",
            why, test_injections, test_stops, test_initial_stops, test_violations,
            test_consumed, test_gup_requests, test_gup_max, test_short_results,
            test_gup_pending, test_checks, test_checks_irqs_on, test_entries,
            test_launcher_first);
}
#define TEST_HOOK(at) test_hook(at)
#define TEST_GUP(pages) test_gup(pages)
#define TEST_EXPIRED() test_expired
#define TEST_ENTRY() do { test_entry(); ++test_entries; } while (0)
#define TEST_CHECK() do { ++test_checks; if (!irqs_disabled()) ++test_checks_irqs_on; } while (0)
#define TEST_STOPPED(initial) do { if (test_pending) { ++test_stops; \
        if (initial) ++test_initial_stops; } } while (0)
#define TEST_STEP_BEGIN() do { test_in_step = true; } while (0)
#define TEST_IOCTL_DONE() do { test_pending = test_expired = test_in_step = false; } while (0)
#define TEST_RESOLVE(a, i) do { if (!test_in_step && !(a)->pages[i]) ++test_launcher_first; } while (0)
#else
#define TEST_HOOK(at) do { } while (0)
#define TEST_GUP(pages) do { } while (0)
#define TEST_EXPIRED() false
#define TEST_ENTRY() do { } while (0)
#define TEST_CHECK() do { } while (0)
#define TEST_STOPPED(initial) do { } while (0)
#define TEST_STEP_BEGIN() do { } while (0)
#define TEST_IOCTL_DONE() do { } while (0)
#define TEST_RESOLVE(a, i) do { } while (0)
#endif

/* Pending work for this task, or the end of its budget: return to the
 * launcher (and release vm_lock) instead of entering the domain again. */
static bool cv_step_stop(u64 deadline)
{
    return signal_pending(current) || need_resched() ||
           ktime_get_ns() >= deadline || TEST_EXPIRED();
}

/* A pin-ahead window is pinned PIN_AHEAD_BATCH pages at a time; before each
 * batch it stops on cv_step_stop(), so the extra work under vm_lock and
 * mmap_lock stays bounded whatever the window. */
#define PIN_AHEAD_BATCH 16

/* Called with mmap_read_lock held, after the faulting page passed resolve()'s
 * checks in VMA v. Each further page is pinned exactly as a fault on it
 * would pin it, in the same VMA, so it gets the same policy; one that fails
 * a check ends the window unpinned and faults later as it does today. */
static void pin_ahead(struct context *c, struct arena *a,
                      struct vm_area_struct *v, unsigned first, u64 deadline)
{
    unsigned long start = a->address + (unsigned long)first * PAGE_SIZE;
    unsigned long count = 0, done;
    while ((arena_pin || count + 1 < pin_ahead_pages) && first + count < a->count &&
           start + count * PAGE_SIZE < v->vm_end && !a->pages[first + count])
        ++count;
    for (done = 0; done < count;) {
        unsigned long batch = min(count - done, (unsigned long)PIN_AHEAD_BATCH);
        struct page **pages = a->pages + first + done;
        long n;
        if (cv_step_stop(deadline))
            return;
        TEST_GUP(batch);
#ifdef CV_TEST_HOOKS
        /* A short result, as when GUP stops at a page it cannot fault in:
         * only the first half of the batch is faulted in and pinned. */
        n = pin_user_pages_remote(c->mm, start + done * PAGE_SIZE,
                                  test_short_gup && batch > 1 ? batch / 2 : batch,
                                  FOLL_WRITE | ((v->vm_flags & VM_WRITE) ? 0 : FOLL_FORCE),
                                  pages, NULL, NULL);
        if (n >= 0 && n < batch) ++test_short_results;
#else
        n = pin_user_pages_remote(c->mm, start + done * PAGE_SIZE, batch,
                                  FOLL_WRITE | ((v->vm_flags & VM_WRITE) ? 0 : FOLL_FORCE),
                                  pages, NULL, NULL);
#endif
        for (long j = max(n, 0L); j < batch; ++j)
            pages[j] = NULL;
        for (long j = 0; j < max(n, 0L); ++j) {
            if (!PageAnon(pages[j]) || page_mapcount(pages[j]) != 1) {
                for (long k = j; k < n; ++k) {
                    unpin_user_page(pages[k]);
                    pages[k] = NULL;
                }
                return;
            }
            ++c->stats.pinned_pages;
        }
        if (n != batch)
            return;
        done += batch;
    }
}

/* r8's terminal store reads as a bare 139 in the suite: the signal cannot say WHICH of
 * resolve()'s five refusals fired, and the board lane's "!same" attribution is a
 * STRUCTURAL deduction from this source, not a reading. So every terminal return names
 * itself. Each of these paths ends the step, so at most one line is emitted per step --
 * unlike a print per fault, which would be ~426 UART lines per step (measured) and could
 * perturb or wedge the very boot it exists to observe.
 *
 * RETRACTED PREMISE (2026-10-18). This comment used to say "pr_warn, not pr_info: the board
 * console drops KERN_INFO". That was wrong, and it was my own repetition of a peer lane's claim
 * rather than something I checked. The board driver's cold_boot runs
 * `echo 1 > /proc/sys/kernel/printk` before its first rung, so from then on the console carries
 * KERN_EMERG and nothing else -- r7's pr_info and r11's pr_warn were voided by the same cause,
 * the console LEVEL, and the level of these lines was never the variable. Every line below is
 * level 4 and the one pr_err is level 3, so NONE of them reaches that console.
 * They are therefore read from the kernel ring buffer with dmesg, which the console level does
 * not filter. The level is kept at pr_warn deliberately: raising these to KERN_EMERG would put
 * them on a console the driver quiets on purpose, where an unexpected line can break the
 * ladder's own parsing. The one exception is the load banner -- see init().
 * page_to_pfn, not %p: %p is hashed, so two values are comparable for equality within a
 * single boot and for nothing else.
 *
 * The prev(...) fields cost nothing -- they are already on the thread for the -ELOOP
 * bound -- and they carry the discriminator: prev addr == this addr with prev cause 13,
 * and cause 15 here, is the read-then-write COW signature that is the hypothesis. */
static bool diag_escapes;
module_param(diag_escapes, bool, 0644);
static unsigned escape_prints;
#define CV_ESCAPE_BUDGET 16U

#define CV_DIAG_TAG "gen-push-1"

/* WHICH LEVEL a diagnostic prints at is now a runtime choice, because the two readers disagree
 * and the disagreement is settleable by measurement rather than by argument. During board rungs
 * the driver sets the console to level 1, so only KERN_EMERG reaches the UART; but `dmesg` reads
 * the ring buffer, which the console level does NOT filter, so a KERN_WARNING line is fully
 * readable that way. Default 0 keeps these at pr_warn and off a console the driver quiets on
 * purpose, where an unexpected line can break the harness's own parsing. Setting it to 1 puts
 * every diagnostic on the console directly, for a boot where the dmesg route turns out not to
 * work. 0644, so one boot can try both without a rebuild. */
static bool diag_emerg;
module_param(diag_emerg, bool, 0644);
static unsigned eloop_prints, refuse_prints, childframe_prints, refused_prints;

/* r12's reading: for `vm`, NONE of resolve()'s five refusal sites fired and the S-24 bound did --
 * the same address and cause 65 times with no change. So the !same deduction is refuted and the
 * remaining question is disjunct 1 vs 2 of the A/D narrowing, which that line cannot separate.
 *
 * resolve_sfence is the behavioural discriminator, and it exists because of the pte_same guard in
 * GUP rather than as a guess: follow_page_pte only calls set_pte_at + update_mmu_cache (and hence
 * sfence.vma) when the entry actually CHANGES, so on the second and later faults of this loop A
 * and D are already set, nothing is written, and NO fence is issued at all. Under disjunct 1 that
 * is exactly the missing operation. The loop ending with this on is disjunct 1; the loop
 * persisting is disjunct 2, and the question leaves the module.
 *
 * Default off so the production path is unchanged, 0644 so one boot tries both. */
static bool resolve_sfence;
module_param(resolve_sfence, bool, 0644);

/* The resume boundary's TLB workaround for bf3db3175 (the satp ruling, 2026-10-18). The kernel's
 * own flush for a PTE it changes is local to the hart that resolved the fault (riscv
 * update_mmu_cache = local_flush_tlb_page), resolve and resume can be separate ioctls, and a repeat
 * fault on an unchanged entry gets no flush at all (GUP's flush sits inside its pte_same guard). So
 * the RESUMING hart flushes the resolved address itself, with interrupts off immediately before the
 * resume CSRUNV, so a migration in between cannot defeat it: `sfence.vma addr, x0`, all ASIDs, one
 * page. It cannot cover the ESCAPE's restore of the caller, which happens inside the CSRUNV with no
 * instruction of ours in between. On today's platforms (one hart each) the boundary is already
 * closed by the single hart; this makes it a property of the code. resume_sfences counts the
 * flushes, read-only in /sys/module/capstone_vm/parameters/ (CV_STATS is the launcher's ABI). */
static unsigned long resume_sfences;
module_param(resume_sfences, ulong, 0444);

/* The resume boundary's satp refresh for bf3db3175 (the RTL lane, 2026-10-18): the ENTRY reads satp
 * from frame[0] (switcher anvil:168-169), the ESCAPE saves the context's live satp to +928 (:479) and
 * the RESUME restores it from +928 (:313). CVA6's ASID_WIDTH is 16 on this build
 * (build_config_pkg.sv:41), so the kernel's ASID allocator is live, and if it rolls its generation
 * while this process is switched out the mm gets a new ASID while +928 still holds the old one --
 * which the kernel may have handed to another mm. So before every resume the module writes the mm's
 * CURRENT satp, read on the resuming hart in the process's own context, into +928; +936 (srevroot)
 * and +944 (options) are left as the escape saved them. resume_satp_refreshes counts the writes;
 * resume_satp_changed counts those where a NONZERO saved word differed, the instrument that shows a
 * rollover. capstone-qemu's escape does not write +928 (its frame ends at x31) and its resume restores
 * an internal copy, so on QEMU the word reads 0 until the first refresh, the refresh is a no-op, and
 * changed stays 0 there by construction. */
#define CV_FRAME_SAVED_SATP (928 / 8)
static unsigned long resume_satp_refreshes, resume_satp_changed;
module_param(resume_satp_refreshes, ulong, 0444);
module_param(resume_satp_changed, ulong, 0444);

/* CV_THREAD_CREATE on a frame that already backs a thread is refused BEFORE the frame is pinned (the
 * revision's negative test, 2026-10-18). Before this, the second create pinned the same page, so its
 * t->frame WAS the live child's frame; its ancestor retire then failed (already retired) and the unwind's
 * clear_thread_frame() zeroed bytes 0..655 of the LIVE child's frame (satp, root, event words, PCC,
 * x1..x31), whose next CV_STEP failed -EINVAL. Only the domain's own thread could be harmed, and on vp2
 * every create is refused by the one slot, so it was self-inflicted and multi-slot only. Counted by
 * busy_frame_refusals; the line is pr_emerg with a count budget so the board console shows it. */
static unsigned long busy_frame_refusals;
module_param(busy_frame_refusals, ulong, 0444);

/* resolve_pte prints the faulting PTE once, when the bound trips. It is the check on this lane's
 * OWN conclusion that GUP sets A and D -- which was read out of the source, never measured on the
 * board -- so a boot where A and D come back clear refutes that reading rather than confirming a
 * walker defect. NOT version-guarded: follow_pte has the same signature and is EXPORT_SYMBOL_GPL
 * in both 6.1.0 and 6.4.14, so the instrument is identical on QEMU and the board and can be
 * validated before a board boot is spent on it, which a board-only guard would have prevented. */
static bool resolve_pte;
module_param(resolve_pte, bool, 0644);

/* The bound as a PARAMETER, so the -ELOOP path can be made to FIRE before a board boot is spent
 * on it. On QEMU `vm`'s store passes, so at the production bound of 64 this path never executes
 * and the PTE read above would reach silicon unvalidated -- which is the shape this project has
 * repeatedly paid for. Repeats do occur on QEMU (the launcher's idempotent CV_RESOLVE re-presents
 * a fault, visible as `prev(addr == addr, rep ...)` in the site lines), so a low value trips the
 * bound there and exercises the read end to end. Default is the production value. */
static unsigned long fault_repeat_max = CV_FAULT_REPEAT_MAX;
module_param(fault_repeat_max, ulong, 0644);

/* S-24's FIX, and the reason it is a kernel call and not a fence.
 *
 * GUP does NOT set the PTE's A or D bits for an anonymous page: follow_page_pte's FOLL_TOUCH block
 * (mm/gup.c:624-633) marks the struct page with set_page_dirty/mark_page_accessed and says in its
 * own comment that pte_mkyoung "would be more correct here" and is deliberately avoided. (The
 * block that DOES set them, with set_pte_at and update_mmu_cache, is follow_pfn_pte -- the
 * no-struct-page path. Quoting that one instead is the error RETRACTION-gup-does-not-set-A-or-D-
 * on-an-anon-page.md withdraws.)
 *
 * So a store to a present, writable page whose D is clear faults; the hardware installs no TLB
 * entry and does not set D itself; resolve() pins a page that is already pinned and reports
 * success having changed nothing; and the store faults again. That is r12's reading exactly:
 * "resolved 65 times with no change".
 *
 * fixup_user_fault drives the kernel's OWN write-fault path, which reaches handle_pte_fault's
 * writable-pte case and performs pte_mkdirty + ptep_set_access_flags + update_mmu_cache -- D set
 * and the sfence issued. EXPORT_SYMBOL_GPL with the same signature in 6.1.0 and 6.4.14, so no
 * version guard and the fix is testable on QEMU: under svadu=false QEMU faults the same way
 * (cpu_helper.c:1124-1131 returns TRANSLATE_FAIL when A/D need updating and hade is false).
 *
 * Default off so the production path is unchanged and the arm is two-sided in one boot. */
static bool resolve_dirty;
module_param(resolve_dirty, bool, 0644);

#define CV_EMIT(fmt, ...) do {                      \
    if (diag_emerg) pr_emerg(fmt, ##__VA_ARGS__);   \
    else            pr_warn(fmt, ##__VA_ARGS__);    \
} while (0)

/* A DETERMINISTIC budget, never pr_*_ratelimited, for any line a verdict reads. Rate limiting is
 * time-based: it drops lines unpredictably, and the line it drops is as likely as not the first
 * one -- which is the only one that matters for a must-fire control. A count always emits the
 * first CV_DIAG_BUDGET occurrences, so "absent" means "did not happen" rather than "was dropped".
 * This module had three verdict-bearing lines on pr_warn_ratelimited: the -ELOOP message that a
 * branch returns on, the thread-create refusal that is a positive control for the next boot, and
 * the child-frame status. All three are now budgeted instead. */
#define CV_DIAG_BUDGET 4U
#define CV_EMIT_FIRST(counter, fmt, ...) do {       \
    if ((counter) < CV_DIAG_BUDGET) {               \
        ++(counter);                                \
        CV_EMIT(fmt, ##__VA_ARGS__);                \
    }                                               \
} while (0)
#define CV_RESOLVE_FAIL(site, fmt, ...)                                            \
    CV_EMIT("capstone_vm: resolve refused at %s: addr 0x%lx cause %lu need 0x%lx "  \
            "pc 0x%lx prev(addr 0x%lx cause %lu rep %u) " fmt "\n",                 \
            site, address, cause, required, t->frame[4],                            \
            t->last_fault_addr, t->last_fault_cause, t->fault_repeat,               \
            ##__VA_ARGS__)

static int resolve(struct context *c, struct cv_thread *t, u64 deadline)
{
    unsigned long address = t->frame[5], cause = t->frame[3];
    struct page *page;
    struct vm_area_struct *v;
    unsigned long vmflags = 0;   /* snapshot under the mmap lock; v is unsafe to read after unlock */
    unsigned long required = cause == 15 ? VM_WRITE : cause == 12 ? VM_EXEC : VM_READ;
    if (!t->started || t->event != 4 || (cause != 12 && cause != 13 && cause != 15))
        return -EINVAL;
    for (unsigned i = 0; i < CV_MAX_ARENAS; ++i) {
        struct arena *a = &c->arenas[i];
        unsigned index;
        long n;
        if (!a->pages || !a->lazy || address < a->address ||
            address - a->address >= a->bytes) continue;
        index = (address - a->address) / PAGE_SIZE;
        mmap_read_lock(c->mm);
        v = find_vma(c->mm, address);
        if (!v || v->vm_start > address || !(v->vm_flags & required) ||
            v->vm_file || (v->vm_flags & VM_SHARED)) {
            unsigned long f = v ? v->vm_flags : 0UL;
            int file = v ? !!v->vm_file : -1;
            mmap_read_unlock(c->mm);
            CV_RESOLVE_FAIL("vma", "found %d flags 0x%lx file %d", !!v, f, file);
            return -EACCES;
        }
        vmflags = v->vm_flags;
        /* Check actual PTE policy first. FORCE then materializes private
         * anonymous backing even for a read-only zero-page first touch. The
         * application still sees its original VMA/PTE protection. */
        TEST_GUP(1);
        n = pin_user_pages_remote(c->mm, address & PAGE_MASK, 1,
                                  FOLL_WRITE | ((v->vm_flags & VM_WRITE) ? 0 : FOLL_FORCE),
                                  &page, NULL, NULL);
        if (n != 1) {
            mmap_read_unlock(c->mm);
            CV_RESOLVE_FAIL("pin", "pin_user_pages_remote=%ld vm_flags 0x%lx", n, vmflags);
            return -EFAULT;
        }
        TEST_HOOK(1);
        if (!PageAnon(page) || page_mapcount(page) != 1) {
            CV_RESOLVE_FAIL("anon", "pfn 0x%lx anon %d mapcount %d refs %d vm_flags 0x%lx",
                            page_to_pfn(page), !!PageAnon(page), page_mapcount(page),
                            page_ref_count(page), vmflags);
            mmap_read_unlock(c->mm); unpin_user_page(page); return -EACCES;
        }
        TEST_RESOLVE(a, index);
        if (!a->pages[index])
            pin_ahead(c, a, v, index + 1, deadline);
        mmap_read_unlock(c->mm);
        if (a->pages[index]) {
            /* mprotect may leave a write fault for Linux to resolve. It must
             * resolve to the same backing: tag-preserving moves are not part
             * of this pinned profile. Drop only the additional pin. */
            int same = a->pages[index] == page;
            if (!same)
                CV_RESOLVE_FAIL("remap",
                                "pfn_old 0x%lx pfn_new 0x%lx mapcount %d refs %d vm_flags 0x%lx",
                                page_to_pfn(a->pages[index]), page_to_pfn(page),
                                page_mapcount(page), page_ref_count(page), vmflags);
            unpin_user_page(page);
            if (!same) return -EACCES;
            /* S-24. THIS PATH CHANGES NOTHING, so reporting success is only correct if the
             * hardware will make progress on the retry. On vp2 it need not: the RTL's walker
             * faults on a clear accessed bit and never sets it (cva6_ptw.sv), while QEMU with
             * svadu=true sets A and D in hardware, so the same fault can be re-raised for ever
             * with an identical PTE. The module touches no PTE anywhere -- there is not one
             * pte_mkyoung, ptep_* or follow_pte in this file -- so nothing here changes the
             * condition the walker is unhappy about.
             *
             * vp2's Tier 2 stalled exactly so: 25 minutes of heartbeats after
             * vm_contract()'s mmap(8193, PROT_READ) first touch, with no record able to show why,
             * because every CV_STEP was "resolving" a fault successfully.
             *
             * A resolve that changes nothing must not be able to report success indefinitely. The
             * repeat is counted per (address, cause) and the run is failed with a named error and
             * one rate-limited line, which turns a silent 25-minute hang into a reading. The bound
             * is deliberately generous: a legitimate repeat happens when a fault is resolved here
             * and the quantum stops the entry before the retry, and the launcher's idempotent
             * CV_RESOLVE then repeats it. */
            if (address == t->last_fault_addr && cause == t->last_fault_cause) {
                if (++t->fault_repeat > fault_repeat_max) {
                    unsigned long pv = 0;
                    int prc = -ENOSYS;
                    if (resolve_pte) {
                        pte_t *ptep;
                        spinlock_t *ptl;
                        mmap_read_lock(c->mm);
                        prc = follow_pte(c->mm, address, &ptep, &ptl);
                        if (!prc) {
                            pv = pte_val(*ptep);
                            pte_unmap_unlock(ptep, ptl);
                        }
                        mmap_read_unlock(c->mm);
                    }
                    CV_EMIT_FIRST(eloop_prints,
                                  "capstone_vm: fault at 0x%lx cause %lu resolved %u times "
                                  "with no change; the page is pinned and the PTE is "
                                  "untouched by this module; pte rc=%d val=0x%lx "
                                  "present=%d write=%d young=%d dirty=%d pfn=0x%lx\n",
                                  address, cause, t->fault_repeat, prc, pv,
                                  prc ? -1 : !!pte_present(__pte(pv)),
                                  prc ? -1 : !!pte_write(__pte(pv)),
                                  prc ? -1 : !!pte_young(__pte(pv)),
                                  prc ? -1 : !!pte_dirty(__pte(pv)),
                                  prc ? 0UL : (unsigned long)pte_pfn(__pte(pv)));
                    return -ELOOP;
                }
            } else {
                t->last_fault_addr = address;
                t->last_fault_cause = cause;
                t->fault_repeat = 1;
            }
            /* S-24: make the kernel set D for a STORE before reporting success, so the retry can
             * make progress. Only for cause 15 -- a read fault needs no D, and A is set by the
             * populate on the path that allocated the page. A failure is not fatal here; the repeat
             * bound still catches a loop and the PTE print says what the entry looked like.
             *
             * THE CALLER MUST HOLD mmap_read_lock. An earlier version of this did not, on the
             * strength of a comment I wrote asserting the function takes the lock itself; it does
             * not, and QEMU answered with `kernel BUG at include/linux/mmap_lock.h:156` --
             * find_vma's mmap_assert_locked -- through find_extend_vma from fixup_user_fault. The
             * contract is in the source: "This function will not return with an unlocked
             * mmap_lock" (mm/gup.c:1260), so a plain lock/unlock pair around it is correct even
             * though it may drop and retake the lock internally when `unlocked` is non-NULL.
             *
             * That same docstring is why this function is the right one rather than a hand-rolled
             * fixup: it "will in turn perform all the necessary SW fixup of the dirty and young
             * bits in the PTE, while get_user_pages() only guarantees to update these in the struct
             * page... on such architectures, gup() will not be enough to make a subsequent access
             * succeed" (mm/gup.c:1251-1258). That is S-24, described by the kernel. */
            if (resolve_dirty && cause == 15) {
                bool unlocked = false;   /* fixup_user_fault takes bool *, in both kernels */
                mmap_read_lock(c->mm);
                fixup_user_fault(c->mm, address & PAGE_MASK, FAULT_FLAG_WRITE, &unlocked);
                mmap_read_unlock(c->mm);
            }
            /* The operation GUP skips on a repeat: its sfence.vma is inside the pte_same guard, so
             * an unchanged entry gets none. Two facts for whoever reads the result: the domain runs
             * under the SAME satp as this ioctl -- frame[0] is read from CSR_SATP and nothing else
             * assigns it, while select_root() writes CSR 0x5c1, the capability-table root, not satp
             * -- and this is the all-ASID form (`sfence.vma rs1`, rs2 = x0), so an ASID mismatch
             * cannot explain a missed flush in either direction. */
            if (resolve_sfence)
                asm volatile("sfence.vma %0" : : "r"(address) : "memory");
            return 0;
        }
        t->last_fault_addr = address;
        t->last_fault_cause = cause;
        t->fault_repeat = 0;
        a->pages[index] = page;
        ++c->stats.pinned_pages;
        c->stats.peak_pages = max(c->stats.peak_pages, c->stats.pinned_pages);
        return 0;
    }
    CV_RESOLVE_FAIL("arena", "no lazy arena covers this address");
    return -EFAULT;
}

static void node_stats(struct context *c)
{
    c->stats.nodes = c->table[3];
    c->stats.nodes_high_water = c->table[1] - CV_NODE_FIRST;
    c->stats.nodes_retired = node_retired(c);
    c->stats.nodes_live = c->stats.nodes_high_water - node_retired(c) - (c->table[2] >> 32);
    c->stats.node_capacity = node_capacity(c);
    high_water = max(high_water, c->stats.nodes_high_water);
}

static long vm_ioctl(struct file *f, unsigned int op, unsigned long arg)
{
    struct context *c = f->private_data;
    long rc = -EINVAL;
    unsigned long flags;
    if (current->mm != c->mm)
        return -EPERM;
retry:
    if (mutex_lock_interruptible(&vm_lock)) return -EINTR;
    if (c->remapping && op != CV_REMAP_END) {
        mutex_unlock(&vm_lock);
        if (wait_event_interruptible(c->remap_wait, !READ_ONCE(c->remapping))) return -EINTR;
        goto retry;
    }
    if (op == CV_GLOBAL) {
        struct context *other;
        struct cv_global stats = {.nodes_allocated = allocated_total,
            .collections = collections_total, .reclaimed = reclaimed_total};
        list_for_each_entry(other, &contexts, link) {
            node_stats(other);
            stats.arenas += other->stats.arenas;
            stats.pinned_pages += other->stats.pinned_pages;
            stats.nodes_allocated += other->stats.nodes;
            stats.nodes_live += other->stats.nodes_live;
            stats.nodes_retired += other->stats.nodes_retired;
            stats.collections += other->stats.collections;
            stats.reclaimed += other->stats.reclaimed;
            stats.node_capacity += other->stats.node_capacity;
            stats.node_bytes += other->stats.node_bytes;
            if (other->stats.arenas) ++stats.contexts;
        }
        stats.nodes_high_water = high_water;
        rc = copy_to_user((void __user *)arg, &stats, sizeof(stats)) ? -EFAULT : 0;
    } else if (op == CV_PROFILE) {
        u64 profile = c->exact_bounds ? 0x100 : 0;
        rc = copy_to_user((void __user *)arg, &profile, sizeof(profile)) ? -EFAULT : 0;
    } else if (op == CV_REMAP_BEGIN || op == CV_REMAP_END) {
        struct cv_remap r;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        rc = op == CV_REMAP_BEGIN ? remap_begin(c, &r) : remap_end(c, &r);
    } else if (op == CV_NODES) {
        struct cv_nodes r;
        struct cv_thread *t;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        t = find_thread(c, r.thread);
        if (!t || t->terminal) { rc = -ESRCH; goto out; }
        if (r.available > CAP_REV_TABLE_NAMESPACE_SIZE - CAP_REV_TABLE_RESERVE) { rc = -ENOMEM; goto out; }
        if (node_max_records &&
            r.available > node_max_records - CAP_REV_TABLE_RESERVE) { rc = -ENOMEM; goto out; }
        rc = ensure_nodes(c, r.available + CAP_REV_TABLE_RESERVE);
    } else if (op == CV_ADD) {
        struct cv_map r;
        struct cv_thread *t;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        t = find_thread(c, r.thread);
        if (!t || t->terminal) { rc = -ESRCH; goto out; }
        rc = add(c, t, &r);
        if (!rc && copy_to_user((void __user *)arg, &r, sizeof(r))) {
            /* The caller did not receive the handle. Reclaim the entire
             * context on close; never silently remove a delivered grant. */
            t->terminal = true; rc = -EFAULT;
        }
    } else if (op == CV_THREAD_CREATE) {
        struct cv_thread_create r;
        struct cv_thread *t;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        if (!r.frame || !find_arena(c, r.frame, PAGE_SIZE)) { rc = -EINVAL; goto out; }
        if (arena_has_thread_frame(c, find_arena(c, r.frame, PAGE_SIZE))) {
            if (++busy_frame_refusals <= CV_DIAG_BUDGET)
                pr_emerg("capstone_vm: thread create refused: frame 0x%llx already backs a thread "
                         "(refusals %lu)\n", (unsigned long long)r.frame, busy_frame_refusals);
            rc = -EBUSY; goto out;
        }
        if (thread_count(c) >= 32) { rc = -ENOSPC; goto out; }
        /* Refuse at CREATE, not at the join. The caller is itself a live frame -- it escaped to
         * make this ioctl -- so on a one-slot machine the child could never be entered. Admitting
         * it anyway is the illegible failure: contract.dom's `threads` busy-waits 100,000,000
         * iterations on a value the child will never set and then fails an assertion that says
         * nothing about slots, and `heap-threads` blocks in join for ever. -EBUSY here fails at the
         * domain's own require(tid > 0, "thread create"), which a pre-registration can state and a
         * judge can check. */
        if (slot_live(NULL) + 1 > supervisor_slots) {
            /* RETRACTED PREMISE (2026-10-18): this comment said the console "does not show
             * KERN_INFO ... while the kernel's own pr_warn lines do appear". Wrong, and not mine
             * to have repeated. The driver sets the console to level 1 before its first rung, so
             * it shows KERN_EMERG only; the pr_warn lines that were cited as appearing were all
             * printed during boot, before the quieting. r6/r7's absence of this line was the
             * console level, not the printk level.
             *
             * This line is the must-fire positive control for the next boot, so it is budgeted
             * rather than rate-limited: the launcher retries a refused create, and a time-based
             * limiter would drop exactly the occurrence a judge reads. */
            CV_EMIT_FIRST(refuse_prints,
                          "capstone_vm: thread create refused, %lu supervisor slot%s in use\n",
                          supervisor_slots, supervisor_slots == 1 ? "" : "s");
            rc = -EBUSY; goto out;
        }
        t = kvzalloc(sizeof(*t), GFP_KERNEL);
        if (!t) { rc = -ENOMEM; goto out; }
        t->context = c; t->id = ++c->next_thread;
        rc = pin_thread_frame(c, r.frame, t);
        if (rc) { kvfree(t); goto out; }
        /* The kernel owns the process address space and lifetime root. The
         * application supplies only the entry/register capabilities. */
        t->frame[0] = csr_read(CSR_SATP);
        t->frame[1] = virt_to_phys(c->table);
        t->frame[2] = t->frame[3] = t->frame[4] = t->frame[5] = t->frame[6] = 0;
        t->frame[7] = 3;
        t->frame[80] = t->frame[81] = 0;
        /* S-21: MAKE THE CHILD'S FRAME UNREACHABLE BY THE DOMAIN BEFORE IT IS EVER ENTERED.
         *
         * On the RTL every escape restores the S caller's snapshot from frame+1024 unchecked, and a
         * child frame is a page the DOMAIN allocated and still holds a NONLIN RW copy of
         * (thread.c's cap_vm_acquire + cap_vm_copyable; this ioctl only find_arena()s and pins it).
         * So without this the domain could overwrite the kernel's saved pc/sp/satp/stvec in its
         * child's frame and have the next escape load them. QEMU cannot show it at any slot count,
         * because the caller stays in save_state -- so this is silicon-only and no record exhibits
         * it; the reasoning is the RTL source, not a reading.
         *
         * Retiring the arena's ancestor kills the domain's copy (25 on access) while the page stays
         * pinned and kernel-only. That is safe to do at whole-arena granularity BECAUSE a thread
         * frame is its own arena: thread.c acquires exactly 4096 bytes for it, and
         * arena_has_thread_frame() matches on t->frame_user == a->address. A frame sharing an arena
         * with live objects would make this retire kill them too, which is why the granularity is
         * checked here rather than assumed.
         *
         * The later teardown retires the same ancestor again. That is a tolerated no-op, not a
         * second effect: cap_rev_retire returns CAP_REV_DEAD on an invalid node BEFORE any mutation
         * (cap_rev_tree.c:249-251), helper_csretire maps that to status 1, retire() ignores a
         * non-zero status when destroying, and ++c->retired_flat only fires on success. So no flag
         * is needed to suppress it.
         *
         * A failed retire fails the create: a frame the domain can still write is the whole defect.
         * Note this is unreachable on vp2 today -- with supervisor_slots = 1 the create is already
         * refused above -- so it is the ABI's property for a multi-slot bitstream, not a fix for
         * the current board. */
        {
            struct arena *fa = find_arena(c, r.frame, PAGE_SIZE);
            unsigned long st, old_root;
            unsigned long free_before;
            if (!fa) { rc = -EINVAL; goto thread_unwind; }
            free_before = node_free_count(c);
            preempt_disable(); local_irq_save(flags);
            old_root = select_root(c);
            asm volatile(".insn r 0x5b, 1, 0x53, %0, %1, zero"
                         : "=&r"(st) : "r"(fa->ancestor) : "memory");
            restore_root(old_root);
            local_irq_restore(flags); preempt_enable();
            if (st) {
                CV_EMIT_FIRST(childframe_prints,
                              "capstone_vm: child frame grant not retired (status %lu); "
                              "refusing the thread\n", st);
                rc = -EACCES;
                goto thread_unwind;
            }
            if (!st) {
                ++c->retired_flat;
                if (node_free_count(c) > free_before) ++c->pushed_flat;
            }
            fa->ancestor_retired = true;
        }
        INIT_LIST_HEAD(&t->link);
        list_add_tail(&t->link, &c->threads);
        r.thread = t->id;
        rc = copy_to_user((void __user *)arg, &r, sizeof(r)) ? -EFAULT : 0;
        if (rc) {
            list_del(&t->link); clear_thread_frame(t); drop_thread_frame(t); kvfree(t);
        }
        if (0) {
thread_unwind:
            /* The frame is pinned and the thread is not on c->threads yet, so the teardown is the
             * same minus the list_del. */
            clear_thread_frame(t); drop_thread_frame(t); kvfree(t);
        }
    } else if (op == CV_THREAD_EXIT) {
        struct cv_thread_control r;
        struct cv_thread *t;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        t = find_thread(c, r.thread);
        if (!t || t == c->main_thread) { rc = -EINVAL; goto out; }
        r.frame = t->frame_user;
        if (t->started) {
            preempt_disable(); local_irq_save(flags);
            run(t->frame_pa, 1);
            local_irq_restore(flags); preempt_enable();
        }
        list_del(&t->link); clear_thread_frame(t); drop_thread_frame(t);
        kvfree(t);
        rc = copy_to_user((void __user *)arg, &r, sizeof(r)) ? -EFAULT : 0;
    } else if (op == CV_STEP) {
        struct cv_step r;
        struct cv_thread *t;
        u64 budget_end;
        bool resolved = false;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        t = find_thread(c, r.thread);
        if (!t || t->terminal) { rc = -ESRCH; goto out; }
        if (r.reply > 1 || (r.reply != (t->started && t->event == 3))) goto out;
        if (!t->started && (!t->frame[8] || !t->frame[12])) goto out;
        /* A frame that has not started needs a free slot to enter. Waiting is safe ACROSS
         * contexts -- two vexec processes never join each other, so run.py's parallel pair simply
         * serialises and still prints both results -- and is NOT safe within one context, where a
         * live sibling is the thing that would have to finish first. The create refusal above means
         * the same-context case cannot normally arise; it is rejected here too rather than
         * deadlocking if it ever does. vm_lock must be dropped while waiting, so the wait is
         * outside it and the ioctl restarts. */
        if (!t->started) {
            struct cv_thread *holder = NULL;
            if (slot_live(&holder) >= supervisor_slots) {
                if (holder && holder->context == c) { rc = -EBUSY; goto out; }
                mutex_unlock(&vm_lock);
                if (wait_event_interruptible(slot_wait, slot_free_now()))
                    return -EINTR;
                return -EAGAIN;   /* the launcher's loop retries EAGAIN, as it does for the quantum */
            }
        }
        if (r.reply && !t->reply_cap) {
            t->frame[28] = r.result; t->frame[29] = 0;
        }
        /* Each CSRUNV restarts the 5 ms quantum, so a domain that keeps
         * faulting would never be preempted: bound the resolved faults by
         * one quantum of this ioctl, then return and release vm_lock. */
        budget_end = ktime_get_ns() + CV_STEP_BUDGET_NS;
        TEST_STEP_BEGIN();
        TEST_HOOK(3);
        for (unsigned long action = r.reply ? 2 : 0, entries = 0;;
             action = 0, ++entries) {
            if (entries) TEST_HOOK(2);
            preempt_disable(); local_irq_save(flags);
            /* Checked with interrupts and preemption off, immediately before
             * CSRUNV: no interrupt can set need_resched or a signal between
             * this check and the entry. Before the first entry nothing has
             * changed, so the launcher simply repeats the same CV_STEP (its
             * loop retries EINTR and EAGAIN; the reply above is rewritten
             * identically). Before a later one the fault just resolved goes
             * back as it was, already counted. */
            TEST_CHECK();
            if (cv_step_stop(budget_end)) {
                local_irq_restore(flags); preempt_enable();
                TEST_STOPPED(!entries);
                if (!entries) {
                    rc = signal_pending(current) ? -EINTR : -EAGAIN;
                    goto out;
                }
                resolved = true;
                break;
            }
            TEST_ENTRY();
            if (!t->started) {
                t->frame[0] = csr_read(CSR_SATP);
                t->frame[1] = virt_to_phys(c->table);
                t->frame[7] = 3;
            }
            if (t->sfence_pending) {
                asm volatile("sfence.vma %0, zero" : : "r"(t->last_fault_addr) : "memory");
                t->sfence_pending = false;
                ++resume_sfences;
            }
            if (t->started) {
                /* run_event's fence orders this store before the CSRUNV, with the sentinel's. */
                unsigned long satp = csr_read(CSR_SATP), saved = t->frame[CV_FRAME_SAVED_SATP];
                if (saved && saved != satp) ++resume_satp_changed;
                WRITE_ONCE(t->frame[CV_FRAME_SAVED_SATP], satp);
                ++resume_satp_refreshes;
            }
            t->event = run_event(t, action);
            t->started = true; t->reply_cap = false;
            local_irq_restore(flags); preempt_enable();
            ++c->stats.steps;
            if (t->event == CV_EVENT_UNWRITTEN) {
                /* REFUSED: the CSRUNV wrote no event record. End the context, never loop. The
                 * report below reads frame[2..5], so set them to a terminal kind and a cause no
                 * real fault carries. Must-fire, budgeted, at KERN_EMERG: the board console shows
                 * nothing lower. */
                if (refused_prints < CV_DIAG_BUDGET) {
                    ++refused_prints;
                    pr_emerg("capstone_vm: CSRUNV action %lu wrote no event record (refused), "
                             "thread %llu entries %lu: context ended\n",
                             action, (unsigned long long)t->id, entries);
                }
                t->frame[2] = 2; t->frame[3] = CV_CAUSE_REFUSED;
                t->frame[4] = t->frame[5] = 0;
                t->event = 2;
            }
            if (diag_escapes && escape_prints < CV_ESCAPE_BUDGET) {
                ++escape_prints;
                CV_EMIT("capstone_vm: escape %u action %lu event %lu frame kind 0x%lx cause 0x%lx "
                        "pc 0x%lx tval 0x%lx a0 0x%lx\n",
                        escape_prints, action, t->event,
                        t->frame[2], t->frame[3], t->frame[4], t->frame[5], t->frame[6]);
            }
            if (t->event != 4)
                break;
            /* Resolve a page fault here, as CV_RESOLVE would, and resume
             * without the two system calls a return through the launcher
             * costs. resolve() can sleep in mmap_lock and GUP while vm_lock
             * is held, so a stop condition already raised (or a quantum
             * already spent) returns the fault unresolved, as without this
             * loop. A fault resolve() refuses goes back to the launcher,
             * whose own CV_RESOLVE fails the same way and delivers it. A
             * fault resolved here but stopped before the next entry is
             * resolved again, idempotently, by the launcher's CV_RESOLVE. */
            TEST_HOOK(4);
            if (cv_step_stop(budget_end)) {
                TEST_STOPPED(false);
                break;
            }
            if (resolve(c, t, budget_end))
                break;
            t->sfence_pending = true;
            ++c->stats.faults;
        }
        if (t->event == 5) {
            int error = ensure_nodes(c, CAP_REV_TABLE_RESERVE + 1);
            /* Return through Linux between retries. Refuse endless retries
             * when the namespace consists of live or pinned identities. */
            t->frame[2] = error ? 2 : 1;
        }
        r.kind = t->frame[2]; r.cause = t->frame[3];
        r.pc = t->frame[4]; r.address = t->frame[5];
        memcpy(r.args, t->frame + 72, sizeof(r.args));
        r.thread = t->id;
        t->terminal = r.kind == 2;
        if (r.kind == 4 && !resolved) ++c->stats.faults;
        rc = copy_to_user((void __user *)arg, &r, sizeof(r)) ? -EFAULT : 0;
        if (rc) t->terminal = true;
    } else if (op == CV_RESOLVE) {
        struct cv_thread_control r;
        struct cv_thread *t;
        if (copy_from_user(&r, (void __user *)arg, sizeof(r))) { rc = -EFAULT; goto out; }
        t = find_thread(c, r.thread);
        rc = t ? resolve(c, t, ktime_get_ns() + CV_STEP_BUDGET_NS) : -ESRCH;
        if (!rc) t->sfence_pending = true;
    } else if (op == CV_RETIRE) {
        u64 id;
        if (copy_from_user(&id, (void __user *)arg, sizeof(id))) { rc = -EFAULT; goto out; }
        for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
            if (c->ids[i] == id && c->arenas[i].pages) { rc = retire(c, &c->arenas[i], false); break; }
    } else if (op == CV_STATS) {
        node_stats(c);
        rc = copy_to_user((void __user *)arg, &c->stats, sizeof(c->stats)) ? -EFAULT : 0;
    } else rc = -ENOTTY;
out:
    TEST_IOCTL_DONE();
    /* Q-26: one wake covers every way a frame can stop occupying the slot --
     * terminal exit, a discard, an error path -- instead of enumerating them and
     * missing one. A spurious wake costs a re-check. */
    wake_up_all(&slot_wait);
    mutex_unlock(&vm_lock);
    return rc;
}
static int vm_open(struct inode *inode, struct file *f)
{
    struct context *c;
    struct context *existing;
    struct cv_thread *t;
    unsigned long available;
    if (num_online_cpus() != 1 || !current->mm || PAGE_SIZE != 4096)
        return -EOPNOTSUPP;
    if (node_initial_pages < 2 || node_initial_pages > CAP_REV_TABLE_MAX_PAGES ||
        !node_batch_pages || node_batch_pages > CAP_REV_TABLE_MAX_PAGES ||
        node_max_pages > CAP_REV_TABLE_MAX_PAGES ||
        (node_max_pages && node_max_pages < node_initial_pages)) return -EINVAL;
    c = kvzalloc(sizeof(*c), GFP_KERNEL);
    if (!c) return -ENOMEM;
    INIT_LIST_HEAD(&c->threads);
    init_waitqueue_head(&c->remap_wait);
    c->next_thread = 0;
    t = kvzalloc(sizeof(*t), GFP_KERNEL);
    if (!t) { kvfree(c); return -ENOMEM; }
    c->frame = (void *)get_zeroed_page(GFP_KERNEL);
    c->table = node_table_flat ? node_new_flat_table(c) : node_new_page(c);
    if (!c->frame || !c->table) {
        if (c->frame) free_page((unsigned long)c->frame);
        node_free_table(c);
        kvfree(t); kvfree(c); return -ENOMEM;
    }
    if (node_table_flat) {
        /* RECYCLE set, PAGED clear, capacity fixed and published immediately. Deliberately NO
         * MAGIC at table[4] and no retired word at table[5]: in this format root+32 and root+40
         * are record id 2, the first allocatable id, and QEMU reads neither outside the paged
         * branch. Writing them would publish a table whose first record is the magic's ASCII.
         * Since the S-23 floor, root+32 and root+40 are record 2, inside the never-allocated gap
         * below CV_NODE_FIRST, and they stay zero for the same reason the whole gap does. */
        c->table[0] = CAP_REV_TABLE_RECYCLE | CV_FLAT_RECORDS;
        c->table[1] = CV_NODE_FIRST;
    } else {
        c->table[0] = CAP_REV_TABLE_FLAGS;
        c->table[1] = 2;
        c->table[4] = CAP_REV_TABLE_MAGIC;
        for (unsigned long i = 0; i < node_initial_pages; ++i) {
            if (node_add_page(c, i * 256)) {
                node_free_table(c); free_page((unsigned long)c->frame);
                kvfree(t); kvfree(c); return -ENOMEM;
            }
        }
    }
    /* An older processor rejects the versioned format through urevavail.
     * Detect that before CSMINT could fault in the kernel context.
     *
     * NOT IN FLAT MODE. urevavail (0xcc0) does not exist on vp2 -- csr_regfile.sv falls through
     * to its default and a csrr is an ILLEGAL INSTRUCTION, so this probe would trap in S the
     * first time a context is created. QEMU implements 0xcc0 as capacity - next + free_count
     * read live, which is exactly node_available(), so in flat mode the header answers it
     * without touching the CSR. Saturated at CV_FLAT_RECORDS as the hardware saturates. The
     * probe's purpose -- detecting a processor too old for the versioned format -- does not
     * apply to vp2, which is the newer part. Recorded in inputs.json as a computed count. */
#if node_table_flat
    available = min(node_available(c), CV_FLAT_RECORDS);
#else
    /* Braced so the declarations stay at the head of a block: the kernel builds with
     * -Wdeclaration-after-statement, and scoping them here keeps the flat build free of
     * -Wunused-variable, which is how the need for this was caught. */
    {
        unsigned long flags, previous_root;
        preempt_disable(); local_irq_save(flags);
        previous_root = select_root(c);
        asm volatile("csrr %0, 0xcc0" : "=r"(available));
        restore_root(previous_root);
        local_irq_restore(flags); preempt_enable();
    }
#endif
    if (available != node_capacity(c) - CV_NODE_FIRST) {
        node_free_table(c); free_page((unsigned long)c->frame);
        kvfree(t); kvfree(c); return -EOPNOTSUPP;
    }
#if node_table_flat
    /* scapctl (0x5c0) does not exist on vp2 either -- any access, any mode, is illegal. The only
     * thing this read produced was the exact-bounds property, which under option (iii) is false.
     * The module never WROTE 0x5c0 (one access in the file, a read), so nothing else is lost. */
    c->exact_bounds = false;
#else
    { unsigned long profile;
      asm volatile("csrr %0, 0x5c0" : "=r"(profile));
      c->exact_bounds = !!(profile & 0x100); }
#endif
    c->frame[1] = virt_to_phys(c->table);
    t->context = c; t->id = 0; t->frame = c->frame;
    t->frame_pa = virt_to_phys(c->frame); t->user_frame = false;
    c->main_thread = t;
    list_add(&t->link, &c->threads);
    c->mm = current->mm; mmgrab(c->mm);
    mutex_lock(&vm_lock);
    list_for_each_entry(existing, &contexts, link)
        if (existing->mm == c->mm) {
            mutex_unlock(&vm_lock);
            mmdrop(c->mm);
            node_free_table(c);
            memset(c->frame, 0, PAGE_SIZE);
            free_page((unsigned long)c->frame);
            kvfree(t); kvfree(c);
            return -EBUSY;
        }
    f->private_data = c;
    list_add(&c->link, &contexts);
    mutex_unlock(&vm_lock);
    return 0;
}
static int vm_release(struct inode *inode, struct file *f)
{
    struct context *c = f->private_data;
    unsigned long flags;
    struct cv_thread *t, *tmp;
    mutex_lock(&vm_lock);
    kvfree(c->remap_pages);
    list_del(&c->link);
    node_stats(c);
    allocated_total += c->stats.nodes;
    collections_total += c->stats.collections;
    reclaimed_total += c->stats.reclaimed;
    /* Stop every supervisor slot while its frame pages are still pinned. */
    list_for_each_entry(t, &c->threads, link) {
        if (t->started) {
            preempt_disable(); local_irq_save(flags);
            run(t->frame_pa, 1);
            local_irq_restore(flags); preempt_enable();
        }
    }
    /* Teardown still clears tags in user frames (including thread start
     * frames) before the final pins are dropped. */
    {
        /* S-23: retire() ignores a rejected CSRETIRE when destroying (and counts only successes in
         * retired_flat), so a teardown whose retires were all rejected would look exactly like one
         * whose broadcasts no longer collide. This line makes the difference readable: one per
         * context, always KERN_EMERG because a verdict reads it. */
        unsigned long attempted = 0, before = c->retired_flat, pushed_before = c->pushed_flat;
        for (unsigned i = 0; i < CV_MAX_ARENAS; ++i)
            if (c->arenas[i].pages || c->arenas[i].address) {
                if (!c->arenas[i].ancestor_retired) ++attempted;
                /* The task may be in exit_mm() here.  A teardown must never
                 * call pin_user_pages_fast() to materialize lazy holes; only
                 * already pinned pages can safely be revoked and released. */
                retire(c, &c->arenas[i], true);
            }
        if (node_table_flat)
            pr_emerg("capstone_vm: release retires attempted=%lu succeeded=%lu pushed=%lu first_id=%lu\n",
                     attempted, c->retired_flat - before, c->pushed_flat - pushed_before,
                     (unsigned long)CV_NODE_FIRST);
    }
    list_for_each_entry_safe(t, tmp, &c->threads, link) {
        list_del(&t->link);
        if (t->user_frame) { clear_thread_frame(t); drop_thread_frame(t); }
        else if (t->frame) { memset(t->frame, 0, PAGE_SIZE); free_page((unsigned long)t->frame); }
        kvfree(t);
    }
    node_free_table(c);
    mmdrop(c->mm); kvfree(c);
    wake_up_all(&slot_wait);   /* the released context's frames are gone */
    mutex_unlock(&vm_lock);
    return 0;
}
static const struct file_operations ops = {
    .owner = THIS_MODULE, .open = vm_open, .release = vm_release, .unlocked_ioctl = vm_ioctl,
};
static struct miscdevice device = {
    .minor = MISC_DYNAMIC_MINOR, .name = "capstone-vm", .fops = &ops, .mode = 0600,
};
static int __init init(void)
{
#ifdef CAPSTONE_VM_FLAT_TABLE
    /* Refuse the LOAD, so a bad value cannot reach a run. 2 because ids 0 and 1 are the header;
     * the maximum because silicon saturates capacity there; whole pages because the table is one
     * __get_free_pages() allocation whose order is computed from this. */
    if (node_flat_records < 2 || node_flat_records > CV_FLAT_RECORDS_MAX ||
        (node_flat_records * CAP_REV_TABLE_RECORD) % PAGE_SIZE) {
        /* pr_emerg, like the banner and for the same reason: this REFUSES the load, so the
         * banner will be absent, and a reader needs to tell "refused for this reason" apart from
         * "did not load, cause unknown". At pr_err it is level 3 and invisible on a console the
         * driver has quieted, which would make a deliberate refusal indistinguishable from a
         * failure. One line, at insmod, before any rung. */
        pr_emerg("capstone_vm: node_flat_records=%lu must be 2..%lu and a whole number of pages\n",
                 node_flat_records, CV_FLAT_RECORDS_MAX);
        return -EINVAL;
    }
#endif
    {
        /* MUST-FIRE control for every CV_RESOLVE_FAIL above, and the ONE line here at
         * KERN_EMERG. The board driver's cold_boot sets the console to level 1 before its first
         * rung, so a pr_warn banner cannot reach that console -- which is exactly how r11's
         * diagnostic came back void. pr_emerg survives it, and it is affordable precisely
         * because it fires once per insmod, before the rungs begin, where it cannot interleave
         * with the ladder's markers. The site lines above stay at pr_warn and are read with
         * dmesg, because one unexpected EMERG line mid-stream can break the harness's parsing.
         *
         * So: banner present => the module loaded at this revision, and an absent site line
         * below really did not fire. Banner absent => the module did not load at all, which is
         * a different and louder failure than a filtered print. Bump CV_DIAG_TAG whenever the
         * lines above change shape. */
        int rc = misc_register(&device);
        if (rc)
            return rc;
        /* Always KERN_EMERG, independent of diag_emerg: this is the line that tells a reader
         * whether the module loaded at all, so it must survive the quieted console even when the
         * rest are being read through dmesg. One line per insmod, before the rungs begin. */
        pr_emerg("capstone_vm: loaded, diag %s, supervisor_slots=%lu, diag_emerg=%d\n",
                 CV_DIAG_TAG, supervisor_slots, diag_emerg);
        return 0;
    }
}
#ifdef CV_TEST_HOOKS
static void __exit done(void) { test_report("unload"); misc_deregister(&device); }
#else
static void __exit done(void) { misc_deregister(&device); }
#endif
module_init(init); module_exit(done);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Private virtual Capstone execution adapter");
