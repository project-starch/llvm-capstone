// SPDX-License-Identifier: GPL-2.0-only
/* Trusted OS adapter: one shared-mm namespace and explicit C contexts. */
#include <linux/module.h>
#include <linux/of.h>
#include <linux/cpu.h>
#if !defined(CONFIG_RISCV_CAPSTONE_TASK) || defined(MODULE)
#error "The Capstone adapter requires built-in CONFIG_RISCV_CAPSTONE_TASK"
#endif
#define CV_TASK_BACKEND 1
#include <asm/capstone_task.h>
#include <linux/capstone_native_helpers.h>

static bool copyable_exact_bounds;
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/version.h>
#include <linux/uaccess.h>
#include <linux/sched/signal.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/list.h>
#include <linux/wait.h>
#include <linux/ktime.h>
#include <linux/log2.h>
#include <linux/rbtree.h>
#include <linux/file.h>
#include <asm/csr.h>
#include <asm/io.h>
#include <asm/pgtable.h>
#include "wire.h"
_Static_assert(sizeof(struct cv_step) == 136, "cv_step is two user-copy chunks and an 8-byte tail");
#ifndef __uninitialized   /* kernels before 6.9 have no macro for the attribute */
#define __uninitialized __attribute__((uninitialized))
#endif
/* The 63-bit capability, id, record and table formats (sublet-format.h) and the paged table
 * (sublet-table.h), over the kernel's page allocator. Table pages are charged to the process's
 * memory cgroup like the arena records. */
#define SUBLET_PAGE_NEW() ((unsigned long *)get_zeroed_page(GFP_KERNEL_ACCOUNT))
#define SUBLET_PAGE_FREE(p) free_page((unsigned long)(p))
#define SUBLET_PA(p) virt_to_phys(p)
#define SUBLET_VA(pa) ((unsigned long *)phys_to_virt(pa))
#define SUBLET_WMB() smp_wmb()
#include "sublet-table.h"
/* One rule, shared with the allocator. No #include of its own and only
 * __builtin_clzl and unsigned long, so it is valid kernel C. */
#include "representable.h"
#include "native-grant.h"

/* One registered mapping, with its own revocable ancestor. */
struct arena {
    struct rb_node node;
    u64 id;
    unsigned long address, bytes, ancestor;
    /* The image's read+execute prefix (zero for every other arena): mapped
     * from the image file without write permission, so it never holds a tag;
     * not pinned, not swept. pages[] and count cover the pages after it. */
    unsigned long rx_bytes;
    struct page **pages;
    /* The frame set of a NONLIN arena: every frame pages[] holds, once, as pfn+1 in an
     * open-addressing table of frame_slots(count) entries. It is the whole-arena duplicate
     * check (frames_distinct, commit 35fee9127ec2) kept for the arena's lifetime, so a page
     * admitted after the mint -- at a fault, in a pin-ahead window or by the resident scan --
     * is held to the same rule as one pinned at mint. Unpinned arenas have
     * neither pages nor frames; their count still describes range geometry. */
    unsigned long *frames;
    /* Data fault events received on this arena's pages, counted under diag_admit where CV_STEP
     * receives the event (not where CAPSTONE_VM_STATS faults is advanced, at the resolution or
     * the return to the launcher), so the two audit each other: an event whose count the
     * process total misses shows as total < sum(faulted). It is printed with the census and
     * stands in for admits in the unpinned mode, which pins and so admits nothing. u64 like the
     * process counter; the census is meaningful only if diag_admit is on for the whole life of
     * the arena (the parameter is writable). Placed before count so the structure grows by 8
     * bytes (112 to 120 on RV64) and not by the padding of a 32-bit field. */
    u64 faulted;
    unsigned count;
    /* The arena's page census, printed once at its retirement under diag_admit: pages admitted
     * at a fault, by a pin-ahead window, by the resident scan, by CV_TOUCH, or pinned whole at
     * the mint; released before retirement by a remap's truncation. admits + windows + scans +
     * touches + eager - released is what the retirement sweeps. Counted, not printed, inside a
     * window: a printk per page there added work before the next batch's deadline check. */
    unsigned admits, windows, scans, touches, eager, released;
    /* Separate from the pin census: successful raw-mm fault resolutions and
     * launcher writes do not acquire payload pins. Diagnostic only. */
    unsigned unpinned_resolves, unpinned_touches, unpinned_events, unpinned_ahead;
    /* Pages are admitted one by one at the domain's first access (resolve). */
    bool lazy;
    /* S-21: the ancestor was already CSRETIREd at CV_THREAD_CREATE, to make a child's frame
     * unreachable by the domain before its first entry. A second retire is not an error, but the
     * non-destroying path reports one, so it must be skipped rather than tolerated. */
    bool ancestor_retired;
    /* Native-task mappings only; initial transport cannot be retired. */
    bool dom_dynamic, dom_metadata;
    /* Checked-copy holds retain both pages and the VA reservation after a
     * logical native unmap. The node is retired immediately, under vm_lock. */
    unsigned copy_holds;
    bool copy_retired, copy_unmap_pending;
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
    bool slot_exported; /* VM transport image in the kernel-owned frame */
    unsigned long native_satp; /* translation identity of the paused snapshot */
    unsigned long event;
    /* S-24: the same kind-4 fault resolved repeatedly without changing anything. The RTL's walker
     * faults on a clear A bit and never sets it, so a PTE this module leaves untouched can re-fault
     * for ever while resolve() keeps reporting success. Bounded and reported rather than spun. */
    unsigned long last_fault_addr, last_fault_cause;
    /* The unpinned paths count a repeat only within one mapping: a program that maps, touches and
     * unmaps a page at the same address over and over (mallocng's group churn) takes a first fault
     * at the same (address, cause) each time, and none of those is a resolve that changed nothing.
     * The arena's id is never reused, so a new mapping at the old address starts a new count. */
    u64 last_fault_arena;
    unsigned fault_repeat;
    /* A fault was resolved and the context has not been resumed since: the resume flushes the
     * resolved address on the resuming hart first (resume_sfences). */
    bool sfence_pending;
    /* Adaptive fault-ahead (fault_ahead_max): the context's recent sequential fault streams. */
    struct cv_stream {
        u64 arena;             /* the arena's mapping ID: never reused, so a stream cannot outlive
                                * its arena into a new one at the same address */
        unsigned long used;    /* stream_clock at its last fault: the least recent is replaced */
        unsigned next;         /* the page index a sequential next fault lands on */
        unsigned run;          /* faults in the stream so far, each at the previous one's next */
        unsigned window;       /* pages requested ahead at its last fault, including empty cuts */
    } streams[4];
    unsigned long stream_clock;
};
struct context {
    struct list_head link;
    struct mm_struct *mm;
    unsigned long *frame, *table;
    phys_addr_t table_pa;   /* the table's physical root (select_root, the frame); from CMA when it is above the buddy limit */
    struct cv_thread *main_thread;
    struct list_head threads;
    u64 next_thread;
    /* Registered arenas by address, under vm_lock.
     * Arenas never overlap, so the address tree answers containment with
     * one predecessor lookup. registered_bytes is the sum of their sizes. */
    struct rb_root arenas;
    unsigned long registered_bytes;
    struct cv_stats stats;
    u64 next_id;
    bool exact_bounds;
    /* R, the one reference over the user range, and the slot root_derive builds a grant in,
     * both in pages of kernel slots that the context owns until its release. */
    struct list_head ref_pages;
    unsigned long *ref_free, *root_slot, *grant_slot;
#ifdef CAPSTONE_VM_BENCH_COUNTERS
    /* Performance build only: cycle and instret retired between entering and leaving each CSRUNV,
     * summed over every thread of the context. Linux's own work (escapes, delegated calls, fault
     * resolution) is outside the window. */
    atomic64_t bench_cycles, bench_instret, bench_entries;
#endif
    /* The image file, writers denied from CV_IMAGE until release. */
    struct file *image;
    /* Private native-task driver state; never imported from scalar userspace. */
    unsigned long dom_meta, dom_exchange, dom_exchange_bytes;
    struct cv_step dom_step;
    pid_t dom_pid;
    unsigned long dom_entry;
    struct dom_delegate *dom_delegate;
    struct cn_context *native_context;
    struct ns_session *native_session;
};
static DEFINE_MUTEX(vm_lock);
/* Quotas in record pages; zero means the format's limit of 2^31 slots. Read-only after boot,
 * so every context sees one policy. A table starts with one batch and grows by batches
 * (cap_rev_table_abi.h). */
static unsigned long node_initial_pages = CV_TABLE_BATCH_PAGES, node_batch_pages = CV_TABLE_BATCH_PAGES;
static unsigned long node_max_pages;
module_param(node_initial_pages, ulong, 0444);
module_param(node_batch_pages, ulong, 0444);
module_param(node_max_pages, ulong, 0444);
/* S-24: how many times the SAME (address, cause) may be "resolved" without changing anything
 * before the run is failed instead of retried. Generous, because a resolve followed by a quantum
 * stop is legitimately repeated by the launcher's idempotent CV_RESOLVE; a hardware fault the
 * module cannot clear repeats without bound, which is the case this catches. */
#define CV_FAULT_REPEAT_MAX 64U
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

/* Optional tag-aware Linux path. Read-only: a namespace never changes its
 * page-ownership contract while active. Stock kernels reject this option. */
static bool unpinned;
module_param(unpinned, bool, 0444);
MODULE_PARM_DESC(unpinned, "Use the platform's raw tag-aware mm instead of payload pins");
/* Experimental sweep optimization; independently opt in until qualified. */
static bool unpinned_fast_sweep;
module_param(unpinned_fast_sweep, bool, 0444);
MODULE_PARM_DESC(unpinned_fast_sweep, "Hold tagged base pages directly under their PTE lock");
/* EXPERIMENTAL, default OFF, read-only, and only with unpinned=1 (load refuses otherwise): resolve an unpinned
 * fault with handle_mm_fault on the VMA resolve() has already found, instead of fixup_user_fault's second
 * VMA lookup. The counter below is the proof, read back from sysfs, that a run took the path. */
static bool unpinned_fast_fault;
module_param(unpinned_fast_fault, bool, 0444);
MODULE_PARM_DESC(unpinned_fast_fault, "Experiment: reuse the validated anonymous VMA on unpinned faults");
static unsigned long unpinned_fast_faults;
module_param(unpinned_fast_faults, ulong, 0444);
MODULE_PARM_DESC(unpinned_fast_faults, "Faults resolved by the direct path since load (read-only)");
static bool unpinned_access_fault;
module_param(unpinned_access_fault, bool, 0444);
MODULE_PARM_DESC(unpinned_access_fault, "Experiment: preserve read access on lazy anonymous demand faults");
static unsigned long unpinned_access_reads;
module_param(unpinned_access_reads, ulong, 0444);
MODULE_PARM_DESC(unpinned_access_reads, "Anonymous read attempts with write-prefault suppressed; includes retries and failures");
static bool unpinned_exact_bounds;
static unsigned long unpinned_faults, unpinned_sweep_pages;
module_param(unpinned_faults, ulong, 0444);
module_param(unpinned_sweep_pages, ulong, 0444);

/* Pages a lazy-arena fault pins, from the faulting page on, clipped to the
 * arena, the VMA and the first page already pinned. 1 pins only the page. */
static unsigned long pin_ahead_pages = 1;
module_param(pin_ahead_pages, ulong, 0444);

/* ADAPTIVE FAULT-AHEAD: a context whose faults stream through an arena gets the next pages pinned
 * in the same CV_STEP, a window that grows while the stream lasts.
 *
 * A fault continues a stream when it lands on the first unpinned page after the stream's previous
 * fault and window, skipping pages admitted earlier, in the same arena (by its
 * mapping ID, which no later arena reuses, even at the same address), for the same context. Once a stream has fault_ahead_after such faults in a row, each further one
 * pins a window ahead of it: FAULT_AHEAD_FIRST pages, then twice the previous window, up to
 * fault_ahead_max. A fault that continues no stream replaces the least recently used slot
 * with an untrained stream. Other slots survive, including streams in the same arena:
 * a loop interleaving two arrays in one mapping can train both. When two streams converge
 * on the same unpinned page, the most recently used stream continues. Four slots per context.
 *
 * The window is pin_ahead()'s, unchanged: clipped to the arena and to the faulting VMA, ended
 * by the first page already pinned, each page pinned exactly as a fault on it would be, 16 at a
 * time with cv_step_stop() before each batch. So it never takes a page outside the arena or
 * outside the VMA, and a page it cannot take faults later as before.
 *
 * Why not a fixed window (pin_ahead_pages): that pins ahead of EVERY fault, and a program
 * whose faults scatter (502.gcc_r) then pins pages it never touches: pin_ahead_pages=16/64 was
 * measured 1.5-3 M instruction attempts worse on gcc. A window here needs fault_ahead_after
 * consecutive sequential faults first, and a stream's overshoot past its last touched page is
 * at most its last window.
 *
 * pin_ahead_pages > 1 or arena_pin selects the fixed window and turns this off: those are the
 * configurations the pin-ahead and step-hooks gates qualify, page for page. fault_ahead_max=0
 * turns it off as well. */
#define FAULT_AHEAD_FIRST 4UL
static unsigned long fault_ahead_max = 64;
module_param(fault_ahead_max, ulong, 0644);
static unsigned long fault_ahead_after = 2;
module_param(fault_ahead_after, ulong, 0644);
/* Pages pinned and window attempts since load (read-only counters). An attempt can pin zero
 * pages at a VMA boundary or budget cut. Growth follows sequential demand evidence, so an
 * empty attempt does not reset training. */
static unsigned long fault_ahead_pages, fault_ahead_windows;
module_param(fault_ahead_pages, ulong, 0444);
module_param(fault_ahead_windows, ulong, 0444);

/* Lazy NONLIN grants (2026-11-03). A non-linear grant used to be pinned WHOLE at mint:
 * pin_user_pages_fast(FOLL_WRITE) over the arena materialised every page of the stack (1 MiB),
 * the startup block (64 KiB) and the image's data and bss before the domain ran, 894 pages on
 * 502.gcc_r's test input of which the program touches a fraction, each one a Linux fault, a
 * clear_page, a pin, a tag sweep and an unpin -- about 10 K instruction attempts a page. Now a
 * NONLIN grant's pages are admitted one by one at the domain's first access, as a linear
 * arena's have been since the lazy path existed, under the same three checks the whole-arena
 * pin made: exclusive anonymous backing (PageAnon), mapped exactly once (page_mapcount 1), and
 * no frame twice in the arena (the frame set, frame_insert). Pages already resident at the
 * mint -- the startup block the launcher packed before granting it, the image's partial data
 * page it read in -- are pinned at the mint by the resident scan, so a launcher write that
 * preceded the grant is covered exactly as before. The meta grant is still pinned whole, see
 * eager_grant(). 0 restores the whole-arena pin for every NONLIN grant (the A/B arm);
 * read-only, so a run's module-params.txt records which it ran. */
static bool lazy_grants = true;
module_param(lazy_grants, bool, 0444);
MODULE_PARM_DESC(lazy_grants, "admit NONLIN grants page by page at first access (0: pin whole at mint)");
/* One line per page an arena gains outside a window: `admit` at a fault (with cause and pc),
 * `scan` from the resident scan, `touch` from CV_TOUCH; and one `retire` line per arena with its
 * whole census (admit, window, scan, touch, eager, released, swept). The instrument for a run
 * whose pinned page count differs from another run of the same program (which a whole-arena pin
 * could not show and a lazy one can), and what the pin-ahead, step-hooks and pthread gates read
 * to count one arena's pins and to check the module's counters against them. Off by default;
 * 0644 so a boot can turn it on for one run. */
static bool diag_admit;
module_param(diag_admit, bool, 0644);

static unsigned long node_capacity(const struct context *c)
{ return sublet_table_capacity(c->table); }
static unsigned long node_free_count(const struct context *c)
{ return sublet_table_free_count(c->table); }
static unsigned long node_available(const struct context *c)
{ return sublet_table_available(c->table); }
static void node_free_table(struct context *c)
{
    sublet_table_free(c->table);
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
/* Experimental VM backend: requires x-capstone-context-transport. Never
 * enable on the legacy RTL ABI, which uses the extension area differently. */
static bool native_slot_transport;
module_param(native_slot_transport, bool, 0444);
MODULE_PARM_DESC(native_slot_transport,
    "experimental VM protected-frame slot export/import (default off)");
static DECLARE_WAIT_QUEUE_HEAD(slot_wait);
/* Writers and admission hold vm_lock. A waiter snapshots this generation
 * before dropping the lock, then checks it without sleeping in the wait
 * predicate (prepare_to_wait_event has already changed its task state).
 * Publish before waking, including releases before the waiter is enqueued.
 * A changed generation is only a retry hint, never a slot reservation. */
static unsigned long slot_generation;
static void slot_changed(void);

/* Live frames across EVERY context, and one of them if asked. vm_lock is held by every caller. */
static unsigned slot_live(struct cv_thread **holder)
{
    struct context *c;
    struct cv_thread *t;
    unsigned n = 0;
    list_for_each_entry(c, &contexts, link)
        list_for_each_entry(t, &c->threads, link)
            if (t->started && !t->terminal && !t->slot_exported) {
                if (holder && !n) *holder = t;
                ++n;
            }
    return n;
}

static void slot_changed(void)
{
    lockdep_assert_held(&vm_lock);
    /* Waiters returning EAGAIN must not wake one another while all slots
     * remain occupied. Only a possible admission publishes a new epoch. */
    if (slot_live(NULL) < supervisor_slots) {
        WRITE_ONCE(slot_generation, slot_generation + 1);
        wake_up_all(&slot_wait);
    }
}

static u64 allocated_total, high_water, collections_total, reclaimed_total;

static unsigned long run(unsigned long frame, unsigned long action);

/* vm_lock serializes physical-slot ownership and protected-frame lifetime.
 * Only kernel pages can hold an importable transport image. User start frames
 * remain on the legacy admission path until they have a protected copy. */
static int slot_acquire(struct cv_thread *t)
{
    struct cv_thread *holder = NULL, *dead;
    struct context *c;
    unsigned long flags, result;
    lockdep_assert_held(&vm_lock);
    if (!native_slot_transport || (t->started && !t->slot_exported &&
        t->native_satp == csr_read(CSR_SATP))) return 0;
#ifdef CV_RACE_TEST
    /* Test build: the racer's start frame is a user page. It may enter while a
     * slot is free (the legacy admission), never by evicting another frame. */
    if (t->user_frame)
        return !t->started && slot_live(NULL) < supervisor_slots ? 0 : -EOPNOTSUPP;
#else
    if (t->user_frame) return -EOPNOTSUPP;
#endif
    /* Linux can change the ASID while a native I/O sleeps. Rebind the
     * trusted snapshot to this mm's current translation identity through
     * export/import, preserving its immutable register epoch. */
    if (t->started && !t->slot_exported) {
        preempt_disable(); local_irq_save(flags);
        result = run(t->frame_pa, 4);
        local_irq_restore(flags); preempt_enable();
        if (result) return -EIO;
        t->slot_exported = true;
    }
    /* A terminal event can precede close(). Reclaim its occupied slot before
     * admitting another frame, even when the exiting task has not run yet. */
    list_for_each_entry(c, &contexts, link)
        list_for_each_entry(dead, &c->threads, link)
            if (dead->started && dead->terminal && !dead->slot_exported) {
                preempt_disable(); local_irq_save(flags);
                run(dead->frame_pa, 1);
                local_irq_restore(flags); preempt_enable();
                dead->started = false;
            }
    if (slot_live(&holder) >= supervisor_slots) {
        if (!holder || holder->user_frame) return -EOPNOTSUPP;
        preempt_disable(); local_irq_save(flags);
        result = run(holder->frame_pa, 4);
        local_irq_restore(flags); preempt_enable();
        if (result) return -EIO;
        holder->slot_exported = true;
    }
    if (t->slot_exported) {
        preempt_disable(); local_irq_save(flags);
        /* Transport v5/v6: SCALAR_STATE.satp is at byte 1192. This
         * kernel-owned field is not an application authority import. */
        WRITE_ONCE(t->frame[1192 / 8], csr_read(CSR_SATP));
        result = run(t->frame_pa, 5);
        if (!result) t->native_satp = csr_read(CSR_SATP);
        local_irq_restore(flags); preempt_enable();
        if (result) return -EIO;
        t->slot_exported = false;
    }
    return 0;
}

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

/* The arena with the highest start below end, or NULL. */
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
/* The arena containing address, or NULL. */
static __always_inline struct arena *arena_at(struct context *c, unsigned long address)
{
    struct arena *a = address == ULONG_MAX ? NULL : arena_below(c, address + 1);
    return a && address - a->address < a->bytes ? a : NULL;
}
/* A fault in the image's read+execute prefix. Linux resolves it from the
 * file, with fault-around and no pin, so it is not one of the adapter's fault
 * events: how many there are depends on the page cache, not on the arena. */
static bool text_fault(struct context *c, unsigned long address)
{
    struct arena *a = arena_at(c, address);
    return a && address - a->address < a->rx_bytes;
}
/* Does [address, address + bytes) overlap an arena other than skip? */
static bool arena_overlaps(struct context *c, unsigned long address,
                           unsigned long bytes, struct arena *skip)
{
    struct arena *a = arena_below(c, address + bytes);
    if (a && a == skip) {
        struct rb_node *n = rb_prev(&a->node);
        a = n ? rb_entry(n, struct arena, node) : NULL;
    }
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
#define for_each_arena(a, c) \
    for (struct rb_node *n_ = rb_first(&(c)->arenas); \
         n_ && ((a) = rb_entry(n_, struct arena, node)); n_ = rb_next(n_))

static struct arena *find_arena(struct context *c, unsigned long address,
                                unsigned long bytes)
{
    struct arena *a = arena_at(c, address);
    return a && a->address == address && a->bytes == bytes ? a : NULL;
}

static bool arena_has_thread_frame(struct context *c, struct arena *a)
{
    struct cv_thread *t;
    list_for_each_entry(t, &c->threads, link)
        if (t->user_frame && t->frame_user == a->address)
            return true;
    return false;
}

#ifdef CV_RACE_TEST
/* The native race fixture uses a private test frame; production has no user-frame entry. */
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

#endif

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

/* The first address in [address, end) whose PTE is present, written to *at with its frame in
 * *pfn: 1, or 0 at the end, or an error. An absent table level is skipped at its own
 * granularity, so a 1 MiB stack with four resident pages costs one PMD lookup and one walk of
 * that PMD's entries rather than 256 page-table walks. Missing entries hold no application
 * tags; swapped and huge mappings are outside this profile and fail closed, as they always
 * did (-EOPNOTSUPP), a bad table -EFAULT. Called with mmap_read_lock held. */
static long next_resident(struct mm_struct *mm, unsigned long address, unsigned long end,
                          unsigned long *at, unsigned long *pfn)
{
    while (address < end) {
        pgd_t *pgd = pgd_offset(mm, address);
        p4d_t *p4d;
        pud_t *pud;
        pmd_t *pmd;
        pte_t *pte;
        spinlock_t *lock;
        unsigned long stop;
        long found = 0;
        if (pgd_none(*pgd)) { address = pgd_addr_end(address, end); continue; }
        if (pgd_bad(*pgd)) return -EFAULT;
        p4d = p4d_offset(pgd, address);
        if (p4d_none(*p4d)) { address = p4d_addr_end(address, end); continue; }
        if (p4d_bad(*p4d)) return -EFAULT;
        pud = pud_offset(p4d, address);
        if (pud_none(*pud)) { address = pud_addr_end(address, end); continue; }
        if (pud_leaf(*pud)) return -EOPNOTSUPP;
        if (pud_bad(*pud)) return -EFAULT;
        pmd = pmd_offset(pud, address);
        if (pmd_none(*pmd)) { address = pmd_addr_end(address, end); continue; }
        if (pmd_leaf(*pmd)) return -EOPNOTSUPP;
        if (pmd_bad(*pmd)) return -EFAULT;
        pte = pte_offset_map_lock(mm, pmd, address, &lock);
        if (!pte) return -EFAULT;
        stop = pmd_addr_end(address, end);
        for (unsigned long i = 0; address < stop; address += PAGE_SIZE, ++i) {
            pte_t value = READ_ONCE(pte[i]);
            if (pte_none(value)) continue;
            if (!pte_present(value)) { found = -EOPNOTSUPP; break; }
            *at = address; *pfn = pte_pfn(value);
            found = 1;
            break;
        }
        pte_unmap_unlock(pte, lock);
        if (found) return found;
    }
    return 0;
}

/* Slots of a NONLIN arena's frame set for n pages: a power of two of at
 * least 2n, so linear probing stays short. */
static unsigned long frame_slots(long n)
{
    return roundup_pow_of_two(2 * (unsigned long)n);
}

/* Record a page's frame in the arena's frame set: false if the frame is there already, i.e.
 * another page of the arena holds it. Expected O(1): the set is at most half full, since no
 * index of pages[] is filled twice. */
static bool frame_insert(struct arena *a, struct page *page)
{
    unsigned long mask, key, at;
    if (!a->frames) return true;
    mask = frame_slots(a->count) - 1;
    key = page_to_pfn(page) + 1;
    at = (key * 0x9e3779b97f4a7c15UL) >> 32 & mask;
    while (a->frames[at]) {
        if (a->frames[at] == key) return false;
        at = (at + 1) & mask;
    }
    a->frames[at] = key;
    return true;
}

/* Pin, check and record every resident page of the arena that pages[] does not hold yet,
 * materialising none. Linux may populate a page with no fault this module sees -- the
 * launcher's writes before a grant, its reads of a stack it walks after a fault, a mapping
 * moved by mremap -- and such a page can carry a tag only if the domain stored one, so it is
 * swept with the rest. Each admitted page passes the same three checks as one admitted at a
 * fault. gup_flags are the caller's: FOLL_FORCE | FOLL_NOFAULT inspects what is there (retire
 * and collect: a PROT_NONE or read-only page is read, no PTE right changes, no backing is
 * manufactured); FOLL_WRITE | FOLL_FORCE at the mint of a lazy grant also breaks a private file
 * page's copy-on-write, as the whole-arena pin did, so the arena never retains a page it does
 * not own. The zero page is skipped: it holds no tag and the domain's first store to it faults
 * into resolve(), which gives it real backing. */
static int pin_remaining(struct context *c, struct arena *a, unsigned int gup_flags)
{
    int error = 0;
    unsigned long base = a->address + a->rx_bytes, address, pfn;
    unsigned j = 0, run_end = 0;
    mmap_read_lock(c->mm);
    while (j < a->count) {
        struct page *page;
        long resident;
        /* A page already held is skipped without touching the page tables: at the retire of
         * a heap arena that is every page (519.lbm_r's 105 K), and a walk per page there cost
         * 8 M attempts when this was first written. Only the runs of absent entries between
         * them are walked, and those a table level at a time. */
        if (a->pages[j]) { ++j; continue; }
        if (run_end <= j) {
            run_end = j + 1;
            while (run_end < a->count && !a->pages[run_end]) ++run_end;
        }
        resident = next_resident(c->mm, base + (unsigned long)j * PAGE_SIZE,
                                 base + (unsigned long)run_end * PAGE_SIZE, &address, &pfn);
        if (resident < 0) { error = resident; break; }
        if (!resident) { j = run_end; continue; }
        j = (address - base) / PAGE_SIZE;
        if (is_zero_pfn(pfn)) { ++j; continue; }
        if (pin_user_pages_remote(c->mm, address, 1, gup_flags, &page, NULL, NULL) != 1) {
            error = -EFAULT; break;
        }
        if (!PageAnon(page) || page_mapcount(page) != 1 || (a->frames && !frame_insert(a, page))) {
            unpin_user_page(page); error = -EACCES; break;
        }
        if (diag_admit)
            pr_info("capstone_vm: scan arena 0x%lx+0x%lx page %u pfn 0x%lx flags 0x%x\n",
                    a->address, a->bytes, j, page_to_pfn(page), gup_flags);
        ++a->scans;
        a->pages[j] = page;
        ++c->stats.pinned_pages;
        c->stats.peak_pages = max(c->stats.peak_pages, c->stats.pinned_pages);
        ++j;
    }
    mmap_read_unlock(c->mm);
    return error;
}

/* A chain of pages of physical addresses: word 0 links the next page, word 1 counts the
 * entries that follow it. */
#define CV_HELD_ENTRIES (PAGE_SIZE / sizeof(unsigned long) - 2)
struct collect_list {
    unsigned long *head, *tail, count;
};

static int collect_page(struct collect_list *list, unsigned long pa)
{
    if (!list->tail || list->tail[1] == CV_HELD_ENTRIES) {
        unsigned long *page = (void *)get_zeroed_page(GFP_KERNEL_ACCOUNT);
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

static unsigned long unpinned_page_summary(struct page *page)
{
    unsigned long result;
    asm volatile(".insn i 0x0b, 6, %0, %1, 0"
                 : "=r"(result) : "r"(page_to_phys(page)) : "memory");
    return result;
}

/* Return one referenced tagged base page, or zero at the end. Unlike a fault or
 * a data access this walk must neither populate memory nor mark it accessed.
 * Inspect the sticky tag summary and take the reference under the PTE lock:
 * retaining a PFN after dropping that lock would race migration/page reuse.
 * vm_lock stops capability execution; mmap_lock excludes fork and unmap until
 * the caller has swept the COMPLETE mm and recycled identities. This relies on
 * every tag-publication route setting the sticky summary, and on payload grants
 * excluding shared/device backing. vm_lock stops publication in this namespace;
 * it does not stop a different mm from writing shared untagged backing.
 * No allocation occurs under the PTE lock. At most one PTE table is scanned
 * without releasing it, including when every resident page has no tags.
 *
 * Lock hold time. The PTE spinlock (preemption off) is held across up to
 * PTRS_PER_PTE (512) PTE reads and, for each present non-zero page, one
 * unpinned_page_summary() query. The old scanner took the lock once per
 * resident page for one PTE read. On the QEMU model a query is a host table
 * lookup, so this costs nothing measurable here. On silicon a query may be an
 * uncached tag-controller access, which would make the hold up to 512 such
 * accesses per table; that latency has not been measured. The option is
 * therefore experimental and default OFF, and the hold must be measured on the
 * target before any default-on decision. */
static long next_tagged(struct mm_struct *mm, unsigned long address, unsigned long end,
                        unsigned long *at, struct page **held)
{
    while (address < end) {
        pgd_t *pgd = pgd_offset(mm, address);
        p4d_t *p4d;
        pud_t *pud;
        pmd_t *pmd;
        pte_t *pte;
        spinlock_t *lock;
        unsigned long stop;
        long found = 0;
        if (pgd_none(*pgd)) { address = pgd_addr_end(address, end); continue; }
        if (pgd_bad(*pgd)) return -EFAULT;
        p4d = p4d_offset(pgd, address);
        if (p4d_none(*p4d)) { address = p4d_addr_end(address, end); continue; }
        if (p4d_bad(*p4d)) return -EFAULT;
        pud = pud_offset(p4d, address);
        if (pud_none(*pud)) { address = pud_addr_end(address, end); continue; }
        if (pud_leaf(*pud)) return -EOPNOTSUPP;
        if (pud_bad(*pud)) return -EFAULT;
        pmd = pmd_offset(pud, address);
        if (pmd_none(*pmd)) { address = pmd_addr_end(address, end); continue; }
        if (pmd_leaf(*pmd)) return -EOPNOTSUPP;
        if (pmd_bad(*pmd)) return -EFAULT;
        pte = pte_offset_map_lock(mm, pmd, address, &lock);
        if (!pte) return -EFAULT;
        stop = pmd_addr_end(address, end);
        for (unsigned long i = 0; address < stop; address += PAGE_SIZE, ++i) {
            pte_t value = READ_ONCE(pte[i]);
            struct page *page;
            unsigned long pfn;
            if (pte_none(value)) continue;
            if (!pte_present(value)) { found = -EOPNOTSUPP; break; }
            pfn = pte_pfn(value);
            if (is_zero_pfn(pfn)) continue;
            /* Device/special PTEs are not ordinary RAM references. Refuse
             * instead of treating their PFNs as struct page objects. */
            if (pte_special(value) || pte_devmap(value) || !pfn_valid(pfn)) {
                found = -EOPNOTSUPP; break;
            }
            page = pfn_to_page(pfn);
            if (!unpinned_page_summary(page)) continue;
            /* Only private anonymous base pages are supported. These checks
             * precede get_page, so no device/compound reference is acquired. */
            if (PageCompound(page) || !PageAnon(page) || page_mapcount(page) != 1) {
                found = -EOPNOTSUPP; break;
            }
            get_page(page);
            *at = address;
            *held = page;
            found = 1;
            break;
        }
        pte_unmap_unlock(pte, lock);
        if (found) return found;
        cond_resched();
    }
    return 0;
}

/* vm_lock and mmap_lock are held. Temporary page references stabilize the sweep's physical
 * list, including pages outside currently registered arenas. Shared tagged
 * backing is refused: this namespace cannot sweep another process's copies.
 * Migration entries, swapped PTEs and huge mappings fail closed as well.
 * FOLL_GET references prevent migration's expected-reference freeze. Unlike
 * FOLL_PIN they do not require unsharing clean launcher pages during a read;
 * the tagged pages we retain must already be private and exclusively mapped. */
static int unpinned_hold_mm_with(struct mm_struct *mm, struct collect_list *held, bool fast)
{
    struct vm_area_struct *v;
    unsigned long cursor = 0;

    while ((v = find_vma(mm, cursor))) {
        unsigned long address = v->vm_start, end = v->vm_end;
        /* Device/PFN mappings cannot be granted as payload arenas. They are
         * not ordinary tagged RAM and have no GUP page to stabilize. */
        if (v->vm_flags & (VM_IO | VM_PFNMAP | VM_MIXEDMAP)) {
            cursor = end;
            continue;
        }
        while (address < end) {
            struct page *page;
            unsigned long at, pfn;
            long found;
            int error;
            if (fast)
                found = next_tagged(mm, address, end, &at, &page);
            else
                found = next_resident(mm, address, end, &at, &pfn);
            if (found < 0) return found;
            if (!found) break;
            address = at + PAGE_SIZE;
            if (!fast) {
                if (is_zero_pfn(pfn)) continue;
                found = get_user_pages_remote(mm, at, 1, FOLL_FORCE | FOLL_NOFAULT,
                                              &page, NULL, NULL);
                if (found != 1) {
                    pr_warn("capstone_vm: unpinned sweep cannot hold %lx flags %lx: %ld\n",
                            at, v->vm_flags, found);
                    return found < 0 ? found : -EFAULT;
                }
                if (!unpinned_page_summary(page)) {
                    put_page(page);
                    continue;
                }
            }
            if (!PageAnon(page) || page_mapcount(page) != 1 ||
                (v->vm_flags & VM_SHARED)) {
                put_page(page);
                return -EOPNOTSUPP;
            }
            error = collect_page(held, page_to_phys(page));
            if (error) { put_page(page); return error; }
            ++unpinned_sweep_pages;
        }
        cursor = end;
    }
    return 0;
}

/* The first sweep of a module load says which scanner ran, once, so a guest log shows what was
 * executed and not only what the parameters asked for. */
static int unpinned_hold_mm(struct mm_struct *mm, struct collect_list *held)
{
    static bool announced;
    if (unlikely(!announced)) {
        announced = true;
        pr_emerg("capstone_vm: first unpinned sweep uses the %s scanner\n",
                 unpinned_fast_sweep ? "fast" : "old");
    }
    return unpinned_hold_mm_with(mm, held, unpinned_fast_sweep);
}

static void unpinned_release_mm(struct collect_list *held);

#ifdef CV_TEST_HOOKS
/* Test build only (make CV_TEST_HOOKS=1). With unpinned_sweep_diff=1 every sweep runs the OTHER
 * scanner as well, on the same mm under the same locks, and compares the two held-page lists:
 * the old whole-mm GUP walk against the PTE-locked walk. Both traverse VMAs and addresses in
 * the same order, so equal lists mean equal sequences; a count, a position-wise comparison and
 * order-independent sum/xor digests of the physical addresses are all kept. The counters are
 * read-only parameters, so a runner reads them from /sys/module/capstone_vm/parameters/. A
 * scanner that omits a page, stops early at a PTE table or VMA boundary, or holds a different
 * page shows up as a mismatch. The extra scan holds its own references and releases them. It
 * changes no state the caller uses and does not alter unpinned_sweep_pages. */
static bool unpinned_sweep_diff;
module_param(unpinned_sweep_diff, bool, 0644);
static unsigned long diff_runs, diff_active_pages, diff_other_pages, diff_mismatch, diff_other_errors,
                     diff_digest_mismatch, diff_first_bad;
module_param(diff_runs, ulong, 0444);
module_param(diff_active_pages, ulong, 0444);
module_param(diff_other_pages, ulong, 0444);
module_param(diff_mismatch, ulong, 0444);
module_param(diff_other_errors, ulong, 0444);
module_param(diff_digest_mismatch, ulong, 0444);
module_param(diff_first_bad, ulong, 0444);

struct diff_cursor { unsigned long *page; unsigned long j; };

static bool diff_next(struct diff_cursor *c, unsigned long *pa)
{
    while (c->page && c->j >= c->page[1]) {
        c->page = c->page[0] ? phys_to_virt(c->page[0]) : NULL;
        c->j = 0;
    }
    if (!c->page) return false;
    *pa = c->page[2 + c->j++];
    return true;
}

static void unpinned_sweep_diff_check(struct mm_struct *mm, struct collect_list *active)
{
    struct collect_list other = {0};
    struct diff_cursor a = {active->head, 0}, b = {other.head, 0};
    unsigned long saved = unpinned_sweep_pages, pa, pb, index = 0, bad = 0;
    unsigned long sum_a = 0, sum_b = 0, xor_a = 0, xor_b = 0;
    bool more_a, more_b;
    int error;
    if (!unpinned || !unpinned_sweep_diff) return;
    error = unpinned_hold_mm_with(mm, &other, !unpinned_fast_sweep);
    unpinned_sweep_pages = saved;
    ++diff_runs;
    if (error) {
        ++diff_other_errors;
        pr_emerg("capstone_vm: sweep diff: other scanner failed %d after %lu pages\n", error, other.count);
        unpinned_release_mm(&other);
        return;
    }
    b.page = other.head;
    for (;;) {
        more_a = diff_next(&a, &pa);
        more_b = diff_next(&b, &pb);
        if (more_a) { sum_a += pa; xor_a ^= pa * 0x9e3779b97f4a7c15UL; }
        if (more_b) { sum_b += pb; xor_b ^= pb * 0x9e3779b97f4a7c15UL; }
        if (!more_a && !more_b) break;
        if (!bad && (!more_a || !more_b || pa != pb)) bad = index + 1;
        ++index;
    }
    diff_active_pages += active->count;
    diff_other_pages += other.count;
    if (bad || active->count != other.count) {
        if (!diff_mismatch) diff_first_bad = bad;
        if (++diff_mismatch <= 16)
            pr_emerg("capstone_vm: sweep diff MISMATCH run %lu: %s %lu pages, other %lu pages, first differing index %lu\n",
                     diff_runs, unpinned_fast_sweep ? "fast" : "old", active->count, other.count, bad);
    }
    if (sum_a != sum_b || xor_a != xor_b) ++diff_digest_mismatch;
    if (!(diff_runs & 63))
        pr_emerg("capstone_vm: sweep diff: runs %lu pages %lu/%lu mismatches %lu digest %lu errors %lu\n",
                 diff_runs, diff_active_pages, diff_other_pages, diff_mismatch, diff_digest_mismatch,
                 diff_other_errors);
    unpinned_release_mm(&other);
}
#else
static inline void unpinned_sweep_diff_check(struct mm_struct *mm, struct collect_list *active) {}
#endif

static void unpinned_release_mm(struct collect_list *held)
{
    unsigned long *page = held->head;
    while (page) {
        for (unsigned long j = 0; j < page[1]; ++j)
            put_page(pfn_to_page(page[2 + j] >> PAGE_SHIFT));
        page = page[0] ? phys_to_virt(page[0]) : NULL;
    }
    collect_free(held);
}

/* Remove an old namespace's storage tags without projecting or changing its
 * bytes. A raw load supplies the original low word to a scalar store (rule 1).
 * This runs only before a fresh context is made visible, with mmap_lock held.
 * Interrupts cannot interpose while an opaque register is live. */
static int unpinned_scrub_mm(struct mm_struct *mm, unsigned long only_pages, unsigned long only_granules)
{
    struct collect_list held = {0};
    unsigned long *page, scrubbed = 0;
    int error;
    mmap_read_lock(mm);
    error = unpinned_hold_mm(mm, &held);
    if (error) goto out;
    unpinned_sweep_diff_check(mm, &held);
    page = held.head;
    while (page) {
        for (unsigned long j = 0; j < page[1]; ++j) {
            void *at = phys_to_virt(page[2 + j]);
            unsigned long flags, count = PAGE_SIZE / 16;
            /* only_pages / only_granules are zero outside the test build's broken variants */
            if (only_pages && scrubbed++ >= only_pages) continue;
            if (only_granules) count = min(count, only_granules);
            local_irq_save(flags);
            asm volatile("1: .insn i 0x0b, 4, t0, %0, 0\n\t"
                         "sd t0, 0(%0)\n\taddi %0, %0, 16\n\t"
                         "addi %1, %1, -1\n\tbnez %1, 1b\n\tli t0, 0"
                         : "+r"(at), "+r"(count) : : "t0", "memory");
            local_irq_restore(flags);
        }
        page = page[0] ? phys_to_virt(page[0]) : NULL;
    }
out:
    unpinned_release_mm(&held);
    mmap_read_unlock(mm);
    return error;
}

/* ---- The growth protocol (cap_rev_table_abi.h) ----
 *
 * The processor raises cause 30 when an allocation finds no slot and the kernel can still grow
 * the table. Every growth attempt is one call of ensure_nodes(c, required), and only
 * ensure_nodes writes NOGROW. It adds record pages when fewer than required slots are available,
 * at least one batch, published before the larger capacity. Then it sets NOGROW exactly when
 * the table still has fewer than required slots, so the answer to cause 30 always makes
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
    c->stats.node_bytes += pages * PAGE_SIZE;
    if (node_capacity(c) != before) {
        ++c->stats.node_growths;
        ++table_growths;
    }
    nogrow = node_available(c) < required;
    if (nogrow != sublet_table_nogrow(c->table)) {
        sublet_table_set_nogrow(c->table, nogrow);
        if (nogrow) ++nogrow_sets; else ++nogrow_clears;
    }
    return error;
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
#if defined(CV_TASK_BACKEND) && defined(CV_TEST_HOOKS)
/* Deliberate S-origin admission exception, once, to exercise CT + extable. */
static bool task_refuse_once;
module_param(task_refuse_once, bool, 0600);
#endif
static unsigned long run_event(struct cv_thread *t, unsigned long action)
{
    WRITE_ONCE(t->frame[2], CV_EVENT_UNWRITTEN);
    /* The sentinel must reach memory before the switcher can write the record over it; a kernel
     * store still draining after the escape's write would turn a real event into a refusal. */
    asm volatile("fence rw,rw" ::: "memory");
#ifdef CV_TASK_BACKEND
    if (current->thread.capstone) {
#ifdef CV_TEST_HOOKS
        if (task_refuse_once) {
            task_refuse_once = false;
            action = ~0UL;
        }
#endif
        if (riscv_capstone_run_user(t->frame_pa, action))
            return CV_EVENT_UNWRITTEN;
    } else
#endif
        run_x0(t->frame_pa, action);
    return READ_ONCE(t->frame[2]);
}
static unsigned long select_root(struct context *c)
{
    unsigned long old;
    asm volatile("csrr %0, 0x5c1\ncsrw 0x5c1, %1"
                 : "=&r"(old) : "r"(c->table_pa) : "memory");
    return old;
}
/* 0, not the `root` select_root returned: outside a window S is in the physical namespace, so the
 * window's end is an edge and its flush fires (R-58; see run()). */
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
                       const struct cv_map *r)
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

/* Clear every capability tag in a page; its bytes are dead (see below). A scalar store
 * clears the tag of the 16-byte granule it writes into, whatever its width
 * (capstone-qemu's store hook rounds both ends to granules; on the RTL a
 * plain store writes ctag 0 for the granule), so rewriting one 8-byte word
 * of each granule is enough: half the loads and stores of rewriting every
 * word, which was most of 519.lbm_r's module instructions. */
/* A store to any word of a granule clears its tag; the data does not matter here, because a
 * page reaches this function only on its way back to Linux (retire of an unmapped or destroyed
 * arena, the tail a remap truncates), where every pinned page is exclusive anonymous memory
 * (resolve() admits nothing else) and nobody reads it again. So one store of zero per granule,
 * eight granules per iteration: the load-and-rewrite form it replaces cost a load, a store and
 * the loop's three instructions per granule (2.9M attempts of a gcc test run's 2129 pages). */
static void clear_page_tags(struct page *page)
{
    unsigned long *p = page_address(page);
    for (unsigned j = 0; j < PAGE_SIZE / sizeof(*p); j += 8 * 16 / sizeof(*p)) {
        WRITE_ONCE(p[j], 0UL);      WRITE_ONCE(p[j + 2], 0UL);
        WRITE_ONCE(p[j + 4], 0UL);  WRITE_ONCE(p[j + 6], 0UL);
        WRITE_ONCE(p[j + 8], 0UL);  WRITE_ONCE(p[j + 10], 0UL);
        WRITE_ONCE(p[j + 12], 0UL); WRITE_ONCE(p[j + 14], 0UL);
    }
}

static int retire(struct context *c, struct arena *a, bool destroying)
{
    unsigned i, swept;
    if (!destroying && arena_has_thread_frame(c, a))
        return -EBUSY;
    if (destroying && WARN_ON_ONCE(a->copy_holds)) return -EBUSY;
    if (!destroying && !unpinned && !a->copy_holds) {
        int error = pin_remaining(c, a, FOLL_FORCE | FOLL_NOFAULT);
        if (error) return error;
    }
    /* S-21: skip a retire already done at CV_THREAD_CREATE. Without this, the domain unmapping
     * its child's frame page takes the NON-destroying path here, cap_rev_retire returns
     * CAP_REV_DEAD for the invalid node, status is 1, and this function reports -EIO -- which the
     * launcher surfaces as "service: Input/output error" and a 125 exit. Measured: the child thread
     * ran and printed VIRTUAL_THREAD_CHILD, and the step then failed on the unmap. An earlier
     * version of this change reasoned that no flag was needed because retire(destroying=true)
     * ignores the status; that enumerated only one of the two paths. */
    /* The 63-bit format has no CSRETIRE: a block ends with CREVOKE through a reference to it,
     * and the kernel keeps none. So a program's unmap is refused until the trap route ends a
     * block through the reference its request carries (plan v5 step 6). Teardown ends every
     * lifetime at once by freeing the table. */
    if (!destroying && !a->ancestor_retired) return -EOPNOTSUPP;
    if (a->copy_holds) {
        a->ancestor_retired = true;
        a->copy_retired = true;
        return 0;
    }
    /* Clear every tag before returning frames to Linux. Later collection can
     * reuse dead IDs only after sweeping all remaining namespace storage.
     * The pages are then unpinned in ONE batched call instead of one per page.
     * unpin_user_pages_dirty_lock() runs a single sanity_check_pinned_pages over
     * the run and, for a folio that is already dirty, skips the folio_lock +
     * folio_mark_dirty that the per-page set_page_dirty_lock() took
     * unconditionally. make_dirty stays true, so the dirty guarantee for a page
     * whose PTE was cleaned while pinned is unchanged. a->pages is compacted in
     * place (kvfree'd just below): the batched call needs its pages contiguous
     * and non-NULL, and n<=i throughout so the compaction never overwrites an
     * entry it has not already read. */
    {
        unsigned n = 0;
        for (i = 0; !unpinned && i < a->count; ++i) {
            if (!a->pages[i]) continue;
            clear_page_tags(a->pages[i]);
            a->pages[n++] = a->pages[i];
        }
        if (n) {
            unpin_user_pages_dirty_lock(a->pages, n, true);
            c->stats.pinned_pages -= n;
        }
        swept = n;
    }
    if (diag_admit)
        pr_info("capstone_vm: retire arena 0x%lx+0x%lx admit=%u window=%u scan=%u touch=%u eager=%u "
                "released=%u swept=%u faulted=%llu\n", a->address, a->bytes, a->admits, a->windows,
                a->scans, a->touches, a->eager, a->released, swept,
                (unsigned long long)a->faulted);
    if (unpinned && diag_admit)
        pr_info("capstone_vm: unpinned retire arena 0x%lx events=%u resolves=%u touch=%u ahead=%u\n",
                a->address, a->unpinned_events, a->unpinned_resolves, a->unpinned_touches,
                a->unpinned_ahead);
    --c->stats.arenas;
    rb_erase(&a->node, &c->arenas);
    c->registered_bytes -= a->bytes;
    kvfree(a->pages);
    kvfree(a->frames);
    kfree(a);
    return 0;
}



/* The meta block (reg 10 before the start) stays eager: asynchronous signal publication writes
 * it and cannot issue a sleeping CV_TOUCH. The main exchange (reg 11) is admitted through the
 * delegate dispatcher's prepare_write callback before every possible output span; worker
 * transports are touched at each attachment and held until detach/exit. In pinned mode a
 * launcher write must not populate an unheld page: exit_mm can free it before context_destroy
 * sweeps its tags. In experimental unpinned mode arch_free_page clears released pages and
 * the raw copy hooks preserve tags across COW/migration. CV_TOUCH faults pages in without
 * retaining them. The launcher's cache records admission of the immutable main-exchange
 * mapping, not residency: eviction does not invalidate that admission or its tag coverage.
 * The stack (reg 2), the image (3) and the startup block (12)
 * are written by the launcher only BEFORE their grant, and those pages are pinned at the mint
 * by the resident scan; after it, only the domain's own accesses populate them, every one of
 * which faults into resolve(). The launcher reads the stack after a fault (its frame chain),
 * which maps the zero page at most. Reply grants are copyable and use lazy admission. */
static bool eager_grant(const struct cv_thread *t, const struct cv_map *r)
{
    return !t->started && r->reg == 10;
}

static int add(struct context *c, struct cv_thread *t, struct cv_map *r)
{
    unsigned i, count;
    struct arena *a;
    unsigned long *slot;
    struct page **pages;
    long n;
    /* Runtime-created mappings may reserve inaccessible pages and later change
     * their PTE permissions. Capability rights remain a ceiling on access. */
    bool mapping_reply = t->started && r->reg == 10;
    if (!r->bytes || r->bytes > CV_MAX_REGION_BYTES ||
        (!mapping_reply && r->bytes > CV_MAX_EAGER_BYTES) ||
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
        r->permissions > 7 || r->reg > 31 ||
        r->cursor < r->address || r->cursor > r->address + r->bytes ||
        (t->started && (t->event != 3 || r->reg != 10 || t->reply_cap)) ||
        (!t->started && r->reg != 2 && r->reg != 3 &&
         r->reg != 10 && r->reg != 11 && r->reg != 12 && r->reg != 13) ||
        (r->reg == 3 && !(r->permissions & 1)) ||
        /* Only the initial image grant has a read+execute prefix, a whole
         * number of pages strictly inside it, and it starts there. */
        (r->rx_bytes && (r->reg != 3 || t->started || (r->rx_bytes & (PAGE_SIZE - 1)) ||
                         r->rx_bytes >= r->bytes || r->cursor >= r->address + r->rx_bytes)))
        return -EINVAL;
    if (arena_overlaps(c, r->address, r->bytes, NULL)) return -EINVAL;
    if (c->registered_bytes + r->bytes > CV_MAX_BYTES) return -ENOMEM;
    if (grant_nodes(c)) return -ENOMEM;
    mmap_read_lock(c->mm);
    for (unsigned long at = r->address; at < r->address + r->bytes;) {
        struct vm_area_struct *v = find_vma(c->mm, at);
        unsigned long rx_end = r->address + r->rx_bytes;
        bool rx = at < rx_end;
        bool bad;
        if (!v || v->vm_start > at || (v->vm_flags & VM_SHARED)) {
            mmap_read_unlock(c->mm); return -EACCES;
        }
        if (rx)
            /* Image text: a private mapping of the image file whose writers
             * CV_IMAGE denied, readable and executable, never writable, so its
             * bytes stay the bytes validated. */
            bad = !v->vm_file || !c->image ||
                  file_inode(v->vm_file) != file_inode(c->image) || (v->vm_flags & VM_WRITE) ||
                  (v->vm_flags & (VM_READ | VM_EXEC)) != (VM_READ | VM_EXEC);
        else if (r->rx_bytes)
            /* Image data: private and writable, never executable. A file
             * mapping is allowed: pinning breaks its copy-on-write below and
             * only exclusive anonymous pages are retained. */
            bad = (v->vm_flags & (VM_READ | VM_WRITE)) != (VM_READ | VM_WRITE) ||
                  (v->vm_flags & VM_EXEC);
        else
            bad = v->vm_file ||
                  ((r->permissions & 4) && !(v->vm_flags & (mapping_reply ? VM_MAYREAD : VM_READ))) ||
                  ((r->permissions & 2) && !(v->vm_flags & (mapping_reply ? VM_MAYWRITE : VM_WRITE))) ||
                  ((r->permissions & 1) && !(v->vm_flags & (mapping_reply ? VM_MAYEXEC : VM_EXEC)));
        if (bad) { mmap_read_unlock(c->mm); return -EACCES; }
#ifdef CONFIG_TRANSPARENT_HUGEPAGE
        if (!(v->vm_flags & VM_NOHUGEPAGE)) {
            mmap_read_unlock(c->mm); return -EOPNOTSUPP;
        }
#endif
        if (unpinned && !mapping_reply && !rx && (!lazy_grants || eager_grant(t, r))) {
            /* Unpinned: populate every page of a NONLIN grant through the kernel's own
             * fault path, as pin_user_pages_fast(FOLL_WRITE) below would, but take no pin. Write
             * access where the VMA grants it, so the PTE comes out writable and dirty as GUP's did. */
            unsigned long next = min((unsigned long)(r->address + r->bytes), v->vm_end);
            unsigned int flags = (v->vm_flags & VM_WRITE) ? FAULT_FLAG_WRITE : 0;
            for (unsigned long p = at; p < next; p += PAGE_SIZE)
                if (fixup_user_fault(c->mm, p, flags, NULL)) {
                    mmap_read_unlock(c->mm); return -EFAULT;
                }
        }
        at = min(min((unsigned long)(r->address + r->bytes), v->vm_end),
                 rx ? rx_end : ULONG_MAX);
    }
    mmap_read_unlock(c->mm);
    /* Everything that can fail is prepared before the grant is minted. The
     * record and its page array are charged to the process's
     * memory cgroup: there is no count limit, only the registered-bytes one,
     * and a million single-page arenas fit under it. */
    /* An arena id names one registration for the context's whole life: the launcher releases
     * a worker's transport hold only on the id it recorded at the attachment (exec.c,
     * transport_mapping), so ids are never reused. The counter advances only on success and
     * would wrap after 2^64 registrations; the registration that would wrap it is refused. */
    if (c->next_id == U64_MAX) return -ENOSPC;
    a = kzalloc(sizeof(*a), GFP_KERNEL_ACCOUNT);
    if (!a) return -ENOMEM;
    a->id = c->next_id + 1;
    count = (r->bytes - r->rx_bytes) / PAGE_SIZE;
    pages = NULL;
    if (!unpinned) {
        pages = kvcalloc(count, sizeof(*pages), GFP_KERNEL_ACCOUNT);
        if (!pages) { kfree(a); return -ENOMEM; }
    }
    /* The arena's geometry, before anything is pinned against it. The frame set lives as long
     * as the arena (a page admitted at any later fault is checked against it), so it is
     * accounted like the page array. */
    a->address = r->address; a->bytes = r->bytes; a->rx_bytes = r->rx_bytes;
    a->pages = pages; a->count = count;
    a->lazy = mapping_reply || (lazy_grants && !eager_grant(t, r));
    if (!unpinned) {
        a->frames = kvcalloc(frame_slots(count), sizeof(*a->frames), GFP_KERNEL_ACCOUNT);
        if (!a->frames) { kvfree(pages); kfree(a); return -ENOMEM; }
    }
    n = 0;
    if (!unpinned && !a->lazy) {
        /* Pinned whole at the mint (lazy_grants off, or eager_grant): Linux resolves each first
         * touch through GUP, then the adapter pins that backing until retirement. Each page:
         * exclusive anonymous backing, mapped once, its frame not held by another page of the
         * arena. The last check matters here because GUP-fast takes no mmap lock: a mapping
         * moved while it walks can show one frame at two addresses, each with mapcount 1. */
        n = pin_user_pages_fast(r->address + r->rx_bytes, count, FOLL_WRITE, pages);
        if (n != count) goto unpin;
        for (i = 0; i < n; ++i)
            if (!PageAnon(pages[i]) || page_mapcount(pages[i]) != 1 || !frame_insert(a, pages[i]))
                goto unpin;
    } else if (!unpinned && !mapping_reply) {
        /* Lazy: the pages already resident -- what the launcher wrote before the grant -- are
         * pinned and checked now, exactly as the whole-arena pin would have (FOLL_WRITE breaks
         * a private file page's copy-on-write); every absent page is admitted at the domain's
         * first access by resolve(), under the same checks. */
        if (pin_remaining(c, a, FOLL_WRITE | FOLL_FORCE)) goto unpin;
    }
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
    /* The block: P, derived from R over the whole arena, in the frame slot. The image also
     * appears in register slot 0, which run() requires filled before it starts a thread. */
    if (root_derive(c, slot, r->reg == 3 ? t->frame + 8 : NULL, r)) goto unpin;
    arena_insert(c, a);
    c->registered_bytes += a->bytes;
    r->id = c->next_id = a->id;
    /* The whole-arena pin's n pages are counted here; the resident scan counted its own. */
    a->eager = n;
    ++c->stats.arenas; c->stats.pinned_pages += n;
    c->stats.peak_pages = max(c->stats.peak_pages, c->stats.pinned_pages);
    if (t->started) t->reply_cap = true;
    return 0;
unpin:
    for (i = 0; !unpinned && i < count; ++i) {
        if (!pages[i]) continue;
        unpin_user_page(pages[i]);
        if (a->lazy) --c->stats.pinned_pages;
    }
    /* The scan's lines were printed; a census reading them needs the unwind too. */
    if (diag_admit && a->scans)
        pr_info("capstone_vm: unwind arena 0x%lx+0x%lx scan=%u\n", a->address, a->bytes, a->scans);
    kvfree(a->frames);
    kvfree(pages);
    kfree(a);
    return -EFAULT;
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
                                           4 after CSRUNV returns a fault, before resolve(),
                                           5 before native syscall service */
static unsigned long test_inject_every = 1;
static unsigned long test_short_gup;    /* pin-ahead GUP pins only half of a batch */
/* Policy-only tests need exact windows despite unrelated scheduling/deadline cuts. The
 * ordinary and injected-stop configurations leave this off and qualify those cuts separately.
 * No page/VMA/backing check is bypassed, and the normal window cap still bounds the work. */
static bool test_full_window;
static bool test_skip_namespace_scrub;
/* Deliberately broken variants for the negative controls (gates/README of unpinned/): each
 * removes exactly one guard of the unpinned fault-ahead window, and the ON oracles must then fail
 * a named check. They exist only in this build; a production module has none of these strings. */
static bool test_skip_vma_clip;          /* the window may cross the demand fault's VMA end */
static bool test_skip_arena_clip;        /* the window may cross the arena's last page */
static bool test_skip_stop_check;        /* no cv_step_stop() between batches */
static bool test_skip_resident_stop;     /* the window does not end at the first resident PTE */
static bool test_fast_noflags;           /* the direct fault path (unpinned_fast_fault) drops FAULT_FLAG_WRITE */
static bool test_skip_repeat_arena;      /* the unpinned no-change bound counts repeats across mappings, by (address, cause) alone */
static unsigned long test_skip_scrub_after_pages;     /* the namespace scrub stops after N pages (0: all) */
static unsigned long test_skip_scrub_after_granules;  /* ... and after N granules of each page (0: all) */
module_param(test_inject, ulong, 0444);
module_param(test_inject_at, ulong, 0444);
module_param(test_inject_every, ulong, 0444);
module_param(test_short_gup, ulong, 0444);
module_param(test_full_window, bool, 0444);
module_param(test_skip_namespace_scrub, bool, 0444);
module_param(test_skip_vma_clip, bool, 0444);
module_param(test_skip_arena_clip, bool, 0444);
module_param(test_skip_stop_check, bool, 0444);
module_param(test_skip_resident_stop, bool, 0444);
module_param(test_fast_noflags, bool, 0444);
module_param(test_skip_repeat_arena, bool, 0444);
module_param(test_skip_scrub_after_pages, ulong, 0444);
module_param(test_skip_scrub_after_granules, ulong, 0444);
static u64 test_events, test_injections, test_stops, test_initial_stops, test_violations;
static u64 test_consumed, test_checks, test_checks_irqs_on, test_entries;
static u64 test_gup_requests, test_gup_max, test_short_results, test_gup_pending;
static u64 test_fault_requests, test_fault_deferred, test_fault_pending;
static u64 test_fault_batches, test_fault_max, test_fault_short;
static bool test_pending, test_expired, test_in_step;
/* Faults whose first resolution came from the launcher's CV_RESOLVE: a stop
 * fell between the fault and its resolution in CV_STEP, so it got the full
 * pin-ahead window. Real quantum ends and signals cause these too. */
static u64 test_launcher_first;
/* Stops nobody injected: a stop acted on while no injection is pending is a real quantum end, a
 * real signal or a real reschedule. real_resolve counts those that fall between a fault and its
 * resolution in CV_STEP (the fault then goes to the launcher, whose CV_RESOLVE pins the FULL
 * window: more pins, fewer faults than the injected run implies); real_window counts those that
 * cut a pin-ahead window before a batch (fewer pins, more faults). The kinds say which
 * condition was set at that moment. Accounting checks may allow for exactly these events. */
static u64 test_real_resolve, test_real_window;
static u64 test_real_resched, test_real_signal, test_real_deadline;
static void test_real_stop(bool window, u64 deadline)
{
    if (test_pending)
        return;
    if (window) ++test_real_window; else ++test_real_resolve;
    if (need_resched()) ++test_real_resched;
    if (signal_pending(current)) ++test_real_signal;
    if (ktime_get_ns() >= deadline) ++test_real_deadline;
}

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
/* Unpinned counterpart of short GUP: refuse the in-step attempt before
 * changing a PTE. CV_RESOLVE must then complete the same fault. This does
 * not turn a failed Linux resolution into success. */
static bool test_fault(void)
{
    if (test_pending && test_condition()) ++test_fault_pending;
    ++test_fault_requests;
    if (test_short_gup && test_in_step) {
        ++test_fault_deferred;
        return true;
    }
    return false;
}
static void test_report(const char *why)
{
    pr_info("capstone_vm test %s: injections=%llu stops=%llu initial_stops=%llu "
            "violations=%llu consumed=%llu gup_requests=%llu gup_max=%llu short_results=%llu "
            "gup_pending=%llu checks=%llu checks_irqs_on=%llu entries=%llu "
            "launcher_first=%llu unpinned=%u fault_requests=%llu "
            "fault_deferred=%llu fault_pending=%llu fault_batches=%llu "
            "fault_max=%llu fault_short=%llu real_resolve=%llu real_window=%llu real_resched=%llu "
            "real_signal=%llu real_deadline=%llu unpinned_fast_faults=%lu\n",
            why, test_injections, test_stops, test_initial_stops, test_violations,
            test_consumed, test_gup_requests, test_gup_max, test_short_results,
            test_gup_pending, test_checks, test_checks_irqs_on, test_entries,
            test_launcher_first, unpinned, test_fault_requests,
            test_fault_deferred, test_fault_pending, test_fault_batches,
            test_fault_max, test_fault_short, test_real_resolve, test_real_window,
            test_real_resched, test_real_signal, test_real_deadline, unpinned_fast_faults);
}
#define TEST_HOOK(at) test_hook(at)
#define TEST_GUP(pages) test_gup(pages)
#define TEST_FAULT() test_fault()
#define TEST_EXPIRED() test_expired
#define TEST_FULL_WINDOW() test_full_window
#define TEST_SKIP_NAMESPACE_SCRUB() test_skip_namespace_scrub
#define TEST_SKIP_VMA_CLIP() test_skip_vma_clip
#define TEST_SKIP_ARENA_CLIP() test_skip_arena_clip
#define TEST_SKIP_STOP_CHECK() test_skip_stop_check
#define TEST_SKIP_RESIDENT_STOP() test_skip_resident_stop
#define TEST_FAST_NOFLAGS() test_fast_noflags
#define TEST_SKIP_REPEAT_ARENA() test_skip_repeat_arena
#define TEST_SCRUB_PAGES() test_skip_scrub_after_pages
#define TEST_SCRUB_GRANULES() test_skip_scrub_after_granules
#define TEST_ENTRY() do { test_entry(); ++test_entries; } while (0)
#define TEST_CHECK() do { ++test_checks; if (!irqs_disabled()) ++test_checks_irqs_on; } while (0)
#define TEST_STOPPED(initial) do { if (test_pending) { ++test_stops; \
        if (initial) ++test_initial_stops; } } while (0)
#define TEST_STEP_BEGIN() do { test_in_step = true; } while (0)
#define TEST_IOCTL_DONE() do { test_pending = test_expired = test_in_step = false; } while (0)
#define TEST_RESOLVE(a, i) do { if (!test_in_step && !(a)->pages[i]) ++test_launcher_first; } while (0)
#define TEST_REAL_STOP(window, deadline) test_real_stop(window, deadline)
/* Which route resolved the fault: CV_STEP's own loop or the launcher's CV_RESOLVE. */
#define TEST_BY() (test_in_step ? " by=step" : " by=resolve")
#else
#define TEST_HOOK(at) do { } while (0)
#define TEST_GUP(pages) do { } while (0)
#define TEST_FAULT() false
#define TEST_EXPIRED() false
#define TEST_FULL_WINDOW() false
#define TEST_SKIP_NAMESPACE_SCRUB() false
#define TEST_SKIP_VMA_CLIP() false
#define TEST_SKIP_ARENA_CLIP() false
#define TEST_SKIP_STOP_CHECK() false
#define TEST_SKIP_RESIDENT_STOP() false
#define TEST_FAST_NOFLAGS() false
#define TEST_SKIP_REPEAT_ARENA() false
#define TEST_SCRUB_PAGES() 0UL
#define TEST_SCRUB_GRANULES() 0UL
#define TEST_ENTRY() do { } while (0)
#define TEST_CHECK() do { } while (0)
#define TEST_STOPPED(initial) do { } while (0)
#define TEST_STEP_BEGIN() do { } while (0)
#define TEST_IOCTL_DONE() do { } while (0)
#define TEST_RESOLVE(a, i) do { } while (0)
#define TEST_REAL_STOP(window, deadline) do { } while (0)
#define TEST_BY() ""
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
/* Returns the pages it pinned from `first` on. `extra` is the adaptive window (stream_window):
 * the window is the larger of it and pin_ahead_pages - 1. */
static unsigned long pin_ahead(struct context *c, struct arena *a,
                               struct vm_area_struct *v, unsigned first, unsigned long extra,
                               u64 deadline)
{
    unsigned long start = a->address + a->rx_bytes + (unsigned long)first * PAGE_SIZE;
    unsigned long count = 0, done;
    unsigned long fixed = pin_ahead_pages ? pin_ahead_pages - 1 : 0;
    unsigned long limit = max(fixed, extra);
    while ((arena_pin || count < limit) && first + count < a->count &&
           start + count * PAGE_SIZE < v->vm_end && !a->pages[first + count])
        ++count;
    for (done = 0; done < count;) {
        unsigned long batch = min(count - done, (unsigned long)PIN_AHEAD_BATCH);
        struct page **pages = a->pages + first + done;
        long n;
        if (cv_step_stop(deadline) && !TEST_FULL_WINDOW()) {
            TEST_REAL_STOP(true, deadline);
            return done;
        }
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
            if (!PageAnon(pages[j]) || page_mapcount(pages[j]) != 1 ||
                (a->frames && !frame_insert(a, pages[j]))) {
                for (long k = j; k < n; ++k) {
                    unpin_user_page(pages[k]);
                    pages[k] = NULL;
                }
                return done + j;
            }
            ++a->windows;
            ++c->stats.pinned_pages;
        }
        if (n != batch)
            return done + max(n, 0L);
        done += batch;
    }
    return done;
}

/* The adaptive window for a fresh fault at page `index` of arena a by context t (see
 * fault_ahead_max), and the stream it belongs to, whose `next` the caller sets once it knows
 * how many pages the window took. 0 while the stream is too short, or when it is off. */
static unsigned long stream_window(struct cv_thread *t, const struct arena *a, unsigned index,
                                   struct cv_stream **out)
{
    struct cv_stream *s, *match = NULL, *victim = t->streams;
    unsigned long maximum = min_t(unsigned long, READ_ONCE(fault_ahead_max), UINT_MAX);
    *out = NULL;
    if (!maximum || arena_pin || pin_ahead_pages > 1)
        return 0;
    for (s = t->streams; s < t->streams + ARRAY_SIZE(t->streams); ++s) {
        if (s->run && s->arena == a->id && s->next == index) {
            if (!match || s->used > match->used)
                match = s;
        }
        if (s->used < victim->used)
            victim = s;
    }
    s = match ? match : victim;
    if (match) {
        if (s->run < UINT_MAX) ++s->run;
    } else {
        s->arena = a->id;
        s->run = 1;
        s->window = 0;
    }
    s->used = ++t->stream_clock;
    if (s->run > fault_ahead_after)
        s->window = (unsigned)min(s->window ? 2UL * s->window : FAULT_AHEAD_FIRST, maximum);
    *out = s;
    return s->window;
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
static unsigned eloop_prints, refused_prints;

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

#ifdef CAPSTONE_VM_BENCH_COUNTERS
/* Performance build only: cycle and instret retired inside CSRUNV windows, cumulative over every context since the
 * module loaded. A program reads them through sysfs (delegated read, outside every window) immediately before and
 * after its timed region, so the difference is the region's own count: no start-up, no Linux. The board runs one hart
 * and one benchmark process at a time, so the module-wide sum is that process's. */
static unsigned long long bench_cycles, bench_instret, bench_entries;
module_param(bench_cycles, ullong, 0444);
module_param(bench_instret, ullong, 0444);
/* CSRUNV entries: each one after the first in a timed region is an escape whose resume (switcher walk,
 * the satp write's TLB flush, the refill) lands inside the next window, so a reading carries its count. */
module_param(bench_entries, ullong, 0444);
#endif

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

/* A file-backed VMA is admitted only as the image's data: add()'s image-data rule, a private,
 * readable and writable, never executable mapping of the image file. Its pages are then admitted
 * like any other, because the FOLL_WRITE pin breaks the copy-on-write and only an exclusive
 * anonymous copy is ever retained. */
static bool image_data_vma(const struct arena *a, const struct vm_area_struct *v)
{
    return a->rx_bytes && !(v->vm_flags & VM_EXEC) &&
           (v->vm_flags & (VM_READ | VM_WRITE)) == (VM_READ | VM_WRITE);
}

/* vm_lock keeps the grant alive; mmap_lock keeps this already-validated VMA
 * stable. Only anonymous faults take the direct path: file faults retain
 * fixup_user_fault's lookup and filesystem-specific completion semantics.
 * As with fixup_user_fault(..., NULL), do not set ALLOW_RETRY or KILLABLE:
 * the anonymous handler retains mmap_lock, including while waiting for a
 * migration entry. The caller still performs the bounded repeat and stop checks.
 * The caller holds mmap_read_lock and v is the VMA it found (or, for a
 * fault-ahead page, the same VMA); an address outside v takes the old call.
 */
static inline int unpinned_fault_vma(struct vm_area_struct *v, unsigned long address,
                                    unsigned int flags)
{
    vm_fault_t fault;

    if (!unpinned_fast_fault || !vma_is_anonymous(v) ||
        address < v->vm_start || address >= v->vm_end ||
        (v->vm_flags & (VM_SHARED | VM_IO | VM_PFNMAP | VM_MIXEDMAP | VM_HUGETLB)))
        return fixup_user_fault(v->vm_mm, address & PAGE_MASK, flags, NULL);

    /* Preserve fixup_user_fault's read/write admission even for an execute
     * fault. handle_mm_fault also performs arch_vma_access_permitted. */
    if (!(v->vm_flags & ((flags & FAULT_FLAG_WRITE) ? VM_WRITE : VM_READ)))
        return -EFAULT;
    ++unpinned_fast_faults;
    if (TEST_FAST_NOFLAGS()) flags = 0;   /* broken variant: a store is resolved as a read */
    pr_info_once("capstone_vm: unpinned fault path: direct handle_mm_fault (unpinned_fast_fault=1), first at 0x%lx\n",
                 address);
    fault = handle_mm_fault(v, address & PAGE_MASK, flags, NULL);
    if (fault & VM_FAULT_ERROR) {
        int error = vm_fault_to_errno(fault, 0);
        return error ? error : -EFAULT;
    }
    return 0;
}

/* Keep successful resolution and counted escape events distinct: a stop
 * after resolution can cause CV_RESOLVE to repeat it without a new escape.
 * Point 1 is the completed demand operation in either memory mode. */
static bool unpinned_resident(struct mm_struct *mm, unsigned long address)
{
    unsigned long at, pfn;
    /* A residency hint only: migration or a later permission fault can
     * invalidate it. It never replaces demand resolution or retains a PFN. */
    return next_resident(mm, address, address + PAGE_SIZE, &at, &pfn) == 1;
}

/* Whether this unpinned fault repeats the thread's previous one with nothing changed: the same address
 * and cause in the same mapping. A new mapping at the old address is not a repeat. */
static inline bool unpinned_fault_repeats(struct cv_thread *t, struct arena *a, unsigned long address,
                                          unsigned long cause)
{
    return address == t->last_fault_addr && cause == t->last_fault_cause &&
           (TEST_SKIP_REPEAT_ARENA() || a->id == t->last_fault_arena);
}

static int unpinned_resolve(struct context *c, struct cv_thread *t, struct arena *a,
                            struct vm_area_struct *v, unsigned long address, u64 deadline)
{
    struct cv_stream *stream;
    unsigned index = (address - a->address - a->rx_bytes) / PAGE_SIZE;
    unsigned long ahead, limit, done = 0;
    unsigned int flags = (v->vm_flags & VM_WRITE) ? FAULT_FLAG_WRITE : 0;
    int rc;
#ifdef CONFIG_TRANSPARENT_HUGEPAGE
    /* add() required VM_NOHUGEPAGE on every VMA it admitted, but a later madvise(MADV_HUGEPAGE)
     * clears it. A huge page would then be faulted in as one PMD that next_resident() cannot
     * describe (-EOPNOTSUPP): refuse the fault instead of widening a window over it. */
    if (!(v->vm_flags & VM_NOHUGEPAGE)) return -EOPNOTSUPP;
#endif
    if (TEST_FAULT()) return -EAGAIN;
    /* The event and VMA access were authenticated by resolve(). Only read
     * materialization changes: stores, execution, image mappings and eager
     * bootstrap retain their existing policy. Speculation uses the same access
     * as this demand fault; a later store must still resolve write/COW. */
    if (unpinned_access_fault && a->lazy && (flags & FAULT_FLAG_WRITE) &&
        vma_is_anonymous(v) && t->frame[3] == 13) {
        flags = 0;
        ++unpinned_access_reads;
    }
    rc = unpinned_fault_vma(v, address, flags);
    if (rc) return rc;
    ++unpinned_faults;
    if (diag_admit) ++a->unpinned_resolves;
    TEST_HOOK(1);
    /* CV_RESOLVE may repeat a completed demand operation after a stop. It
     * must not train a second stream or repeat the speculative work. */
    if (t->sfence_pending) return 0;
#ifdef CV_TEST_HOOKS
    if (!test_in_step) ++test_launcher_first;
#endif
    ahead = stream_window(t, a, index, &stream);
    limit = max(ahead, pin_ahead_pages ? pin_ahead_pages - 1 : 0);
    if (!TEST_SKIP_ARENA_CLIP())
        limit = min_t(unsigned long, limit, a->count - index - 1);
    address = (address & PAGE_MASK) + PAGE_SIZE;
    if (!TEST_SKIP_VMA_CLIP())
        limit = min(limit, (v->vm_end - address) / PAGE_SIZE);
    /* mmap_read_lock remains held. Every page is inside this same validated
     * VMA and arena; Linux applies its normal access/COW policy. No payload
     * reference or pin survives the call. A failed speculative resolution
     * leaves the remaining pages for demand faults and does not fail the
     * already completed demand operation. Stop checks bound each batch. */
    while (done < limit) {
        unsigned long batch = min_t(unsigned long, limit - done, PIN_AHEAD_BATCH);
        unsigned long target = batch;
        if (!TEST_SKIP_STOP_CHECK() && cv_step_stop(deadline) && !TEST_FULL_WINDOW()) {
            TEST_REAL_STOP(true, deadline);
            break;
        }
#ifdef CV_TEST_HOOKS
        if (test_pending && test_condition()) ++test_fault_pending;
        ++test_fault_batches;
        test_fault_max = max_t(u64, test_fault_max, batch);
        if (test_short_gup && batch > 1) target = batch / 2;
        if (target != batch) ++test_fault_short;
#endif
        for (unsigned long j = 0; j < target; ++j) {
            if (!TEST_SKIP_RESIDENT_STOP() && unpinned_resident(c->mm, address + done * PAGE_SIZE))
                goto finished;
            if (unpinned_fault_vma(v, address + done * PAGE_SIZE, flags))
                goto finished;
            ++done;
        }
        if (target != batch) break;
    }
finished:
    if (diag_admit) {
        a->unpinned_ahead += done;
        pr_info("capstone_vm: unpinned window arena 0x%lx address 0x%lx pages %lu limit %lu\n",
                a->address, address, done, limit);
    }
    if (stream) {
        unsigned long skip_end;
        stream->next = index + 1 + done;
        skip_end = min_t(unsigned long, a->count, (unsigned long)stream->next + 64);
        while (stream->next < skip_end && unpinned_resident(c->mm,
                a->address + a->rx_bytes + (unsigned long)stream->next * PAGE_SIZE))
            ++stream->next;
        if (ahead) {
            ++fault_ahead_windows;
            fault_ahead_pages += done;
            if (diag_admit)
                pr_info("capstone_vm: unpinned ahead arena 0x%lx page %u want %lu resolved %lu fault 0x%lx%s\n",
                        a->address, index, ahead, done, address - PAGE_SIZE, TEST_BY());
        }
    }
    return 0;
}

/* Exactly one diagnostic for each increment of stats.faults. Unlike the
 * pinned admission census, this also records repeated faults and faults
 * returned to the launcher before resolution. No timing allowance is used
 * to reconcile it with the public counter. */
static void unpinned_fault_event(struct context *c, struct cv_thread *t,
                                 const char *route)
{
    struct arena *a;
    if (!unpinned || !diag_admit) return;
    a = arena_at(c, t->frame[5]);
    if (a) ++a->unpinned_events;
    pr_info("capstone_vm: unpinned fault arena 0x%lx address 0x%lx cause %lu pc 0x%lx route %s number %llu\n",
            a ? a->address : 0, t->frame[5], t->frame[3], t->frame[4], route,
            (unsigned long long)c->stats.faults);
}

static int resolve(struct context *c, struct cv_thread *t, u64 deadline)
{
    unsigned long address = t->frame[5], cause = t->frame[3];
    struct page *page;
    struct vm_area_struct *v;
    unsigned long vmflags = 0;   /* snapshot under the mmap lock; v is unsafe to read after unlock */
    unsigned long required = cause == 15 ? VM_WRITE : cause == 12 ? VM_EXEC : VM_READ;
    struct arena *a;
    if (!t->started || cv_fault_classify(t->event, cause) == CV_FAULT_DENIED)
        return -EINVAL;
    a = arena_at(c, address);
    if (a && address - a->address < a->rx_bytes) {
        /* Image text: Linux maps the file page (and its neighbours, by
         * fault-around), with no copy and no pin. A store is refused: these
         * pages are never writable. fixup_user_fault checks read access
         * only, so execute permission is checked here. */
        int rc;
        if (cause == 15) return -EACCES;
        mmap_read_lock(c->mm);
        v = find_vma(c->mm, address);
        if (!v || v->vm_start > address || !v->vm_file || !c->image ||
            file_inode(v->vm_file) != file_inode(c->image) ||
            (v->vm_flags & (VM_WRITE | VM_SHARED)) ||
            !(v->vm_flags & (cause == 12 ? VM_EXEC : VM_READ))) {
            mmap_read_unlock(c->mm); return -EACCES;
        }
        rc = fixup_user_fault(c->mm, address, cause == 12 ? FAULT_FLAG_INSTRUCTION : 0, NULL);
        mmap_read_unlock(c->mm);
        return rc;
    }
    if (a && !a->lazy && unpinned) {
        /* Unpinned: an eager grant's page was moved (migration, compaction) or
         * write-protected (a fork's COW) under the domain. The grant was validated at add();
         * here only what the kernel's fault path needs: a private VMA with the access. A
         * private file mapping is allowed (image data: its pages are anonymous COW copies
         * since add() populated them for write). Same repeat bound as the unpinned path. */
        int rc;
        mmap_read_lock(c->mm);
        v = find_vma(c->mm, address);
        if (!v || v->vm_start > address || !(v->vm_flags & required) ||
            (v->vm_flags & VM_SHARED)) {
            unsigned long f = v ? v->vm_flags : 0UL;
            mmap_read_unlock(c->mm);
            CV_RESOLVE_FAIL("unpinned-vma", "found %d flags 0x%lx", !!v, f);
            return -EACCES;
        }
        vmflags = v->vm_flags;
        if (unpinned_fault_repeats(t, a, address, cause)) {
            if (++t->fault_repeat > fault_repeat_max) {
                mmap_read_unlock(c->mm);
                CV_RESOLVE_FAIL("unpinned-loop", "resolved %u times with no change", t->fault_repeat);
                return -ELOOP;
            }
        } else {
            t->last_fault_addr = address;
            t->last_fault_cause = cause;
            t->last_fault_arena = a->id;
            t->fault_repeat = 0;
        }
        rc = unpinned_resolve(c, t, a, v, address, deadline);
        mmap_read_unlock(c->mm);
        if (rc) CV_RESOLVE_FAIL("unpinned", "fixup_user_fault=%d vm_flags 0x%lx", rc, vmflags);
        return rc;
    }
    if (a && a->lazy) {
        unsigned index;
        long n;
        /* pages[] covers the pages after the read+execute prefix (zero except for the image). */
        index = (address - a->address - a->rx_bytes) / PAGE_SIZE;
        mmap_read_lock(c->mm);
        v = find_vma(c->mm, address);
        if (!v || v->vm_start > address || !(v->vm_flags & required) ||
            (v->vm_flags & VM_SHARED) || (v->vm_file && !image_data_vma(a, v))) {
            unsigned long f = v ? v->vm_flags : 0UL;
            int file = v ? !!v->vm_file : -1;
            mmap_read_unlock(c->mm);
            CV_RESOLVE_FAIL("vma", "found %d flags 0x%lx file %d", !!v, f, file);
            return -EACCES;
        }
        vmflags = v->vm_flags;
        if (unpinned) {
            /* Unpinned: the kernel's own fault path, as the image's RX path and the
             * resolve_dirty path use it, and no pin. Write access where the VMA grants it (GUP's
             * FOLL_WRITE did the same), so a read fault still materializes private writable
             * dirty backing rather than the zero page, unless the anonymous-read
             * experiment selects actual read access. The repeat bound stays: a resolve that
             * changes nothing must not report success for ever. */
            int rc;
            if (unpinned_fault_repeats(t, a, address, cause)) {
                if (++t->fault_repeat > fault_repeat_max) {
                    mmap_read_unlock(c->mm);
                    CV_RESOLVE_FAIL("unpinned-loop", "resolved %u times with no change", t->fault_repeat);
                    return -ELOOP;
                }
            } else {
                t->last_fault_addr = address;
                t->last_fault_cause = cause;
                t->last_fault_arena = a->id;
                t->fault_repeat = 0;
            }
            rc = unpinned_resolve(c, t, a, v, address, deadline);
            mmap_read_unlock(c->mm);
            if (rc) CV_RESOLVE_FAIL("unpinned", "fixup_user_fault=%d vm_flags 0x%lx", rc, vmflags);
            return rc;
        }
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
        /* A page admitted at a new index may not share its frame with one already admitted:
         * the whole-arena duplicate check of the mint, applied to this page. A repeat at an
         * index already filled is judged by the backing comparison below instead. */
        if (!a->pages[index] && a->frames && !frame_insert(a, page)) {
            CV_RESOLVE_FAIL("frame", "pfn 0x%lx already backs another page of the arena "
                            "(mapcount %d refs %d vm_flags 0x%lx)", page_to_pfn(page),
                            page_mapcount(page), page_ref_count(page), vmflags);
            mmap_read_unlock(c->mm); unpin_user_page(page); return -EACCES;
        }
        if (!a->pages[index]) {
            struct cv_stream *stream;
            unsigned long ahead = stream_window(t, a, index, &stream);
            unsigned long pinned = pin_ahead(c, a, v, index + 1, ahead, deadline);
            if (stream) {
                /* The next demand fault lands past pages admitted earlier (a stream crosses
                 * them without faulting). Scan at most 64 pages, independently of
                 * fault_ahead_max: a longer pinned run ends the stream rather
                 * than costing an unbounded scan inside the step. */
                unsigned skip_end = stream->next = index + 1 + pinned;
                skip_end = min_t(unsigned long, a->count, (unsigned long)skip_end + 64);
                while (stream->next < skip_end && a->pages[stream->next])
                    ++stream->next;
                if (ahead) {
                    ++fault_ahead_windows;
                    fault_ahead_pages += pinned;
                    if (diag_admit)
                        pr_info("capstone_vm: fault-ahead arena 0x%lx page %u want %lu pinned %lu fault 0x%lx%s\n",
                                a->address, index, ahead, pinned,
                                a->address + a->rx_bytes + (unsigned long)index * PAGE_SIZE,
                                TEST_BY());
                }
            }
        }
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
        if (diag_admit)
            pr_info("capstone_vm: admit arena 0x%lx+0x%lx page %u cause %lu pc 0x%lx pfn 0x%lx\n",
                    a->address, a->bytes, index, cause, t->frame[4], page_to_pfn(page));
        ++a->admits;
        a->pages[index] = page;
        ++c->stats.pinned_pages;
        c->stats.peak_pages = max(c->stats.peak_pages, c->stats.pinned_pages);
        return 0;
    }
    CV_RESOLVE_FAIL("arena", "no lazy arena covers this address");
    return -EFAULT;
}

/* Admit one absent page of a lazy arena outside a fault, for CV_TOUCH: the launcher is about to
 * write it through Linux. The same rule as resolve()'s for a page admitted at a fault: the VMA
 * present, readable, private, a file only as the image's data; the page pinned with FOLL_WRITE
 * (FOLL_FORCE where the VMA is not writable) so a zero page, a shared page or a file page is
 * replaced by an exclusive copy; then PageAnon, mapcount 1 and the frame set. Caller holds
 * vm_lock and has checked that pages[index] is empty. */
static int admit_page(struct context *c, struct arena *a, unsigned index)
{
    unsigned long address = a->address + a->rx_bytes + (unsigned long)index * PAGE_SIZE;
    struct vm_area_struct *v;
    struct page *page;
    mmap_read_lock(c->mm);
    v = find_vma(c->mm, address);
    if (!v || v->vm_start > address || !(v->vm_flags & VM_READ) || (v->vm_flags & VM_SHARED) ||
        (v->vm_file && !image_data_vma(a, v))) {
        mmap_read_unlock(c->mm); return -EACCES;
    }
    if (unpinned) {
        int rc = fixup_user_fault(c->mm, address,
                      (v->vm_flags & VM_WRITE) ? FAULT_FLAG_WRITE : 0, NULL);
        mmap_read_unlock(c->mm);
        if (!rc && diag_admit) {
            ++a->unpinned_touches;
            pr_info("capstone_vm: unpinned touch arena 0x%lx+0x%lx page %u\n",
                    a->address, a->bytes, index);
        }
        return rc;
    }
    if (pin_user_pages_remote(c->mm, address, 1,
                              FOLL_WRITE | ((v->vm_flags & VM_WRITE) ? 0 : FOLL_FORCE),
                              &page, NULL, NULL) != 1) {
        mmap_read_unlock(c->mm); return -EFAULT;
    }
    if (!PageAnon(page) || page_mapcount(page) != 1 || (a->frames && !frame_insert(a, page))) {
        mmap_read_unlock(c->mm); unpin_user_page(page); return -EACCES;
    }
    mmap_read_unlock(c->mm);
    if (diag_admit)
        pr_info("capstone_vm: touch arena 0x%lx+0x%lx page %u pfn 0x%lx\n",
                a->address, a->bytes, index, page_to_pfn(page));
    ++a->touches;
    a->pages[index] = page;
    ++c->stats.pinned_pages;
    c->stats.peak_pages = max(c->stats.peak_pages, c->stats.pinned_pages);
    return 0;
}

/* CV_TOUCH: every page of [address, address + bytes) inside a lazy arena is admitted before the
 * launcher writes it. A page already held, or in an arena pinned whole, needs nothing; a page
 * outside every arena is refused, as is one in the image's read+execute prefix, which the
 * launcher never writes. */
static int touch(struct context *c, const struct cv_touch *r)
{
    unsigned long at, end;
    if (!r->bytes || r->address > ULONG_MAX - r->bytes) return -EINVAL;
    at = r->address & PAGE_MASK;
    end = PAGE_ALIGN(r->address + r->bytes);
    if (end < at) return -EINVAL;
    for (; at < end; at += PAGE_SIZE) {
        struct arena *a = arena_at(c, at);
        unsigned index;
        int error;
        if (!a) return -EINVAL;
        if (at - a->address < a->rx_bytes) return -EACCES;
        if (!a->lazy) continue;
        index = (at - a->address - a->rx_bytes) / PAGE_SIZE;
        if (!unpinned && a->pages[index]) continue;
        error = admit_page(c, a, index);
        if (error) return error;
    }
    return 0;
}

static void node_stats(struct context *c)
{
    c->stats.nodes = c->table[CV_HDR_ALLOCATIONS];
    c->stats.nodes_high_water = c->table[CV_HDR_NEXT] - CV_TABLE_FIRST_SLOT;
    c->stats.nodes_retired = c->table[CV_HDR_RETIRED];
    c->stats.nodes_live = c->stats.nodes_high_water - c->stats.nodes_retired - node_free_count(c);
    c->stats.node_capacity = node_capacity(c);
    high_water = max(high_water, c->stats.nodes_high_water);
}

/* Does the image's filesystem keep one inode per file, so the writers Linux
 * counts on it (i_writecount) are all the file's writers? Named by the
 * superblock's type, which the filesystem cannot misreport: statfs() on 9p
 * returns the HOST's filesystem magic. 9p with cache=none or mmap makes a new
 * inode at each lookup, so a writer through a hard link is counted elsewhere;
 * NFS, FUSE and overlayfs make no such promise either. tmpfs covers memfds,
 * whose own descriptor is not counted: the launcher requires their seals. */
static bool one_inode_per_file(struct file *file)
{
    static const char *const fs[] = {"tmpfs", "ext2", "ext3", "ext4", "btrfs",
                                     "xfs", "erofs", "squashfs"};
    const char *name = file_inode(file)->i_sb->s_type->name;
    for (unsigned i = 0; i < ARRAY_SIZE(fs); ++i)
        if (!strcmp(name, fs[i])) return true;
    return false;
}

/* vm_lock is held on entry and return, including a supervisor-slot wait.
 * Both the launcher ioctl and the experimental native-task driver use this
 * operation; request/result storage belongs to the trusted caller. */
/* `jdl` non-NULL selects the native loop's coarse bound: a jiffies deadline, one load per
 * check instead of a clock read. It bounds how long vm_lock is held exactly as the
 * nanosecond budget does, to within one tick. The test-hook build keeps nanoseconds. */
static bool cv_step_stop_j(unsigned long jdl)
{
    return signal_pending(current) || need_resched() ||
           time_after_eq(jiffies, jdl) || TEST_EXPIRED();
}
#define CV_STEP_BUDGET_J (msecs_to_jiffies(CV_STEP_BUDGET_NS / NSEC_PER_MSEC) + 1)
static long cv_step_locked(struct context *c, struct cv_step *request, const unsigned long *jdl)
{
    /* Edited in place: nothing is stored before the last line, so a refusal leaves the
     * request as it came. */
    struct cv_step *const r = request;
    long rc = -EINVAL;
    unsigned long flags;
    struct cv_thread *t;
    u64 budget_end;
    bool resolved = false, text = false;
    t = c->main_thread && c->main_thread->id == r->thread ? c->main_thread : find_thread(c, r->thread);
    if (!t || t->terminal) { rc = -ESRCH; return rc; }
    if (r->reply > 1 || (r->reply != (t->started && t->event == 3))) return rc;
    if (!t->started && (!t->frame[8] || !t->frame[12])) return rc;
    /* A frame that has not started needs a free slot to enter. Waiting is safe ACROSS
     * contexts -- two vexec processes never join each other, so run.py's parallel pair simply
     * serialises and still prints both results -- and is NOT safe within one context, where a
     * live sibling is the thing that would have to finish first. The create refusal above means
     * the same-context case cannot normally arise; it is rejected here too rather than
     * deadlocking if it ever does. vm_lock must be dropped while waiting, so the wait is
     * outside it and the ioctl restarts. */
    if (!native_slot_transport && !t->started) {
        struct cv_thread *holder = NULL;
        if (slot_live(&holder) >= supervisor_slots) {
            unsigned long generation = slot_generation;
            if (holder && holder->context == c) { rc = -EBUSY; return rc; }
            mutex_unlock(&vm_lock);
            rc = wait_event_interruptible(slot_wait,
                    READ_ONCE(slot_generation) != generation);
            mutex_lock(&vm_lock);
            return rc ? -EINTR : -EAGAIN;   /* the launcher's loop retries EAGAIN, as it does for the quantum */
        }
    }
    if (native_slot_transport) {
        rc = slot_acquire(t);
        if (rc) return rc;
    }
    if (r->reply && !t->reply_cap) {
        t->frame[28] = r->result; t->frame[29] = 0;
    }
    /* Each CSRUNV restarts the 5 ms quantum, so a domain that keeps
     * faulting would never be preempted: bound the resolved faults by
     * one quantum of this ioctl, then return and release vm_lock. */
    budget_end = jdl ? 0 : ktime_get_ns() + CV_STEP_BUDGET_NS;
    TEST_STEP_BEGIN();
    TEST_HOOK(3);
    for (unsigned long action = r->reply ? 2 : 0, entries = 0;;
         action = 0, ++entries) {
        if (entries && !text) TEST_HOOK(2);  /* after a data fault only, see below */
        preempt_disable(); local_irq_save(flags);
        /* Checked with interrupts and preemption off, immediately before
         * CSRUNV: no interrupt can set need_resched or a signal between
         * this check and the entry. Before the first entry nothing has
         * changed, so the launcher simply repeats the same CV_STEP (its
         * loop retries EINTR and EAGAIN; the reply above is rewritten
         * identically). Before a later one the fault just resolved goes
         * back as it was, already counted. */
        TEST_CHECK();
        if (jdl ? cv_step_stop_j(*jdl) : cv_step_stop(budget_end)) {
            local_irq_restore(flags); preempt_enable();
            TEST_STOPPED(!entries);
            if (!entries) {
                rc = signal_pending(current) ? -EINTR : -EAGAIN;
                return rc;
            }
            resolved = true;
            break;
        }
        TEST_ENTRY();
        if (!t->started) {
            t->frame[0] = csr_read(CSR_SATP);
            t->frame[1] = c->table_pa;
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
#ifdef CAPSTONE_VM_BENCH_COUNTERS
        {
            /* Interrupts are off: both reads bracket the CSRUNV alone. */
            unsigned long cy0 = csr_read(CSR_CYCLE), in0 = csr_read(CSR_INSTRET);
            t->event = run_event(t, action);
            unsigned long dcy = csr_read(CSR_CYCLE) - cy0, din = csr_read(CSR_INSTRET) - in0;
            atomic64_add(dcy, &c->bench_cycles);
            atomic64_add(din, &c->bench_instret);
            WRITE_ONCE(bench_cycles, bench_cycles + dcy);
            WRITE_ONCE(bench_instret, bench_instret + din);
            WRITE_ONCE(bench_entries, bench_entries + 1);
            atomic64_inc(&c->bench_entries);
        }
#else
        t->event = run_event(t, action);
#endif
        if (!t->started) t->native_satp = csr_read(CSR_SATP);
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
        /* A text fault is not a fault event (text_fault), nor an
         * injection point of the test build's stop hooks: neither this
         * one nor the re-entry hook above. */
        text = text_fault(c, t->frame[5]);
        if (!text && diag_admit) {
            struct arena *fa = arena_at(c, t->frame[5]);
            if (fa) ++fa->faulted;
        }
        if (!text)
            TEST_HOOK(4);
        if (jdl ? cv_step_stop_j(*jdl) : cv_step_stop(budget_end)) {
            TEST_STOPPED(false);
            TEST_REAL_STOP(false, budget_end);
            break;
        }
        if (resolve(c, t, jdl ? ktime_get_ns() + CV_STEP_BUDGET_NS : budget_end))
            break;
        t->sfence_pending = true;
        if (!text) {
            ++c->stats.faults;
            unpinned_fault_event(c, t, "step");
        }
    }
    if (t->event == 5) {
        int error = ensure_nodes(c, 1);
        /* Return through Linux between retries. Refuse endless retries
         * when the namespace consists of live or pinned identities. */
        t->frame[2] = error ? 2 : 1;
    }
    r->kind = t->frame[2]; r->cause = t->frame[3];
    r->pc = t->frame[4]; r->address = t->frame[5];
    /* The low word of register slots 1 and 8 (byte 80 and 192: x1 = ra,
     * x8 = fp; see wire.h cv_step). The escape writes them on a terminal
     * step; otherwise they hold what ENTRY left, zero. */
    r->ra = t->frame[10]; r->fp = t->frame[24];
    /* Eight loads and stores: for 64 aligned bytes the kernel's memcpy is
     * a call around a four-byte loop. */
    r->args[0] = t->frame[72]; r->args[1] = t->frame[73];
    r->args[2] = t->frame[74]; r->args[3] = t->frame[75];
    r->args[4] = t->frame[76]; r->args[5] = t->frame[77];
    r->args[6] = t->frame[78]; r->args[7] = t->frame[79];
    r->thread = t->id;
    t->terminal = r->kind == 2;
    if (r->kind == 4 && !resolved && !text_fault(c, r->address)) {
        ++c->stats.faults;
        unpinned_fault_event(c, t, "return");
    }
    return 0;
}

static int context_create(struct context **out)
{
    struct context *c;
    struct context *existing;
    struct cv_thread *t;
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
    INIT_LIST_HEAD(&c->threads);
    INIT_LIST_HEAD(&c->ref_pages);
    c->arenas = RB_ROOT;
    c->next_thread = 0;
    t = kvzalloc(sizeof(*t), GFP_KERNEL);
    if (!t) { kvfree(c); return -ENOMEM; }
    c->frame = (void *)get_zeroed_page(GFP_KERNEL);
    /* The table starts with node_initial_pages record pages, published before the processor
     * sees the table, and grows by batches from there (ensure_nodes). */
    c->table = sublet_table_create();
    error = -ENOMEM;
    if (!c->frame || !c->table) goto fail;
    c->table_pa = virt_to_phys(c->table);
    if (sublet_table_grow(c->table, 1, node_initial_pages, node_initial_pages, &pages)) goto fail;
    c->stats.node_bytes += (pages + 1) * PAGE_SIZE;
    /* Capacity is authoritative in the table we have just allocated. */
    error = -EOPNOTSUPP;
    if (node_available(c) != node_capacity(c) - CV_TABLE_FIRST_SLOT) goto fail;
    c->exact_bounds = copyable_exact_bounds;
    error = root_create(c);
    if (error) goto fail;
    c->frame[1] = c->table_pa;
    t->context = c; t->id = 0; t->frame = c->frame;
    t->frame_pa = virt_to_phys(c->frame); t->user_frame = false;
    c->main_thread = t;
    list_add(&t->link, &c->threads);
    c->mm = current->mm; mmgrab(c->mm);
    mutex_lock(&vm_lock);
    list_for_each_entry(existing, &contexts, link)
        if (existing->mm == c->mm) {
            error = -EBUSY;
            goto fail_locked;
        }
    if (unpinned) {
        error = TEST_SKIP_NAMESPACE_SCRUB() ? 0 :
                unpinned_scrub_mm(c->mm, TEST_SCRUB_PAGES(), TEST_SCRUB_GRANULES());
        if (error) goto fail_locked;
    }
    *out = c;
    list_add(&c->link, &contexts);
    mutex_unlock(&vm_lock);
    return 0;
fail_locked:
    mutex_unlock(&vm_lock);
    mmdrop(c->mm);
fail:
    ref_pages_free(c);
    node_free_table(c);
    if (c->frame) { memset(c->frame, 0, PAGE_SIZE); free_page((unsigned long)c->frame); }
    kvfree(t); kvfree(c);
    return error;
}
static int context_destroy(struct context *c)
{
    unsigned long flags;
    struct cv_thread *t, *tmp;
    mutex_lock(&vm_lock);
    list_del(&c->link);
    node_stats(c);
    allocated_total += c->stats.nodes;
    collections_total += c->stats.collections;
    reclaimed_total += c->stats.reclaimed;
    /* Stop every supervisor slot while its frame pages are still pinned. */
    list_for_each_entry(t, &c->threads, link) {
        if (t->started && !t->slot_exported) {
            preempt_disable(); local_irq_save(flags);
            run(t->frame_pa, 1);
            local_irq_restore(flags); preempt_enable();
        }
    }
    /* Teardown still clears tags in user frames (including thread start
     * frames) before the final pins are dropped. */
    /* One pass over the arenas present now, the next node taken before each retire. The task
     * may be in exit_mm() here: a teardown never materializes lazy holes, it releases only
     * pages already pinned. */
    for (struct rb_node *n = rb_first(&c->arenas), *next; n; n = next) {
        struct arena *a = rb_entry(n, struct arena, node);
        next = rb_next(n);
        retire(c, a, true);
    }
    WARN_ON(!RB_EMPTY_ROOT(&c->arenas));
#ifdef CAPSTONE_VM_BENCH_COUNTERS
    /* Must-fire, one per context, KERN_EMERG because the board console shows nothing lower. */
    pr_emerg("capstone_vm: bench cycles=%lld instret=%lld entries=%lld\n",
             (long long)atomic64_read(&c->bench_cycles), (long long)atomic64_read(&c->bench_instret),
             (long long)atomic64_read(&c->bench_entries));
#endif
    list_for_each_entry_safe(t, tmp, &c->threads, link) {
        list_del(&t->link);
        if (t->user_frame) { clear_thread_frame(t); drop_thread_frame(t); }
        else if (t->frame) { memset(t->frame, 0, PAGE_SIZE); free_page((unsigned long)t->frame); }
        kvfree(t);
    }
    ref_pages_free(c);
    node_free_table(c);
    if (c->image) { allow_write_access(c->image); fput(c->image); }
    mmdrop(c->mm); kvfree(c);
    slot_changed();   /* the released context's frames are gone */
    mutex_unlock(&vm_lock);
    return 0;
}
#include "checked-copy-dom.h"
#include "binfmt-dom.h"

static int __init init(void)
{
    struct device_node *profile;
    u32 id_bits, exact, page_size, granule_size, checked_abi = 0, transport_version = 0;
    int error;
    /* Admission switches remain accepted by existing boot recipes. Disabling
     * one disables this adapter; it can never select a legacy entry path. */
    if (!native_task || !native_dispatch || !dom_binfmt || !checked_copy ||
        !native_slot_transport || !unpinned) return -EOPNOTSUPP;
    profile = of_find_compatible_node(NULL, NULL, "capstone,copyable-v1");
    if (!profile) return -EOPNOTSUPP;
    error = of_property_read_u32(profile, "id-bits", &id_bits) ||
            of_property_read_u32(profile, "exact-bounds", &exact) ||
            of_property_read_u32(profile, "page-size", &page_size) ||
            of_property_read_u32(profile, "granule-size", &granule_size);
    of_property_read_u32(profile, "checked-copy-abi", &checked_abi);
    of_property_read_u32(profile, "context-transport-version", &transport_version);
    of_node_put(profile);
    if (error || id_bits != CAP_REV_SUBLET_ID_BITS || exact > 1 ||
        page_size != PAGE_SIZE || granule_size != 16)
        return -EOPNOTSUPP;
    copyable_exact_bounds = exact;
    /* Refuse mismatched models at load, before an experimental instruction
     * can trap in S mode. No test-only CSR probing or legacy ABI fallback. */
    if ((checked_copy && (checked_abi != 1 || !native_slot_transport)) ||
        (native_slot_transport && transport_version != (checked_copy ? 6 : 5)) ||
        (native_dispatch && (!dom_binfmt || !checked_copy || !native_slot_transport)) ||
        ((native_slot_transport || checked_copy || dom_binfmt) && num_possible_cpus() != 1))
        return -EOPNOTSUPP;


#ifdef CV_TASK_BACKEND
    if (native_task && (!native_dispatch || !unpinned)) return -EOPNOTSUPP;
    if (native_io && !native_task) return -EOPNOTSUPP;
#endif
    if (unpinned_fast_sweep && !unpinned) return -EINVAL;
    if (unpinned_fast_fault && !unpinned) {
        pr_emerg("capstone_vm: unpinned_fast_fault requires unpinned=1\n");
        return -EINVAL;
    }
    if (unpinned) {
        int error;
        /* Tag-copy interrupt exclusion and namespace sweeps are currently
         * single-hart protocols. Offline-but-possible CPUs are not safe:
         * hotplug must not silently invalidate the load-time contract. */
        if (num_possible_cpus() != 1) {
            pr_err("capstone_vm: unpinned backing requires one possible CPU\n");
            return -EOPNOTSUPP;
        }
        error = capstone_unpinned_config(CAP_REV_SUBLET_ID_BITS, &unpinned_exact_bounds);
        if (error) return error;
        if (unpinned_exact_bounds != copyable_exact_bounds) return -EOPNOTSUPP;
    }
#ifdef CONFIG_TRANSPARENT_HUGEPAGE
    if (dom_binfmt) return -EOPNOTSUPP;
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
        /* Always KERN_EMERG, independent of diag_emerg: this is the line that tells a reader
         * whether the module loaded at all, so it must survive the quieted console even when the
         * rest are being read through dmesg. One line per insmod, before the rungs begin. */
        pr_emerg("capstone_vm: loaded, diag %s, supervisor_slots=%lu, diag_emerg=%d, unpinned=%d, fast_sweep=%d, unpinned_fast_fault=%d\n",
                 CV_DIAG_TAG, supervisor_slots, diag_emerg, unpinned, unpinned_fast_sweep, unpinned_fast_fault);
        if (dom_binfmt) insert_binfmt(&capstone_dom);
        return 0;
    }
}
static void __exit done(void)
{
    if (dom_binfmt) unregister_binfmt(&capstone_dom);
#ifdef CV_TEST_HOOKS
    test_report("unload");
#endif
}
module_init(init); module_exit(done);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Built-in native Capstone task adapter");
