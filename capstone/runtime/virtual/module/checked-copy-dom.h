/* SPDX-License-Identifier: GPL-2.0-only */
/* Internal native-task adapter, consumed by native-syscall-dom.h.
 * Call with vm_lock held and a kernel-resident, prefaulted buffer. Neither the
 * context nor its thread may be destroyed until this synchronous call ends.
 * c->dom_pid admits only the owning native thread; shared-mm native threads
 * need the context-owner/slot virtualization integration before admission. */
#include <capstone/checked-copy.h>
#include "checked-copy-core.c"

static bool checked_copy;
module_param(checked_copy, bool, 0444);
MODULE_PARM_DESC(checked_copy, "Experimental CSCHECK helper (matching QEMU required; default off)");

#ifdef CV_RACE_TEST
/* Test-only build (make CV_RACE_TEST=1; tests/native-race). The production
 * module compiles none of this. A racer thread B, a real second domain thread
 * of the same context, is run deterministically by the kernel at a chosen
 * point of a checked copy: the points where the copy leaves its exclusion
 * window and another thread could run on the single hart. The four switches
 * are deliberately broken variants (negative controls) of the safe protocol. */
enum { RT_SITE_NONE, RT_SITE_HOLD, RT_SITE_BETWEEN, RT_SITE_FAULT_PRE,
       RT_SITE_FAULT_POST, RT_SITE_WINDOW };
static bool rt_norecheck, rt_noexcl, rt_dummy, rt_nohold;
module_param(rt_norecheck, bool, 0644);
MODULE_PARM_DESC(rt_norecheck, "CONTROL: authenticate only the first chunk of a copy");
module_param(rt_noexcl, bool, 0644);
MODULE_PARM_DESC(rt_noexcl, "CONTROL: no irq/preempt/pagefault exclusion around check+copy");
module_param(rt_dummy, bool, 0644);
MODULE_PARM_DESC(rt_dummy, "CONTROL: prefault by a dummy store instead of handle_mm_fault");
module_param(rt_nohold, bool, 0644);
MODULE_PARM_DESC(rt_nohold, "CONTROL: no hold on the mapping across the copy");
struct rt_state {
    struct context *c;
    struct cv_thread *b;
    bool armed, fired, pending, b_parked, map_for_b;
    int site;
    u64 b_runs, copies, copy_done, hold_seen, unmap_held, quarantine_ok;
    u32 copy_cause;
    long b_result;
};
static struct rt_state rt;
static void rt_site(struct context *, int);
static void rt_pending(struct context *);
static void rt_record(struct context *, const struct cu_result *, u64);
static void rt_probe(struct context *, unsigned long, int);
#define RT_SITE(c, where) rt_site(c, where)
#define RT_SITE_UNLOCKED(c, where) do { \
        if (rt.armed && !rt.fired && rt.site == (where)) { \
            mutex_lock(&vm_lock); rt_site(c, where); mutex_unlock(&vm_lock); } } while (0)
#define RT_PENDING(c) rt_pending(c)
#define RT_PROBE(c, addr, which) rt_probe(c, addr, which)
#define RT_NOEXCL() (rt_noexcl)
#else
#define RT_SITE(c, site) do { } while (0)
#define RT_SITE_UNLOCKED(c, site) do { } while (0)
#define RT_PENDING(c) do { } while (0)
#define RT_PROBE(c, addr, which) do { } while (0)
#define RT_NOEXCL() false
#endif

struct cv_copy_state {
    struct context *c;
    struct cv_thread *t;
    struct arena *arena;
    enum cu_direction direction;
    u64 length;
    unsigned long irq_flags, root;
    cn_u32 admission_cause;
    /* kmalloc storage, never a VMAP_STACK physical-address conversion. */
    u64 descriptor[6] __aligned(16);
#ifdef CV_RACE_TEST
    bool rt_have;
    u64 rt_off0, rt_addr0;
#endif
};

static void cv_copy_enter(void *opaque)
{
    struct cv_copy_state *s = opaque;
    /* slot_acquire() returns 0 at once for a started, resident frame: skip the call. */
    s->admission_cause = (!s->t->started || s->t->terminal) ? CU_REFUSED :
                         slot_acquire(s->t) ? CU_REFUSED : 0;
    mmap_read_lock(s->c->mm);
    if (!RT_NOEXCL()) {
        preempt_disable();
        local_irq_save(s->irq_flags);
        pagefault_disable();
    }
    s->root = select_root(s->c);
}
static void cv_copy_drain(void *opaque)
{
    (void)opaque;
    asm volatile("fence rw, rw" : : : "memory");
}
static void cv_copy_leave(void *opaque)
{
    struct cv_copy_state *s = opaque;
    restore_root(s->root);
    if (!RT_NOEXCL()) {
        pagefault_enable();
        local_irq_restore(s->irq_flags);
        preempt_enable();
    }
    mmap_read_unlock(s->c->mm);
    /* A reschedule that became pending inside the window runs here. */
    RT_PENDING(s->c);
}
static cn_u32 cv_copy_check(void *opaque, const struct cn_capref *ref,
                            cn_u64 offset, cn_u64 bytes,
                            enum cu_direction dir, cn_u64 *address)
{
    struct cv_copy_state *s = opaque;
    unsigned long cause;
    u64 *d = s->descriptor;
    if (s->admission_cause) return s->admission_cause;
#ifdef CV_RACE_TEST
    if (rt_norecheck && s->rt_have) {
        /* CONTROL: trust the first authentication for the rest of the call. */
        *address = s->rt_addr0 + (offset - s->rt_off0);
        return 0;
    }
#endif
    if (ref->reserved || ref->context_id != s->t->id ||
        !ref->context_epoch || ref->context_epoch != READ_ONCE(s->t->frame[82]) ||
        !ref->frame_slot || ref->frame_slot >= 32) return CU_REFUSED;
    if (s->arena && s->arena->copy_retired) return CU_LIFETIME;
    d[0] = ref->context_epoch; d[1] = ref->frame_slot;
    d[2] = offset; d[3] = bytes; d[4] = 0; d[5] = 0;
    if (dir == CU_TO_DOMAIN)
        asm volatile(".insn r 0x5b, 1, 0x26, %0, %1, %2"
                     : "=&r"(cause) : "r"(s->t->frame_pa), "r"(virt_to_phys(d)) : "memory");
    else
        asm volatile(".insn r 0x5b, 1, 0x25, %0, %1, %2"
                     : "=&r"(cause) : "r"(s->t->frame_pa), "r"(virt_to_phys(d)) : "memory");
#ifdef CV_NATIVE_UNSAFE_BOUNDS
    /* Test-only broken variant: the guest sentinel oracle must reject it. */
    if (cause == CU_BOUNDS) {
        d[4] = s->t->frame[8 + 2 * ref->frame_slot] + offset;
        cause = 0;
    }
#endif
    if (cause) return cause >= CU_TAG && cause <= CU_REFUSED ? cause : CU_REFUSED;
    *address = d[4];
    if (!access_ok((void __user *)*address, bytes)) return CU_BOUNDS;
    /* Preflight has no hold yet. Require a registered arena before any
     * irreversible I/O; a later copy still acquires its own mapping hold. */
    if (bytes && !s->arena) {
        struct arena *a = arena_at(s->c, *address);
        if (!a || a->copy_retired || *address - a->address > a->bytes ||
            bytes > a->bytes - (*address - a->address)) return CU_BOUNDS;
    }

    if (s->arena && (*address < s->arena->address ||
        *address - s->arena->address > s->arena->bytes ||
        bytes > s->arena->bytes - (*address - s->arena->address))) return CU_BOUNDS;
#ifdef CV_RACE_TEST
    if (!s->rt_have) { s->rt_have = true; s->rt_off0 = offset; s->rt_addr0 = *address; }
#endif
    return 0;
}
static cn_u32 cv_copy_hold(void *opaque, const struct cn_capref *ref)
{
    struct cv_copy_state *s = opaque;
    cn_u64 address = 0;
    cn_u32 cause;
    RT_SITE(s->c, RT_SITE_HOLD);
    /* The engine still authenticates the zero-length access inside its
     * checked window. No byte is accessed, so no arena hold is necessary;
     * in particular a one-past cursor must not depend on its neighbor. */
    if (!s->length) return 0;
    /* The ISA supplies the cursor. Arena lookup only retains the mapping;
     * it is never a substitute for the subsequent per-chunk authority check. */
    cv_copy_enter(s);
    cause = cv_copy_check(s, ref, 0, 0, s->direction, &address);
    cv_copy_drain(s);
    cv_copy_leave(s);
    if (cause) return cause;
#ifdef CV_RACE_TEST
    if (rt_nohold) return 0;    /* CONTROL: the mapping is not held */
#endif
    s->arena = arena_at(s->c, address);
    if (!s->arena || s->arena->copy_retired ||
        s->arena->copy_holds == UINT_MAX) return CU_REFUSED;
    /* Complete resident-page accounting before publishing the hold, so a
     * held unmap can retire immediately without a fallible pin walk. */
    if (!unpinned && pin_remaining(s->c, s->arena, FOLL_FORCE | FOLL_NOFAULT))
        return CU_REFUSED;
    ++s->arena->copy_holds;
    return 0;
}
static void cv_copy_release(void *opaque)
{
    struct cv_copy_state *s = opaque;
    struct arena *a = s->arena;
    unsigned long base, bytes;
    bool unmap;
    if (!a) return;
    base = a->address; bytes = a->bytes; unmap = a->copy_unmap_pending;
    if (WARN_ON_ONCE(!a->copy_holds)) return;
    if (--a->copy_holds || !a->copy_retired) return;
    /* retire already invalidated the node; this pass clears/releases pages
     * and removes the arena. Holding vm_lock prevents mint/reuse until done. */
    if (retire(s->c, a, true)) return;
    if (unmap) {
        int rc = vm_munmap(base, bytes);
        WARN_ON_ONCE(rc);
    }
}
static cn_u64 cv_copy_inatomic(void *opaque, cn_u64 address, void *buffer,
                              cn_u64 bytes, enum cu_direction dir)
{
    struct cv_copy_state *s = opaque;
    (void)s;
    /* The point between the authentication and the access: only a
     * preemption or interrupt can place another thread here. */
    RT_SITE(s->c, RT_SITE_WINDOW);
    if (dir == CU_TO_DOMAIN)
        return __copy_to_user_inatomic((void __user *)address, buffer, bytes);
    return __copy_from_user_inatomic(buffer, (void __user *)address, bytes);
}
static cn_u32 cv_copy_bad_mapping(struct cv_copy_state *s)
{
    unsigned long flags;
    /* handle_mm_fault may already have installed backing. If pinned-mode
     * accounting cannot represent it, merely returning EFAULT would let a
     * resumed domain put tags in an untracked page. Refuse all further entry.
     * Only the single native main thread is admitted by this experiment. */
    /* slot_live excludes terminal threads: discard the paused hardware slot
     * before making it disappear from software admission accounting. */
    preempt_disable();
    local_irq_save(flags);
    if (!s->t->slot_exported) run(s->t->frame_pa, 1);
    else memset(s->t->frame + 1024 / 8, 0, PAGE_SIZE - 1024);
    s->t->slot_exported = false;
    s->t->started = false;
    s->t->terminal = true;
    local_irq_restore(flags);
    preempt_enable();
    s->t->event = 2;
    WRITE_ONCE(s->t->frame[2], 2);
    WRITE_ONCE(s->t->frame[3], CU_REFUSED);
    slot_changed();
    return CU_REFUSED;
}
static cn_u32 cv_copy_resolve(void *opaque, cn_u64 address, enum cu_direction dir)
{
    struct cv_copy_state *s = opaque;
    struct vm_area_struct *vma;
    vm_fault_t fault;
    cn_u32 cause = CU_REFUSED;
    bool attempted = false;
    unsigned flags = FAULT_FLAG_USER | (dir == CU_TO_DOMAIN ? FAULT_FLAG_WRITE : 0);
    /* The held arena survives sleeping fault handling and concurrent logical
     * retirement. ALLOW_RETRY is intentionally absent: mmap_lock stays ours.
     * Only anonymous memory and this context's immutable image are admitted;
     * arbitrary file/device mappings remain unsupported. */
    mutex_unlock(&vm_lock);
    RT_PROBE(s->c, address, 1);
    RT_SITE_UNLOCKED(s->c, RT_SITE_FAULT_PRE);
    RT_PROBE(s->c, address, 2);
#ifdef CV_RACE_TEST
    if (rt_dummy && dir == CU_TO_DOMAIN) {
        /* CONTROL: the destructive prefault the contract forbids. */
        char zero = 0;
        cause = copy_to_user((void __user *)address, &zero, 1) ? CU_REFUSED : 0;
        mutex_lock(&vm_lock);
        return cause;
    }
#endif
    mmap_read_lock(s->c->mm);
    vma = find_vma(s->c->mm, address);
    if (!vma || address < vma->vm_start ||
        (!vma_is_anonymous(vma) && vma->vm_file != s->c->image) ||
        (vma->vm_flags & (VM_IO | VM_PFNMAP | VM_MIXEDMAP | VM_HUGETLB | VM_SHARED)) ||
        !(vma->vm_flags & (dir == CU_TO_DOMAIN ? VM_WRITE : VM_READ))) goto out;
    attempted = true;
    fault = handle_mm_fault(vma, address, flags, NULL);
    if (!(fault & (VM_FAULT_ERROR | VM_FAULT_RETRY))) cause = 0;
out:
    mmap_read_unlock(s->c->mm);
    RT_SITE_UNLOCKED(s->c, RT_SITE_FAULT_POST);
    mutex_lock(&vm_lock);
    if (cause && attempted) return cv_copy_bad_mapping(s);
    if (!cause && !unpinned) {
        struct arena *a = s->arena;
        unsigned long offset = address - a->address;
        unsigned index;
        if (address < a->address || offset >= a->bytes)
            return cv_copy_bad_mapping(s);
        /* Sealed image text is read-only and cannot contain domain tags.
         * It is excluded from pinned data-page accounting by add/resolve. */
        if (offset < a->rx_bytes)
            return dir == CU_FROM_DOMAIN ? 0 : cv_copy_bad_mapping(s);
        index = (offset - a->rx_bytes) / PAGE_SIZE;
        /* Normal pinned-mode resolve also records pages made resident by
         * the fault handler. Otherwise a later tagged domain store could
         * escape the retirement sweep because it no longer faults. */
        if (index >= a->count || (!a->pages[index] && admit_page(s->c, a, index)))
            return cv_copy_bad_mapping(s);
        {
            struct page *page;
            long n;
            bool same;
            /* The pinned profile refuses a COW/backing replacement, just
             * like resolve(). Tag-preserving COW needs the unpinned profile. */
            mmap_read_lock(s->c->mm);
            n = pin_user_pages_remote(s->c->mm, address & PAGE_MASK, 1,
                                      FOLL_WRITE | FOLL_FORCE, &page, NULL, NULL);
            mmap_read_unlock(s->c->mm);
            if (n != 1) return cv_copy_bad_mapping(s);
            same = page == a->pages[index];
            unpin_user_page(page);
            if (!same) return cv_copy_bad_mapping(s);
        }
    }
    return cause;
}
static void cv_copy_between(void *opaque)
{
    (void)opaque;
#ifndef CV_RACE_TEST
    /* Hand vm_lock over only when the scheduler wants this CPU (single hart): with nothing
     * else runnable no other task can take it, and the unlock/lock pair costs more than a
     * page of copying. A pending signal or the end of the quantum also yields. */
    if (!need_resched() && !signal_pending(current)) return;
#endif
    mutex_unlock(&vm_lock);
    cond_resched();
    mutex_lock(&vm_lock);
    RT_SITE(((struct cv_copy_state *)opaque)->c, RT_SITE_BETWEEN);
}

/* The admission rule of every checked copy; the caller holds vm_lock. */
static bool cv_copy_admitted(struct context *c, struct cv_thread *t)
{
    return checked_copy && c && t && t->context == c && c->mm == current->mm &&
        c->dom_pid == task_pid_nr(current) && t == c->main_thread && !t->user_frame &&
        t->started && !t->terminal && t->event == 3 &&
#ifdef CV_RACE_TEST
        thread_count(c) == (rt.b ? 2 : 1) &&
#else
        thread_count(c) == 1 &&
#endif
        num_possible_cpus() == 1;
}
static void cv_copy_state_init(struct cv_copy_state *s, struct context *c, struct cv_thread *t,
                               enum cu_direction dir, u64 length)
{
    s->c = c; s->t = t; s->arena = NULL; s->direction = dir; s->length = length;
    s->irq_flags = 0; s->root = 0; s->admission_cause = 0;
#ifdef CV_RACE_TEST
    s->rt_have = false; s->rt_off0 = s->rt_addr0 = 0;
#endif
}
static const struct cu_ops cv_copy_ops = {
    cv_copy_hold, cv_copy_release, cv_copy_enter, cv_copy_check,
    cv_copy_inatomic, cv_copy_drain, cv_copy_leave, cv_copy_resolve, cv_copy_between
};
/* The native syscall loop's copy. The synchronous task backend owns its context
 * until step returns (callbacks cannot destroy/transfer/retain it). Otherwise
 * this runs inside the context's own CV_STEP_NATIVE ioctl, whose
 * file reference outlives it (the control descriptor is never an application descriptor), and
 * reuses the context's state object: the protocol (hold, per-chunk check inside the exclusion
 * window, non-destructive fault retry, recheck, release) is the same cu_copy. */
struct cu_result cv_checked_copy_reuse(struct cv_copy_state *, struct context *, struct cv_thread *,
                                       const struct cn_capref *, u64, void *, u64, enum cu_direction);
struct cu_result cv_checked_copy_reuse(struct cv_copy_state *s, struct context *c, struct cv_thread *t,
                                       const struct cn_capref *ref, u64 offset,
                                       void *buffer, u64 length, enum cu_direction dir)
{
    lockdep_assert_held(&vm_lock);
    if (!cv_copy_admitted(c, t) || !IS_ALIGNED((unsigned long)s->descriptor, 16))
        return (struct cu_result){0, CU_REFUSED};
    cv_copy_state_init(s, c, t, dir, length);
    {
        struct cu_result result = cu_copy(&cv_copy_ops, s, ref, offset, buffer, length, dir, PAGE_SIZE);
#ifdef CV_RACE_TEST
        rt_record(c, &result, length);
#endif
        return result;
    }
}

struct cu_result cv_checked_copy(struct context *, struct cv_thread *,
                                const struct cn_capref *, u64, void *, u64,
                                enum cu_direction);
struct cu_result cv_checked_copy(struct context *c, struct cv_thread *t,
                                const struct cn_capref *ref, u64 offset,
                                void *buffer, u64 length, enum cu_direction dir)
{
    struct cv_copy_state *s;
    struct cu_result result = {0, CU_REFUSED};
    lockdep_assert_held(&vm_lock);
    if (!cv_copy_admitted(c, t)) return result;
    /* The current native step owns c across sleeping callbacks. There is
     * no control-FD reference or scalar entry path that can substitute for it. */
    if (!riscv_capstone_owns(c)) return result;
    s = kzalloc(sizeof(*s), GFP_KERNEL);
    if (!s) return result;
    cv_copy_state_init(s, c, t, dir, length);
    result = cu_copy(&cv_copy_ops, s, ref, offset, buffer, length, dir, PAGE_SIZE);
#ifdef CV_RACE_TEST
    rt_record(c, &result, length);
#endif
    kfree(s);
    return result;
}
