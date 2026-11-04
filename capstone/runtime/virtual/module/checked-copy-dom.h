/* SPDX-License-Identifier: GPL-2.0-only */
/* Internal native-task adapter. The syscall/SDK consumer is not wired yet.
 * Call with vm_lock held and a kernel-resident, prefaulted buffer. Neither the
 * context nor its thread may be destroyed until this synchronous call ends.
 * c->dom_pid admits only the owning native thread; shared-mm native threads
 * need the context-owner/slot virtualization integration before admission. */
#include <capstone/checked-copy.h>
#include "checked-copy-core.c"

static bool checked_copy;
module_param(checked_copy, bool, 0444);
MODULE_PARM_DESC(checked_copy, "Experimental CSCHECK helper (matching QEMU required; default off)");

struct cv_copy_state {
    struct context *c;
    struct cv_thread *t;
    struct arena *arena;
    enum cu_direction direction;
    u64 length;
    unsigned long irq_flags, root;
    /* kmalloc storage, never a VMAP_STACK physical-address conversion. */
    u64 descriptor[6] __aligned(16);
};

static void cv_copy_enter(void *opaque)
{
    struct cv_copy_state *s = opaque;
    mmap_read_lock(s->c->mm);
    preempt_disable();
    local_irq_save(s->irq_flags);
    pagefault_disable();
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
    pagefault_enable();
    local_irq_restore(s->irq_flags);
    preempt_enable();
    mmap_read_unlock(s->c->mm);
}
static cn_u32 cv_copy_check(void *opaque, const struct cn_capref *ref,
                            cn_u64 offset, cn_u64 bytes,
                            enum cu_direction dir, cn_u64 *address)
{
    struct cv_copy_state *s = opaque;
    unsigned long cause;
    u64 *d = s->descriptor;
    if (ref->reserved || ref->context_id != s->t->id ||
        !ref->context_epoch || ref->context_epoch != READ_ONCE(s->t->frame[82]) ||
        !ref->frame_slot || ref->frame_slot >= 32) return CU_REFUSED;
    if (s->arena && s->arena->copy_retired) return CU_LIFETIME;
    d[0] = ref->context_epoch; d[1] = ref->frame_slot;
    d[2] = offset; d[3] = bytes; d[4] = 0; d[5] = 0;
    if (dir == CU_TO_DOMAIN)
        asm volatile(".insn r 0x0b, 6, 1, %0, %1, %2"
                     : "=&r"(cause) : "r"(s->t->frame_pa), "r"(virt_to_phys(d)) : "memory");
    else
        asm volatile(".insn r 0x0b, 6, 0, %0, %1, %2"
                     : "=&r"(cause) : "r"(s->t->frame_pa), "r"(virt_to_phys(d)) : "memory");
    if (cause) return cause >= CU_TAG && cause <= CU_REFUSED ? cause : CU_REFUSED;
    *address = d[4];
    if (!access_ok((void __user *)*address, bytes)) return CU_BOUNDS;
    if (s->arena && (*address < s->arena->address ||
        *address - s->arena->address > s->arena->bytes ||
        bytes > s->arena->bytes - (*address - s->arena->address))) return CU_BOUNDS;
    return 0;
}
static cn_u32 cv_copy_hold(void *opaque, const struct cn_capref *ref)
{
    struct cv_copy_state *s = opaque;
    cn_u64 address = 0;
    cn_u32 cause;
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
    s->arena = arena_at(s->c, address);
    if (!s->arena || s->arena->copy_retired || s->c->remapping == s->arena ||
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
    (void)opaque;
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
    run(s->t->frame_pa, 1);
    s->t->started = false;
    s->t->terminal = true;
    local_irq_restore(flags);
    preempt_enable();
    s->t->event = 2;
    WRITE_ONCE(s->t->frame[2], 2);
    WRITE_ONCE(s->t->frame[3], CU_REFUSED);
    wake_up_all(&slot_wait);
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
     * File-backed / device mappings await the grant-aware fault adapter. */
    mutex_unlock(&vm_lock);
    mmap_read_lock(s->c->mm);
    vma = find_vma(s->c->mm, address);
    if (!vma || address < vma->vm_start || !vma_is_anonymous(vma) ||
        (vma->vm_flags & (VM_IO | VM_PFNMAP | VM_SHARED)) ||
        !(vma->vm_flags & (dir == CU_TO_DOMAIN ? VM_WRITE : VM_READ))) goto out;
    attempted = true;
    fault = handle_mm_fault(vma, address, flags, NULL);
    if (!(fault & (VM_FAULT_ERROR | VM_FAULT_RETRY))) cause = 0;
out:
    mmap_read_unlock(s->c->mm);
    mutex_lock(&vm_lock);
    if (cause && attempted) return cv_copy_bad_mapping(s);
    if (!cause && !unpinned) {
        struct arena *a = s->arena;
        unsigned long offset = address - a->address;
        unsigned index;
        if (address < a->address || offset < a->rx_bytes || offset >= a->bytes)
            return cv_copy_bad_mapping(s);
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
    mutex_unlock(&vm_lock);
    cond_resched();
    mutex_lock(&vm_lock);
}

struct cu_result cv_checked_copy(struct context *, struct cv_thread *,
                                const struct cn_capref *, u64, void *, u64,
                                enum cu_direction);
struct cu_result cv_checked_copy(struct context *c, struct cv_thread *t,
                                const struct cn_capref *ref, u64 offset,
                                void *buffer, u64 length, enum cu_direction dir)
{
    static const struct cu_ops ops = {
        cv_copy_hold, cv_copy_release, cv_copy_enter, cv_copy_check,
        cv_copy_inatomic, cv_copy_drain, cv_copy_leave, cv_copy_resolve, cv_copy_between
    };
    struct cv_copy_state *s;
    struct file *owner;
    struct cu_result result = {0, CU_REFUSED};
    lockdep_assert_held(&vm_lock);
    if (!checked_copy || !c || !t || t->context != c || c->mm != current->mm ||
        c->dom_pid != task_pid_nr(current) || t != c->main_thread || t->user_frame ||
        !t->started || t->terminal || t->event != 3 ||
        thread_count(c) != 1 || num_possible_cpus() != 1) return result;
    /* Keep the backing context alive across both lock-dropping callbacks.
     * The caller also owns its invocation reference until it handles result. */
    owner = fget(c->dom_fd);
    if (!owner) return result;
    if (owner->private_data != c) { fput(owner); return result; }
    s = kzalloc(sizeof(*s), GFP_KERNEL);
    if (!s) { fput(owner); return result; }
    s->c = c; s->t = t; s->direction = dir; s->length = length;
    result = cu_copy(&ops, s, ref, offset, buffer, length, dir, PAGE_SIZE);
    kfree(s);
    fput(owner);
    return result;
}
