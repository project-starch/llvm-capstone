/* SPDX-License-Identifier: GPL-2.0-only */
/* Check-then-copy race experiment (CV_RACE_TEST builds only).
 *
 * Included after dom_map/dom_unmap. A racer thread B is a real second domain
 * thread of the native context: its register frame is a user start page built
 * by the domain exactly as the launcher's thread.c builds one, and it runs
 * domain code (CREVOKE, CDERIVE, stores, and the same MAP/UNMAP services the
 * SDK issues). The kernel runs B on the single hart at the deterministic
 * points of a checked copy where another thread could have run: while the
 * copy sleeps in fault resolution, between chunks, after the blocking issue,
 * and at a reschedule pending inside the exclusion window. */

#define RT_SVC_ARM 0x7a10
#define RT_SVC_QUERY 0x7a11
#define RT_SVC_DISARM 0x7a12
#define RT_SVC_PREP 0x7a13
#define RT_SVC_IDS 0x7a14
#define RT_SVC_COLLECT 0x7a15

static u64 rt_old_base, rt_new_base, rt_resolves, rt_pte1, rt_pte2;
static long rt_munmap_rc, rt_mmap_rc, rt_collect_rc;
static u64 rt_collections;

/* 0 absent, 1 the shared zero page, 2 present writable, 3 present read-only */
static int rt_pte_state(struct mm_struct *mm, unsigned long addr)
{
    pte_t *ptep;
    spinlock_t *ptl;
    int st = 0;
    mmap_read_lock(mm);
    if (!follow_pte(mm, addr, &ptep, &ptl)) {
        pte_t pte = *ptep;
        st = is_zero_pfn(pte_pfn(pte)) ? 1 : pte_write(pte) ? 2 : 3;
        pte_unmap_unlock(ptep, ptl);
    }
    mmap_read_unlock(mm);
    return st;
}

static void rt_probe(struct context *c, unsigned long addr, int which)
{
    if (!rt.armed || rt.c != c) return;
    if (which == 1) {
        ++rt_resolves;
        if (rt_resolves == 1) rt_pte1 = rt_pte_state(c->mm, addr);
    } else if (rt_resolves == 1) {
        rt_pte2 = rt_pte_state(c->mm, addr);
    }
}

/* Make the page at addr present by a READ fault: an anonymous page then maps
 * the shared zero page and the first write must take a copy-on-write fault. */
static long rt_prep_cow(struct context *c, unsigned long addr)
{
    struct vm_area_struct *vma;
    mmap_read_lock(c->mm);
    vma = find_vma(c->mm, addr);
    if (vma && vma->vm_start <= addr && vma_is_anonymous(vma))
        handle_mm_fault(vma, addr, FAULT_FLAG_USER, NULL);
    mmap_read_unlock(c->mm);
    return rt_pte_state(c->mm, addr);
}

static long rt_thread_create(struct context *c, u64 frame_addr)
{
    struct cv_thread *t;
    struct arena *fa;
    unsigned long flags, st, old_root;
    long rc;
    if (rt.b) return -EEXIST;
    fa = frame_addr ? find_arena(c, frame_addr, PAGE_SIZE) : NULL;
    if (!fa) return -EINVAL;
    if (arena_has_thread_frame(c, fa)) return -EBUSY;
    if (slot_live(NULL) + 1 > supervisor_slots) return -EBUSY;
    t = kvzalloc(sizeof(*t), GFP_KERNEL);
    if (!t) return -ENOMEM;
    t->context = c; t->id = ++c->next_thread;
    rc = pin_thread_frame(c, frame_addr, t);
    if (rc) { kvfree(t); return rc; }
    t->frame[0] = csr_read(CSR_SATP);
    t->frame[1] = c->table_pa;
    t->frame[2] = t->frame[3] = t->frame[4] = t->frame[5] = t->frame[6] = 0;
    t->frame[7] = 3;
    t->frame[80] = t->frame[81] = 0;
    /* S-21 as in CV_THREAD_CREATE: the domain must lose its copy of the frame. */
    preempt_disable(); local_irq_save(flags);
    old_root = select_root(c);
    asm volatile(".insn r 0x5b, 1, 0x53, %0, %1, zero"
                 : "=&r"(st) : "r"(fa->ancestor) : "memory");
    restore_root(old_root);
    local_irq_restore(flags); preempt_enable();
    if (st) { clear_thread_frame(t); drop_thread_frame(t); kvfree(t); return -EACCES; }
    ++c->retired_flat;
    fa->ancestor_retired = true;
    INIT_LIST_HEAD(&t->link);
    list_add_tail(&t->link, &c->threads);
    rt.b = t;
    return t->id;
}

/* Run B until it parks (CV_SERVICE_WAIT). vm_lock is held. */
static int rt_run_b(struct context *c)
{
    struct cv_thread *b = rt.b;
    struct cv_step q = {0};
    unsigned guard;
    int rc = 0;
    if (!b || b->terminal) return -ESRCH;
    q.thread = b->id;
    q.result = rt.b_result;
    ++rt.b_runs;
    for (guard = 0; guard < 4096; ++guard) {
        q.reply = b->started && b->event == 3;
        rc = cv_step_locked(c, &q, NULL);
        if (rc == -EAGAIN) { cond_resched(); continue; }
        if (rc) break;
        if (q.kind == 1) { q.result = 0; continue; }
        if (q.kind == 4) {
            rc = resolve(c, b, ktime_get_ns() + CV_STEP_BUDGET_NS);
            if (rc) break;
            b->sfence_pending = true;
            q.result = 0;
            continue;
        }
        if (q.kind != 3) { rc = -EIO; break; }
        switch (q.args[7]) {
        case CV_SERVICE_WAIT:
            rt.b_result = 0;
            return 0;
        case CV_SERVICE_UNMAP: {
            struct arena *a = find_arena(c, q.args[0], q.args[1]);
            rt.hold_seen = a ? a->copy_holds : 0;
            rt_old_base = q.args[0];
            rt_munmap_rc = q.result = dom_unmap(c, q.args[0], q.args[1]);
            rt.unmap_held = rt.hold_seen != 0 && q.result == 0;
            break;
        }
        case CV_SERVICE_MAP: {
            struct arena *a = NULL, *candidate;
            rt.map_for_b = true;
            rt_mmap_rc = q.result = dom_map(c, &q);
            rt.map_for_b = false;
            /* Only the race oracle needs lookup by registration ID. Keep
             * this test-only scan out of production mapping bookkeeping. */
            if (!q.result)
                for_each_arena(candidate, c)
                    if (candidate->id == c->next_id) {
                        a = candidate;
                        break;
                    }
            rt_new_base = a ? a->address : 0;
            break;
        }
        case RT_SVC_COLLECT: {
            /* The table collection ensure_nodes runs when identities are scarce:
             * dead identities are swept and become free for reuse. */
            struct cv_thread *owner = NULL, *t;
            list_for_each_entry(t, &c->threads, link)
                if (t->started && !t->terminal) { owner = t; break; }
            rt_collect_rc = q.result = owner ? collect(c, owner) : -ESRCH;
            if (!q.result) { c->retired_flat = c->pushed_flat = 0; ++rt_collections; }
            break;
        }
        default:
            q.result = -ENOSYS;
            break;
        }
    }
    pr_info("capstone_rt: racer stopped rc=%d kind=%llu cause=%llu pc=0x%llx\n",
            rc, q.kind, q.cause, q.pc);
    return rc ? rc : -ELOOP;
}

static void rt_fire(struct context *c, int site)
{
    int rc;
    rt.fired = true;
    rc = rt_run_b(c);
    pr_info("capstone_rt: racer ran at site %d rc=%d runs=%llu\n", site, rc, rt.b_runs);
}

static void rt_site(struct context *c, int site)
{
    if (!rt.armed || rt.fired || rt.c != c || rt.site != site || !rt.b) return;
    /* With interrupts off nothing else can run here: the reschedule that
     * would let B in stays pending and is taken at the end of the window. */
    if (site == RT_SITE_WINDOW && irqs_disabled()) { rt.pending = true; return; }
    rt_fire(c, site);
}

static void rt_pending(struct context *c)
{
    if (!rt.pending || rt.fired || rt.c != c) return;
    rt.pending = false;
    rt_fire(c, RT_SITE_WINDOW);
}

static void rt_record(struct context *c, const struct cu_result *r, u64 length)
{
    if (!rt.armed || rt.c != c || length < PAGE_SIZE || rt.copies) return;
    ++rt.copies;
    rt.copy_done = r->bytes_done;
    rt.copy_cause = r->cause;
    pr_info("capstone_rt: copy length=%llu done=%llu cause=%u fired=%d\n",
            length, r->bytes_done, r->cause, rt.fired);
}

static bool rt_service(struct context *c, struct cv_step *s)
{
    switch (s->args[7]) {
    case CV_SERVICE_THREAD_CREATE:
        s->result = rt_thread_create(c, s->args[0]);
        return true;
    case RT_SVC_ARM:
        rt.armed = false;
        rt.fired = rt.pending = false;
        rt.copies = rt.copy_done = rt.copy_cause = rt.hold_seen = rt.unmap_held = 0;
        rt.b_runs = rt_resolves = rt_pte1 = rt_pte2 = rt_collections = 0;
        rt_collect_rc = 0;
        rt_old_base = rt_new_base = 0;
        rt_munmap_rc = rt_mmap_rc = 0;
        rt.c = c;
        rt.site = s->args[0];
        rt.armed = true;
        s->result = 0;
        return true;
    case RT_SVC_IDS: {
        /* Raw words of two capability slots in domain memory (the revoked
         * object and its replacement), for the record only. The wide format
         * keeps the table slot in lo[63:39] and the generation in hi[63:34]. */
        u64 w[2][2];
        if (copy_from_user(w[0], (void __user *)s->args[0], 16) ||
            copy_from_user(w[1], (void __user *)s->args[1], 16)) { s->result = -EFAULT; return true; }
        pr_info("capstone_rt: ids revoked lo=%llx hi=%llx slot=%llu gen=%llu replacement lo=%llx hi=%llx slot=%llu gen=%llu\n",
                w[0][0], w[0][1], w[0][0] >> 39, w[0][1] >> 34,
                w[1][0], w[1][1], w[1][0] >> 39, w[1][1] >> 34);
        s->result = 0;
        return true;
    }
    case RT_SVC_PREP:
        s->result = rt_prep_cow(c, s->args[0]);
        return true;
    case RT_SVC_DISARM:
        rt.armed = false;
        s->result = 0;
        return true;
    case RT_SVC_QUERY:
        switch (s->args[0]) {
        case 0: s->result = rt.fired; break;
        case 1: s->result = rt.copy_cause; break;
        case 2: s->result = rt.copy_done; break;
        case 3: s->result = rt.b_runs; break;
        case 4: s->result = rt.hold_seen; break;
        case 5: s->result = rt.unmap_held; break;
        case 6: s->result = rt_old_base; break;
        case 7: s->result = rt_new_base; break;
        case 8: s->result = rt_munmap_rc; break;
        case 9: s->result = rt_mmap_rc; break;
        case 10: s->result = rt.copies; break;
        case 11: s->result = rt_resolves; break;
        case 12: s->result = rt_pte1; break;
        case 13: s->result = rt_pte2; break;
        case 14: s->result = rt_collections; break;
        case 15: s->result = rt_collect_rc; break;
        default: s->result = -EINVAL; break;
        }
        return true;
    }
    return false;
}
