/* Included by capstone_vm.c; native allocator policy never enters this file.
 * The retained page set includes freed slots until their VMA is unmapped, so
 * complete tag sweeps cannot miss stale tags in mallocng's retained groups. */
static struct heap_object *heap_find(struct context *c, unsigned long address)
{
    struct rb_node *n = c->heap_objects.rb_node;
    while (n) {
        struct heap_object *o = rb_entry(n, struct heap_object, link);
        if (address == o->address) return o;
        n = address < o->address ? n->rb_left : n->rb_right;
    }
    return NULL;
}
static void heap_insert(struct context *c, struct heap_object *o)
{
    struct rb_node **link = &c->heap_objects.rb_node, *parent = NULL;
    while (*link) {
        struct heap_object *at = rb_entry(*link, struct heap_object, link);
        parent = *link;
        link = o->address < at->address ? &(*link)->rb_left : &(*link)->rb_right;
    }
    rb_link_node(&o->link, parent, link); rb_insert_color(&o->link, &c->heap_objects);
}
static struct heap_page *heap_page_find(struct context *c, unsigned long address)
{
    struct rb_node *n = c->heap_pages.rb_node;
    while (n) {
        struct heap_page *p = rb_entry(n, struct heap_page, link);
        if (address == p->address) return p;
        n = address < p->address ? n->rb_left : n->rb_right;
    }
    return NULL;
}
static void heap_page_insert(struct context *c, struct heap_page *p)
{
    struct rb_node **link = &c->heap_pages.rb_node, *parent = NULL;
    while (*link) {
        struct heap_page *at = rb_entry(*link, struct heap_page, link);
        parent = *link;
        link = p->address < at->address ? &(*link)->rb_left : &(*link)->rb_right;
    }
    rb_link_node(&p->link, parent, link); rb_insert_color(&p->link, &c->heap_pages);
}
static int heap_pin(struct context *c, unsigned long address, struct page **result)
{
    struct heap_page *p;
    struct page *page;
    struct vm_area_struct *v;
    long n;
    address &= PAGE_MASK;
    p = heap_page_find(c, address);
    if (p) { *result = p->page; return 0; }
    p = kzalloc(sizeof(*p), GFP_KERNEL);
    if (!p) return -ENOMEM;
    mmap_read_lock(c->mm);
    v = find_vma(c->mm, address);
    if (!v || v->vm_start > address || v->vm_file || (v->vm_flags & VM_SHARED) ||
        !(v->vm_flags & VM_WRITE)) {
        mmap_read_unlock(c->mm); kfree(p); return -EACCES;
    }
    n = pin_user_pages_remote(c->mm, address, 1, FOLL_WRITE, &page, NULL, NULL);
    mmap_read_unlock(c->mm);
    if (n != 1) { kfree(p); return -EFAULT; }
    if (!PageAnon(page) || page_mapcount(page) != 1 || !page_address(page)) {
        unpin_user_page(page); kfree(p); return -EACCES;
    }
    p->page = page; p->address = address; heap_page_insert(c, p);
    ++c->stats.pinned_pages;
    c->stats.peak_pages = max(c->stats.peak_pages, c->stats.pinned_pages);
    *result = page;
    return 0;
}
static unsigned long heap_info(struct context *c, unsigned long *slot, unsigned long info[6])
{
    unsigned long flags, root, result;
    preempt_disable(); local_irq_save(flags); root = select_root(c);
    asm volatile(".insn r 0x5b, 1, 0x54, %0, %1, %2"
                 : "=r"(result) : "r"(slot), "r"(info) : "memory");
    restore_root(root); local_irq_restore(flags); preempt_enable();
    return result;
}
static int heap_resident(struct context *c, struct heap_object *o)
{
    unsigned long end = PAGE_ALIGN(o->address + o->bytes);
    for (unsigned long at = o->address & PAGE_MASK; at < end; at += PAGE_SIZE) {
        struct page *page;
        int resident, error;
        mmap_read_lock(c->mm);
        resident = resident_page(c->mm, at);
        mmap_read_unlock(c->mm);
        if (resident < 0) return resident;
        if (!resident) continue;
        error = heap_pin(c, at, &page);
        if (error) return error;
    }
    return 0;
}
static void heap_retire(struct context *c, struct heap_object *o)
{
    unsigned long flags, root, result;
    preempt_disable(); local_irq_save(flags); root = select_root(c);
    asm volatile(".insn r 0x5b, 1, 0x53, %0, %1, zero"
                 : "=r"(result) : "r"(o->ancestor) : "memory");
    restore_root(root); local_irq_restore(flags); preempt_enable();
    /* A failed retirement must never be followed by slot reuse. */
    BUG_ON(result);
    ++c->heap_stats[1]; ++c->heap_stats[7]; --c->heap_live;
    rb_erase(&o->link, &c->heap_objects); kfree(o);
}
static void heap_clear(struct context *c, unsigned long base, unsigned long bytes)
{
    unsigned long end = base + bytes;
    /* Only complete capability granules can carry an application tag. Keep
     * payload bytes and mallocng's adjacent prefix/footer bytes unchanged. */
    for (unsigned long at = ALIGN(base, 16); at <= end && end - at >= 16; at += 16) {
        struct heap_page *page = heap_page_find(c, at & PAGE_MASK);
        unsigned long *p;
        if (!page) continue;
        p = page_address(page->page) + (at & ~PAGE_MASK);
        WRITE_ONCE(p[0], READ_ONCE(p[0])); WRITE_ONCE(p[1], READ_ONCE(p[1]));
    }
}
static int heap_release(struct context *c, unsigned long address)
{
    struct heap_object *o = c->heap_old;
    if (!c->heap_busy || !o || o->address != address) return -EINVAL;
    heap_clear(c, o->address, o->bytes);
    heap_retire(c, o); c->heap_old = NULL;
    return 0;
}
static int heap_begin(struct context *c, struct cv_heap *r)
{
    struct cv_thread *t = find_thread(c, r->thread);
    struct heap_object *o = NULL;
    unsigned long info[6];
    int error;
    if (!c->heap_enabled) return -EOPNOTSUPP;
    if (r->bytes > CV_MAX_REGION_BYTES) return -ENOMEM;
    if (c->heap_busy || !t || t->terminal || t->event != 3 ||
        r->op > CV_HEAP_FREE_LINEAR)
        return -EINVAL;
    if (r->address) {
        o = heap_find(c, r->address);
        if (!o) return -EINVAL;
        if (r->op == CV_HEAP_FREE_LINEAR) {
            if (!o->linear) return -EINVAL;
        } else if (!heap_info(c, t->frame + 28, info) ||
                   info[2] != o->address || info[3] != o->node ||
                   info[0] != o->address || info[1] > o->address + o->bytes ||
                   o->linear) return -EINVAL;
    } else if (r->op == CV_HEAP_FREE_LINEAR) return -EINVAL;
    if (o) {
        error = heap_resident(c, o);
        if (error) return error;
    }
    if (r->op != CV_HEAP_FREE && r->op != CV_HEAP_FREE_LINEAR) {
        error = ensure_nodes(c, 2 + CAP_REV_TABLE_RESERVE);
        if (error) return error;
        c->heap_pending = kzalloc(sizeof(struct heap_object), GFP_KERNEL);
        if (!c->heap_pending) return -ENOMEM;
    }
    c->heap_busy = true; c->heap_owner = r->thread;
    c->heap_old = o; c->heap_op = r->op;
    return 0;
}
static int heap_commit(struct context *c, struct cv_heap *r)
{
    struct cv_thread *t = find_thread(c, r->thread);
    struct heap_object *o = c->heap_pending;
    unsigned long flags, root, desc[3], info[6], *slot;
    bool frees = c->heap_op == CV_HEAP_FREE || c->heap_op == CV_HEAP_FREE_LINEAR;
    if (!c->heap_busy || c->heap_owner != r->thread || !t || (!o && !frees) ||
        (r->address && (r->address > ULONG_MAX - max(r->bytes, 1ULL) ||
                        r->bytes > CV_MAX_REGION_BYTES))) return -EINVAL;
    /* Failure preserves the old lifetime. Successful in-place realloc still
     * ends it and gives the same address a fresh identity. */
    if (c->heap_old && (frees || r->address)) {
        if (r->address == c->heap_old->address) {
            unsigned long kept = max(r->bytes, 1ULL);
            if (kept < c->heap_old->bytes)
                heap_clear(c, r->address + kept, c->heap_old->bytes - kept);
        } else heap_clear(c, c->heap_old->address, c->heap_old->bytes);
        heap_retire(c, c->heap_old);
    }
    c->heap_old = NULL;
    if (r->address && !frees) {
        if (heap_find(c, r->address)) return -EINVAL;
        o->address = r->address; o->bytes = max(r->bytes, 1ULL);
        o->linear = c->heap_op == CV_HEAP_LINEAR;
        slot = t->frame + 28;
        desc[0] = o->address; desc[1] = o->address + o->bytes; desc[2] = 6;
        preempt_disable(); local_irq_save(flags); root = select_root(c);
        asm volatile(".insn r 0x5b, 1, 0x50, %0, %1, %2"
                     : "=r"(o->ancestor) : "r"(slot), "r"(desc) : "memory");
        if (!o->linear)
            asm volatile(".insn i 0x5b, 3, t0, %0, 0\n"
                         ".insn r 0x5b, 1, 3, t0, zero, zero\n"
                         ".insn s 0x5b, 4, t0, 0(%0)\nli t0, 0"
                         : : "r"(slot) : "t0", "memory");
        restore_root(root); local_irq_restore(flags); preempt_enable();
        if (!heap_info(c, slot, info)) return -EIO;
        o->node = info[3]; heap_insert(c, o); t->reply_cap = true;
        ++c->heap_stats[0]; ++c->heap_stats[5];
        if (!o->linear) ++c->heap_stats[6];
        ++c->heap_live; c->heap_stats[3] = max(c->heap_stats[3], c->heap_live);
    } else kfree(o);
    c->heap_pending = NULL; c->heap_busy = false;
    return 0;
}
static int heap_copy(struct context *c, struct cv_heap_copy *r)
{
    unsigned long offset = 0, flags, status;
    struct page *src, *dst;
    int error;
    if (!c->heap_busy || r->bytes > CV_MAX_REGION_BYTES ||
        r->src > ULONG_MAX - r->bytes || r->dst > ULONG_MAX - r->bytes ||
        ((r->src | r->dst) & 15)) return -EINVAL;
    /* Fault in and validate both complete ranges before writing anything. */
    for (offset = 0; offset < r->bytes;) {
        error = heap_pin(c, r->src + offset, &src); if (error) return error;
        error = heap_pin(c, r->dst + offset, &dst); if (error) return error;
        offset += min(r->bytes - offset,
                      min(PAGE_SIZE - ((r->src + offset) & ~PAGE_MASK),
                          PAGE_SIZE - ((r->dst + offset) & ~PAGE_MASK)));
    }
    for (offset = 0; offset < r->bytes;) {
        unsigned long sa = r->src + offset, da = r->dst + offset;
        unsigned long n = min_t(unsigned long, r->bytes - offset,
                              min(PAGE_SIZE - (sa & ~PAGE_MASK), PAGE_SIZE - (da & ~PAGE_MASK)));
        char *sp = page_address(heap_page_find(c, sa & PAGE_MASK)->page) + (sa & ~PAGE_MASK);
        char *dp = page_address(heap_page_find(c, da & PAGE_MASK)->page) + (da & ~PAGE_MASK);
        while (n >= 16) {
            preempt_disable(); local_irq_save(flags);
            asm volatile(".insn r 0x5b, 1, 0x55, %0, %1, %2"
                         : "=r"(status) : "r"(dp), "r"(sp) : "memory");
            local_irq_restore(flags); preempt_enable();
            if (status) return -EIO;
            sp += 16; dp += 16; n -= 16; offset += 16;
        }
        while (n--) { WRITE_ONCE(*dp++, READ_ONCE(*sp++)); ++offset; }
    }
    return 0;
}
static void heap_drop_page(struct context *c, struct heap_page *p, bool clear)
{
    if (clear) {
        unsigned long *at = page_address(p->page);
        for (unsigned i = 0; i < PAGE_SIZE / sizeof(*at); ++i)
            WRITE_ONCE(at[i], READ_ONCE(at[i]));
    }
    set_page_dirty_lock(p->page); unpin_user_page(p->page);
    rb_erase(&p->link, &c->heap_pages); kfree(p); --c->stats.pinned_pages;
}
static int heap_range(struct context *c, struct cv_heap_range *r)
{
    struct rb_node *n = rb_first(&c->heap_pages);
    struct rb_root moved = RB_ROOT;
    if (!r->bytes || ((r->address | r->bytes | r->replacement | r->replacement_bytes) & ~PAGE_MASK) ||
        r->address > ULONG_MAX - r->bytes ||
        r->replacement > ULONG_MAX - r->replacement_bytes) return -EINVAL;
    if (r->replacement && !c->heap_busy) {
        for (struct rb_node *at = rb_first(&c->heap_pages); at; at = rb_next(at)) {
            struct heap_page *p = rb_entry(at, struct heap_page, link);
            if (p->address >= r->address && p->address - r->address < r->bytes) return -EBUSY;
        }
        return 0;
    }
    /* Validate destination keys before removing any source record. */
    if (r->replacement) {
        for (struct rb_node *at = rb_first(&c->heap_pages); at; at = rb_next(at)) {
            struct heap_page *p = rb_entry(at, struct heap_page, link), *other;
            if (p->address < r->address || p->address - r->address >= r->bytes ||
                p->address - r->address >= r->replacement_bytes) continue;
            other = heap_page_find(c, r->replacement + p->address - r->address);
            if (other && (other->address < r->address ||
                          other->address - r->address >= r->bytes)) return -EACCES;
        }
    }
    while (n) {
        struct heap_page *p = rb_entry(n, struct heap_page, link);
        n = rb_next(n);
        if (p->address < r->address || p->address - r->address >= r->bytes) continue;
        if (r->replacement && p->address - r->address < r->replacement_bytes) {
            unsigned long offset = p->address - r->address;
            rb_erase(&p->link, &c->heap_pages);
            /* Temporary list via rb_left; do not mix old and moved keys. */
            p->address = r->replacement + offset;
            p->link.rb_left = moved.rb_node; moved.rb_node = &p->link;
        } else heap_drop_page(c, p, true);
    }
    while (moved.rb_node) {
        struct heap_page *p = rb_entry(moved.rb_node, struct heap_page, link);
        moved.rb_node = p->link.rb_left;
        heap_page_insert(c, p);
    }
    return 0;
}
static int heap_resolve(struct context *c, unsigned long address)
{
    struct rb_node *n = c->heap_objects.rb_node;
    struct page *page;
    while (n) {
        struct heap_object *o = rb_entry(n, struct heap_object, link);
        if (address >= o->address && address - o->address < o->bytes)
            return heap_pin(c, address, &page);
        n = address < o->address ? n->rb_left : n->rb_right;
    }
    return -EFAULT;
}
