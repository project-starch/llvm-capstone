/* SPDX-License-Identifier: MIT */
/* Lifecycle model, lane L. Executable specification only. Mutation macros
 * (NL_MUTATE_*) exist for the negative controls and must never enter a
 * production backend. */
#include <capstone/native-life.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#define SLOT_BITS 20
#define SLOT_MASK ((1ULL << SLOT_BITS) - 1)
#define GEN_MAX SLOT_MASK
#define FLOOR 2U

cn_u64 nl_cap_slot(const struct nl_cap *c) { return c->w[1] & SLOT_MASK; }
cn_u64 nl_cap_gen(const struct nl_cap *c) { return (c->w[1] >> 20) & SLOT_MASK; }
unsigned nl_cap_rights(const struct nl_cap *c) { return (c->w[1] >> 40) & 15; }
cn_u64 nl_cap_length(const struct nl_cap *c) { return (c->w[1] >> 44) * 16; }

/* ---- world and pages ---- */
struct nl_world *nl_world_new(void)
{
    struct nl_world *w = calloc(1, sizeof(*w));
    if (w) { w->next_root = 0x1000; w->next_tid = 100; }
    return w;
}
static struct nl_page *page_get(struct nl_world *w)
{
    struct nl_page *p;
    ++w->page_allocs;
    if (w->pool_n) { p = w->pool[--w->pool_n]; ++w->page_reuses; }
    else if (!(p = malloc(sizeof(*p)))) return NULL;
    memset(p, 0, sizeof(*p)); /* anonymous zero fill: data and tags */
    p->refs = 1;
    return p;
}
static void page_put(struct nl_ns *ns, struct nl_page *p)
{
    struct nl_world *w = ns->world;
    ++ns->stats.page_puts;
    if (--p->refs) return;
    /* Rule 1: tags are cleared before the page can be reused, gated on the
     * "may hold tags" flag; this is the page free path, not the namespace's. */
    if (p->may_hold_tags) {
        memset(p->tag, 0, sizeof(p->tag));
        p->may_hold_tags = 0; ++ns->stats.free_tag_clears; ++w->tag_clear_on_free;
    }
    if (w->pool_n == w->pool_cap) {   /* kept alive: a stale raw pointer must not fault the model */
        unsigned cap = w->pool_cap ? w->pool_cap * 2 : 1024;
        struct nl_page **np = realloc(w->pool, cap * sizeof(*np));
        if (!np) { free(p); return; }
        w->pool = np; w->pool_cap = cap;
    }
    w->pool[w->pool_n++] = p;
}
void nl_world_free(struct nl_world *w)
{
    if (!w) return;
    for (unsigned i = 0; i < NL_MAX_SPACES; ++i) {
        struct nl_ns *ns = w->spaces[i];
        if (!ns) continue;
        for (unsigned j = 0; j < ns->npages; ++j) if (ns->map[j] && !--ns->map[j]->refs) free(ns->map[j]);
        if (ns->table && !--ns->table->refs) { free(ns->table->rec); free(ns->table); }
        free(ns->arenas); free(ns);
    }
    while (w->pool_n) free(w->pool[--w->pool_n]);
    free(w->pool);
    while (w->all_tasks) { struct nl_task *t = w->all_tasks; w->all_tasks = t->next_alloc; free(t); }
    free(w);
}

/* ---- lifetime table ---- */
static struct nl_table *table_new(struct nl_world *w, unsigned capacity)
{
    struct nl_table *t = calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->rec = calloc(capacity, sizeof(*t->rec));
    if (!t->rec) { free(t); return NULL; }
    t->capacity = capacity; t->next_unused = FLOOR; t->refs = 1;
    t->root_id = w->nfree_roots ? w->free_roots[--w->nfree_roots] : (w->next_root += 0x1000);
    return t;
}
static void table_release(struct nl_ns *ns, struct nl_table *t)
{
    struct nl_world *w = ns->world;
#ifdef NL_POISON_TABLE
    memset(t->rec, 0xAA, t->capacity * sizeof(*t->rec)); /* debug only: the free itself is O(1) */
#endif
    if (w->nfree_roots < 16) w->free_roots[w->nfree_roots++] = t->root_id;
    free(t->rec); free(t);
    ++ns->stats.table_frees;
}
struct nl_ns *nl_ns_new(struct nl_world *w, cn_u64 id, unsigned capacity, unsigned npages)
{
    struct nl_ns *ns;
    unsigned slot = 0;
    if (npages > NL_MAX_PAGES || capacity <= FLOOR || capacity > SLOT_MASK) return NULL;
    while (slot < NL_MAX_SPACES && w->spaces[slot]) ++slot;
    if (slot == NL_MAX_SPACES) return NULL;
    ns = calloc(1, sizeof(*ns));
    if (!ns) return NULL;
    ns->world = w; ns->id = id; ns->epoch = 1; ns->npages = npages;
    ns->base_va = 0x10000000ULL + id * 0x10000000ULL;
    ns->table = table_new(w, capacity);
    for (unsigned i = 0; ns->table && i < npages; ++i)
        if (!(ns->map[i] = page_get(w))) { ns->npages = i; goto fail; }
    if (!ns->table) { free(ns); return NULL; }
    w->spaces[slot] = ns;
    return ns;
fail:
    for (unsigned i = 0; i < ns->npages; ++i) page_put(ns, ns->map[i]);
    table_release(ns, ns->table);
    free(ns);
    return NULL;
}
struct nl_ns *nl_ns_find(struct nl_world *w, cn_u64 id)
{
    for (unsigned i = 0; i < NL_MAX_SPACES; ++i) if (w->spaces[i] && w->spaces[i]->id == id && !w->spaces[i]->freed) return w->spaces[i];
    return NULL;
}
cn_u64 nl_ns_root(const struct nl_ns *ns) { return ns->table->root_id; }

int nl_mint(struct nl_ns *ns, cn_u64 base, cn_u64 len, unsigned rights, struct nl_cap *out)
{
    struct nl_table *t = ns->table;
    unsigned slot;
    if (ns->gate) return -EAGAIN;       /* a sibling blocks on the fork's exclusion */
    if (ns->dying || ns->freed) return -ESTALE;
#ifdef NL_MUTATE_BOUNDS_WRAP
    if (base < ns->base_va || len > (SLOT_MASK << 4) || !len ||
        base + len > ns->base_va + (cn_u64)ns->npages * NL_PAGE_BYTES) return -EINVAL;
#else
    if (base < ns->base_va || len > (SLOT_MASK << 4) || !len ||
        base - ns->base_va > (cn_u64)ns->npages * NL_PAGE_BYTES ||
        len > (cn_u64)ns->npages * NL_PAGE_BYTES - (base - ns->base_va)) return -EINVAL;
#endif
    if (t->free_head) { slot = t->free_head; t->free_head = t->rec[slot].next; --t->free_count; }
    else if (t->next_unused < t->capacity) { slot = t->next_unused++; t->rec[slot].gen = 1; }
    else return -ENOMEM;
    t->rec[slot].state = NL_LIVE; t->rec[slot].next = 0; ++t->live;
    out->w[0] = base;
    out->w[1] = slot | ((cn_u64)t->rec[slot].gen << 20) | ((cn_u64)(rights & 15) << 40) |
                ((len + 15) / 16) << 44;
    out->tag = 1;
    return 0;
}
static void hw_invalidate_slot(struct nl_ns *ns, cn_u64 slot)
{
    for (unsigned i = 0; i < NL_CACHE_WAYS; ++i) {
        struct nl_hwentry *e = &ns->world->hw[i];
        if (e->valid && e->root == ns->table->root_id && e->slot == slot) e->valid = 0;
    }
}
int nl_retire(struct nl_ns *ns, const struct nl_cap *c)
{
    struct nl_table *t = ns->table;
    cn_u64 slot = nl_cap_slot(c);
    ++ns->stats.retire_calls;
    if (ns->gate) return -EAGAIN;
    if (!c->tag || slot < FLOOR || slot >= t->next_unused || t->rec[slot].state != NL_LIVE ||
        t->rec[slot].gen != nl_cap_gen(c)) return -ESTALE;
    hw_invalidate_slot(ns, slot);       /* the retire broadcast */
    --t->live;
    if (++t->rec[slot].gen > GEN_MAX) { t->rec[slot].state = NL_DEAD; return 0; }
    t->rec[slot].state = NL_FREE; t->rec[slot].next = t->free_head;
    t->free_head = slot; ++t->free_count;
    return 0;
}
int nl_table_valid(const struct nl_ns *ns, const struct nl_cap *c)
{
    const struct nl_table *t = ns->table;
    cn_u64 slot = nl_cap_slot(c);
    return c->tag && slot >= FLOOR && slot < t->next_unused && t->rec[slot].state == NL_LIVE &&
           t->rec[slot].gen == nl_cap_gen(c);
}
int nl_valid(struct nl_ns *ns, const struct nl_cap *c)
{
    struct nl_hwentry *e;
    cn_u64 slot = nl_cap_slot(c), gen = nl_cap_gen(c);
    if (!c->tag) return 0;
    e = &ns->world->hw[(ns->table->root_id / 0x1000 + slot) % NL_CACHE_WAYS];
    if (e->valid && e->root == ns->table->root_id && e->slot == slot && e->gen == gen) return 1;
    if (!nl_table_valid(ns, c)) return 0;
    *e = (struct nl_hwentry){ns->table->root_id, slot, gen, 1};
    return 1;
}
void nl_csfence_rev(struct nl_ns *ns)
{
    struct nl_world *w = ns->world;
    ++ns->stats.hw_invalidates;
    for (unsigned i = 0; i < NL_CACHE_WAYS; ++i)
        if (w->hw[i].valid && w->hw[i].root == ns->table->root_id) w->hw[i].valid = 0;
}

/* ---- memory ---- */
static int locate(struct nl_ns *ns, cn_u64 va, unsigned align, unsigned *page, unsigned *off)
{
    if (ns->freed || va < ns->base_va || (va & (align - 1))) return -EFAULT;
    va -= ns->base_va;
    if (va / NL_PAGE_BYTES >= ns->npages) return -EFAULT;
    *page = va / NL_PAGE_BYTES; *off = va % NL_PAGE_BYTES;
    return 0;
}
/* Copy on write. The tag words travel with the data (tag-aware COW). */
static struct nl_page *page_for_write(struct nl_ns *ns, unsigned idx)
{
    struct nl_page *p = ns->map[idx], *q;
    if (p->refs == 1) { ns->cow[idx] = 0; return p; }
    q = page_get(ns->world);
    if (!q) return NULL;
    memcpy(q->w, p->w, sizeof(p->w));
#ifndef NL_MUTATE_COW_DROP_TAGS
    memcpy(q->tag, p->tag, sizeof(p->tag)); q->may_hold_tags = p->may_hold_tags;
    ns->stats.cow_tag_words += NL_GRANULES / 64;
#endif
    ++ns->stats.cow_breaks;
    --p->refs; ns->map[idx] = q; ns->cow[idx] = 0;
    return q;
}
static int store_words(struct nl_ns *ns, cn_u64 va, unsigned align, const cn_u64 *v, unsigned n,
                       int tag, int keep_tag)
{
    unsigned idx, off;
    struct nl_page *p;
    int rc = locate(ns, va, align, &idx, &off);
    unsigned g;
    if (rc) return rc;
    p = page_for_write(ns, idx);
    if (!p) return -ENOMEM;
    memcpy(&p->w[off / 8], v, n * 8);
    g = off / 16;
    if (!keep_tag || tag >= 0) {
        if (tag > 0) { p->tag[g / 64] |= 1ULL << (g % 64); p->may_hold_tags = 1; }
        else p->tag[g / 64] &= ~(1ULL << (g % 64));
    }
    return 0;
}
int nl_store_cap(struct nl_ns *ns, cn_u64 va, const struct nl_cap *c)
{ return store_words(ns, va, 16, c->w, 2, c->tag ? 1 : 0, 0); }
int nl_store_u64(struct nl_ns *ns, cn_u64 va, cn_u64 v)
{
#ifdef NL_MUTATE_POKE_KEEP_TAG
    return store_words(ns, va, 8, &v, 1, -1, 1);   /* leaves the granule's tag as it was */
#else
    return store_words(ns, va, 8, &v, 1, 0, 0);    /* rule 1: clears the granule's tag */
#endif
}
int nl_load_cap(struct nl_ns *ns, cn_u64 va, struct nl_cap *out)
{
    unsigned idx, off, g;
    int rc = locate(ns, va, 16, &idx, &off);
    struct nl_page *p;
    if (rc) return rc;
    p = ns->map[idx]; g = off / 16;
    out->w[0] = p->w[off / 8]; out->w[1] = p->w[off / 8 + 1];
    out->tag = !!(p->tag[g / 64] & (1ULL << (g % 64)));
    return 0;
}
int nl_load_u64(struct nl_ns *ns, cn_u64 va, cn_u64 *v)
{
    unsigned idx, off;
    int rc = locate(ns, va, 8, &idx, &off);
    if (rc) return rc;
    *v = ns->map[idx]->w[off / 8];
    return 0;
}
int nl_arena_add(struct nl_ns *ns, cn_u64 base, cn_u64 pages, const struct nl_cap *ancestor)
{
    struct nl_arena *a = realloc(ns->arenas, (ns->narenas + 1) * sizeof(*a));
    if (!a) return -ENOMEM;
    ns->arenas = a;
    a[ns->narenas++] = (struct nl_arena){base, pages, ancestor ? *ancestor : (struct nl_cap){{0, 0}, 0}};
    return 0;
}

/* ---- checked access ---- */
int nl_check(struct nl_ns *ns, const struct nl_cap *c, cn_u64 off, cn_u64 len, int write)
{
    cn_u64 l = nl_cap_length(c);
    if (ns->dying || ns->freed) return -ESTALE;
    if (!c->tag) return -EACCES;
    if (!nl_valid(ns, c)) return -ESTALE;
    if (!(nl_cap_rights(c) & (write ? NL_RIGHT_W : NL_RIGHT_R))) return -EACCES;
    if (off > l || len > l - off) return -EFAULT;
#ifdef NL_MUTATE_BOUNDS_WRAP
    if (c->w[0] < ns->base_va || c->w[0] + off + len > ns->base_va + (cn_u64)ns->npages * NL_PAGE_BYTES)
        return -EFAULT;
#else
    {
        cn_u64 span = (cn_u64)ns->npages * NL_PAGE_BYTES, at;
        if (c->w[0] < ns->base_va) return -EFAULT;
        at = c->w[0] - ns->base_va;
        if (at > span || off > span - at || len > span - at - off) return -EFAULT;
    }
#endif
    return 0;
}
int nl_hold_begin(struct nl_ns *ns, const struct nl_cap *c, cn_u64 off, cn_u64 len, int write,
                  struct nl_hold *h)
{
    unsigned idx, o;
    int rc = nl_check(ns, c, off, len, write);
    memset(h, 0, sizeof(*h));
    if (rc) return rc;
    if ((rc = locate(ns, c->w[0] + off, 1, &idx, &o))) return rc;
    if (o + len > NL_PAGE_BYTES) return -EINVAL;   /* one chunk stays inside one page */
    h->page = write ? page_for_write(ns, idx) : ns->map[idx];
    if (!h->page) return -ENOMEM;
    h->ns = ns; h->off = o; h->len = len; h->write = write; h->active = 1;
    ++ns->inflight;
    return 0;
}
void nl_hold_do(struct nl_hold *h, const void *src, void *dst)
{
    /* No revalidation here: the chunk is non-preemptible and holds the raw page. */
    if (h->write && src) memcpy((char *)h->page->w + h->off, src, h->len);
    if (!h->write && dst) memcpy(dst, (char *)h->page->w + h->off, h->len);
    if (h->write) { /* a data write clears the granule tags it touches */
        unsigned first = h->off / 16, last = h->len ? (h->off + h->len - 1) / 16 : first;
        for (unsigned g = first; g <= last; ++g) h->page->tag[g / 64] &= ~(1ULL << (g % 64));
    }
}
void nl_hold_end(struct nl_hold *h)
{
    if (!h->active) return;
    --h->ns->inflight; h->active = 0;
}

/* ---- fork ---- */
int nl_ns_fork(struct nl_ns *p, cn_u64 child_id, const struct nl_fork_hooks *hk, struct nl_ns **out)
{
    struct nl_world *w = p->world;
    struct nl_ns *c;
    struct nl_table *pt = p->table, *ct;
    unsigned slot = 0, n, half;
    *out = NULL;
    if (p->dying || p->freed) return -ESTALE;
    if (p->gate) return -EBUSY;
#ifndef NL_MUTATE_FORK_INFLIGHT
    if (p->inflight) return -EAGAIN;   /* a chunk holds a raw page pointer: sharing it would bypass COW */
#endif
    if (nl_ns_find(w, child_id)) return -EEXIST;
    while (slot < NL_MAX_SPACES && w->spaces[slot]) ++slot;
    if (slot == NL_MAX_SPACES) return -ENOMEM;
    c = calloc(1, sizeof(*c));
    if (!c) return -ENOMEM;
    c->world = w; c->id = child_id; c->epoch = 1; c->npages = p->npages;
    c->base_va = p->base_va;   /* identical layout: capabilities in copied pages stay valid */
#ifndef NL_MUTATE_NO_GATE
    ++p->gate;                 /* siblings are held out from before the mm copy to after the table copy */
#endif
    /* Step 1: the address space. Pages are shared and write-protected; the
     * first write on either side copies the page with its tags. */
    for (unsigned i = 0; i < p->npages; ++i) {
        c->map[i] = p->map[i]; ++p->map[i]->refs;
        c->cow[i] = p->cow[i] = 1; c->priv[i] = p->priv[i];
        ++p->stats.page_refs_taken;
    }
    c->arenas = p->narenas ? malloc(p->narenas * sizeof(*c->arenas)) : NULL;
    if (p->narenas && !c->arenas) { c->freed = 1; goto fail; }
    if (p->narenas) memcpy(c->arenas, p->arenas, p->narenas * sizeof(*c->arenas));
    c->narenas = p->narenas;
    if (hk && hk->after_mm) hk->after_mm(hk->opaque, p);
    /* Step 2: the namespace. Same slots, generations and free list, in a
     * distinct table, so a revoke in the child can never reach the parent. */
#ifdef NL_MUTATE_SHARE_ROOT
    c->table = pt; ++pt->refs; c->table_shared = 1;
    (void)ct; (void)n; (void)half;
#else
    ct = table_new(w, pt->capacity);
    if (!ct) { c->freed = 1; goto fail; }
    n = pt->next_unused; half = n / 2;
    memcpy(ct->rec, pt->rec, half * sizeof(*pt->rec));
    if (hk && hk->mid_table) hk->mid_table(hk->opaque, p);
    memcpy(ct->rec + half, pt->rec + half, (n - half) * sizeof(*pt->rec));
    ct->next_unused = pt->next_unused; ct->free_head = pt->free_head;
    ct->free_count = pt->free_count; ct->live = pt->live;
    c->table = ct;
    p->stats.table_words_copied += n * 2 + 5;
#endif
    w->spaces[slot] = c;
#ifndef NL_MUTATE_NO_GATE
    --p->gate;
#endif
    *out = c;
    return 0;
fail:
#ifndef NL_MUTATE_NO_GATE
    --p->gate;
#endif
    for (unsigned i = 0; i < c->npages; ++i) if (c->map[i]) page_put(c, c->map[i]);
    if (c->table) { nl_csfence_rev(c); if (!--c->table->refs) table_release(c, c->table); }
    free(c->arenas); free(c);
    return -ENOMEM;
}
void nl_ns_abort_fork(struct nl_ns *c)
{
    if (!c) return;
    for (unsigned i = 0; i < NL_MAX_SPACES; ++i) if (c->world->spaces[i] == c) c->world->spaces[i] = NULL;
    for (unsigned i = 0; i < c->npages; ++i) if (c->map[i]) page_put(c, c->map[i]);
#ifndef NL_MUTATE_ABORT_NO_INVALIDATE
    if (c->table) nl_csfence_rev(c);   /* the root goes back to the allocator */
#endif
    if (c->table && !--c->table->refs) table_release(c, c->table);
    free(c->arenas); free(c);
}
struct nl_ns *nl_ns_get(struct nl_ns *ns) { ++ns->refs; return ns; }

/* ---- exit ---- */
static int ns_teardown(struct nl_ns *ns)
{
    ns->dying = 1;
#ifndef NL_MUTATE_EXIT_NO_DRAIN
    if (ns->inflight) return -EAGAIN;  /* accesses in flight: the caller sleeps and retries */
#endif
#ifdef NL_MUTATE_EXIT_ARENA_LOOP
    for (unsigned i = 0; i < ns->narenas; ++i) {
        ++ns->stats.arena_walks;
        ++ns->stats.retire_calls;      /* a CSRETIRE per arena */
        for (cn_u64 j = 0; j < ns->arenas[i].pages; ++j) ++ns->stats.teardown_tag_clears;
    }
#endif
    /* 1. invalidate cached authority while the table still exists */
#ifndef NL_MUTATE_EXIT_NO_INVALIDATE
    nl_csfence_rev(ns);
#endif
    /* 2. free the table: one release, whatever the number of arenas */
    if (--ns->table->refs == 0) table_release(ns, ns->table);
    ns->table = NULL;
    /* 3. the address space (Linux's exit_mmap): page puts, with tag clearing
     * owned by the page free path and gated on the may-hold-tags flag */
    for (unsigned i = 0; i < ns->npages; ++i) { page_put(ns, ns->map[i]); ns->map[i] = NULL; }
    free(ns->arenas); ns->arenas = NULL;
    ns->freed = 1;
    return 0;
}
int nl_ns_put(struct nl_ns *ns)
{
    if (ns->refs) --ns->refs;
    if (ns->refs && !ns->dying) return 0;
    if (ns->freed) return -ESTALE;
    return ns_teardown(ns);
}
int nl_ns_audit(const struct nl_ns *ns)
{
    const struct nl_table *t = ns->table;
    unsigned steps = 0, slot, live = 0;
    if (!t) return -ESTALE;
    for (slot = t->free_head; slot; slot = t->rec[slot].next) {
        if (++steps > t->capacity || slot < FLOOR || slot >= t->next_unused ||
            t->rec[slot].state != NL_FREE) return -EUCLEAN;
    }
    if (steps != t->free_count) return -EUCLEAN;
    for (slot = FLOOR; slot < t->next_unused; ++slot) {
        if (t->rec[slot].gen < 1 || t->rec[slot].state > NL_DEAD) return -EUCLEAN;
        live += t->rec[slot].state == NL_LIVE;
    }
    return live == t->live ? 0 : -EUCLEAN;
}
int nl_world_audit_hw(const struct nl_world *w)
{
    for (unsigned i = 0; i < NL_CACHE_WAYS; ++i) {
        const struct nl_hwentry *e = &w->hw[i];
        const struct nl_ns *owner = NULL;
        if (!e->valid) continue;
        for (unsigned j = 0; j < NL_MAX_SPACES; ++j)
            if (w->spaces[j] && !w->spaces[j]->freed && w->spaces[j]->table &&
                w->spaces[j]->table->root_id == e->root) owner = w->spaces[j];
        if (!owner) return -ESTALE;
        if (e->slot >= owner->table->next_unused || owner->table->rec[e->slot].state != NL_LIVE ||
            owner->table->rec[e->slot].gen != e->gen) return -ESTALE;
    }
    return 0;
}

/* ---- context engine bindings ---- */
static void translation_of(const struct nl_ns *ns, struct cn_translation *t)
{
    *t = (struct cn_translation){0x8000000000000000ULL | ns->id, 0x1000 * (ns->id + 1), 1,
                                 ns->id + 1000, ns->id, ns->epoch};
}
static int op_validate(void *o, const struct cn_owner *owner, const struct cn_translation *t)
{
    struct nl_ns *ns = nl_ns_find(o, owner->namespace_id);
    struct cn_translation want;
    if (ns && ns->id == ((struct nl_world *)o)->fail_validate_ns) return -ESTALE;
    if (!ns || ns->dying || ns->epoch != owner->namespace_epoch ||
        owner->address_space_id != ns->id + 1000) return -ESTALE;
    translation_of(ns, &want);
    return memcmp(&want, t, sizeof(want)) ? -ESTALE : 0;
}
static int op_clone(void *o, const struct cn_owner *parent, const struct cn_translation *from,
                    const struct cn_owner *child, struct cn_translation *to)
{
    struct nl_world *w = o;
    struct nl_ns *p = nl_ns_find(w, parent->namespace_id), *c;
    int rc;
    (void)from;
    if (!p) return -ESTALE;
    rc = nl_ns_fork(p, child->namespace_id, w->hooks, &c);
    if (rc) return rc;
    translation_of(c, to);
    return 0;
}
static void op_abort(void *o, const struct cn_owner *child)
{
    nl_ns_abort_fork(nl_ns_find(o, child->namespace_id));
}
void nl_cap_from_reg(struct nl_cap *c, const struct cn_register *r)
{
    c->w[0] = r->payload[0]; c->w[1] = r->payload[1]; c->tag = !!(r->flags & CN_REG_TAG);
}
void nl_reg_from_cap(struct cn_register *r, const struct nl_cap *c)
{
    memset(r, 0, sizeof(*r));
    r->payload[0] = c->w[0]; r->payload[1] = c->w[1]; r->flags = c->tag ? CN_REG_TAG : 0;
}
static int op_authorize(void *o, const struct cn_owner *owner, const struct cn_register *r,
                        cn_u64 off, cn_u64 len, int write)
{
    struct nl_ns *ns = nl_ns_find(o, owner->namespace_id);
    struct nl_cap c;
    if (!ns) return -ESTALE;
    nl_cap_from_reg(&c, r);
    return nl_check(ns, &c, off, len, write);
}
static int op_step(void *o, struct cn_frame *f, struct cn_event *e)
{
    struct nl_world *w = o;
    e->kind = w->step_kind ? w->step_kind : CN_EVENT_QUANTUM;
    e->fault_pc = f->pc; e->cause = 0; e->address = 0;
    return 0;
}
static int op_complete(void *o, struct cn_frame *f, cn_u64 reply)
{
    (void)o;
    memset(&f->x[10], 0, sizeof(f->x[10])); f->x[10].payload[0] = reply;
    f->pc += 4; f->ccsr[0].payload[0] = f->pc;
    return 0;
}
const struct cn_ops nl_cn_ops = {op_validate, op_clone, op_abort, op_authorize, op_step, op_complete};

void nl_frame_init(struct cn_frame *f, const struct nl_ns *ns, cn_u64 pc)
{
    memset(f, 0, sizeof(*f));
    f->version = CN_ABI_VERSION; f->bytes = sizeof(*f); f->features = CN_FEATURES_V1;
    translation_of(ns, &f->translation);
    f->pc = pc; f->status = 0x4000;
    f->ccsr[0].payload[0] = pc;
}

/* ---- tasks ---- */
struct nl_task *nl_task_new(struct nl_world *w, struct nl_ns *ns, cn_u64 tid, int uid,
                            const struct cn_frame *initial)
{
    struct nl_task *t = calloc(1, sizeof(*t));
    struct cn_frame f;
    if (!t) return NULL;
    if (!initial) { nl_frame_init(&f, ns, 0x8000); initial = &f; }
    t->world = w; t->ns = ns; t->tid = tid; t->uid = uid;
    t->owner = (struct cn_owner){tid, ns->id + 1000, ns->id, ns->epoch};
    if (cn_create(w->engine, &t->owner, initial, &t->ctx)) { free(t); return NULL; }
    nl_ns_get(ns);
    t->next_alloc = w->all_tasks; w->all_tasks = t;
    return t;
}
void nl_task_frame(struct nl_task *t, struct cn_frame *f)
{
    if (cn_snapshot(t->ctx, &t->owner, f)) memset(f, 0, sizeof(*f));
}
int nl_task_access(struct nl_task *t, unsigned slot, cn_u64 off, cn_u64 len, int write)
{
    struct cn_capref ref;
    int rc = cn_get_capref(t->ctx, &t->owner, slot, &ref);
    if (rc) return rc;
    return cn_authorize(t->ctx, &t->owner, &ref, off, len, write);
}
int nl_task_copy(struct nl_task *t, unsigned slot, cn_u64 off, void *buf, cn_u64 len, int write)
{
    struct cn_frame f;
    struct nl_cap c;
    cn_u64 done = 0;
    int rc = nl_task_access(t, slot, off, len, write);
    if (rc) return rc;
    nl_task_frame(t, &f);
    nl_cap_from_reg(&c, &f.x[slot]);
    while (done < len) {
        struct nl_hold h;
        cn_u64 at = c.w[0] + off + done, in_page = NL_PAGE_BYTES - at % NL_PAGE_BYTES;
        cn_u64 n = len - done < in_page ? len - done : in_page;
        /* Each chunk revalidates: the previous chunk's result is not a permission. */
        rc = nl_hold_begin(t->ns, &c, off + done, n, write, &h);
        if (rc) return rc;
        nl_hold_do(&h, write ? (char *)buf + done : NULL, write ? NULL : (char *)buf + done);
        nl_hold_end(&h);
        done += n;
    }
    return 0;
}

/* threads and TLS */
static int slot_cap(struct nl_task *t, const struct cn_frame *f, unsigned slot, unsigned need,
                    struct nl_cap *c)
{
    if (!slot || slot >= CN_REGISTER_COUNT) return -EINVAL;
    nl_cap_from_reg(c, &f->x[slot]);
    if (!c->tag || !nl_valid(t->ns, c)) return -EACCES;
    return (nl_cap_rights(c) & need) == need ? 0 : -EACCES;
}
int nl_thread_create(struct nl_task *p, unsigned flags, unsigned stack_slot, unsigned tls_slot,
                     cn_u64 tls_scalar, unsigned ct_slot, struct nl_task **out)
{
    struct cn_frame f, cf;
    struct nl_cap c;
    struct nl_task *t;
    cn_u64 tid = p->world->next_tid++;
    int rc;
    *out = NULL;
    nl_task_frame(p, &f);
    cf = f;
    if ((rc = slot_cap(p, &f, stack_slot, NL_RIGHT_W, &c))) return rc;
    cf.x[2] = f.x[stack_slot];
    if (flags & NL_CLONE_SETTLS) {
        if (!tls_slot) {
#ifdef NL_MUTATE_TLS_SCALAR
            struct nl_cap amb;   /* forges authority from a scalar address */
            if ((rc = nl_mint(p->ns, p->ns->base_va, (cn_u64)p->ns->npages * NL_PAGE_BYTES,
                              NL_RIGHT_R | NL_RIGHT_W, &amb))) return rc;
            amb.w[0] = tls_scalar;
            nl_reg_from_cap(&cf.x[4], &amb);
#else
            (void)tls_scalar;
            return -EINVAL;      /* a thread pointer is a capability, never a number */
#endif
        } else {
            if ((rc = slot_cap(p, &f, tls_slot, NL_RIGHT_R, &c))) return rc;
            cf.x[4] = f.x[tls_slot];
        }
    }
    memset(&cf.x[10], 0, sizeof(cf.x[10]));    /* the child's clone returns 0 */
    cf.pc += 4; cf.ccsr[0].payload[0] = cf.pc;
    t = nl_task_new(p->world, p->ns, tid, p->uid, &cf);
    if (!t) return -ENOMEM;
    t->mask = p->mask; t->domain = p->domain;
    memcpy(t->handler, p->handler, sizeof(t->handler));
    memcpy(t->handler_flags, p->handler_flags, sizeof(t->handler_flags));
    if (flags & NL_CLONE_CLEARTID) {
        if ((rc = slot_cap(p, &f, ct_slot, NL_RIGHT_W, &c))) { nl_task_exit(t); return rc; }
        t->clear_tid = f.x[ct_slot]; t->has_clear_tid = 1;
    }
    if (flags & NL_CLONE_SETTLS && tls_slot) { t->tls = f.x[tls_slot]; t->has_tls = 1; }
    *out = t;
    return 0;
}

int nl_task_fork(struct nl_task *p, cn_u64 child_tid, cn_u64 child_ns, const struct nl_fork_hooks *hk,
                 struct nl_task **out)
{
    struct cn_owner co = {child_tid, child_ns + 1000, child_ns, 1};
    struct cn_context *cctx;
    struct nl_task *t;
    int rc;
    *out = NULL;
    p->world->hooks = hk;
    rc = cn_clone(p->ctx, &p->owner, &co, &cctx);
    p->world->hooks = NULL;
    if (rc) return rc;
    t = calloc(1, sizeof(*t));
    if (!t) return -ENOMEM;
    *t = *p;                                  /* kernel-held copies: handlers, TLS, robust, clear_tid */
    t->ns = nl_ns_find(p->world, child_ns); t->ctx = cctx; t->owner = co; t->tid = child_tid;
    memset(t->tokens, 0, sizeof(t->tokens));
    t->exited = 0; t->stopped = 0;
    nl_ns_get(t->ns);
    t->next_alloc = p->world->all_tasks; p->world->all_tasks = t;
    *out = t;
    return 0;
}

/* futex */
static int futex_key(struct nl_task *t, unsigned slot, cn_u64 off, cn_u64 *va)
{
    struct cn_frame f;
    int rc = nl_task_access(t, slot, off, 8, 0);
    if (rc) return rc;
    nl_task_frame(t, &f);
    *va = f.x[slot].payload[0] + off;
    return 0;
}
int nl_futex_wait(struct nl_task *t, unsigned slot, cn_u64 off, cn_u64 expected, int *blocked)
{
    struct nl_world *w = t->world;
    cn_u64 va, cur;
    int rc = futex_key(t, slot, off, &va);
    *blocked = 0;
    if (rc) return rc;
    if ((rc = nl_task_copy(t, slot, off, &cur, 8, 0))) return rc;
    if (cur != expected) return -EAGAIN;
    for (unsigned i = 0; i < 64; ++i)
        if (!w->futex[i].used) {
#ifdef NL_MUTATE_FUTEX_KEY_VA
            w->futex[i] = (struct nl_futex_waiter){0, va, t->tid, 0, 1};
#else
            w->futex[i] = (struct nl_futex_waiter){t->ns->id, va, t->tid, 0, 1};
#endif
            *blocked = 1;
            return 0;
        }
    return -ENOMEM;
}
static int wake_va(struct nl_task *t, cn_u64 va, unsigned count)
{
    int woken = 0;
    for (unsigned i = 0; i < 64 && (unsigned)woken < count; ++i) {
        struct nl_futex_waiter *x = &t->world->futex[i];
#ifdef NL_MUTATE_FUTEX_KEY_VA
        if (x->used && !x->woken && x->va == va) { x->woken = 1; ++woken; }
#else
        if (x->used && !x->woken && x->ns == t->ns->id && x->va == va) { x->woken = 1; ++woken; }
#endif
    }
    return woken;
}
int nl_futex_collect(struct nl_task *t)
{
    for (unsigned i = 0; i < 64; ++i) {
        struct nl_futex_waiter *x = &t->world->futex[i];
        if (x->used && x->task == t->tid && x->woken) { x->used = 0; return 1; }
    }
    return 0;
}
int nl_futex_wake(struct nl_task *t, unsigned slot, cn_u64 off, unsigned count)
{
    cn_u64 va;
    int rc = futex_key(t, slot, off, &va);
    return rc ? rc : wake_va(t, va, count);
}
int nl_set_clear_tid(struct nl_task *t, unsigned slot)
{
    struct cn_frame f;
    struct nl_cap c;
    int rc;
    nl_task_frame(t, &f);
    if ((rc = slot_cap(t, &f, slot, NL_RIGHT_W, &c))) return rc;
    t->clear_tid = f.x[slot]; t->has_clear_tid = 1;
    return 0;
}
int nl_set_robust_list(struct nl_task *t, unsigned slot, cn_u64 futex_offset)
{
    struct cn_frame f;
    struct nl_cap c;
    int rc;
    nl_task_frame(t, &f);
    if ((rc = slot_cap(t, &f, slot, NL_RIGHT_R | NL_RIGHT_W, &c))) return rc;
    t->robust = f.x[slot]; t->robust_offset = futex_offset; t->has_robust = 1;
    return 0;
}

/* Robust list walk at exit. Entry pointers are capabilities loaded from user
 * memory WITH their tags and used under their own bounds; an untagged or dead
 * entry ends the walk. Layout: head+0 next cap, head+16 futex offset. */
#define FUTEX_TID_MASK 0x3fffffffULL
#define FUTEX_OWNER_DIED 0x40000000ULL
#define FUTEX_WAITERS 0x80000000ULL
static void robust_walk(struct nl_task *t)
{
    struct nl_cap head, cur, next;
    cn_u64 off, word;
    unsigned n = 0;
#ifdef NL_MUTATE_ROBUST_UNBOUNDED
    const unsigned limit = 20000;
#else
    const unsigned limit = NL_ROBUST_LIMIT;
#endif
    nl_cap_from_reg(&head, &t->robust);
    if (nl_check(t->ns, &head, 0, 32, 0)) return;
    if (nl_load_cap(t->ns, head.w[0], &cur) || nl_load_u64(t->ns, head.w[0] + 16, &off)) return;
    while (n < limit) {
        t->world->robust_steps++; ++n;
#ifdef NL_MUTATE_ROBUST_RAW
        /* trusts the stored address: ambient authority over the namespace */
        if (!cur.w[0] || cur.w[0] == head.w[0]) return;
        if (nl_load_u64(t->ns, cur.w[0] + off, &word)) return;
        if ((word & FUTEX_TID_MASK) == t->tid)
            { nl_store_u64(t->ns, cur.w[0] + off, (word & FUTEX_WAITERS) | FUTEX_OWNER_DIED);
              wake_va(t, cur.w[0] + off, 1); }
        if (nl_load_cap(t->ns, cur.w[0], &next)) return;
        cur = next;
        continue;
#endif
        if (!cur.tag || cur.w[0] == head.w[0]) return;
        if (nl_check(t->ns, &cur, 0, 16, 0) || nl_check(t->ns, &cur, off, 8, 1)) return;
        if (nl_load_u64(t->ns, cur.w[0] + off, &word)) return;
        if ((word & FUTEX_TID_MASK) == t->tid) {
            nl_store_u64(t->ns, cur.w[0] + off, (word & FUTEX_WAITERS) | FUTEX_OWNER_DIED);
            wake_va(t, cur.w[0] + off, 1);
        }
        if (nl_load_cap(t->ns, cur.w[0], &next)) return;
        cur = next;
    }
}
int nl_task_exit(struct nl_task *t)
{
    struct nl_cap c;
    int rc;
    if (t->exited && t->exit_pending) {      /* the namespace was still draining: finish the put */
        rc = nl_ns_put(t->ns);
        t->exit_pending = rc == -EAGAIN;
        return rc;
    }
    if (t->exited) return -ESTALE;
    /* Order: robust list, clear_child_tid + wake, waiters, context, namespace. */
    if (t->has_robust) robust_walk(t);
    if (t->has_clear_tid) {
        nl_cap_from_reg(&c, &t->clear_tid);
#ifdef NL_MUTATE_CLEARTID_RAW
        nl_store_u64(t->ns, c.w[0], 0);          /* raw address, no lifetime check */
        wake_va(t, c.w[0], 1);
#else
        if (!nl_check(t->ns, &c, 0, 8, 1)) { nl_store_u64(t->ns, c.w[0], 0); wake_va(t, c.w[0], 1); }
#endif
    }
    for (unsigned i = 0; i < 64; ++i)
        if (t->world->futex[i].used && t->world->futex[i].task == t->tid) t->world->futex[i].used = 0;
    cn_destroy(t->ctx, &t->owner);
    t->exited = 1;
    rc = nl_ns_put(t->ns);
    t->exit_pending = rc == -EAGAIN;
    return rc;
}
int nl_process_kill(struct nl_ns *ns, struct nl_task **tasks, unsigned n)
{
    int rc = 0;
#ifndef NL_MUTATE_EXIT_NO_STOP
    (void)ns;
#endif
#ifdef NL_MUTATE_EXIT_NO_STOP
    (void)tasks; (void)n; (void)rc;
    ns->refs = 0;
    return ns_teardown(ns);                   /* threads never stopped */
#else
    for (unsigned i = 0; i < n; ++i) {
        int r = nl_task_exit(tasks[i]);
        if (r != -ESTALE) rc = r;
    }
    return rc;
#endif
}

/* ---- signals ---- */
int nl_sigaction(struct nl_task *t, unsigned signo, unsigned slot, unsigned flags)
{
    struct cn_frame f;
    struct nl_cap c;
    int rc;
    if (!signo || signo >= NL_NSIG || signo == 9 || signo == 19) return -EINVAL;
    nl_task_frame(t, &f);
    if ((rc = slot_cap(t, &f, slot, NL_RIGHT_X, &c))) return rc;
    t->handler[signo] = f.x[slot]; t->handler_flags[signo] = flags;
    return 0;
}
int nl_sigaction_scalar(struct nl_task *t, unsigned signo, cn_u64 address, unsigned flags)
{
    if (!signo || signo >= NL_NSIG) return -EINVAL;
#ifndef NL_MUTATE_SIGACT_SCALAR
    (void)t;
#endif
#ifdef NL_MUTATE_SIGACT_SCALAR
    struct nl_cap amb;
    int rc = nl_mint(t->ns, t->ns->base_va, (cn_u64)t->ns->npages * NL_PAGE_BYTES,
                     NL_RIGHT_R | NL_RIGHT_W | NL_RIGHT_X, &amb);
    if (rc) return rc;
    amb.w[0] = address;
    nl_reg_from_cap(&t->handler[signo], &amb); t->handler_flags[signo] = flags;
    return 0;
#else
    (void)address; (void)flags;
    return -EINVAL;      /* a handler is a code capability registered from a slot */
#endif
}
int nl_sig_deliver(struct nl_task *t, unsigned signo, struct nl_sigctx *out, struct cn_signal_token *tok)
{
    struct cn_frame f, h;
    cn_u64 epoch;
    int rc;
    if (!signo || signo >= NL_NSIG || !(t->handler[signo].flags & CN_REG_TAG)) return -ENOENT;
#ifndef NL_MUTATE_SIG_STALE_HANDLER
    {   /* the registered capability is revalidated at every use */
        struct nl_cap hc;
        nl_cap_from_reg(&hc, &t->handler[signo]);
        if (!nl_valid(t->ns, &hc) || !(nl_cap_rights(&hc) & NL_RIGHT_X)) return -ESTALE;
    }
#endif
    nl_task_frame(t, &f);
    h = f;
    memset(&h.x[10], 0, sizeof(h.x[10])); h.x[10].payload[0] = signo;
    h.pc = t->handler[signo].payload[0];
    h.ccsr[0] = t->handler[signo]; h.ccsr[0].payload[0] = h.pc;
    if ((rc = cn_get_epoch(t->ctx, &t->owner, &epoch))) return rc;
    if ((rc = cn_signal_push(t->ctx, &t->owner, epoch, &h, tok))) return rc;
    t->tokens[tok->depth - 1] = *tok;
    t->mask_saved[tok->depth - 1] = t->mask;
    t->mask |= 1ULL << (signo - 1);
    memset(out, 0, sizeof(*out));
    out->magic = NL_SIGCTX_MAGIC; out->nonce = tok->nonce; out->depth = tok->depth;
    out->ctx_id = tok->context_id; out->pc = f.pc; out->sigmask = t->mask_saved[tok->depth - 1];
    for (unsigned i = 0; i < CN_REGISTER_COUNT; ++i) {
        out->x[i] = f.x[i].payload[0];
        out->tag_hint[i] = !!(f.x[i].flags & CN_REG_TAG);
    }
    return 0;
}
int nl_sig_return(struct nl_task *t, const struct nl_sigctx *u)
{
    struct cn_signal_token tok;
    cn_u64 epoch;
    unsigned depth;
    int rc;
    if (u->magic != NL_SIGCTX_MAGIC) return -EINVAL;
    tok = (struct cn_signal_token){u->ctx_id, u->nonce, (unsigned)u->depth, 0};
    depth = (unsigned)u->depth;
    if ((rc = cn_get_epoch(t->ctx, &t->owner, &epoch))) return rc;
#ifdef NL_MUTATE_SIG_TRUST_USER
    /* Restores honestly, then rebuilds the resumed frame from the user-writable sigcontext. */
    {
        struct cn_frame g;
        if ((rc = cn_signal_return(t->ctx, &t->owner, epoch, &tok))) return rc;
        nl_task_frame(t, &g);
        for (unsigned i = 1; i < CN_REGISTER_COUNT; ++i) {
            g.x[i].payload[0] = u->x[i];
            if (u->tag_hint[i]) g.x[i].flags |= CN_REG_TAG;
        }
        g.pc = u->pc; g.ccsr[0].payload[0] = u->pc;
        cn_destroy(t->ctx, &t->owner);
        if ((rc = cn_create(t->world->engine, &t->owner, &g, &t->ctx))) return rc;
        (void)depth;
        return 0;
    }
#else
    if ((rc = cn_signal_return(t->ctx, &t->owner, epoch, &tok))) return rc;
    {
    struct cn_frame f;
    nl_task_frame(t, &f);
    /* Only plain integer registers take the handler's edits. Capability
     * registers, PCC and the PC are the kernel's, whatever the sigcontext says. */
    for (unsigned i = 1; i < CN_REGISTER_COUNT; ++i) {
        if (f.x[i].flags & (CN_REG_TAG | CN_REG_RAW_TAG)) continue;
        if (f.x[i].payload[0] == u->x[i]) continue;
        if ((rc = cn_get_epoch(t->ctx, &t->owner, &epoch))) return rc;
        if ((rc = cn_scalar_write(t->ctx, &t->owner, epoch, i, u->x[i]))) return rc;
    }
    t->mask = u->sigmask & ~((1ULL << 8) | (1ULL << 18));   /* SIGKILL and SIGSTOP stay unblockable */
    }
    memset(&t->tokens[depth - 1], 0, sizeof(t->tokens[0]));
    return 0;
#endif
}
enum nl_sig_policy nl_sig_restart_policy(long err, unsigned flags, int handler_ran)
{
    switch (err) {
    case NL_ERESTARTNOINTR: return NL_SIG_REISSUE;
    case NL_ERESTARTSYS: return handler_ran && !(flags & NL_SA_RESTART) ? NL_SIG_EINTR : NL_SIG_REISSUE;
    case NL_ERESTARTNOHAND: return handler_ran ? NL_SIG_EINTR : NL_SIG_REISSUE;
    case NL_ERESTART_RESTARTBLOCK: return handler_ran ? NL_SIG_EINTR : NL_SIG_BLOCK;
    default: return NL_SIG_EINTR;
    }
}

int nl_sig_finish_service(struct nl_task *t, long err, unsigned sa_flags, int handler_ran,
                          struct nl_service_args *out)
{
    enum nl_sig_policy p = nl_sig_restart_policy(err, sa_flags, handler_ran);
    struct cn_frame f;
    cn_u64 epoch;
    int rc;
    if (p == NL_SIG_EINTR) {
        struct cn_event ev;
        if ((rc = cn_get_epoch(t->ctx, &t->owner, &epoch))) return rc;
        return cn_run(t->ctx, &t->owner, epoch, CN_RESUME_REPLY, (cn_u64)-4, &ev);
    }
    nl_task_frame(t, &f);
    for (unsigned i = 0; i < 6; ++i) out->a[i] = f.x[10 + i];   /* a0..a5 of the protected frame */
    out->restart_block = p == NL_SIG_BLOCK;
    return 0;
}

/* ---- ptrace and /proc/PID/mem ---- */
static int may_access(const struct nl_actor *a, const struct nl_task *t, int need_stopped)
{
#ifndef NL_MUTATE_PTRACE_DOMAIN
    /* A confined native task holds authority only through capabilities; a
     * debugger's ambient access to another address space is not among them. */
    if (a->is_domain) return -EPERM;
#endif
    if (a->uid != 0 && a->uid != t->uid) return -EPERM;
    if (t->exited) return -ESRCH;
    if (need_stopped && !t->stopped) return -ESRCH;
    return 0;
}
int nl_ptrace_may_attach(const struct nl_actor *a, const struct nl_task *t) { return may_access(a, t, 1); }
int nl_procmem_open(const struct nl_actor *a, const struct nl_task *t)
{
#ifdef NL_MUTATE_PROCMEM_DOMAIN
    if (a->uid != 0 && a->uid != t->uid) return -EPERM;
    return t->exited ? -ESRCH : 0;
#else
    return may_access(a, t, 0);
#endif
}
int nl_ptrace_peek(const struct nl_actor *a, struct nl_task *t, cn_u64 va, cn_u64 *out)
{
    int rc = may_access(a, t, 1);
    return rc ? rc : nl_load_u64(t->ns, va, out);   /* data only: a tag cannot be read out */
}
int nl_ptrace_poke(const struct nl_actor *a, struct nl_task *t, cn_u64 va, cn_u64 v)
{
    int rc = may_access(a, t, 1);
#ifdef NL_MUTATE_POKE_NO_COW
    /* writes through a shared page: the other process sees the poke */
    unsigned idx, off;
    if (rc || (rc = locate(t->ns, va, 8, &idx, &off))) return rc;
    t->ns->map[idx]->w[off / 8] = v;
    t->ns->map[idx]->tag[(off / 16) / 64] &= ~(1ULL << ((off / 16) % 64));
    return 0;
#else
    return rc ? rc : nl_store_u64(t->ns, va, v);
#endif
}
int nl_ptrace_getcaps(const struct nl_actor *a, struct nl_task *t, struct nl_sigctx *out)
{
    struct cn_frame f;
    int rc = may_access(a, t, 1);
    if (rc) return rc;
    nl_task_frame(t, &f);
    memset(out, 0, sizeof(*out));
    out->pc = f.pc; out->sigmask = t->mask;
    for (unsigned i = 0; i < CN_REGISTER_COUNT; ++i) {
        out->x[i] = f.x[i].payload[0];
        out->tag_hint[i] = !!(f.x[i].flags & CN_REG_TAG);
    }
#ifdef NL_MUTATE_PTRACE_LEAK_NONCE
    for (unsigned d = CN_MAX_SIGNAL_DEPTH; d; --d)
        if (t->tokens[d - 1].nonce) { out->nonce = t->tokens[d - 1].nonce; out->depth = d; break; }
#endif
    return 0;
}
int nl_ptrace_setreg(const struct nl_actor *a, struct nl_task *t, unsigned slot, cn_u64 v)
{
    cn_u64 epoch;
    int rc = may_access(a, t, 1);
    if (rc) return rc;
    if ((rc = cn_get_epoch(t->ctx, &t->owner, &epoch))) return rc;
    return cn_scalar_write(t->ctx, &t->owner, epoch, slot, v);
}
