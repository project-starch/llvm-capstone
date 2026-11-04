/* SPDX-License-Identifier: MIT */
/* Lane L lifecycle controls. Every CHECK is a positive control that must pass
 * on the correct model and fail on at least one NL_MUTATE_* / CN_MUTATE_* variant. */
#include <capstone/native-life.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
    fprintf(stderr, "FAIL line %u: %s\n", (unsigned)__LINE__, #x); exit(1); } } while (0)

#define PG NL_PAGE_BYTES
static struct nl_world *world(void)
{
    struct nl_world *w = nl_world_new();
    CHECK(w && cn_engine_create(CN_BACKEND_IOCTL, &nl_cn_ops, w, &w->engine) == 0);
    return w;
}
static struct nl_cap mint(struct nl_ns *ns, unsigned page, unsigned pages, unsigned rights)
{
    struct nl_cap c;
    CHECK(nl_mint(ns, ns->base_va + (cn_u64)page * PG, (cn_u64)pages * PG, rights, &c) == 0);
    return c;
}
static cn_u64 va(const struct nl_ns *ns, unsigned page, unsigned off) { return ns->base_va + (cn_u64)page * PG + off; }
static void set_cap(struct cn_frame *f, unsigned slot, const struct nl_cap *c) { nl_reg_from_cap(&f->x[slot], c); }
static void set_scalar(struct cn_frame *f, unsigned slot, cn_u64 v)
{
    memset(&f->x[slot], 0, sizeof(f->x[slot])); f->x[slot].payload[0] = v;
}
static struct nl_task *task(struct nl_world *w, struct nl_ns *ns, cn_u64 tid, const struct cn_frame *f)
{
    struct nl_task *t = nl_task_new(w, ns, tid, 1000, f);
    CHECK(t != NULL);
    return t;
}
static void cleanup(struct nl_world *w) { cn_engine_free(w->engine); nl_world_free(w); }

/* ===== fork (I6) ===== */
struct sib { struct nl_ns *ns; struct nl_cap *victim; int mint_rc, retire_rc, retire_rc2; };
static void sib_after_mm(void *o, struct nl_ns *p)
{
    struct sib *s = o; struct nl_cap c;
    s->mint_rc = nl_mint(p, va(p, 7, 0), 64, NL_RIGHT_R, &c);
}
static void sib_mid_table(void *o, struct nl_ns *p)
{
    struct sib *s = o;
    s->retire_rc = nl_retire(p, s->victim);   /* a record in the half already copied */
}

static void test_fork(void)
{
    struct nl_world *w = world();
    struct nl_ns *A = nl_ns_new(w, 1, 4096, 8), *B;
    struct nl_cap c0 = mint(A, 0, 2, 7), c1 = mint(A, 2, 1, 7), c2 = mint(A, 3, 1, 7), tmp;
    struct cn_frame f;
    struct nl_task *P, *C;
    struct nl_cap m0, m1;
    cn_u64 v;
    nl_retire(A, &(struct nl_cap){{0, 0}, 0});        /* untagged: refused, no effect */
    tmp = mint(A, 6, 1, 7); CHECK(nl_retire(A, &tmp) == 0);    /* leaves slot on the free list */
    CHECK(nl_store_cap(A, va(A, 4, 0), &c0) == 0 && nl_store_cap(A, va(A, 4, 16), &c1) == 0);
    CHECK(nl_store_u64(A, va(A, 5, 0), 0x1111) == 0);
    nl_frame_init(&f, A, 0x8000); set_cap(&f, 10, &c0); set_cap(&f, 11, &c1); set_cap(&f, 12, &c2);
    set_scalar(&f, 13, 77);
    P = task(w, A, 10, &f);
    CHECK(nl_task_fork(P, 11, 2, NULL, &C) == 0);
    B = C->ns;
    CHECK(B != A && B->id == 2 && nl_ns_audit(A) == 0 && nl_ns_audit(B) == 0);
    CHECK(B->table->next_unused == A->table->next_unused && B->table->free_head == A->table->free_head);
    /* same slots, same generations, in both namespaces */
    CHECK(nl_valid(A, &c0) && nl_valid(B, &c0) && nl_valid(A, &c1) && nl_valid(B, &c1));
    /* the child's registers are the parent's, tags included */
    { struct cn_frame g; nl_task_frame(C, &g); CHECK(g.x[10].flags & CN_REG_TAG);
      CHECK(g.x[10].payload[0] == c0.w[0] && g.x[10].payload[1] == c0.w[1]); }
    /* tag COW: a write elsewhere on a shared page copies it WITH its tags */
    { cn_u64 cb = B->stats.cow_breaks;
      CHECK(nl_store_u64(B, va(B, 4, 64), 5) == 0 && B->stats.cow_breaks == cb + 1); }
    CHECK(nl_load_cap(B, va(B, 4, 0), &m0) == 0 && m0.tag && m0.w[1] == c0.w[1]);
    CHECK(nl_load_cap(B, va(B, 4, 16), &m1) == 0 && m1.tag && nl_valid(B, &m1));
    /* data isolation after fork */
    CHECK(nl_store_u64(A, va(A, 5, 0), 0x2222) == 0);
    CHECK(nl_load_u64(B, va(B, 5, 0), &v) == 0 && v == 0x1111);
    CHECK(nl_store_u64(B, va(B, 5, 8), 0x3333) == 0);
    CHECK(nl_load_u64(A, va(A, 5, 8), &v) == 0 && v == 0);
    /* parent and child alternate through the engine's single slot, each in its own namespace */
    for (unsigned r = 0; r < 4; ++r) {
        struct nl_task *who = r & 1 ? C : P;
        struct cn_event ev;
        cn_u64 e;
        CHECK(cn_get_epoch(who->ctx, &who->owner, &e) == 0);
        CHECK(cn_run(who->ctx, &who->owner, e, CN_RESUME_RETRY, 0, &ev) == 0 && ev.kind == CN_EVENT_QUANTUM);
        { struct cn_frame g; nl_task_frame(who, &g); CHECK(g.translation.namespace_id == who->ns->id); }
    }
    /* child CREVOKE leaves the parent's capability live */
    CHECK(nl_retire(B, &c0) == 0);
    CHECK(!nl_valid(B, &c0) && !nl_valid(B, &m0));
    CHECK(nl_valid(A, &c0));
    { struct nl_cap pm; CHECK(nl_load_cap(A, va(A, 4, 0), &pm) == 0 && pm.tag && nl_valid(A, &pm)); }
    CHECK(nl_task_access(P, 10, 0, 16, 0) == 0);                         /* the parent still reads through it */
    CHECK(nl_task_access(C, 10, 0, 16, 0) == -ESTALE);                  /* the child no longer does */
    CHECK(nl_ns_root(A) != nl_ns_root(B));                              /* a distinct srevroot */
    /* and the other way round */
    CHECK(nl_retire(A, &c1) == 0 && !nl_valid(A, &c1) && nl_valid(B, &c1));
    /* parent and child mint independently from identical free lists */
    { struct nl_cap pa = mint(A, 1, 1, 7), cb = mint(B, 1, 1, 7);
      CHECK(nl_valid(A, &pa) && nl_valid(B, &cb));
      CHECK(nl_cap_slot(&pa) == nl_cap_slot(&c1) && nl_cap_slot(&cb) == nl_cap_slot(&c0));   /* each reused its own retired slot */
      CHECK(nl_cap_gen(&pa) == nl_cap_gen(&c1) + 1 && nl_cap_gen(&cb) == nl_cap_gen(&c0) + 1); }
    CHECK(nl_ns_audit(A) == 0 && nl_ns_audit(B) == 0);
    nl_task_exit(C); nl_task_exit(P);
    cleanup(w);

    /* ---- consistent snapshot: a sibling runs between the steps ---- */
    w = world();
    A = nl_ns_new(w, 1, 4096, 8);
    {
        struct nl_cap caps[8], victim;
        struct nl_ns *Ch;
        struct sib s = {0};
        struct nl_fork_hooks hk = {sib_after_mm, sib_mid_table, &s};
        int valid_at_fork[8], any_alias = 0;
        for (unsigned i = 0; i < 8; ++i) { caps[i] = mint(A, i % 6, 1, 7); }
        CHECK(nl_retire(A, &caps[1]) == 0);              /* a free-list entry exists */
        victim = caps[0]; s.victim = &victim;      /* slot 2: a record the first half-copy has already taken */
        for (unsigned i = 0; i < 8; ++i) valid_at_fork[i] = nl_valid(A, &caps[i]);
        CHECK(nl_ns_fork(A, 2, &hk, &Ch) == 0);
        /* with the gate the sibling is held out (EAGAIN) and replays afterwards */
        CHECK(nl_ns_audit(Ch) == 0);
        for (unsigned i = 0; i < 8; ++i) CHECK(nl_valid(Ch, &caps[i]) == valid_at_fork[i]);
        { struct nl_cap n = mint(Ch, 1, 1, 7);
          for (unsigned i = 0; i < 8; ++i)
              if (valid_at_fork[i] && nl_cap_slot(&n) == nl_cap_slot(&caps[i]) &&
                  nl_cap_gen(&n) == nl_cap_gen(&caps[i])) any_alias = 1;
          CHECK(!any_alias); }                         /* no two live objects share (slot, generation) */
        CHECK(s.mint_rc == -EAGAIN && s.retire_rc == -EAGAIN);   /* the sibling was held out */
        /* the deferred sibling operations now apply to the parent only */
        CHECK(nl_retire(A, &victim) == 0 && nl_valid(Ch, &victim) && !nl_valid(A, &victim));
        CHECK(nl_ns_audit(A) == 0);
        nl_ns_abort_fork(Ch);
    }
    cleanup(w);

    /* ---- fork while a checked write chunk is in flight ---- */
    w = world();
    A = nl_ns_new(w, 1, 4096, 8);
    {
        struct nl_cap d = mint(A, 0, 1, 7);
        struct nl_hold h;
        struct nl_ns *Ch = NULL;
        cn_u64 v, junk = 0xbad;
        CHECK(nl_store_u64(A, va(A, 0, 0), 0x600d) == 0);
        CHECK(nl_hold_begin(A, &d, 0, 8, 1, &h) == 0);
        { int rc = nl_ns_fork(A, 2, NULL, &Ch);
          if (rc == 0) {     /* the chunk now writes through a page it shares with the child */
              nl_hold_do(&h, &junk, NULL); nl_hold_end(&h);
              CHECK(nl_load_u64(Ch, va(Ch, 0, 0), &v) == 0 && v == 0x600d);
          }
          CHECK(rc == -EAGAIN && !Ch); }
        nl_hold_do(&h, &junk, NULL); nl_hold_end(&h);
        CHECK(nl_ns_fork(A, 2, NULL, &Ch) == 0);
        CHECK(nl_load_u64(Ch, va(Ch, 0, 0), &v) == 0 && v == 0xbad);   /* the finished write is in the snapshot */
    }
    cleanup(w);

    /* ---- an aborted clone leaves no cached authority behind ---- */
    w = world();
    A = nl_ns_new(w, 1, 4096, 8);
    {
        struct nl_cap d = mint(A, 0, 1, 7);
        struct nl_ns *C1 = NULL, *C2 = NULL;
        CHECK(nl_ns_fork(A, 2, NULL, &C1) == 0);
        cn_u64 root1 = nl_ns_root(C1);
        CHECK(nl_valid(C1, &d));                      /* warms the cache for C1's table root */
        CHECK(nl_retire(A, &d) == 0);
        nl_ns_abort_fork(C1);
        CHECK(nl_ns_fork(A, 3, NULL, &C2) == 0);
        CHECK(nl_ns_root(C2) == root1);               /* the root was handed out again: the control is live */
        CHECK(!nl_valid(C2, &d) && nl_world_audit_hw(w) == 0);
    }
    cleanup(w);

    /* ---- bounds arithmetic does not wrap ---- */
    w = world();
    A = nl_ns_new(w, 1, 4096, 8);
    {
        struct nl_cap d = mint(A, 0, 1, 7), forged, out;
        CHECK(nl_mint(A, ~0ULL - 7, 16, 7, &out) == -EINVAL);
        CHECK(nl_mint(A, A->base_va + 7 * PG, 2 * PG, 7, &out) == -EINVAL);
        forged = d; forged.w[0] = ~0ULL - 7;
        CHECK(nl_check(A, &forged, 0, 16, 0) == -EFAULT);
        CHECK(nl_check(A, &d, ~0ULL - 3, 16, 0) == -EFAULT);
        CHECK(nl_ns_new(w, 9, 1U << 21, 1) == NULL);       /* capacity beyond the slot field */
    }
    cleanup(w);

    /* ---- rollback: a failure after the namespace was cloned ---- */
    w = world();
    A = nl_ns_new(w, 1, 4096, 8);
    c0 = mint(A, 0, 1, 7);
    nl_frame_init(&f, A, 0x8000); set_cap(&f, 10, &c0);
    P = task(w, A, 10, &f);
    w->fail_validate_ns = 2;
    CHECK(nl_task_fork(P, 11, 2, NULL, &C) != 0);
    CHECK(nl_ns_find(w, 2) == NULL);
    for (unsigned i = 0; i < A->npages; ++i) CHECK(A->map[i]->refs == 1);
    CHECK(nl_store_u64(A, va(A, 0, 0), 9) == 0 && A->stats.cow_breaks == 0);   /* no leftover sharing */
    w->fail_validate_ns = 0;
    CHECK(nl_task_fork(P, 11, 2, NULL, &C) == 0);                 /* and a retry works */
    nl_task_exit(C); nl_task_exit(P);
    cleanup(w);

    /* ---- threads share the namespace; TLS is a capability ---- */
    w = world();
    A = nl_ns_new(w, 1, 4096, 8);
    {
        struct nl_cap stack = mint(A, 0, 1, 7), tls = mint(A, 1, 1, 7), ct = mint(A, 2, 1, 7), ro = mint(A, 3, 1, NL_RIGHT_R);
        struct nl_task *T1, *T2;
        nl_frame_init(&f, A, 0x8000);
        set_cap(&f, 10, &stack); set_cap(&f, 11, &tls); set_cap(&f, 12, &ct); set_cap(&f, 13, &ro);
        set_scalar(&f, 14, va(A, 1, 0));
        T1 = task(w, A, 10, &f);
        CHECK(nl_thread_create(T1, NL_CLONE_SETTLS, 10, 11, 0, 0, &T2) == 0);
        CHECK(T2->ns == T1->ns && A->refs == 2);
        { struct cn_frame g; nl_task_frame(T2, &g);
          CHECK((g.x[4].flags & CN_REG_TAG) && g.x[4].payload[1] == tls.w[1]);
          CHECK((g.x[2].flags & CN_REG_TAG) && g.x[10].payload[0] == 0 && !(g.x[10].flags & CN_REG_TAG)); }
        /* a scalar address is not a thread pointer; a read-only stack or a scalar slot is refused */
        { struct nl_task *T3 = NULL;
          CHECK(nl_thread_create(T1, NL_CLONE_SETTLS, 10, 0, va(A, 5, 0), 0, &T3) == -EINVAL && !T3);
          CHECK(nl_thread_create(T1, NL_CLONE_SETTLS, 13, 11, 0, 0, &T3) == -EACCES && !T3);
          CHECK(nl_thread_create(T1, NL_CLONE_SETTLS, 10, 14, 0, 0, &T3) == -EACCES && !T3); }
        /* one table: a retire by one thread is seen by the other */
        CHECK(nl_task_access(T2, 2, 0, 8, 1) == 0);
        CHECK(nl_retire(A, &stack) == 0);
        CHECK(nl_task_access(T2, 2, 0, 8, 1) == -ESTALE && nl_task_access(T1, 10, 0, 8, 1) == -ESTALE);
        CHECK(nl_task_exit(T1) == 0 && !A->freed);          /* a thread's exit does not tear the namespace down */
        CHECK(nl_task_exit(T2) == 0 && A->freed);
    }
    cleanup(w);
}

/* ===== exit (I6) ===== */
static void arena_run(unsigned arenas, struct nl_stats *out, cn_u64 *root_out)
{
    struct nl_world *w = world();
    struct nl_ns *A = nl_ns_new(w, 1, 4096, 64);
    struct nl_task *T;
    struct nl_cap c;
    struct cn_frame f;
    for (unsigned i = 0; i < arenas; ++i) {
        c = mint(A, i % 60, 1, 7);
        CHECK(nl_arena_add(A, va(A, i % 60, 0), 1, &c) == 0);
    }
    CHECK(nl_store_cap(A, va(A, 61, 0), &c) == 0);          /* the pages hold tags: the free path will clear them */
    nl_frame_init(&f, A, 0x8000);
    T = task(w, A, 10, &f);
    { struct nl_stats before = A->stats;
      CHECK(nl_task_exit(T) == 0 && A->freed);
      *out = A->stats;
      out->retire_calls -= before.retire_calls;
      if (root_out) *root_out = 0; }
    cleanup(w);
}
static void exit_hold_scenario(void)
{
    struct nl_world *w = world();
    struct nl_ns *A = nl_ns_new(w, 1, 4096, 4), *B;
    struct nl_cap c = mint(A, 0, 1, 7);
    struct nl_task *T;
    struct cn_frame f;
    struct nl_hold h;
    cn_u64 sentinel = 0x5e17e1, got = 0, junk = 0xbadbadbadULL;
    int rc;
    nl_frame_init(&f, A, 0x8000); set_cap(&f, 10, &c);
    T = task(w, A, 10, &f);
    CHECK(nl_hold_begin(A, &c, 0, 8, 1, &h) == 0 && A->inflight == 1);
    rc = nl_task_exit(T);
    if (A->freed) {   /* the table and pages are gone while the chunk still holds its raw page */
        B = nl_ns_new(w, 2, 4096, 4);
        for (unsigned i = 0; i < 4; ++i)
            if (B->map[i] == h.page) { CHECK(nl_store_u64(B, va(B, i, 0), sentinel) == 0); }
        nl_hold_do(&h, &junk, NULL);
        for (unsigned i = 0; i < 4; ++i)
            if (B->map[i] == h.page) { CHECK(nl_load_u64(B, va(B, i, 0), &got) == 0); CHECK(got == sentinel); }
    }
    CHECK(rc == -EAGAIN && !A->freed);                       /* exit waits for the access to drain */
    nl_hold_do(&h, &junk, NULL); nl_hold_end(&h);
    CHECK(A->inflight == 0 && nl_task_exit(T) == 0 && A->freed);   /* the exit that was told to wait finishes */
    cleanup(w);
}
static void test_exit(void)
{
    struct nl_stats one, many;
    struct nl_world *w;
    struct nl_ns *A, *B;
    /* cached authority is invalidated before its table can be reused */
    w = world();
    A = nl_ns_new(w, 1, 4096, 4);
    { struct nl_cap old = mint(A, 0, 1, 7), seen;
      cn_u64 root = nl_ns_root(A);
      CHECK(nl_valid(A, &old) && nl_world_audit_hw(w) == 0);
      A->refs = 1;
      CHECK(nl_ns_put(A) == 0 && A->freed);
      B = nl_ns_new(w, 2, 4096, 4);
      CHECK(nl_ns_root(B) == root);                           /* the table storage was reused: the control is live */
      CHECK(nl_world_audit_hw(w) == 0);
      seen = old;
      CHECK(!nl_valid(B, &seen)); }
    cleanup(w);

    arena_run(1, &one, NULL);
    arena_run(400, &many, NULL);
    /* O(1): no per-arena retire loop, no teardown tag clearing, one table free, one invalidation */
    CHECK(many.retire_calls == 0 && one.retire_calls == 0);
    CHECK(many.teardown_tag_clears == 0 && one.teardown_tag_clears == 0);
    CHECK(many.arena_walks == 0);
    CHECK(many.table_frees == 1 && one.table_frees == 1 && many.hw_invalidates == 1 && one.hw_invalidates == 1);
    CHECK(many.free_tag_clears == one.free_tag_clears && one.free_tag_clears >= 1);   /* the page free path, once per tagged page */
    CHECK(many.page_puts == one.page_puts + 0 && many.page_puts == 64);                 /* exit_mmap is per page, same either way */

    exit_hold_scenario();

    /* exit stops every thread before the namespace is freed */
    w = world();
    A = nl_ns_new(w, 1, 4096, 4);
    { struct nl_cap st = mint(A, 0, 1, 7);
      struct cn_frame f;
      struct nl_task *T[3];
      nl_frame_init(&f, A, 0x8000); set_cap(&f, 10, &st);
      T[0] = task(w, A, 10, &f);
      CHECK(nl_thread_create(T[0], 0, 10, 0, 0, 0, &T[1]) == 0 && nl_thread_create(T[0], 0, 10, 0, 0, 0, &T[2]) == 0);
      CHECK(A->refs == 3);
      nl_process_kill(A, T, 3);
      CHECK(A->freed);
      for (unsigned i = 0; i < 3; ++i) { cn_u64 e; CHECK(cn_get_epoch(T[i]->ctx, &T[i]->owner, &e) == -ESTALE); }
    }
    cleanup(w);
    w = world();
    A = nl_ns_new(w, 1, 4096, 4);
    { struct nl_cap st = mint(A, 0, 1, 7);
      struct cn_frame f;
      struct nl_task *T[2];
      struct nl_hold h;
      nl_frame_init(&f, A, 0x8000); set_cap(&f, 10, &st);
      T[0] = task(w, A, 10, &f);
      CHECK(nl_thread_create(T[0], 0, 10, 0, 0, 0, &T[1]) == 0);
      CHECK(nl_hold_begin(A, &st, 0, 8, 0, &h) == 0);
      CHECK(nl_process_kill(A, T, 2) == -EAGAIN && !A->freed);     /* waits for the access */
      nl_hold_end(&h);
      CHECK(nl_process_kill(A, T, 2) == 0 && A->freed);            /* and the retry completes */
    }
    cleanup(w);

    /* rule 1: pages leave the pool without tags */
    w = world();
    A = nl_ns_new(w, 1, 4096, 4);
    { struct nl_cap c = mint(A, 0, 1, 7);
      CHECK(nl_store_cap(A, va(A, 1, 0), &c) == 0);
      A->refs = 1; CHECK(nl_ns_put(A) == 0);
      CHECK(w->pool_n == 4 && w->tag_clear_on_free == 1);
      for (unsigned i = 0; i < w->pool_n; ++i) {
          CHECK(!w->pool[i]->may_hold_tags);
          for (unsigned k = 0; k < NL_GRANULES / 64; ++k) CHECK(w->pool[i]->tag[k] == 0);
      } }
    cleanup(w);
}

/* ===== signals (I5) ===== */
static void test_signals(void)
{
    struct nl_world *w = world();
    struct nl_ns *A = nl_ns_new(w, 1, 4096, 8);
    struct nl_cap d0 = mint(A, 0, 1, 7), d1 = mint(A, 1, 1, 7), code = mint(A, 2, 1, NL_RIGHT_X | NL_RIGHT_R), ro = mint(A, 3, 1, NL_RIGHT_R);
    struct cn_frame f0, f1, f2;
    struct nl_task *T, *U;
    struct nl_sigctx u, a, b;
    struct cn_signal_token tk, tk2;
    cn_u64 ep;
    nl_frame_init(&f0, A, 0x8000);
    set_cap(&f0, 10, &d0); set_cap(&f0, 11, &d1); set_cap(&f0, 12, &code); set_cap(&f0, 14, &ro);
    set_scalar(&f0, 13, 77);
    f0.ccsr[0].flags = CN_REG_TAG; f0.ccsr[0].payload[1] = 0xc0de;
    T = task(w, A, 10, &f0);
    CHECK(nl_sigaction(T, 10, 12, NL_SA_RESTART) == 0 && nl_sigaction(T, 11, 12, 0) == 0);
    CHECK(nl_sigaction(T, 10, 14, 0) == -EACCES);                /* a data capability is not a handler */
    CHECK(nl_sigaction(T, 9, 12, 0) == -EINVAL);                 /* SIGKILL cannot be caught */
    CHECK(nl_sigaction_scalar(T, 10, va(A, 2, 0), 0) == -EINVAL); /* a number is not a handler */
    CHECK(T->handler[10].flags & CN_REG_TAG);
    { struct nl_cap hc2 = mint(A, 5, 1, NL_RIGHT_X | NL_RIGHT_R);
      struct cn_frame hf;
      struct nl_task *H;
      nl_frame_init(&hf, A, 0x8000); set_cap(&hf, 12, &hc2);
      H = task(w, A, 40, &hf);
      CHECK(nl_sigaction(H, 12, 12, 0) == 0);
      CHECK(nl_retire(A, &hc2) == 0);                      /* the handler's lifetime ended */
      CHECK(nl_sig_deliver(H, 12, &u, &tk) == -ESTALE);    /* never installed as PCC */
      nl_task_exit(H); }

    /* a plain delivery and an honest return restore the whole frame */
    nl_task_frame(T, &f1);
    CHECK(nl_sig_deliver(T, 10, &u, &tk) == 0 && tk.depth == 1 && u.nonce == tk.nonce);
    nl_task_frame(T, &f2);
    CHECK(f2.pc == code.w[0] && f2.x[10].payload[0] == 10 && !(f2.x[10].flags & CN_REG_TAG));
    CHECK(nl_sig_return(T, &u) == 0);
    nl_task_frame(T, &f2);
    CHECK(!memcmp(&f1, &f2, sizeof(f1)));

    /* user sigcontext mutation cannot forge the kernel frame */
    CHECK(nl_sig_deliver(T, 10, &u, &tk) == 0);
    u.x[10] = 0x4141414141414141ULL; u.tag_hint[10] = 1;           /* try to retarget a capability register */
    u.x[11] ^= 0xffff; u.x[12] = 0; u.tag_hint[12] = 1;
    u.pc = 0x1234; u.x[13] = 99;                                   /* a scalar register and the PC */
    u.sigmask = ~0ULL;
    CHECK(nl_sig_return(T, &u) == 0);
    nl_task_frame(T, &f2);
    CHECK(!memcmp(&f2.x[10], &f1.x[10], sizeof(f1.x[10])));
    CHECK(!memcmp(&f2.x[11], &f1.x[11], sizeof(f1.x[11])) && !memcmp(&f2.x[12], &f1.x[12], sizeof(f1.x[12])));
    CHECK(f2.pc == f1.pc && !memcmp(&f2.ccsr[0], &f1.ccsr[0], sizeof(f1.ccsr[0])));
    CHECK(f2.x[13].payload[0] == 99 && !(f2.x[13].flags & CN_REG_TAG));    /* the legitimate edit lands */
    CHECK(!(T->mask & ((1ULL << 8) | (1ULL << 18))));                      /* unblockable signals stay so */
    T->mask = 0;

    /* replay and wrong-frame tokens fail */
    CHECK(nl_sig_deliver(T, 10, &a, &tk) == 0);
    CHECK(nl_sig_return(T, &a) == 0);
    CHECK(nl_sig_deliver(T, 10, &b, &tk2) == 0 && tk2.depth == 1 && tk2.nonce != tk.nonce);
    CHECK(nl_sig_return(T, &a) == -EPERM);                        /* the consumed token, at the same depth */
    CHECK(nl_sig_return(T, &b) == 0);

    /* nested frames unwind in order only */
    CHECK(nl_sig_deliver(T, 10, &a, &tk) == 0);
    CHECK(nl_sig_deliver(T, 11, &b, &tk2) == 0 && tk2.depth == 2);
    CHECK(nl_sig_return(T, &a) == -EPERM);                        /* out of order */
    CHECK(nl_sig_return(T, &b) == 0);
    nl_task_frame(T, &f2);
    CHECK(f2.pc == code.w[0] && f2.x[10].payload[0] == 10);       /* back in the first handler */
    CHECK(nl_sig_return(T, &a) == 0);
    nl_task_frame(T, &f2);
    CHECK(f2.pc == f1.pc);
    for (unsigned i = 0; i < CN_MAX_SIGNAL_DEPTH; ++i) CHECK(nl_sig_deliver(T, 10, &a, &tk) == 0);
    CHECK(nl_sig_deliver(T, 10, &a, &tk) == -EOVERFLOW);
    nl_task_exit(T);
    cleanup(w);

    w = world();
    A = nl_ns_new(w, 1, 4096, 8);
    d0 = mint(A, 0, 1, 7); d1 = mint(A, 1, 1, 7); code = mint(A, 2, 1, NL_RIGHT_X | NL_RIGHT_R);
    nl_frame_init(&f0, A, 0x8000); set_cap(&f0, 10, &d0); set_cap(&f0, 11, &d1); set_cap(&f0, 12, &code);
    T = task(w, A, 10, &f0);
    CHECK(nl_thread_create(T, 0, 10, 0, 0, 0, &U) == 0);
    CHECK(nl_sigaction(T, 10, 12, 0) == 0 && nl_sigaction(U, 10, 12, 0) == 0);
    CHECK(nl_sig_deliver(T, 10, &a, &tk) == 0 && nl_sig_deliver(U, 10, &b, &tk2) == 0);
    CHECK(nl_sig_return(U, &a) == -EPERM);                         /* thread T's token on thread U */
    CHECK(nl_sig_return(U, &b) == 0 && nl_sig_return(T, &a) == 0);

    /* restart rules, from the protected frame */
    CHECK(nl_sig_restart_policy(NL_ERESTARTSYS, 0, 1) == NL_SIG_EINTR);
    CHECK(nl_sig_restart_policy(NL_ERESTARTSYS, NL_SA_RESTART, 1) == NL_SIG_REISSUE);
    CHECK(nl_sig_restart_policy(NL_ERESTARTSYS, 0, 0) == NL_SIG_REISSUE);
    CHECK(nl_sig_restart_policy(NL_ERESTARTNOHAND, NL_SA_RESTART, 1) == NL_SIG_EINTR);
    CHECK(nl_sig_restart_policy(NL_ERESTARTNOHAND, 0, 0) == NL_SIG_REISSUE);
    CHECK(nl_sig_restart_policy(NL_ERESTARTNOINTR, 0, 1) == NL_SIG_REISSUE);
    CHECK(nl_sig_restart_policy(NL_ERESTART_RESTARTBLOCK, NL_SA_RESTART, 1) == NL_SIG_EINTR);
    CHECK(nl_sig_restart_policy(NL_ERESTART_RESTARTBLOCK, 0, 0) == NL_SIG_BLOCK);
    CHECK(nl_sig_restart_policy(4, 0, 1) == NL_SIG_EINTR);
    /* a blocked service is interrupted by a signal, the handler edits its sigcontext,
     * and the reissued request uses the kernel's arguments */
    cleanup(w);
    w = world();
    A = nl_ns_new(w, 1, 4096, 8);
    d0 = mint(A, 0, 1, 7); d1 = mint(A, 1, 1, 7); code = mint(A, 2, 1, NL_RIGHT_X | NL_RIGHT_R);
    nl_frame_init(&f0, A, 0x8000); set_cap(&f0, 10, &d0); set_cap(&f0, 11, &d1); set_cap(&f0, 12, &code);
    set_scalar(&f0, 13, 4096);
    T = task(w, A, 20, &f0);
    {
        struct cn_event ev;
        struct cn_frame g;
        w->step_kind = CN_EVENT_SERVICE;
        CHECK(cn_get_epoch(T->ctx, &T->owner, &ep) == 0 && cn_run(T->ctx, &T->owner, ep, CN_RESUME_RETRY, 0, &ev) == 0);
        CHECK(ev.kind == CN_EVENT_SERVICE);
        w->step_kind = 0;
        CHECK(nl_sigaction(T, 10, 12, 0) == 0);
        nl_task_frame(T, &f1);
        CHECK(nl_sig_deliver(T, 10, &u, &tk) == 0);
        u.x[11] = 0xdead; u.tag_hint[11] = 1; u.x[13] = 8192;      /* buffer capability and a length */
        CHECK(nl_sig_return(T, &u) == 0);
        nl_task_frame(T, &g);
        CHECK(!memcmp(&g.x[11], &f1.x[11], sizeof(g.x[11])));         /* the buffer argument is still the kernel's */
        CHECK(g.x[13].payload[0] == 8192);                            /* the scalar edit applies, as in Linux */
        /* the pending service survives the signal: ERESTARTSYS with SA_RESTART reissues from the
         * protected frame, without it the same event is answered with -EINTR */
        { struct nl_service_args sa;
          CHECK(nl_sig_finish_service(T, NL_ERESTARTSYS, NL_SA_RESTART, 1, &sa) == 0 && !sa.restart_block);
          CHECK((sa.a[1].flags & CN_REG_TAG) && sa.a[1].payload[1] == d1.w[1] && sa.a[3].payload[0] == 8192);
          CHECK(nl_sig_finish_service(T, NL_ERESTART_RESTARTBLOCK, 0, 0, &sa) == 0 && sa.restart_block); }
        CHECK(nl_sig_finish_service(T, NL_ERESTARTSYS, 0, 1, NULL) == 0);       /* EINTR reply */
        nl_task_frame(T, &g);
        CHECK(g.x[10].payload[0] == (cn_u64)-4 && g.pc == f1.pc + 4);
        (void)ep;
    }
    cleanup(w);
}

/* ===== ptrace and /proc (I5) ===== */
static void test_ptrace(void)
{
    struct nl_world *w = world();
    struct nl_ns *A = nl_ns_new(w, 1, 4096, 8);
    struct nl_cap d0 = mint(A, 0, 1, 7), cell = mint(A, 4, 1, 7), code = mint(A, 2, 1, NL_RIGHT_X | NL_RIGHT_R);
    struct cn_frame f;
    struct nl_task *V, *V2;
    struct nl_actor dbg = {1000, 0}, dom = {1000, 1}, other = {1001, 0};
    struct nl_sigctx u, g;
    struct cn_signal_token tk;
    struct nl_cap m;
    cn_u64 v;
    nl_ns_find(w, 1)->priv[7] = 1;                                         /* a scalar-only page outside every grant */
    CHECK(nl_store_u64(A, va(A, 7, 0), 0x5ec2e7) == 0);
    CHECK(nl_store_cap(A, va(A, 4, 0), &d0) == 0 && nl_store_u64(A, va(A, 4, 64), 0x77) == 0);
    nl_frame_init(&f, A, 0x8000); set_cap(&f, 10, &d0); set_cap(&f, 12, &code); set_cap(&f, 11, &cell);
    V = task(w, A, 10, &f);
    CHECK(nl_sigaction(V, 10, 12, 0) == 0);

    CHECK(nl_ptrace_may_attach(&dbg, V) == -ESRCH);                        /* not stopped */
    V->stopped = 1;
    CHECK(nl_ptrace_may_attach(&dbg, V) == 0 && nl_ptrace_may_attach(&other, V) == -EPERM);
    CHECK(nl_ptrace_may_attach(&dom, V) == -EPERM);                        /* a confined task is not a debugger */
    /* the debugger reads data; a capability's tag does not travel with the value */
    CHECK(nl_ptrace_peek(&dbg, V, va(A, 4, 0), &v) == 0 && v == d0.w[0]);
    CHECK(nl_ptrace_peek(&dbg, V, va(A, 7, 0), &v) == 0 && v == 0x5ec2e7);
    v = 0;
    CHECK(nl_ptrace_peek(&dom, V, va(A, 7, 0), &v) == -EPERM && v == 0);   /* no ambient reach for a domain */
    CHECK(nl_ptrace_peek(&other, V, va(A, 7, 0), &v) == -EPERM);
    CHECK(nl_procmem_open(&dom, V) == -EPERM && nl_procmem_open(&other, V) == -EPERM && nl_procmem_open(&dbg, V) == 0);
    /* a poke is a scalar store: it can never create or keep a capability */
    CHECK(nl_ptrace_poke(&dbg, V, va(A, 4, 0), 0xbeef) == 0);
    CHECK(nl_load_cap(A, va(A, 4, 0), &m) == 0 && !m.tag);
    CHECK(nl_check(A, &m, 0, 8, 0) == -EACCES);
    CHECK(nl_ptrace_poke(&dbg, V, va(A, 4, 8), d0.w[1]) == 0);              /* copy the payload bits elsewhere */
    CHECK(nl_load_cap(A, va(A, 4, 0), &m) == 0 && !m.tag);                  /* still not a capability */
    /* register writes go through the scalar path and clear the register's tag */
    CHECK(nl_ptrace_setreg(&dbg, V, 10, 0x1000) == 0);
    nl_task_frame(V, &f);
    CHECK(!(f.x[10].flags & CN_REG_TAG) && f.x[10].payload[0] == 0x1000);
    CHECK(nl_ptrace_setreg(&dom, V, 11, 1) == -EPERM);
    /* poking a page shared with another process copies it first */
    { struct nl_task *W2; struct nl_ns *B; cn_u64 pv;
      CHECK(nl_task_fork(V, 11, 2, NULL, &V2) == 0);
      V2->stopped = 1; B = V2->ns;
      CHECK(nl_ptrace_poke(&dbg, V2, va(B, 4, 64), 0x99) == 0);
      CHECK(nl_load_u64(A, va(A, 4, 64), &pv) == 0 && pv == 0x77);          /* the parent's page is untouched */
      CHECK(nl_load_u64(B, va(B, 4, 64), &pv) == 0 && pv == 0x99);
      W2 = V2; nl_task_exit(W2); }
    /* a pending signal frame's token is not exposed to the debugger */
    CHECK(nl_sig_deliver(V, 10, &u, &tk) == 0 && tk.nonce != 0);
    CHECK(nl_ptrace_getcaps(&dbg, V, &g) == 0 && g.nonce == 0 && g.depth == 0 && g.ctx_id == 0);
    CHECK(nl_ptrace_getcaps(&dom, V, &g) == -EPERM);
    V->exited = 1;
    CHECK(nl_ptrace_peek(&dbg, V, va(A, 4, 64), &v) == -ESRCH && nl_procmem_open(&dbg, V) == -ESRCH);
    V->exited = 0;
    nl_task_exit(V);
    cleanup(w);
}

/* ===== futex, robust list, clear_child_tid (I5/I6) ===== */
static void test_futex(void)
{
    struct nl_world *w = world();
    struct nl_ns *A = nl_ns_new(w, 1, 4096, 8), *B;
    struct nl_cap word = mint(A, 0, 1, 7), wr = mint(A, 1, 1, NL_RIGHT_W), stk = mint(A, 2, 1, 7);
    struct cn_frame f;
    struct nl_task *T1, *T2, *C;
    int blocked = 0;
    nl_frame_init(&f, A, 0x8000); set_cap(&f, 10, &word); set_cap(&f, 11, &wr); set_cap(&f, 12, &stk); set_cap(&f, 13, &word);
    T1 = task(w, A, 10, &f);
    CHECK(nl_thread_create(T1, 0, 12, 0, 0, 0, &T2) == 0);
    CHECK(nl_store_u64(A, va(A, 0, 0), 0) == 0);
    CHECK(nl_futex_wait(T1, 10, 0, 1, &blocked) == -EAGAIN && !blocked);   /* value changed: no sleep */
    CHECK(nl_futex_wait(T1, 11, 0, 0, &blocked) == -EACCES);                /* no read right on the word */
    CHECK(nl_futex_wait(T1, 10, 0, 0, &blocked) == 0 && blocked);
    /* the same address in a forked process is a different futex */
    CHECK(nl_task_fork(T2, 50, 2, NULL, &C) == 0);
    B = C->ns;
    CHECK(nl_futex_wake(C, 13, 0, 1) == 0);
    CHECK(nl_futex_wake(T2, 13, 0, 1) == 1);
    CHECK(nl_retire(A, &word) == 0 && nl_futex_wake(T2, 13, 0, 1) == -ESTALE);  /* a dead capability wakes nothing */
    CHECK(nl_futex_collect(T1) == 1 && nl_futex_collect(T1) == 0);
    nl_task_exit(C); (void)B;
    nl_task_exit(T2); nl_task_exit(T1);
    cleanup(w);

    /* completed waits do not accumulate */
    w = world();
    A = nl_ns_new(w, 1, 4096, 8);
    { struct nl_cap word = mint(A, 0, 1, 7);
      struct cn_frame f;
      struct nl_task *T;
      int bl;
      nl_frame_init(&f, A, 0x8000); set_cap(&f, 10, &word);
      T = task(w, A, 10, &f);
      for (unsigned i = 0; i < 200; ++i) {
          CHECK(nl_futex_wait(T, 10, 0, 0, &bl) == 0 && bl);
          CHECK(nl_futex_wake(T, 10, 0, 1) == 1 && nl_futex_collect(T) == 1);
      }
      nl_task_exit(T); }
    cleanup(w);

    /* clear_child_tid: the thread's own word is cleared and woken at exit; a dead
     * mapping is never written */
    w = world();
    A = nl_ns_new(w, 1, 4096, 8);
    {
        struct nl_cap ct = mint(A, 0, 1, 7), ct2 = mint(A, 1, 1, 7), stk = mint(A, 2, 1, 7), re;
        struct nl_task *P, *T, *T3;
        const cn_u64 S = 0x5e17e1;
        int bl;
        nl_frame_init(&f, A, 0x8000); set_cap(&f, 10, &ct); set_cap(&f, 11, &ct2); set_cap(&f, 12, &stk);
        P = task(w, A, 10, &f);
        CHECK(nl_thread_create(P, NL_CLONE_CLEARTID, 12, 0, 0, 10, &T) == 0);
        CHECK(nl_store_u64(A, va(A, 0, 0), T->tid) == 0);
        CHECK(nl_futex_wait(P, 10, 0, T->tid, &bl) == 0 && bl);
        CHECK(nl_task_exit(T) == 0);
        { cn_u64 v; CHECK(nl_load_u64(A, va(A, 0, 0), &v) == 0 && v == 0); }
        CHECK(w->futex[0].woken == 1);                                        /* the joiner was woken */
        CHECK(nl_thread_create(P, NL_CLONE_CLEARTID, 12, 0, 0, 11, &T3) == 0);
        CHECK(nl_retire(A, &ct2) == 0);                                       /* the mapping's lifetime ended ... */
        re = mint(A, 1, 1, 7);                                                /* ... and the range is reused */
        CHECK(nl_cap_slot(&re) == nl_cap_slot(&ct2) && nl_cap_gen(&re) != nl_cap_gen(&ct2));
        CHECK(nl_store_u64(A, va(A, 1, 0), S) == 0);
        CHECK(nl_task_exit(T3) == 0);
        { cn_u64 v; CHECK(nl_load_u64(A, va(A, 1, 0), &v) == 0 && v == S); }  /* the new owner's data is intact */
        nl_task_exit(P);
    }
    cleanup(w);

    /* robust list */
    w = world();
    A = nl_ns_new(w, 1, 4096, 8);
    {
        struct nl_cap head = mint(A, 0, 1, 7), e1 = mint(A, 1, 1, 7), stk = mint(A, 3, 1, 7);
        struct nl_task *P, *T;
        const cn_u64 off = 64, waiters = 0x80000000ULL;
        cn_u64 v;
        nl_frame_init(&f, A, 0x8000); set_cap(&f, 10, &head); set_cap(&f, 11, &stk); set_cap(&f, 13, &head);
        P = task(w, A, 10, &f);
        CHECK(nl_thread_create(P, 0, 11, 0, 0, 0, &T) == 0);
        /* head -> e1 -> head (end of list); e1's lock word is owned by T */
        CHECK(nl_store_cap(A, va(A, 0, 0), &e1) == 0 && nl_store_u64(A, va(A, 0, 16), off) == 0);
        CHECK(nl_store_cap(A, va(A, 1, 0), &head) == 0 && nl_store_u64(A, va(A, 1, off), T->tid | waiters) == 0);
        CHECK(nl_set_robust_list(T, 13, off) == 0);                          /* T inherited the head capability in x13 */
        w->futex[5] = (struct nl_futex_waiter){1, va(A, 1, off), 777, 0, 1};     /* a waiter on e1's lock word */
        CHECK(nl_task_exit(T) == 0);
        CHECK(nl_load_u64(A, va(A, 1, off), &v) == 0 && v == (waiters | 0x40000000ULL));                      /* tid cleared, OWNER_DIED set, WAITERS kept */
        CHECK(w->futex[5].woken == 1);
        nl_task_exit(P);
    }
    cleanup(w);

    /* robust list: forged and hostile lists */
    w = world();
    A = nl_ns_new(w, 1, 4096, 8);
    {
        struct nl_cap head = mint(A, 0, 1, 7), e1 = mint(A, 1, 1, 7), e2 = mint(A, 2, 1, 7), stk = mint(A, 3, 1, 7);
        struct nl_cap short_e = mint(A, 5, 1, 7);
        struct nl_task *P, *T;
        cn_u64 v;
        const cn_u64 off = 64;
        nl_frame_init(&f, A, 0x8000); set_cap(&f, 10, &head); set_cap(&f, 11, &stk); set_cap(&f, 13, &head);
        P = task(w, A, 10, &f);
        /* (a) the list's next pointer is a forged, untagged copy of the address of a lock word
         * that names this very thread, in a page the thread has no capability over */
        CHECK(nl_thread_create(P, 0, 11, 0, 0, 0, &T) == 0);
        CHECK(nl_store_u64(A, va(A, 6, off), T->tid) == 0);
        CHECK(nl_store_cap(A, va(A, 0, 0), &e1) == 0 && nl_store_u64(A, va(A, 0, 16), off) == 0);
        CHECK(nl_store_cap(A, va(A, 1, 0), &head) == 0);
        CHECK(nl_store_u64(A, va(A, 1, 0), va(A, 6, 0)) == 0);              /* scalar store over the tagged next: clears the tag */
        CHECK(nl_store_u64(A, va(A, 1, off), T->tid) == 0);
        CHECK(nl_set_robust_list(T, 13, off) == 0);
        CHECK(nl_task_exit(T) == 0);
        CHECK(nl_load_u64(A, va(A, 6, off), &v) == 0 && v == T->tid);       /* the victim is untouched */
        /* (b) a tagged entry whose bounds do not reach the lock word */
        CHECK(nl_thread_create(P, 0, 11, 0, 0, 0, &T) == 0);
        CHECK(nl_store_u64(A, va(A, 5, off), T->tid) == 0);
        { struct nl_cap narrow = short_e; narrow.w[1] = (narrow.w[1] & ((1ULL << 44) - 1)) | (2ULL << 44);   /* 32 bytes */
          CHECK(nl_store_cap(A, va(A, 0, 0), &narrow) == 0); }
        CHECK(nl_set_robust_list(T, 13, off) == 0);
        CHECK(nl_task_exit(T) == 0);
        CHECK(nl_load_u64(A, va(A, 5, off), &v) == 0 && v == T->tid);       /* not marked: out of bounds */
        /* (c) a cyclic list is bounded */
        CHECK(nl_thread_create(P, 0, 11, 0, 0, 0, &T) == 0);
        CHECK(nl_store_cap(A, va(A, 0, 0), &e2) == 0);
        CHECK(nl_store_cap(A, va(A, 2, 0), &e2) == 0);                      /* e2 -> e2 forever */
        CHECK(nl_store_u64(A, va(A, 2, off), 4242) == 0);
        CHECK(nl_set_robust_list(T, 13, off) == 0);
        w->robust_steps = 0;
        CHECK(nl_task_exit(T) == 0);
        CHECK(w->robust_steps <= NL_ROBUST_LIMIT && w->robust_steps > 0);
        nl_task_exit(P);
    }
    cleanup(w);
}

int main(void)
{
    test_fork();
    test_exit();
    test_signals();
    test_ptrace();
    test_futex();
    printf("PASS native lifecycle: %u checks\n", checks);
    return 0;
}
