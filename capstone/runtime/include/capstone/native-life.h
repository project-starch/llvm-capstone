/* SPDX-License-Identifier: MIT */
#ifndef CAPSTONE_NATIVE_LIFE_H
#define CAPSTONE_NATIVE_LIFE_H

/* Native-task lifecycle layer, version 1 (lane L). An executable model of the
 * process-lifecycle contract (I5 signals/ptrace, I6 fork/exit, thread TLS,
 * futex and robust lists) on top of the context double in native-context.h.
 *
 * It is a software model: a lifetime table with generations and a free list,
 * 4 KiB pages with a per-granule tag bit, a hardware authority cache, a futex
 * table and per-task kernel state. It does not decode Capstone capabilities
 * and it does not touch Linux. Production tag authentication, page COW and
 * the arch signal path belong to the ISA, tag and kernel backends; what this
 * layer fixes is the ORDER of operations, the exclusion they need and the
 * state the kernel must keep for itself. Every operation is default off in
 * the production module: nothing here is linked into capstone_vm.ko. */
#include <capstone/native-context.h>

#define NL_PAGE_BYTES 4096U
#define NL_GRANULES 256U /* 16-byte granules per page */
#define NL_MAX_PAGES 2048U
#define NL_CACHE_WAYS 64U
#define NL_ROBUST_LIMIT 2048U
#define NL_MAX_SPACES 64U

#define NL_RIGHT_R 1U
#define NL_RIGHT_W 2U
#define NL_RIGHT_X 4U

enum nl_state { NL_FREE = 0, NL_LIVE = 1, NL_DEAD = 2 };
struct nl_record { cn_u64 gen; unsigned state, next; };
struct nl_table {
    cn_u64 root_id; /* identity of the table's storage: reused after a free */
    unsigned capacity, next_unused, free_head, free_count, live, refs;
    struct nl_record *rec;
};

/* A capability value as the model sees it: two payload words and a tag. */
struct nl_cap {
    cn_u64 w[2];
    int tag;
};
/* word 1 layout: slot 0..19, gen 20..39, rights 40..43, length 44..63 (bytes) */
cn_u64 nl_cap_slot(const struct nl_cap *);
cn_u64 nl_cap_gen(const struct nl_cap *);
unsigned nl_cap_rights(const struct nl_cap *);
cn_u64 nl_cap_length(const struct nl_cap *);

struct nl_page {
    unsigned refs;
    int may_hold_tags;
    cn_u64 w[NL_PAGE_BYTES / 8];
    cn_u64 tag[NL_GRANULES / 64];
};
struct nl_arena { cn_u64 base, pages; struct nl_cap ancestor; };

struct nl_stats {
    cn_u64 table_words_copied, page_refs_taken, cow_breaks, cow_tag_words;
    cn_u64 retire_calls, teardown_tag_clears, table_frees, hw_invalidates;
    cn_u64 page_puts, free_tag_clears, arena_walks;
};

/* I6 fork. Hooks stand in for a sibling thread that runs while the clone is
 * between steps; with the gate held they must observe -EAGAIN. */
struct nl_ns;
struct nl_fork_hooks {
    void (*after_mm)(void *, struct nl_ns *parent);
    void (*mid_table)(void *, struct nl_ns *parent);
    void *opaque;
};
struct nl_world;
struct nl_task;
struct nl_ns {
    struct nl_world *world;
    cn_u64 id, epoch;
    struct nl_table *table;     /* the srevroot; shared only by a broken fork */
    int table_shared;
    cn_u64 base_va;
    unsigned npages;
    struct nl_page *map[NL_MAX_PAGES];
    unsigned char cow[NL_MAX_PAGES];
    unsigned char priv[NL_MAX_PAGES]; /* scalar-only pages outside any grant */
    struct nl_arena *arenas;
    unsigned narenas;
    unsigned refs;              /* threads (contexts) using this namespace */
    unsigned gate;              /* fork quiesce: siblings are held out */
    unsigned inflight;          /* checked-copy chunks in progress */
    int dying, freed;
    struct nl_stats stats;
};

/* The hardware authority cache: entries name a table by root_id, so a stale
 * entry outlives its table unless it is invalidated before the free. */
struct nl_hwentry { cn_u64 root, slot, gen; int valid; };
struct nl_futex_waiter { cn_u64 ns, va, task; int woken, used; };
struct nl_world {
    struct nl_page *pool[NL_MAX_PAGES * 4]; /* freed pages, reused LIFO */
    unsigned pool_n;
    struct nl_hwentry hw[NL_CACHE_WAYS];
    struct nl_futex_waiter futex[64];
    cn_u64 next_root;
    struct nl_ns *spaces[NL_MAX_SPACES];
    struct cn_engine *engine;
    unsigned long page_allocs, page_reuses;
    unsigned long tag_clear_on_free;
    const struct nl_fork_hooks *hooks;
    cn_u64 next_tid, free_roots[16];
    unsigned nfree_roots;
    struct nl_task *all_tasks;
    cn_u64 fail_validate_ns;    /* fault injection: validate_namespace refuses this namespace */
    unsigned step_kind;         /* synthetic backend: event the next step reports */
    unsigned long robust_steps; /* instrumentation for the bounded-walk control */
};

struct nl_world *nl_world_new(void);
void nl_world_free(struct nl_world *);
struct nl_ns *nl_ns_new(struct nl_world *, cn_u64 id, unsigned capacity, unsigned npages);
struct nl_ns *nl_ns_find(struct nl_world *, cn_u64 id);

int nl_mint(struct nl_ns *, cn_u64 base, cn_u64 len, unsigned rights, struct nl_cap *out);
int nl_retire(struct nl_ns *, const struct nl_cap *);
int nl_valid(struct nl_ns *, const struct nl_cap *);   /* tag, live, via hw cache */
int nl_table_valid(const struct nl_ns *, const struct nl_cap *); /* table only */
int nl_store_cap(struct nl_ns *, cn_u64 va, const struct nl_cap *);
int nl_load_cap(struct nl_ns *, cn_u64 va, struct nl_cap *out);
int nl_store_u64(struct nl_ns *, cn_u64 va, cn_u64 v); /* rule 1: clears the tag */
int nl_load_u64(struct nl_ns *, cn_u64 va, cn_u64 *v);
int nl_arena_add(struct nl_ns *, cn_u64 base, cn_u64 pages, const struct nl_cap *ancestor);
void nl_csfence_rev(struct nl_ns *);

/* Checked access: validates tag, lifetime, bounds, rights, then moves bytes
 * through the capability. The hold form models one non-preemptible copy chunk:
 * begin validates and pins the raw page pointer, do moves, end releases. */
int nl_check(struct nl_ns *, const struct nl_cap *, cn_u64 off, cn_u64 len, int write);
struct nl_hold { struct nl_ns *ns; struct nl_page *page; unsigned off, len; int write, active; };
int nl_hold_begin(struct nl_ns *, const struct nl_cap *, cn_u64 off, cn_u64 len, int write,
                  struct nl_hold *);
void nl_hold_do(struct nl_hold *, const void *src_or_null, void *dst_or_null);
void nl_hold_end(struct nl_hold *);

int nl_ns_fork(struct nl_ns *parent, cn_u64 child_id, const struct nl_fork_hooks *,
               struct nl_ns **child);
void nl_ns_abort_fork(struct nl_ns *child);
struct nl_ns *nl_ns_get(struct nl_ns *);
/* Drop one thread reference; the last one tears the namespace down (O(1)). */
int nl_ns_put(struct nl_ns *);
int nl_ns_audit(const struct nl_ns *);            /* table invariants */
int nl_world_audit_hw(const struct nl_world *);   /* cache vs live tables */
cn_u64 nl_ns_root(const struct nl_ns *);

/* ---- tasks (a thread's kernel-held state around a cn context) ---- */
#define NL_ERESTARTSYS 512
#define NL_ERESTARTNOINTR 513
#define NL_ERESTARTNOHAND 514
#define NL_ERESTART_RESTARTBLOCK 516
enum nl_sig_policy { NL_SIG_REISSUE = 1, NL_SIG_EINTR, NL_SIG_BLOCK };
#define NL_SA_RESTART 1U
#define NL_NSIG 32U
struct nl_sigctx {
    cn_u64 magic, nonce, depth, ctx_id;
    cn_u64 pc;
    cn_u64 x[CN_REGISTER_COUNT];     /* payload word 0 of each register */
    cn_u64 tag_hint[CN_REGISTER_COUNT]; /* informational; ignored on return */
    cn_u64 sigmask;
};
#define NL_SIGCTX_MAGIC 0x4e4c534947433031ULL
struct nl_task {
    struct nl_world *world;
    struct nl_ns *ns;
    struct cn_context *ctx;
    struct cn_owner owner;
    struct cn_register handler[NL_NSIG]; unsigned handler_flags[NL_NSIG];
    struct cn_register tls, clear_tid, robust;
    int has_tls, has_clear_tid, has_robust;
    cn_u64 robust_offset;
    cn_u64 tid;
    cn_u64 mask, mask_saved[CN_MAX_SIGNAL_DEPTH];
    struct cn_signal_token tokens[CN_MAX_SIGNAL_DEPTH];
    int stopped, exited, domain;
    int uid;
    struct nl_task *next_alloc;
};
struct nl_actor { int uid; int is_domain; };

struct nl_task *nl_task_new(struct nl_world *, struct nl_ns *, cn_u64 tid, int uid,
                            const struct cn_frame *initial);
void nl_task_frame(struct nl_task *, struct cn_frame *); /* trusted view */
void nl_frame_init(struct cn_frame *, const struct nl_ns *, cn_u64 pc);
void nl_reg_from_cap(struct cn_register *, const struct nl_cap *);
void nl_cap_from_reg(struct nl_cap *, const struct cn_register *);
int nl_task_access(struct nl_task *, unsigned slot, cn_u64 off, cn_u64 len, int write);
int nl_task_copy(struct nl_task *, unsigned slot, cn_u64 off, void *buf, cn_u64 len, int write);

/* threads, TLS */
#define NL_CLONE_SETTLS 1U
#define NL_CLONE_CLEARTID 2U
int nl_thread_create(struct nl_task *parent, unsigned flags, unsigned stack_slot,
                     unsigned tls_slot, cn_u64 tls_scalar, unsigned clear_tid_slot,
                     struct nl_task **child);
/* fork */
int nl_task_fork(struct nl_task *parent, cn_u64 child_tid, cn_u64 child_ns,
                 const struct nl_fork_hooks *, struct nl_task **child);
/* exit */
int nl_task_exit(struct nl_task *);
int nl_process_kill(struct nl_ns *, struct nl_task **tasks, unsigned n);
/* futex (private only; shared futexes fail closed) */
int nl_futex_wait(struct nl_task *, unsigned slot, cn_u64 off, cn_u64 expected, int *blocked);
int nl_futex_wake(struct nl_task *, unsigned slot, cn_u64 off, unsigned count);
int nl_set_robust_list(struct nl_task *, unsigned slot, cn_u64 futex_offset);
int nl_set_clear_tid(struct nl_task *, unsigned slot);
/* signals */
int nl_sigaction(struct nl_task *, unsigned signo, unsigned handler_slot, unsigned flags);
int nl_sigaction_scalar(struct nl_task *, unsigned signo, cn_u64 address, unsigned flags);
int nl_sig_deliver(struct nl_task *, unsigned signo, struct nl_sigctx *user_out,
                   struct cn_signal_token *token);
int nl_sig_return(struct nl_task *, const struct nl_sigctx *user_in);
#define NL_ERESTARTSYS 512
#define NL_ERESTARTNOINTR 513
#define NL_ERESTARTNOHAND 514
#define NL_ERESTART_RESTARTBLOCK 516
enum nl_sig_policy nl_sig_restart_policy(long err, unsigned sa_flags, int handler_ran);
/* ptrace and /proc/PID/mem */
int nl_ptrace_may_attach(const struct nl_actor *, const struct nl_task *tracee);
int nl_ptrace_peek(const struct nl_actor *, struct nl_task *tracee, cn_u64 va, cn_u64 *out);
int nl_ptrace_poke(const struct nl_actor *, struct nl_task *tracee, cn_u64 va, cn_u64 v);
int nl_ptrace_getcaps(const struct nl_actor *, struct nl_task *tracee, struct nl_sigctx *out);
int nl_ptrace_setreg(const struct nl_actor *, struct nl_task *tracee, unsigned slot, cn_u64 v);
int nl_procmem_open(const struct nl_actor *, const struct nl_task *tracee);

/* the engine callbacks that bind a world to cn_engine_create */
extern const struct cn_ops nl_cn_ops;

#endif
