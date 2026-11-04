/* SPDX-License-Identifier: MIT */
/* Executable specification only: no Linux scheduling or ISA implementation. */
#include <capstone/native-context.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

struct signal_frame {
    struct cn_frame frame;
    struct cn_event event;
    struct cn_signal_token ticket;
};
struct cn_context {
    struct cn_engine *engine;
    struct cn_context *next;
    struct cn_owner owner;
    struct cn_frame saved;
    struct cn_event event;
    struct signal_frame signals[CN_MAX_SIGNAL_DEPTH];
    cn_u64 id, epoch, sequence;
    unsigned depth;
    int dead;
};
struct cn_engine {
    struct cn_ops ops;
    void *opaque;
    struct cn_context *contexts, *resident;
    struct cn_frame slot; /* Exactly one execution slot, with move semantics. */
    struct cn_counters counters;
    cn_u64 next_id, next_nonce;
};

static int owner_equal(const struct cn_owner *a, const struct cn_owner *b)
{
    return a->thread_id == b->thread_id && a->address_space_id == b->address_space_id &&
           a->namespace_id == b->namespace_id && a->namespace_epoch == b->namespace_epoch;
}
static int owned(const struct cn_context *c, const struct cn_owner *actor)
{
    if (!c || !actor || !owner_equal(&c->owner, actor)) return -EPERM;
    return c->dead ? -ESTALE : 0;
}
static int epoch_ok(const struct cn_context *c, cn_u64 epoch)
{
    if (epoch != c->epoch) return -ESTALE;
    return epoch == UINT64_MAX ? -EOVERFLOW : 0;
}
static int frame_valid(struct cn_engine *e, const struct cn_owner *o, const struct cn_frame *f)
{
    struct cn_register zero = {0};
    if (!o || !f || f->version != CN_ABI_VERSION || f->bytes != sizeof(*f) ||
        (f->features & ~CN_FEATURES_V1)) return -EPROTONOSUPPORT;
    if (!o->thread_id || !o->address_space_id || !o->namespace_id || !o->namespace_epoch ||
        f->translation.address_space_id != o->address_space_id ||
        f->translation.namespace_id != o->namespace_id ||
        f->translation.namespace_epoch != o->namespace_epoch ||
        !f->translation.root_pa || !f->translation.translation_epoch) return -EINVAL;
    if (memcmp(&f->x[0], &zero, sizeof(zero)) || (f->fcsr & ~255ULL)) return -EINVAL;
    for (unsigned i = 0; i < 4; ++i) if (f->reserved[i]) return -EINVAL;
    for (unsigned i = 0; i < CN_REGISTER_COUNT + CN_CAP_CSR_COUNT; ++i) {
        const struct cn_register *r = i < CN_REGISTER_COUNT ? &f->x[i] : &f->ccsr[i-CN_REGISTER_COUNT];
        if (r->reserved || (r->flags & ~CN_REG_FLAGS) ||
            ((r->flags & CN_REG_TAG) && (r->flags & CN_REG_RAW_TAG)) ||
            ((r->flags & CN_REG_EXACT_BOUNDS) && !(f->features & CN_FEATURE_EXACT_BOUNDS)))
            return -EINVAL;
    }
    if (!(f->features & CN_FEATURE_FP)) {
        if (f->fcsr) return -EINVAL;
        for (unsigned i = 0; i < CN_REGISTER_COUNT; ++i) if (f->f[i]) return -EINVAL;
    }
    return e->ops.validate_namespace(e->opaque, o, &f->translation);
}
static struct cn_frame *frame(struct cn_context *c)
{
    return c->engine->resident == c ? &c->engine->slot : &c->saved;
}
static void move_frame(struct cn_frame *to, struct cn_frame *from)
{
    *to = *from;
    memset(from, 0, sizeof(*from));
}
static int activate(struct cn_context *c)
{
    struct cn_engine *e = c->engine;
    int rc = frame_valid(e, &c->owner, frame(c));
    if (rc) return rc;
    if (e->resident == c) return 0;
#ifdef CN_MUTATE_SKIP_NAMESPACE
    struct cn_translation previous = e->slot.translation;
#endif
    if (e->resident) {
        move_frame(&e->resident->saved, &e->slot);
        ++e->counters.evictions;
        e->counters.bytes_moved += sizeof(e->slot);
    }
    move_frame(&e->slot, &c->saved);
#ifdef CN_MUTATE_SKIP_NAMESPACE
    if (previous.namespace_id) e->slot.translation = previous;
#endif
#ifdef CN_MUTATE_DROP_TAGS
    for (unsigned i = 0; i < CN_REGISTER_COUNT; ++i) e->slot.x[i].flags &= ~CN_REG_TAG;
#endif
    e->resident = c;
    ++e->counters.restorations;
    e->counters.bytes_moved += sizeof(e->slot);
    return 0;
}
int cn_engine_create(enum cn_backend backend, const struct cn_ops *ops, void *opaque,
                     struct cn_engine **out)
{
    if (!out) return -EINVAL;
    *out = NULL;
    /* Arch-return has a distinct dispatcher and is not implemented here. */
    if (backend != CN_BACKEND_IOCTL) return -EOPNOTSUPP;
    if (!ops || !ops->validate_namespace || !ops->step ||
        (!!ops->clone_namespace != !!ops->abort_clone)) return -EINVAL;
    struct cn_engine *e = calloc(1, sizeof(*e));
    if (!e) return -ENOMEM;
    e->ops = *ops; e->opaque = opaque; e->next_id = e->next_nonce = 1;
    *out = e;
    return 0;
}
void cn_engine_free(struct cn_engine *e)
{
    if (!e) return;
    while (e->contexts) {
        struct cn_context *c = e->contexts;
        e->contexts = c->next;
        memset(c, 0, sizeof(*c)); free(c);
    }
    memset(e, 0, sizeof(*e)); free(e);
}
int cn_create(struct cn_engine *e, const struct cn_owner *owner, const struct cn_frame *initial,
              struct cn_context **out)
{
    if (!e || !out) return -EINVAL;
    *out = NULL;
    int rc = frame_valid(e, owner, initial);
    if (rc) return rc;
    if (e->next_id == UINT64_MAX) return -EOVERFLOW;
    for (struct cn_context *p = e->contexts; p; p = p->next)
        if (!p->dead && p->owner.thread_id == owner->thread_id) return -EEXIST;
    struct cn_context *c = calloc(1, sizeof(*c));
    if (!c) return -ENOMEM;
    c->engine = e; c->owner = *owner; c->saved = *initial;
    c->id = e->next_id++; c->epoch = 1;
    c->next = e->contexts; e->contexts = c; *out = c;
    return 0;
}
int cn_get_epoch(struct cn_context *c, const struct cn_owner *actor, cn_u64 *out)
{
    int rc = owned(c, actor);
    if (rc) return rc;
    if (!out) return -EINVAL;
    *out = c->epoch;
    return 0;
}
int cn_snapshot(struct cn_context *c, const struct cn_owner *actor, struct cn_frame *out)
{
    int rc = owned(c, actor);
    if (rc) return rc;
    if (!out) return -EINVAL;
    *out = *frame(c); /* Trusted audit consumer only; cannot import it as resume state. */
    return 0;
}
int cn_run(struct cn_context *c, const struct cn_owner *actor, cn_u64 epoch,
           enum cn_resume resume, cn_u64 reply, struct cn_event *out)
{
    int rc = owned(c, actor);
    if (rc) return rc;
    if (!out) return -EINVAL;
    if ((rc = epoch_ok(c, epoch))) return rc;
    if (resume != CN_RESUME_RETRY && resume != CN_RESUME_REPLY) return -EINVAL;
    if ((resume == CN_RESUME_REPLY) != (c->event.kind == CN_EVENT_SERVICE)) return -EINVAL;
    if (resume == CN_RESUME_REPLY && frame(c)->pc > UINT64_MAX - 4) return -EOVERFLOW;
    if (c->sequence == UINT64_MAX) return -EOVERFLOW;
    if (resume == CN_RESUME_REPLY && !c->engine->ops.complete_service) return -EOPNOTSUPP;
    if ((rc = activate(c))) return rc;
    if (resume == CN_RESUME_REPLY) {
        rc = c->engine->ops.complete_service(c->engine->opaque, &c->engine->slot, reply);
        if (rc) { cn_destroy(c, actor); return rc; }
    }
    struct cn_event event = {.version = CN_ABI_VERSION, .bytes = sizeof(event)};
    ++c->engine->counters.entries;
    rc = c->engine->ops.step(c->engine->opaque, &c->engine->slot, &event);
    ++c->epoch;
    if (!rc) rc = frame_valid(c->engine, &c->owner, &c->engine->slot);
    if (rc || event.version != CN_ABI_VERSION || event.bytes != sizeof(event) ||
        event.kind < CN_EVENT_QUANTUM || event.kind > CN_EVENT_REFUSED ||
        event.fault_class > CN_FAULT_BACKEND ||
        (event.kind == CN_EVENT_PAGE_FAULT && event.fault_class != CN_FAULT_TRANSLATION) ||
        (event.kind == CN_EVENT_CAP_FAULT && (event.fault_class < CN_FAULT_TAG ||
                                             event.fault_class > CN_FAULT_NAMESPACE)) ||
        (event.kind == CN_EVENT_REFUSED && event.fault_class != CN_FAULT_BACKEND) ||
        (event.kind != CN_EVENT_PAGE_FAULT && event.kind != CN_EVENT_CAP_FAULT &&
         event.kind != CN_EVENT_REFUSED && event.fault_class != CN_FAULT_NONE)) {
        cn_destroy(c, actor);
        return rc ? rc : -EPROTO;
    }
    event.context_epoch = c->epoch; event.sequence = ++c->sequence;
    /* The backend, not a scalar reply, selects the restart instruction. */
    event.restart_pc = c->engine->slot.pc;
    c->event = event; *out = event;
    if (event.kind == CN_EVENT_EXIT || event.kind == CN_EVENT_CAP_FAULT ||
        event.kind == CN_EVENT_REFUSED) cn_destroy(c, actor);
    return 0;
}
int cn_get_capref(struct cn_context *c, const struct cn_owner *actor, unsigned slot,
                  struct cn_capref *out)
{
    int rc = owned(c, actor);
    if (rc) return rc;
    if (!out || !slot || slot >= CN_REGISTER_COUNT) return -EINVAL;
    *out = (struct cn_capref){c->id, c->epoch, slot, 0};
    return 0;
}
int cn_authorize(struct cn_context *c, const struct cn_owner *actor, const struct cn_capref *ref,
                  cn_u64 offset, cn_u64 length, int write)
{
    int rc = owned(c, actor);
    if (rc) return rc;
    if (!ref || ref->reserved || !ref->frame_slot || ref->frame_slot >= CN_REGISTER_COUNT ||
        ref->context_id != c->id || ref->context_epoch != c->epoch) return -ESTALE;
    if (write != 0 && write != 1) return -EINVAL;
    if (length > UINT64_MAX - offset) return -EOVERFLOW;
    if ((rc = frame_valid(c->engine, &c->owner, frame(c)))) return rc;
    const struct cn_register *r = &frame(c)->x[ref->frame_slot];
    if (!(r->flags & CN_REG_TAG)) return -EACCES;
    if (!c->engine->ops.authorize) return -EOPNOTSUPP;
    return c->engine->ops.authorize(c->engine->opaque, &c->owner, r, offset, length, write);
}
int cn_scalar_write(struct cn_context *c, const struct cn_owner *actor, cn_u64 epoch,
                     unsigned slot, cn_u64 value)
{
    int rc = owned(c, actor);
    if (rc) return rc;
    if ((rc = epoch_ok(c, epoch))) return rc;
    if (!slot || slot >= CN_REGISTER_COUNT) return -EINVAL;
#ifndef CN_MUTATE_SCALAR_TAG
    memset(&frame(c)->x[slot], 0, sizeof(frame(c)->x[slot]));
#endif
    frame(c)->x[slot].payload[0] = value;
    ++c->epoch;
    return 0;
}
int cn_signal_push(struct cn_context *c, const struct cn_owner *actor, cn_u64 epoch,
                    const struct cn_frame *handler, struct cn_signal_token *out)
{
    int rc = owned(c, actor);
    if (rc) return rc;
    if (!out || !handler) return -EINVAL;
    if ((rc = epoch_ok(c, epoch))) return rc;
    if (c->depth == CN_MAX_SIGNAL_DEPTH) return -EOVERFLOW;
    if (c->engine->next_nonce == UINT64_MAX) return -EOVERFLOW;
    if ((rc = frame_valid(c->engine, &c->owner, handler))) return rc;
    if (memcmp(&handler->translation, &frame(c)->translation, sizeof(handler->translation)))
        return -EPERM;
    struct signal_frame *s = &c->signals[c->depth];
    move_frame(&s->frame, frame(c)); s->event = c->event;
    s->ticket = (struct cn_signal_token){c->id, c->engine->next_nonce++, ++c->depth, 0};
    *frame(c) = *handler; memset(&c->event, 0, sizeof(c->event));
    *out = s->ticket; ++c->epoch;
    return 0;
}
int cn_signal_return(struct cn_context *c, const struct cn_owner *actor, cn_u64 epoch,
                      const struct cn_signal_token *token)
{
    int rc = owned(c, actor);
    if (rc) return rc;
    if ((rc = epoch_ok(c, epoch))) return rc;
    if (!token || token->reserved || !c->depth || token->context_id != c->id ||
        token->depth != c->depth) return -EPERM;
    struct signal_frame *s = &c->signals[c->depth-1];
#ifndef CN_MUTATE_REPLAY
    if (token->nonce != s->ticket.nonce) return -EPERM;
#endif
    if ((rc = frame_valid(c->engine, &c->owner, &s->frame))) return rc;
    memset(frame(c), 0, sizeof(*frame(c)));
    move_frame(frame(c), &s->frame); c->event = s->event;
    memset(s, 0, sizeof(*s)); --c->depth; ++c->epoch;
    return 0;
}
int cn_clone(struct cn_context *c, const struct cn_owner *actor, const struct cn_owner *child,
              struct cn_context **out)
{
    int rc = owned(c, actor);
    if (rc) return rc;
    if (!out || !child) return -EINVAL;
    *out = NULL;
    if (c->depth || c->event.kind == CN_EVENT_SERVICE) return -EOPNOTSUPP;
    if (child->thread_id == actor->thread_id || child->address_space_id == actor->address_space_id ||
        child->namespace_id == actor->namespace_id) return -EPERM;
    for (struct cn_context *p = c->engine->contexts; p; p = p->next)
        if (!p->dead && (p->owner.thread_id == child->thread_id ||
                        p->owner.address_space_id == child->address_space_id ||
                        p->owner.namespace_id == child->namespace_id)) return -EEXIST;
    if (!c->engine->ops.clone_namespace) return -EOPNOTSUPP;
    if ((rc = frame_valid(c->engine, &c->owner, frame(c)))) return rc;
    struct cn_frame copy = *frame(c);
    const struct cn_translation parent_translation = copy.translation;
    rc = c->engine->ops.clone_namespace(c->engine->opaque, actor, &parent_translation,
                                        child, &copy.translation);
    if (rc) return rc;
    rc = copy.translation.root_pa == frame(c)->translation.root_pa ? -EPERM :
         cn_create(c->engine, child, &copy, out);
    if (rc) c->engine->ops.abort_clone(c->engine->opaque, child);
    memset(&copy, 0, sizeof(copy));
    return rc;
}
int cn_destroy(struct cn_context *c, const struct cn_owner *actor)
{
    int rc = owned(c, actor);
    if (rc) return rc;
    if (c->engine->resident == c) {
        memset(&c->engine->slot, 0, sizeof(c->engine->slot));
        c->engine->resident = NULL;
    }
    memset(&c->saved, 0, sizeof(c->saved)); memset(c->signals, 0, sizeof(c->signals));
    memset(&c->event, 0, sizeof(c->event)); c->depth = 0; c->dead = 1;
    return 0;
}
void cn_get_counters(const struct cn_engine *e, struct cn_counters *out)
{
    if (e && out) *out = e->counters;
}
