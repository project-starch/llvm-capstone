/* Consume the native-context lane's actual executable reference backend. */
#include <capstone/native-context.h>
#include "../native-grant.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>

struct scenario { unsigned kind, fault; cn_u64 cause; };
static int validate(void *opaque, const struct cn_owner *o, const struct cn_translation *t)
{
    (void)opaque;
    return o->namespace_id == t->namespace_id && o->namespace_epoch == t->namespace_epoch &&
           o->address_space_id == t->address_space_id && t->root_pa == 4096 ? 0 : -EPERM;
}
static int step(void *opaque, struct cn_frame *f, struct cn_event *e)
{
    struct scenario *s=opaque;
    e->kind=s->kind; e->fault_class=s->fault; e->cause=s->cause;
    e->fault_pc=f->pc; e->address=0x10000;
    return 0;
}
static enum cv_fault_class classify(const struct cn_event *e)
{
#ifndef CV_BRIDGE_CAUSE_ONLY
    if (e->version != CN_ABI_VERSION || e->bytes != sizeof(*e) ||
        e->kind != CN_EVENT_PAGE_FAULT || e->fault_class != CN_FAULT_TRANSLATION)
        return CV_FAULT_DENIED;
#endif
    return cv_fault_classify(4,e->cause);
}
int main(void)
{
    struct cn_owner owner={1,2,3,4}, foreign={1,2,9,4};
    struct cn_frame initial={.version=CN_ABI_VERSION,.bytes=sizeof(initial),.pc=0x2000,
        .translation={.address_space_id=2,.namespace_id=3,.namespace_epoch=4,
                      .translation_epoch=1,.root_pa=4096}};
    struct scenario cases[]={{CN_EVENT_PAGE_FAULT,CN_FAULT_TRANSLATION,13},
        {CN_EVENT_PAGE_FAULT,CN_FAULT_TRANSLATION,15},
        {CN_EVENT_PAGE_FAULT,CN_FAULT_TRANSLATION,12},
        {CN_EVENT_CAP_FAULT,CN_FAULT_LIFETIME,13},
        {CN_EVENT_CAP_FAULT,CN_FAULT_BOUNDS,15},
        {CN_EVENT_QUANTUM,CN_FAULT_NONE,13},
        {CN_EVENT_PAGE_FAULT,CN_FAULT_TRANSLATION,25}};
    for (unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);++i) {
        struct cn_ops ops={.validate_namespace=validate,.step=step};
        struct cn_engine *engine;
        struct cn_context *ctx;
        struct cn_event event;
        cn_u64 epoch;
        assert(!cn_engine_create(CN_BACKEND_IOCTL,&ops,&cases[i],&engine));
        assert(!cn_create(engine,&owner,&initial,&ctx));
        assert(!cn_get_epoch(ctx,&owner,&epoch));
        assert(cn_run(ctx,&foreign,epoch,CN_RESUME_RETRY,0,&event)==-EPERM);
        assert(!cn_run(ctx,&owner,epoch,CN_RESUME_RETRY,0,&event));
        assert(event.restart_pc==initial.pc && event.context_epoch==epoch+1);
        assert((classify(&event)!=CV_FAULT_DENIED)==(i<3));
        cn_engine_free(engine);
    }
    puts("context-fault-peer: protected events, namespace, restart and denial PASS");
    return 0;
}
