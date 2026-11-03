/* Drive the real launcher service with two native workers naming the SAME
 * process transport, bypassing all domain-side locking (as a context that
 * forges its gp identity, or calls the service directly, does). The launcher
 * must never serve one trusted host object (its bounce buffer, its signal
 * ring) from two native threads at once, and a worker's detach must end its
 * thread's attachment to the process signal ring.
 *
 * Each serve waits up to a second for the other caller to enter serve too:
 * a launcher that lets both in reports the overlap deterministically, without
 * relying on a heap race crashing; one that serializes them only waits. */
#define _GNU_SOURCE
#define main vexec_main
#define capstone_delegate_serve contract_serve
#define capstone_signals_attach contract_attach
#define capstone_signals_detach contract_detach
#include EXEC_C
#undef main
#undef capstone_delegate_serve
#undef capstone_signals_attach
#undef capstone_signals_detach
#include <assert.h>
#include <stdatomic.h>

static __thread unsigned caller;
static __thread struct capstone_signal_state *attached;
static struct capstone_delegate_host *_Atomic active[2];
static atomic_int entered, overlap;
static struct capstone_delegate_host *seen[2];
static int attached_after[2], detached[2];
static int legacy;

void contract_attach(struct capstone_signal_state *s) { attached = s; }
void contract_detach(void) { attached = NULL; }
void contract_serve(struct capstone_delegate_host *h, struct capstone_delegate_entry *request)
{
    assert(request == meta);
    seen[caller] = h;
    atomic_store(&active[caller], h);
    if (atomic_load(&active[!caller]) == h) atomic_store(&overlap, 1);
    atomic_fetch_add(&entered, 1);
    for (int i = 0; i < 1000 && atomic_load(&entered) < 2; ++i) usleep(1000);
    if (atomic_load(&active[!caller]) == h) atomic_store(&overlap, 1);
    atomic_store(&active[caller], NULL);
}
static void *call(void *arg)
{
    caller = (unsigned)(uintptr_t)arg;
    struct cv_step step = {0};
    step.args[7] = legacy ? CV_SERVICE_DELEGATE : CV_SERVICE_THREAD_DELEGATE;
    step.args[0] = (uintptr_t)meta;
    step.args[1] = (uintptr_t)host.exchange;
    step.args[2] = host.exchange_bytes;
    pthread_mutex_lock(&service_lock);
    assert(virtual_service(caller + 1, &step) == 0);
    attached_after[caller] = attached == &host.signals;
    transport_detach(caller + 1);
    detached[caller] = attached == NULL && worker_host == NULL;
    pthread_mutex_unlock(&service_lock);
    return NULL;
}
int main(int argc, char **argv)
{
    legacy = argc > 1 && !strcmp(argv[1], "legacy");
    alarm(20);  /* a deadlock is a failure, never an endless test */
    meta = calloc(1, CAPSTONE_DELEGATE_META_BYTES);
    host.exchange = calloc(1, 4096);
    host.exchange_bytes = 4096;
    assert(meta && host.exchange);
    ((struct capstone_delegate_entry *)meta)->version = CAPSTONE_DELEGATE_VERSION;
    capstone_signals_init(&host.signals, (void *)((char *)meta + CAPSTONE_SIGNAL_OFFSET));
    assert(!map_reserve());
    map_insert((struct mapping){.address = meta, .bytes = CAPSTONE_DELEGATE_META_BYTES,
                               .visible = CAPSTONE_DELEGATE_META_BYTES, .id = 1});
    assert(!map_reserve());
    map_insert((struct mapping){.address = host.exchange, .bytes = 4096, .visible = 4096, .id = 2});
    pthread_t a, b;
    assert(!pthread_create(&a, NULL, call, (void *)0));
    assert(!pthread_create(&b, NULL, call, (void *)1));
    assert(!pthread_join(a, NULL) && !pthread_join(b, NULL));
    int served = seen[0] == &host && seen[1] == &host;
    /* The transport service attaches a worker to the process ring; the legacy one does not. */
    int attach_ok = legacy ? !attached_after[0] && !attached_after[1]
                           : attached_after[0] && attached_after[1];
    printf("transport-host kind=%d served=%d overlap=%d attached=%d,%d detached=%d,%d\n",
           legacy ? 0 : 10, served, atomic_load(&overlap), attached_after[0], attached_after[1],
           detached[0], detached[1]);
    return !(served && !atomic_load(&overlap) && attach_ok && detached[0] && detached[1]);
}
