/* Probe: a worker that once served on the process transport exits (transport_detach under
 * service_lock) while main is blocked inside a process-transport serve (host_lock held,
 * service_lock released) waiting for data a third context must deliver through a service. */
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
static __thread int role;          /* 1 = main reader (blocks), 2 = raw context R */
static atomic_int fed, main_in_serve, r_served, detaching;
static int with_exit;
void contract_attach(struct capstone_signal_state *s) { (void)s; }
void contract_detach(void) { }
void contract_serve(struct capstone_delegate_host *h, struct capstone_delegate_entry *request)
{
    (void)h; (void)request;
    if (role == 1) {               /* read(pipe): returns once the writer has written */
        atomic_store(&main_in_serve, 1);
        while (!atomic_load(&fed)) usleep(1000);
    }
}
static void serve_meta(uint64_t tid)
{
    struct cv_step step = {0};
    step.args[7] = CV_SERVICE_THREAD_DELEGATE;
    step.args[0] = (uintptr_t)meta;
    step.args[1] = (uintptr_t)host.exchange;
    step.args[2] = host.exchange_bytes;
    pthread_mutex_lock(&service_lock);
    assert(virtual_service(tid, &step) == 0);
    pthread_mutex_unlock(&service_lock);
}
static void *reader(void *arg) { (void)arg; role = 1; serve_meta(0); return NULL; }
static void *raw_context(void *arg)
{
    (void)arg; role = 2;
    serve_meta(2);                 /* an earlier call on the process transport */
    atomic_store(&r_served, 1);
    while (!atomic_load(&main_in_serve)) usleep(1000);
    pthread_mutex_lock(&service_lock);   /* virtual_service_loop -> CV_SERVICE_THREAD_EXIT */
    atomic_store(&detaching, 1);
    transport_detach(2);
    pthread_mutex_unlock(&service_lock);
    return NULL;
}
static void *writer(void *arg)
{
    (void)arg;
    while (!atomic_load(&main_in_serve)) usleep(1000);
    /* The exiting worker publishes this while holding service_lock. The
     * writer therefore cannot run ahead of detach, even on a busy host. */
    while (with_exit && !atomic_load(&detaching)) usleep(1000);
    pthread_mutex_lock(&service_lock);   /* any service of a private-transport pthread */
    atomic_store(&fed, 1);
    pthread_mutex_unlock(&service_lock);
    return NULL;
}
int main(int argc, char **argv)
{
    with_exit = argc > 1 && !strcmp(argv[1], "exit");
    alarm(10);
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
    pthread_t r, m, w;
    if (with_exit) { assert(!pthread_create(&r, NULL, raw_context, NULL));
                     while (!atomic_load(&r_served)) usleep(1000); }
    assert(!pthread_create(&m, NULL, reader, NULL));
    assert(!pthread_create(&w, NULL, writer, NULL));
    for (int i = 0; i < 300 && !atomic_load(&fed); ++i) usleep(10000);
    printf("detach-deadlock exit=%d fed=%d\n", with_exit, atomic_load(&fed));
    fflush(stdout);
    _exit(atomic_load(&fed) ? 0 : 1);
}
