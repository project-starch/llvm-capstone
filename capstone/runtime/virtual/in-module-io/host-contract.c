/* Exercise the real launcher's module exclusion and its lock ordering.
 * The module stand-in waits for a second service before granting ownership,
 * as an inline pipe read would wait for a private worker's write. */
#define _GNU_SOURCE
#define main vexec_main
#define ioctl contract_ioctl
#define capstone_delegate_serve contract_serve
#include "../exec.c"
#undef main
#undef ioctl
#undef capstone_delegate_serve
#include <assert.h>
#include <stdarg.h>
#include <stdatomic.h>

static atomic_int waiting, fed, owned, served, released;
int contract_ioctl(int fd, unsigned long op, ...)
{
    (void)fd;
    va_list ap; va_start(ap, op); void *arg = va_arg(ap, void *); va_end(ap);
    assert(op == CV_IO_HOST);
    if (*(uint32_t *)arg) {
        atomic_store(&waiting, 1);
        while (!atomic_load(&fed)) usleep(1000);
        assert(!atomic_exchange(&owned, 1));
    } else {
        assert(atomic_exchange(&owned, 0) && atomic_load(&served));
        atomic_store(&released, 1);
    }
    return 0;
}
void contract_serve(struct capstone_delegate_host *h, struct capstone_delegate_entry *request)
{
    assert(h == &host && request == meta && atomic_load(&owned));
    atomic_store(&served, 1);
}
static void *worker(void *unused)
{
    (void)unused;
    struct cv_step step = {0};
    step.args[7] = CV_SERVICE_THREAD_DELEGATE;
    step.args[0] = (uintptr_t)meta;
    step.args[1] = (uintptr_t)host.exchange;
    step.args[2] = host.exchange_bytes;
    pthread_mutex_lock(&service_lock);
    assert(!virtual_service(1, &step));
    pthread_mutex_unlock(&service_lock);
    return NULL;
}
int main(void)
{
    alarm(10);
    meta = calloc(1, CAPSTONE_DELEGATE_META_BYTES);
    host.exchange = calloc(1, 4096); host.exchange_bytes = 4096;
    assert(meta && host.exchange);
    ((struct capstone_delegate_entry *)meta)->version = CAPSTONE_DELEGATE_VERSION;
    capstone_signals_init(&host.signals, (void *)((char *)meta + CAPSTONE_SIGNAL_OFFSET));
    assert(!map_reserve());
    map_insert((struct mapping){.address=meta, .bytes=CAPSTONE_DELEGATE_META_BYTES,
                               .visible=CAPSTONE_DELEGATE_META_BYTES, .id=1});
    assert(!map_reserve());
    map_insert((struct mapping){.address=host.exchange, .bytes=4096, .visible=4096, .id=2});
    io_registered = 1;
    pthread_t thread; assert(!pthread_create(&thread, NULL, worker, NULL));
    while (!atomic_load(&waiting)) usleep(1000);
    pthread_mutex_lock(&service_lock);
    atomic_store(&fed, 1);
    pthread_mutex_unlock(&service_lock);
    assert(!pthread_join(thread, NULL));
    assert(atomic_load(&served) && atomic_load(&released) && !atomic_load(&owned));
    puts("IOV1 host exclusion served=1 released=1 service_lock_free=1 PASS");
    return 0;
}
