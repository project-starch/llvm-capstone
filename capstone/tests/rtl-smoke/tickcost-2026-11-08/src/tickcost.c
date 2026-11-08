// tickcost: does a tick-driven hrtimer callback longer than one tick period (TICK_NSEC = 10 ms at HZ=100) livelock the
// board kernel silently? The board kernel is HZ_PERIODIC with a oneshot clockevent and no high-resolution timers, so
// tick_handle_periodic() replays every missed period in one interrupt, running tick_periodic() -> ... ->
// hrtimer_run_queues() each time; clockevents_program_event() returns -ETIME without an SBI call while the next period
// is already past. If one replay costs more than a period, the loop never ends and nothing traps except rdtime.
//
//   scanbench  (write 1)  time one full miss-scan of tables shaped exactly as cvprof's (8192 x 32 B user, 16384 x 16 B
//                         kernel), three times each, irqs off; printed as KERN_EMERG `tickcost: scan ...`
//   mode=0, spin_us=N     the callback busy-waits N microseconds (ktime_get_ns, which reads rdtime)
//   mode=1                the callback is cvprof's tick() over FULL tables (every slot used, no key can match), so
//                         every tick does the full scan cvprof does once its table is full
//   enable     (write 1/0) start/stop the hrtimer (period_ms, default 10); stopping prints calls and the longest body
// Observation only otherwise. Built like cvprof (no compressed instructions).
#include <linux/module.h>
#include <linux/hrtimer.h>
#include <linux/ktime.h>
#include <linux/sched.h>
#include <asm/irq_regs.h>
#include <linux/ptrace.h>
#include <linux/string.h>
#include <linux/spinlock.h>

#define NK 16384U
#define NU 8192U
struct kent { unsigned long pc; unsigned long n; };
struct uent { unsigned long pc; unsigned long n; char comm[TASK_COMM_LEN]; };
static struct kent kt[NK];
static struct uent ut[NU];
static unsigned long samples, ksamples, usamples, noregs, kfull, ufull, calls;
static u64 maxbody_ns;
static int period_ms = 10;
module_param(period_ms, int, 0644);
static int mode;
module_param(mode, int, 0644);
static int spin_us = 3000;
module_param(spin_us, int, 0644);
static struct hrtimer timer;
static bool running;
static DEFINE_SPINLOCK(lock);

static unsigned slot(unsigned long pc, unsigned n) { return (unsigned)((pc >> 1) * 2654435761UL) & (n - 1); }

static void fill_full(void)
{
    unsigned i;
    for (i = 0; i < NU; ++i) { ut[i].pc = 1; ut[i].n = 1; memset(ut[i].comm, 0, TASK_COMM_LEN); }
    for (i = 0; i < NK; ++i) { kt[i].pc = 1; kt[i].n = 1; }
}

/* cvprof's tick body, verbatim apart from the name */
static void cvprof_body(struct pt_regs *regs)
{
    unsigned long flags;
    spin_lock_irqsave(&lock, flags);
    ++samples;
    if (!regs) { ++noregs; goto out; }
    if (user_mode(regs)) {
        unsigned long pc = instruction_pointer(regs) & ~0xfUL;
        unsigned i = slot(pc ^ (unsigned long)current->comm[0] << 40, NU), k;
        ++usamples;
        for (k = 0; k < NU; ++k, i = (i + 1) & (NU - 1)) {
            if (ut[i].n && ut[i].pc == pc && !strncmp(ut[i].comm, current->comm, TASK_COMM_LEN)) { ++ut[i].n; goto out; }
            if (!ut[i].n) { ut[i].pc = pc; ut[i].n = 1; memcpy(ut[i].comm, current->comm, TASK_COMM_LEN); goto out; }
        }
        ++ufull;
    } else {
        unsigned long pc = instruction_pointer(regs);
        unsigned i = slot(pc, NK), k;
        ++ksamples;
        for (k = 0; k < NK; ++k, i = (i + 1) & (NK - 1)) {
            if (kt[i].n && kt[i].pc == pc) { ++kt[i].n; goto out; }
            if (!kt[i].n) { kt[i].pc = pc; kt[i].n = 1; goto out; }
        }
        ++kfull;
    }
out:
    spin_unlock_irqrestore(&lock, flags);
}

static enum hrtimer_restart tick(struct hrtimer *t)
{
    u64 t0 = ktime_get_ns(), d;
    ++calls;
    if (mode == 1) cvprof_body(get_irq_regs());
    else while (ktime_get_ns() - t0 < (u64)spin_us * 1000ULL) cpu_relax();
    d = ktime_get_ns() - t0;
    if (d > maxbody_ns) maxbody_ns = d;
    hrtimer_forward_now(t, ms_to_ktime(period_ms));
    return HRTIMER_RESTART;
}

static int set_enable(const char *v, const struct kernel_param *kp)
{
    bool on;
    if (kstrtobool(v, &on)) return -EINVAL;
    if (on && !running) {
        samples = ksamples = usamples = noregs = kfull = ufull = calls = 0; maxbody_ns = 0;
        if (mode == 1) fill_full();
        pr_emerg("tickcost: enable mode=%d spin_us=%d period_ms=%d\n", mode, spin_us, period_ms);
        hrtimer_init(&timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
        timer.function = tick;
        hrtimer_start(&timer, ms_to_ktime(period_ms), HRTIMER_MODE_REL);
        running = true;
    } else if (!on && running) {
        hrtimer_cancel(&timer);
        running = false;
        pr_emerg("tickcost: disable mode=%d calls=%lu maxbody_us=%llu samples=%lu user=%lu kernel=%lu ufull=%lu kfull=%lu\n",
                 mode, calls, maxbody_ns / 1000, samples, usamples, ksamples, ufull, kfull);
    }
    return 0;
}
static int get_enable(char *buf, const struct kernel_param *kp) { return sprintf(buf, "%d\n", running); }
static const struct kernel_param_ops enable_ops = { .set = set_enable, .get = get_enable };
module_param_cb(enable, &enable_ops, NULL, 0644);

static noinline unsigned long scan_user(unsigned long pc, const char *comm)
{
    unsigned i = slot(pc ^ (unsigned long)comm[0] << 40, NU), k;
    for (k = 0; k < NU; ++k, i = (i + 1) & (NU - 1)) {
        if (ut[i].n && ut[i].pc == pc && !strncmp(ut[i].comm, comm, TASK_COMM_LEN)) return k;
        if (!ut[i].n) return k;
    }
    return k;
}
static noinline unsigned long scan_kernel(unsigned long pc)
{
    unsigned i = slot(pc, NK), k;
    for (k = 0; k < NK; ++k, i = (i + 1) & (NK - 1)) {
        if (kt[i].n && kt[i].pc == pc) return k;
        if (!kt[i].n) return k;
    }
    return k;
}
static int set_scanbench(const char *v, const struct kernel_param *kp)
{
    unsigned long flags, k;
    u64 t0, t1;
    int r;
    if (running) return -EBUSY;
    fill_full();
    for (r = 0; r < 3; ++r) {
        local_irq_save(flags);
        t0 = ktime_get_ns(); k = scan_user(0x2aca438fa0UL, current->comm); t1 = ktime_get_ns();
        local_irq_restore(flags);
        pr_emerg("tickcost: scan user rep=%d probes=%lu ns=%llu\n", r, k, t1 - t0);
        local_irq_save(flags);
        t0 = ktime_get_ns(); k = scan_kernel(0xffffffff80001234UL); t1 = ktime_get_ns();
        local_irq_restore(flags);
        pr_emerg("tickcost: scan kernel rep=%d probes=%lu ns=%llu\n", r, k, t1 - t0);
    }
    memset(kt, 0, sizeof kt); memset(ut, 0, sizeof ut);
    return 0;
}
static const struct kernel_param_ops scan_ops = { .set = set_scanbench, .get = param_get_int };
static int scanbench;
module_param_cb(scanbench, &scan_ops, &scanbench, 0200);

static int __init tickcost_init(void) { pr_emerg("tickcost: loaded HZ=%d tick_ns=%lu\n", HZ, (unsigned long)TICK_NSEC); return 0; }
static void __exit tickcost_exit(void) { if (running) hrtimer_cancel(&timer); }
module_init(tickcost_init);
module_exit(tickcost_exit);
MODULE_LICENSE("GPL");
