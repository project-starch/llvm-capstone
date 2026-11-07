// cvprof: a tick-driven PC sampler for the board kernel (no CONFIG_PERF_EVENTS, no high-resolution timers).
// An hrtimer fires each period_ms from the timer interrupt; it reads the interrupted pc from get_irq_regs() and counts
// it in one of two open-addressing tables: kernel pcs, and user pcs with the task's comm. Write 1 to `dump` to print
// the totals and the hottest entries (KERN_EMERG, must-fire). Observation only: nothing else is touched.
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
static unsigned long samples, ksamples, usamples, noregs, kfull, ufull;
static int period_ms = 10;
module_param(period_ms, int, 0444);
static struct hrtimer timer;
static bool running;
static DEFINE_SPINLOCK(lock);

static unsigned slot(unsigned long pc, unsigned n) { return (unsigned)((pc >> 1) * 2654435761UL) & (n - 1); }

static enum hrtimer_restart tick(struct hrtimer *t)
{
    struct pt_regs *regs = get_irq_regs();
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
    hrtimer_forward_now(t, ms_to_ktime(period_ms));
    return HRTIMER_RESTART;
}

static int set_enable(const char *v, const struct kernel_param *kp)
{
    bool on;
    if (kstrtobool(v, &on)) return -EINVAL;
    if (on && !running) {
        unsigned long flags;
        spin_lock_irqsave(&lock, flags);
        memset(kt, 0, sizeof kt); memset(ut, 0, sizeof ut);
        samples = ksamples = usamples = noregs = kfull = ufull = 0;
        spin_unlock_irqrestore(&lock, flags);
        hrtimer_init(&timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
        timer.function = tick;
        hrtimer_start(&timer, ms_to_ktime(period_ms), HRTIMER_MODE_REL);
        running = true;
    } else if (!on && running) {
        hrtimer_cancel(&timer);
        running = false;
    }
    return 0;
}
static int get_enable(char *buf, const struct kernel_param *kp) { return sprintf(buf, "%d\n", running); }
static const struct kernel_param_ops enable_ops = { .set = set_enable, .get = get_enable };
module_param_cb(enable, &enable_ops, NULL, 0644);

static int top = 150;
module_param(top, int, 0644);
static int set_dump(const char *v, const struct kernel_param *kp)
{
    static bool kdone[NK], udone[NU];
    int r;
    if (running) { hrtimer_cancel(&timer); running = false; }   /* the tables are quiet from here: no lock while printing */
    pr_emerg("cvprof: total samples=%lu kernel=%lu user=%lu noregs=%lu kfull=%lu ufull=%lu period_ms=%d\n",
             samples, ksamples, usamples, noregs, kfull, ufull, period_ms);
    memset(kdone, 0, sizeof kdone); memset(udone, 0, sizeof udone);
    for (r = 0; r < top; ++r) {
        unsigned i, best = NK; unsigned long bn = 0;
        for (i = 0; i < NK; ++i) if (!kdone[i] && kt[i].n > bn) { bn = kt[i].n; best = i; }
        if (best == NK) break;
        kdone[best] = true;
        pr_emerg("cvprof: k %lu %px %pS\n", bn, (void *)kt[best].pc, (void *)kt[best].pc);
    }
    for (r = 0; r < top; ++r) {
        unsigned i, best = NU; unsigned long bn = 0;
        for (i = 0; i < NU; ++i) if (!udone[i] && ut[i].n > bn) { bn = ut[i].n; best = i; }
        if (best == NU) break;
        udone[best] = true;
        pr_emerg("cvprof: u %lu %lx %s\n", bn, ut[best].pc, ut[best].comm);
    }
    pr_emerg("cvprof: end\n");
    return 0;
}
static const struct kernel_param_ops dump_ops = { .set = set_dump, .get = param_get_int };
static int dump;
module_param_cb(dump, &dump_ops, &dump, 0200);

static int __init cvprof_init(void) { pr_emerg("cvprof: loaded period_ms=%d\n", period_ms); return 0; }
module_init(cvprof_init);
MODULE_LICENSE("GPL");
