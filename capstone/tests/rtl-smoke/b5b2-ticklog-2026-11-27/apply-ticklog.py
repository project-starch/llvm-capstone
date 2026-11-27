#!/usr/bin/env python3
"""apply-ticklog.py LINUX_TREE: add the tick log to drivers/clocksource/timer-riscv.c (runtime lane, 2026-11-27).

At every timer interrupt the handler reads cycle, instret and hpmcounter3 (in that order) and stores their deltas
from the previous interrupt as three u32 in a vmalloc'ed table of 2^20 records (10,485 s at HZ=100). Writing 0 to
/proc/ticklog freezes the table and snapshots the counters and the boot clock; reading it returns a 104-byte header
(magic "TICKLOG1", then u64 n, lost, c0, i0, h0, t0_ns, cf, if, hf, tf_ns, cap, rec_size, all little-endian) and n
records. Record k is the delta from interrupt k to interrupt k+1 after the first recorded one; t0_ns is the boot clock
(ktime_get_boot_fast_ns, the base of /proc/uptime) read once at the first recorded interrupt. Nothing per interrupt
reads the time CSR, which on this kernel's firmware traps to M. Idempotent: refuses a tree that already has it."""
import sys

f = sys.argv[1] + '/drivers/clocksource/timer-riscv.c'
t = open(f).read()
if 'TICKLOG1' in t:
    sys.exit('apply-ticklog: already applied to ' + f)

INC = '#include <asm/timex.h>\n'
CODE = r'''
/* tick log (runtime lane, 2026-11-27): the per-interrupt deltas of cycle, instret and hpmcounter3 */
#include <linux/proc_fs.h>
#include <linux/vmalloc.h>
#include <linux/timekeeping.h>
#include <linux/uaccess.h>
#define TICKLOG_CAP (1UL << 20)
struct ticklog_rec { u32 dc, di, dh; };
struct ticklog_hdr { char magic[8]; u64 n, lost, c0, i0, h0, t0, cf, inf, hf, tf, cap, rsz; };
static struct ticklog_rec *ticklog_buf;
static unsigned long ticklog_n, ticklog_lost;
static bool ticklog_started, ticklog_frozen;
static u64 tl_c, tl_i, tl_h;
static struct ticklog_hdr ticklog_h;

static void ticklog_record(void)
{
	u64 c = csr_read(CSR_CYCLE);
	u64 i = csr_read(CSR_INSTRET);
	u64 h = csr_read(CSR_HPMCOUNTER3);

	if (!ticklog_buf || ticklog_frozen)
		return;
	if (!ticklog_started) {
		ticklog_h.c0 = c; ticklog_h.i0 = i; ticklog_h.h0 = h;
		ticklog_h.t0 = ktime_get_boot_fast_ns();
		ticklog_started = true;
	} else if (ticklog_n < TICKLOG_CAP) {
		ticklog_buf[ticklog_n].dc = (u32)(c - tl_c);
		ticklog_buf[ticklog_n].di = (u32)(i - tl_i);
		ticklog_buf[ticklog_n].dh = (u32)(h - tl_h);
		ticklog_n++;
	} else {
		ticklog_lost++;
	}
	tl_c = c; tl_i = i; tl_h = h;
}

static ssize_t ticklog_write(struct file *file, const char __user *ubuf, size_t len, loff_t *ppos)
{
	char ch;

	if (len < 1 || get_user(ch, ubuf))
		return -EFAULT;
	if (ch != '0')
		return -EINVAL;
	if (!ticklog_frozen) {
		ticklog_frozen = true;
		barrier();
		memcpy(ticklog_h.magic, "TICKLOG1", 8);
		ticklog_h.n = ticklog_n; ticklog_h.lost = ticklog_lost;
		ticklog_h.cf = csr_read(CSR_CYCLE); ticklog_h.inf = csr_read(CSR_INSTRET);
		ticklog_h.hf = csr_read(CSR_HPMCOUNTER3); ticklog_h.tf = ktime_get_boot_fast_ns();
		ticklog_h.cap = TICKLOG_CAP; ticklog_h.rsz = sizeof(struct ticklog_rec);
		pr_info("ticklog: frozen n=%lu lost=%lu\n", ticklog_n, ticklog_lost);
	}
	return len;
}

static ssize_t ticklog_read(struct file *file, char __user *ubuf, size_t len, loff_t *ppos)
{
	size_t hs = sizeof(ticklog_h), total, done = 0, k;
	loff_t pos = *ppos;

	if (!ticklog_frozen)
		return -EBUSY;
	total = hs + ticklog_h.n * sizeof(struct ticklog_rec);
	if (pos >= total)
		return 0;
	if (len > total - pos)
		len = total - pos;
	if (pos < hs) {
		k = min(len, hs - (size_t)pos);
		if (copy_to_user(ubuf, (char *)&ticklog_h + pos, k))
			return -EFAULT;
		done = k;
	}
	if (done < len) {
		if (copy_to_user(ubuf + done, (char *)ticklog_buf + (pos + done - hs), len - done))
			return -EFAULT;
		done = len;
	}
	*ppos = pos + done;
	return done;
}

static const struct proc_ops ticklog_ops = {
	.proc_read = ticklog_read,
	.proc_write = ticklog_write,
	.proc_lseek = default_llseek,
};

static int __init ticklog_init(void)
{
	ticklog_buf = vmalloc(TICKLOG_CAP * sizeof(struct ticklog_rec));
	if (!ticklog_buf) {
		pr_err("ticklog: no table\n");
		return -ENOMEM;
	}
	proc_create("ticklog", 0600, NULL, &ticklog_ops);
	pr_info("ticklog: %lu records of %zu bytes\n", TICKLOG_CAP, sizeof(struct ticklog_rec));
	return 0;
}
late_initcall(ticklog_init);
'''

HOLD = '\tcsr_clear(CSR_IE, IE_TIE);\n\tevdev->event_handler(evdev);\n'
NEW = '\tticklog_record();\n' + HOLD
for old, new in ((INC, INC + CODE), (HOLD, NEW)):
    n = t.count(old)
    if n != 1:
        sys.exit(f'apply-ticklog: {n} matches for {old!r} in {f}')
    t = t.replace(old, new)
open(f, 'w').write(t)
print('apply-ticklog: applied to', f)
