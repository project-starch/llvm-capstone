#!/usr/bin/env python3
"""apply-ticklog.py LINUX_TREE: the tick log, version 3 (runtime lane, 2026-11-27, Amendment 1).

Two parts, both refused if already present:
- arch/riscv/kernel/entry.S: every trap taken from U stamps cycle, instret and hpmcounter3 in ticklog_acct once its
  registers are saved (_save_common, SPP = 0), and every return to U adds the time since that stamp to three
  accumulators and counts the entry (resume_userspace, after the guard and the sscratch write, the point both
  restore_all and restore_user pass). Only t0-t2 are used there; both points run with interrupts off.
- drivers/clocksource/timer-riscv.c: at every timer interrupt the handler reads cycle, instret and hpmcounter3 (in
  that order) and the kernel's own share K(now) = accumulated + (now - the open trap's stamp, if one is open), and
  stores seven u32 deltas from the previous interrupt: cycle, instret, hpm3, K-cycle, K-instret, K-hpm3, K-entries,
  in a vmalloc'ed table of 2^20 records (10,485 s at HZ=100). A program interval's own instructions are
  instret - K-instret: every S trap from U (the tick itself, system calls, page faults) is excluded.
- the firmware's M-path counters (sbi-tick-apply.py part 2): at boot the kernel names a zeroed page to the firmware
  with one ecall (EID 0x0A544B4C); every record adds the deltas of its four counters (machine-timer interrupts from U
  and their instret, other M traps from U and their instret), 11 u32 in all. A firmware that does not answer
  0x544B leaves them 0 and the header's fw_ok 0.
- entry.S also counts every trap from U by scause (32 exception and 32 interrupt codes), into the header at the
  freeze.
Writing 0 to /proc/ticklog freezes the table and snapshots the counters and the boot clock; reading it returns a
624-byte header (magic "TICKLOG3", then u64 n, lost, c0, i0, h0, t0_ns, cf, if, hf, tf_ns, cap, rec_size, fw_ok and
the 64 scause counts, all little-endian) and n records. Record k is the delta from interrupt k to interrupt k+1 after the first recorded one;
t0_ns is the boot clock (ktime_get_boot_fast_ns, the base of /proc/uptime) read once at the first recorded
interrupt. Nothing per interrupt reads the time CSR, which on this kernel's firmware traps to M."""
import sys

root = sys.argv[1]


def patch(rel, pairs, mark):
    f = f'{root}/{rel}'
    t = open(f).read()
    if mark in t:
        sys.exit(f'apply-ticklog: already applied to {f}')
    for old, new in pairs:
        n = t.count(old)
        if n != 1:
            sys.exit(f'apply-ticklog: {n} matches for {old[:60]!r} in {f}')
        t = t.replace(old, new)
    open(f, 'w').write(t)
    print('apply-ticklog: applied to', f)


# entry.S. ticklog_acct: 0 stamp cycle, 8 stamp instret, 16 stamp hpm3, 24 open, 32 K-cycle, 40 K-instret,
# 48 K-hpm3, 56 K-entries
ENTRY_OLD = '\tREG_S s4, PT_CAUSE(sp)\n#ifndef CONFIG_RISCV_CAPSTONE_TRAP\n\tREG_S s5, PT_TP(sp)\n'
ENTRY_NEW = '''\tREG_S s4, PT_CAUSE(sp)
\t/* tick log: a trap from U opens the kernel's share (ticklog_acct, timer-riscv.c) */
\tandi t0, s1, SR_SPP
\tbnez t0, 1f
\tla t1, ticklog_acct
\tcsrr t0, CSR_CYCLE
\tREG_S t0, 0(t1)
\tcsrr t0, CSR_INSTRET
\tREG_S t0, 8(t1)
\tcsrr t0, CSR_HPMCOUNTER3
\tREG_S t0, 16(t1)
\tli t0, 1
\tREG_S t0, 24(t1)
\tandi t0, s4, 31
\tbgez s4, 79f
\taddi t0, t0, 32
79:
\tslli t0, t0, 3
\tla t1, ticklog_cause
\tadd t1, t1, t0
\tREG_L t0, 0(t1)
\taddi t0, t0, 1
\tREG_S t0, 0(t1)
1:
#ifndef CONFIG_RISCV_CAPSTONE_TRAP
\tREG_S s5, PT_TP(sp)
'''
EXIT_OLD = '\tcsrw CSR_SCRATCH, tp\n\n#ifdef CONFIG_RISCV_CAPSTONE_TRAP\n\tREG_L s0, TASK_TI_FLAGS(tp)\n'
EXIT_NEW = '''\tcsrw CSR_SCRATCH, tp

\t/* tick log: the return to U closes the kernel's share */
\tla t1, ticklog_acct
\tREG_L t0, 24(t1)
\tbeqz t0, 78f
\tcsrr t0, CSR_CYCLE
\tREG_L t2, 0(t1)
\tsub t0, t0, t2
\tREG_L t2, 32(t1)
\tadd t2, t2, t0
\tREG_S t2, 32(t1)
\tcsrr t0, CSR_INSTRET
\tREG_L t2, 8(t1)
\tsub t0, t0, t2
\tREG_L t2, 40(t1)
\tadd t2, t2, t0
\tREG_S t2, 40(t1)
\tcsrr t0, CSR_HPMCOUNTER3
\tREG_L t2, 16(t1)
\tsub t0, t0, t2
\tREG_L t2, 48(t1)
\tadd t2, t2, t0
\tREG_S t2, 48(t1)
\tREG_L t2, 56(t1)
\taddi t2, t2, 1
\tREG_S t2, 56(t1)
\tREG_S zero, 24(t1)
78:

#ifdef CONFIG_RISCV_CAPSTONE_TRAP
\tREG_L s0, TASK_TI_FLAGS(tp)
'''

INC = '#include <asm/timex.h>\n'
CODE = r'''
/* tick log (runtime lane, 2026-11-27, version 2): per-interrupt deltas of cycle, instret, hpmcounter3 and of the
 * kernel's share of them over traps from U (entry.S writes ticklog_acct) */
#include <linux/proc_fs.h>
#include <linux/vmalloc.h>
#include <linux/timekeeping.h>
#include <linux/uaccess.h>
#define TICKLOG_CAP (1UL << 20)
struct ticklog_rec { u32 dc, di, dh, kc, ki, kh, kn, mn, mi, on, oi; };
struct ticklog_hdr { char magic[8]; u64 n, lost, c0, i0, h0, t0, cf, inf, hf, tf, cap, rsz, fw_ok, cause[64]; };
u64 ticklog_cause[64];
static volatile u64 *ticklog_fw;  /* the firmware's M-path counters: MTI from U n, instret; other from U n, instret */
struct ticklog_acct_s { u64 sc, si, sh, open, kc, ki, kh, kn; };
struct ticklog_acct_s ticklog_acct;
static struct ticklog_rec *ticklog_buf;
static unsigned long ticklog_n, ticklog_lost;
static bool ticklog_started, ticklog_frozen;
static u64 tl_c, tl_i, tl_h, tl_kc, tl_ki, tl_kh, tl_kn, tl_f[4];
static struct ticklog_hdr ticklog_h;

static void ticklog_record(void)
{
	u64 c = csr_read(CSR_CYCLE);
	u64 i = csr_read(CSR_INSTRET);
	u64 h = csr_read(CSR_HPMCOUNTER3);
	u64 kc = ticklog_acct.kc, ki = ticklog_acct.ki, kh = ticklog_acct.kh, kn = ticklog_acct.kn;

	if (ticklog_acct.open) {
		kc += c - ticklog_acct.sc;
		ki += i - ticklog_acct.si;
		kh += h - ticklog_acct.sh;
	}
	if (!ticklog_buf || !ticklog_fw || ticklog_frozen)
		return;
	if (!ticklog_started) {
		ticklog_h.c0 = c; ticklog_h.i0 = i; ticklog_h.h0 = h;
		ticklog_h.t0 = ktime_get_boot_fast_ns();
		ticklog_started = true;
	} else if (ticklog_n < TICKLOG_CAP) {
		struct ticklog_rec *r = &ticklog_buf[ticklog_n];

		r->dc = (u32)(c - tl_c); r->di = (u32)(i - tl_i); r->dh = (u32)(h - tl_h);
		r->kc = (u32)(kc - tl_kc); r->ki = (u32)(ki - tl_ki); r->kh = (u32)(kh - tl_kh);
		r->kn = (u32)(kn - tl_kn);
		r->mn = (u32)(ticklog_fw[0] - tl_f[0]); r->mi = (u32)(ticklog_fw[1] - tl_f[1]);
		r->on = (u32)(ticklog_fw[2] - tl_f[2]); r->oi = (u32)(ticklog_fw[3] - tl_f[3]);
		ticklog_n++;
	} else {
		ticklog_lost++;
	}
	tl_c = c; tl_i = i; tl_h = h; tl_kc = kc; tl_ki = ki; tl_kh = kh; tl_kn = kn;
	tl_f[0] = ticklog_fw[0]; tl_f[1] = ticklog_fw[1]; tl_f[2] = ticklog_fw[2]; tl_f[3] = ticklog_fw[3];
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
		memcpy(ticklog_h.magic, "TICKLOG3", 8);
		memcpy(ticklog_h.cause, ticklog_cause, sizeof(ticklog_cause));
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
	struct sbiret ret;

	ticklog_fw = (volatile u64 *)get_zeroed_page(GFP_KERNEL);
	if (!ticklog_fw) {
		pr_err("ticklog: no firmware page\n");
		return -ENOMEM;
	}
	ret = sbi_ecall(0x0A544B4C, 0, __pa(ticklog_fw), 0, 0, 0, 0, 0);
	ticklog_h.fw_ok = !ret.error && ret.value == 0x544B;
	pr_info("ticklog: firmware M-path counters %s (error %ld value %#lx)\n", ticklog_h.fw_ok ? "on" : "absent",
		ret.error, ret.value);
	ticklog_buf = vmalloc(TICKLOG_CAP * sizeof(struct ticklog_rec));
	if (!ticklog_buf) {
		pr_err("ticklog: no table\n");
		return -ENOMEM;
	}
	proc_create("ticklog", 0600, NULL, &ticklog_ops);
	pr_info("ticklog: %lu records of %zu bytes (version 3)\n", TICKLOG_CAP, sizeof(struct ticklog_rec));
	return 0;
}
late_initcall(ticklog_init);
'''
HOLD = '\tcsr_clear(CSR_IE, IE_TIE);\n\tevdev->event_handler(evdev);\n'

patch('arch/riscv/kernel/entry.S', [(ENTRY_OLD, ENTRY_NEW), (EXIT_OLD, EXIT_NEW)], 'ticklog_acct')
patch('drivers/clocksource/timer-riscv.c', [(INC, INC + CODE), (HOLD, '\tticklog_record();\n' + HOLD)], 'TICKLOG')
