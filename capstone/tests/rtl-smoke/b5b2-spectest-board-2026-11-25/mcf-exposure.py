"""mcf-exposure.py LABEL:BOARD:SEAL:UART_TXT ... : per run, mcf's exposure on silicon (board lane, 2026-11-25), for the rtl
lane's per-unit-time wedge rate. Entry = the kernel time of mcf_r_base's "enters" line; the end is the first of:
  watchdog  "TRAPTEST: watchdog killed ... mcf_r ... after Ns" -> entry + N (the stage's own bound, which started at
            "stage ... mcf_r begin", a few seconds before entry: the exposure is the time mcf was running);
  done      the stage's "@@QSTEP-END ... rc=" with a later tick/kernel time -> the last kernel time before it;
  fault     a "sublet: fault" line of the mcf pid -> its kernel time;
  silence   none of these: the capture ends mid-mcf -> the last kernel-timestamped or tick time in the capture (the wedge
            happened at or after it).
Prints one row per run and the pooled exposure. A run with no mcf entry is an ERROR row, never a zero.
2026-11-26: the entry, watchdog, fault and QSTEP-END lines are matched anywhere in a line (the console splices printk into program
output); and for "done" the end is the up1 of the stage's last "@@SPEC-END <n> run" line before QSTEP-END (mcf's own end, on the
uptime clock), falling back to the last kernel/tick time before QSTEP-END. The code took the last time in the whole rest of
the capture, which adds the validation, the stage's dmesg dump and later ticks."""
import re, sys
rows, pool, ev = [], 0.0, 0
for a in sys.argv[1:]:
    lab, brd, seal, p = a.split(':', 3)
    t = open(p, errors='replace').read().replace('\r', '')
    m = re.search(r'\[ *([0-9.]+)\] sublet: mcf_r_base\.nati\[(\d+)\] enters', t)
    if not m: print(f'ERROR {lab}: no mcf entry in {p}'); continue
    t0, pid = float(m.group(1)), m.group(2); rest = t[m.end():]
    wd = re.search(r'TRAPTEST: watchdog killed [^\n]*mcf_r[^\n]* after (\d+)s', rest)
    fl = re.search(rf'\[ *([0-9.]+)\] sublet: fault (cause=\d+) [^\n]*pid={pid}\b', rest)
    if fl: end, how = float(fl.group(1)), f'fault {fl.group(2)}'
    elif wd: end, how = t0 + int(wd.group(1)), f'watchdog at {wd.group(1)} s (no wedge)'
    else:
        qe = re.search(r'@@QSTEP-END \S*mcf_r rc=(\d+)', rest); pre = rest[:qe.start()] if qe else rest
        ts = [float(x) for x in re.findall(r'^\[ *([0-9.]+)\] ', pre, re.M)] + [float(x) for x in re.findall(r'^TRAPTEST: tick ([0-9.]+)$', pre, re.M)]
        ru = [float(x) for x in re.findall(r'@@SPEC-END \d+ run \S+ \d+ rc=\d+ up0=[0-9.]+ up1=([0-9.]+)', pre)]
        if qe and ru: end, how = ru[-1], f'stage ended rc={qe.group(1)}; end = up1 of its last @@SPEC-END run line'
        elif qe: end, how = (max(ts) if ts else t0), f'stage ended rc={qe.group(1)}; end = the last kernel/tick time before QSTEP-END'
        else: end, how = (max(ts) if ts else t0), 'SILENCE (a wedge at or after the last kernel time)'
    x = end - t0; pool += x; ev += how.startswith('SILENCE')
    rows.append((lab, brd, seal, t0, end, x, how))
for r in rows: print(f'{r[0]:22s} board {r[1]} {r[2]:10s} mcf entry {r[3]:9.2f} s  end {r[4]:9.2f} s  exposure {r[5]:8.1f} s  {r[6]}')
print(f'pooled: {pool:,.1f} s of mcf, {ev} wedge(s)')
