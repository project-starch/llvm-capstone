#!/usr/bin/env python3
"""read-ticklog.py UART [--keep-list OUT] [--json OUT] [--icount-shift S] [--fw-fixed N] [--k-entry N]:
the tick-log boot's reader (runtime lane, 2026-11-27; version 3 records, Amendment 1).

Reads one boot's console capture and the /proc/ticklog table it carries (PREREG.md, "The reader" and Amendment 1):
- the table: the @@TICKLOG-BEGIN..END block with every foreign line removed (kernel printk "[ t]" lines and /init's
  "TRAPTEST: tick" lines, whole or spliced mid-line), whitespace dropped, base64- and gzip-decoded, and its sha256
  compared with the @@TICKLOG-SHA line. A mismatch is UNREADABLE (exit 2), never a reading;
- interrupt j's boot time is t0 + (cycles since the first recorded interrupt) / f, f calibrated from the header's
  first and freeze snapshots;
- the window: the interrupts whose boot time lies in [up0, up1] of the mcf stage's own @@SPEC-END line;
- an interval's PROGRAM instructions: instret, less the kernel's share over traps from U, less the firmware's measured
  M path of traps from U, less the fixed parts neither measures: --fw-fixed per machine-timer interrupt from U
  (fw-mpath-fixed.py on the booted firmware) and --k-entry per kernel entry from U (entry.o's prologue and epilogue);
- the controls, each PASS or FAIL; any FAIL makes the verdict line VOID;
- the reported quantities, and with --keep-list the window's F_k on program instructions, one per line, in [0, 1).
--icount-shift S divides every instruction and cycle field by 2^S: under QEMU -icount the twin's instret and cycle
CSRs count virtual nanoseconds, 2^S per instruction. Exit 0 with every control PASS, 1 if any control fails, 2 if
the table or a marker cannot be read."""
import base64, collections, gzip, hashlib, json, re, statistics, struct, sys

HDR = struct.Struct('<8s13Q64Q')
REC = struct.Struct('<11I')  # cycle, instret, hpm3; kernel share of each, K-entries; fw MTI-from-U n, instret; other n, instret
KEY_TWIN = 26_644_263_390  # mcf's protected instructions on the twin traces ic6 and ic7 (rtl, nested)
F_NOMINAL = 25_000_000      # the core clock: cycles per second, caplifive-b5b2.dts clock-frequency
HZ = 100
ROW_STAGE_S = 6601.15       # the 165.03 G row's mcf stage, up1 - up0 = 13455.23 - 6854.08


def die(code, msg):
    print('read-ticklog: ' + msg)
    sys.exit(code)


def strip_foreign(block):
    # printk lines ("[ 1234.567890] ...") and /init's heartbeat, wherever they landed, up to their own newline
    block = re.sub(r'\[\s*\d+\.\d+\][^\n]*\n?', '', block)
    block = re.sub(r'TRAPTEST: tick [0-9.]+\n?', '', block)
    return re.sub(r'\s+', '', block)


def table(log):
    m = re.search(r'@@TICKLOG-SHA ([0-9a-f]{64}) bytes=(\d+)', log)
    b = re.search(r'@@TICKLOG-BEGIN\n(.*?)@@TICKLOG-END', log, re.S)
    if not m or not b:
        die(2, 'UNREADABLE: no @@TICKLOG-SHA line or no BEGIN/END block')
    txt = strip_foreign(b[1])
    if re.search(r'[^A-Za-z0-9+/=]', txt):
        die(2, 'UNREADABLE: characters outside base64 remain after removing foreign lines: '
               + repr(re.findall(r'[^A-Za-z0-9+/=]+', txt)[:5]))
    try:
        raw = gzip.decompress(base64.b64decode(txt, validate=True))
    except Exception as e:
        die(2, f'UNREADABLE: decode failed: {e}')
    if hashlib.sha256(raw).hexdigest() != m[1] or len(raw) != int(m[2]):
        die(2, f'UNREADABLE: sha256/size {hashlib.sha256(raw).hexdigest()[:16]}/{len(raw)} != {m[1][:16]}/{m[2]}')
    h = HDR.unpack_from(raw)
    if h[0] != b'TICKLOG3':
        die(2, f'UNREADABLE: magic {h[0]!r}')
    hd = dict(zip('n lost c0 i0 h0 t0 cf inf hf tf cap rsz fw_ok'.split(), h[1:14]))
    hd['cause'] = h[14:]
    if hd['rsz'] != REC.size or len(raw) != HDR.size + hd['n'] * REC.size:
        die(2, f'UNREADABLE: record size {hd["rsz"]} or length {len(raw)} disagrees with n={hd["n"]}')
    recs = list(REC.iter_unpack(raw[HDR.size:]))
    return hd, recs, m[1]


def markers(log):
    up = lambda name: [float(x) for x in re.findall(rf'@@TICK-CTL {name} up=([0-9.]+)', log)]
    ctl = {k: up(k) for k in ('idle-begin', 'idle-end', 'loop-begin', 'loop-end')}
    if any(len(v) != 1 for v in ctl.values()):
        die(2, f'UNREADABLE: control markers {ctl}')
    s = re.findall(r'@@SPEC-END 0 run capstone-lifetimes 0 rc=(\d+) up0=([0-9.]+) up1=([0-9.]+)', log)
    if len(s) != 1:
        die(2, f'UNREADABLE: {len(s)} mcf @@SPEC-END lines')
    return {k: v[0] for k, v in ctl.items()}, int(s[0][0]), float(s[0][1]), float(s[0][2])


def window(times, lo, hi):
    """indices j of the interrupts with lo <= time_j <= hi"""
    return [j for j, t in enumerate(times) if lo <= t <= hi]


def frac_within(xs, centre, tol):
    return sum(1 for x in xs if abs(x - centre) <= tol * centre) / len(xs)


def hist(deltas, cyc=None):
    """delta/mean in 0.1 bins up to 4x plus overflow: per bin (ticks, instruction share, mean CPI)"""
    mean = sum(deltas) / len(deltas); tot = sum(deltas)
    B = collections.defaultdict(lambda: [0, 0, 0])
    for k, d in enumerate(deltas):
        b = max(0, min(int(d / mean * 10), 40))
        B[b][0] += 1; B[b][1] += d; B[b][2] += cyc[k] if cyc else 0
    rows = [(b, n, s / tot, (c / s if s and cyc else None)) for b, (n, s, c) in sorted(B.items())]
    hi = sum(d for d in deltas if d >= 1.8 * mean) / tot
    return mean, rows, hi


def synthetic_control():
    """10 % of the ticks at 2x the others: the reader must find 10 % of ticks in the 1.8x tail and their instruction
    share 2*0.1/(0.9+0.2) = 18.18 %"""
    d = [44_000] * 9000 + [88_000] * 1000
    mean, rows, hi = hist(d)
    tail_ticks = sum(1 for x in d if x >= 1.8 * mean)
    ok = abs(hi - 0.2 / 1.1) < 0.005 and tail_ticks == 1000 and sum(n for b, n, _, _ in rows if b == 18) == 1000
    return ok, hi, tail_ticks


def opt(a, name, default, kind=int):
    return kind(a[a.index(name) + 1]) if name in a else default


def main():
    a = sys.argv[1:]
    if not a:
        die(2, __doc__)
    U = 1 << opt(a, '--icount-shift', 0)
    FWF = opt(a, '--fw-fixed', None)
    KE = opt(a, '--k-entry', None)
    if FWF is None or KE is None:
        die(2, 'give --fw-fixed (fw-mpath-fixed.py on the booted firmware) and --k-entry (entry.o), see PREREG')
    log = open(a[0], 'rb').read().decode(errors='replace').replace('\r', '')
    hd, recs, sha = table(log)
    ctl, rc, up0, up1 = markers(log)

    def prog(r):
        """an interval's program instructions (Amendment 1)"""
        return (r[1] - r[4] - r[8] - r[10]) / U - FWF * r[7] - KE * r[6]

    f = (hd['cf'] - hd['c0']) / ((hd['tf'] - hd['t0']) / 1e9)
    times, c = [hd['t0'] / 1e9], 0
    for dc, *_ in recs:
        c += dc; times.append(hd['t0'] / 1e9 + c / f)
    out = {'sha256': sha, 'n': hd['n'], 'lost': hd['lost'], 'fw_ok': hd['fw_ok'], 'f_cycles_per_s': f, 'mcf_rc': rc,
           'up0': up0, 'up1': up1, 'icount_unit': U, 'fw_fixed': FWF, 'k_entry': KE}
    C = {}
    C['table not full, nothing lost'] = hd['lost'] == 0 and hd['n'] < hd['cap']
    C['firmware M-path counters on'] = hd['fw_ok'] == 1
    C['clock calibration within 0.1 % of 25 MHz'] = abs(f / F_NOMINAL - 1) < 0.001
    W = window(times, up0, up1)
    if len(W) < 2:
        die(2, f'UNREADABLE: {len(W)} interrupts in [{up0}, {up1}]')
    wr = recs[W[0]:W[-1]]
    di = [prog(r) for r in wr]
    dcy = [r[0] / U for r in wr]
    dpc = [(r[0] - r[3]) / U for r in wr]
    dall = [r[1] / U for r in wr]
    kn = [r[6] for r in wr]
    dh = [r[2] for r in wr]
    mti = [r[8] / U for r in wr if r[7] == 1]
    N_s = len(W); N_exp = (up1 - up0) * HZ
    out.update(N_s=N_s, N_expected=N_exp, stage_s=up1 - up0, stage_vs_row=(up1 - up0) / ROW_STAGE_S - 1,
               window_program=sum(di), window_instret_all=sum(dall), window_kernel=sum(r[4] for r in wr) / U,
               window_kentries=sum(kn), window_flagged=sum(1 for x in kn if x > 1),
               window_fw_mti=sum(r[7] for r in wr), window_fw_mti_instret=sum(r[8] for r in wr) / U,
               window_fw_other=sum(r[9] for r in wr), window_fw_other_instret=sum(r[10] for r in wr) / U,
               window_mti_multi=sum(1 for r in wr if r[7] > 1),
               window_hpm3=sum(dh), window_hpm3_kernel=sum(r[5] for r in wr),
               window_cycles=sum(dcy), window_cycles_program=sum(dpc),
               cM_mean=statistics.mean(mti) if mti else None, cM_sd=statistics.pstdev(mti) if mti else None,
               cM_min=min(mti) if mti else None, cM_max=max(mti) if mti else None,
               cause_counts={i: v for i, v in enumerate(hd['cause']) if v})
    C['N_s within 1 % of (up1-up0) x 100'] = abs(N_s / N_exp - 1) <= 0.01
    C['mcycle delta: >= 99 % within 1 % of 250,000'] = frac_within(dcy, F_NOMINAL // HZ, 0.01) >= 0.99
    L = window(times, ctl['loop-begin'] + 1, ctl['loop-end'] - 1)
    ld = [prog(r) for r in recs[L[0]:L[-1]]]
    lmed = statistics.median(ld)
    out.update(loop_ticks=len(ld), loop_median=lmed, loop_within3=frac_within(ld, lmed, 0.03))
    C['awk loop: >= 95 % of ticks within 3 % of its median'] = out['loop_within3'] >= 0.95
    I = window(times, ctl['idle-begin'] + 1, ctl['idle-end'] - 1)
    imed = statistics.median([prog(r) for r in recs[I[0]:I[-1]]])
    out.update(idle_ticks=len(I) - 1, idle_median=imed)
    C['idle ticks resolved: median < 0.1 x the loop median'] = imed < 0.1 * lmed
    ok, shi, stail = synthetic_control()
    out.update(synthetic_tail_share=shi, synthetic_tail_ticks=stail)
    C['synthetic log: 10 % of ticks at 2x found'] = ok
    mean, rows, hi = hist(di, dpc)
    out.update(mean_delta=mean, share_at_1_8x=hi, hist=rows)
    for k, v in C.items():
        print(f'CONTROL {"PASS" if v else "FAIL"} {k}')
    print(f'TABLE sha256 {sha} n={hd["n"]} lost={hd["lost"]} fw_ok={hd["fw_ok"]} f={f:,.0f} Hz unit 1/{U}')
    print(f'WINDOW mcf rc={rc} [{up0}, {up1}] {up1 - up0:.2f} s ({100 * out["stage_vs_row"]:+.2f} % vs the row\'s '
          f'{ROW_STAGE_S} s{"; FLAG > 2 %" if abs(out["stage_vs_row"]) > 0.02 else ""}) N_s={N_s:,} '
          f'(expected {N_exp:,.0f}) program={sum(di):,.0f} all={sum(dall):,.0f} kernel={out["window_kernel"]:,.0f} '
          f'hpm3={sum(dh):,} (kernel {out["window_hpm3_kernel"]:,}) cycles={sum(dcy):,.0f}')
    print(f'MPATH MTI from U {out["window_fw_mti"]:,} ({out["window_mti_multi"]:,} intervals with more than one), '
          f'measured {out["window_fw_mti_instret"]:,.0f} + fixed {FWF} each; c_M per tick mean {out["cM_mean"]} '
          f'sd {out["cM_sd"]} min {out["cM_min"]} max {out["cM_max"]}; other M traps from U {out["window_fw_other"]:,} '
          f'measured {out["window_fw_other_instret"]:,.0f} (their fixed part is not subtracted)')
    print(f'KENTRY {sum(kn):,} entries from U x {KE}; intervals with a non-timer trap {out["window_flagged"]:,} of '
          f'{len(kn):,}; scause counts over the whole log {out["cause_counts"]}')
    off = sum(di) - KEY_TWIN
    print(f'KEY-MATCH program {sum(di):,.0f} against the twin key {KEY_TWIN:,}: {off:+,.0f} = '
          f'{100 * off / KEY_TWIN:+.4f} %: {"MATCH" if abs(off) <= 0.001 * KEY_TWIN else "NO MATCH"} '
          f'(graded on the twin -icount run only; on silicon reported, not graded)')
    print(f'CONTROL-READINGS loop median {lmed:,.0f} within3 {100 * out["loop_within3"]:.2f} % ({len(ld)} ticks); '
          f'idle median {imed:,.0f}; synthetic tail {100 * shi:.2f} % / {stail} ticks')
    print(f'REPORTED mean delta {mean:,.0f}; share of window instructions in ticks >= 1.8x mean: {100 * hi:.3f} %')
    for b, n, s, cpi in rows:
        lab = f'>= 4.0' if b == 40 else f'{b / 10:.1f}-{(b + 1) / 10:.1f}'
        print(f'HIST {lab:>8} ticks {n:>8,} instr-share {100 * s:7.3f} % CPI {cpi:.3f}' if cpi else
              f'HIST {lab:>8} ticks {n:>8,} instr-share {100 * s:7.3f} %')
    print('VERDICT-INPUT ' + ('READY' if all(C.values()) else 'VOID (a control failed)'))
    if '--keep-list' in a:
        # F_k in [0, 1) for the N_s window ticks: the total runs to the first interrupt after up1
        if W[-1] >= len(recs):
            die(2, 'UNREADABLE: no interrupt recorded after the window')
        ps = [prog(r) for r in recs[W[0]:W[-1] + 1]]
        tot, cum = sum(ps), 0
        with open(a[a.index('--keep-list') + 1], 'w') as fo:
            for p in ps:
                fo.write(f'{cum / tot!r}\n'); cum += p
    if '--json' in a:
        json.dump({'controls': C, **out}, open(a[a.index('--json') + 1], 'w'), indent=1)
    sys.exit(0 if all(C.values()) else 1)


main()
