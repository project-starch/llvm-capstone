#!/usr/bin/env python3
"""ST100's pair, judged against its pre-registration (lane 816de2675b21): arm by arm, run-scoped.
  arm 1 k800 retval=4 (the control; a failed control VOIDS the boot);
  arm 2 native size 100: oracle '23674002 573a4409', SPEEDTEST1-CYCLES (the counter §7p's ratio used) and BASELINE-WARM;
  arm 3 k800 retval=4 again (§7p's composition);
  arm 4 domain size 100: oracle hash, arena 128 MiB, SPEEDTEST1-CYCLES ... DROPPED 0 RC 0, no share-trap / NOMEM.
NARMS=3 judges the size-1 pre-boot's composition (k800, native, domain) for the negative test.
Prints the readings and the ratio against the predictions (native ~337.2 G cycles, domain ~398.6 G, ratio ~1.18, band
1.14-1.24). Exit 0: every arm read and the ratio is inside the band; 1: a reading off the pre-registration; 2: no data."""
import re, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from fpga_driver.transcript import read, scope_to_run, arm_segments
import os
ORA = os.environ.get('ST_ORA', '23674002 573a4409'); NARMS = int(os.environ.get('NARMS', '4'))
arms = arm_segments(scope_to_run(read(sys.argv[1])))
if not arms:
    sys.exit(f"NO DATA: no arm segments in {sys.argv[1]}")
print(f"arms read: {len(arms)}")
ok = {f'{NARMS} arms': len(arms) == NARMS}
nat_i, dom_i = 1, NARMS - 1
r = {}
if len(arms) >= 1:
    k = arms[0]
    ok['k800 returned 4'] = k.returned is True and bool(re.search(r'RESULT k800 retval=4\b', k.framed + k.uart))
if NARMS == 4 and len(arms) >= 3:
    ok['k800 (second) returned 4'] = arms[2].returned is True and bool(re.search(r'RESULT k800 retval=4\b', arms[2].framed + arms[2].uart))
if len(arms) > nat_i:
    n = arms[nat_i]
    ok['native returned'] = n.returned is True
    ok['native oracle hash'] = f'Verification Hash: {ORA}' in n.uart
    m = re.search(r'SPEEDTEST1-CYCLES (\d+) HIGHWATER \S+ HEAP 134217728 DROPPED 0 RC 0', n.uart)
    ok['native SPEEDTEST1-CYCLES ... DROPPED 0 RC 0'] = bool(m)
    if m: r['native_cycles'] = int(m.group(1))
    m = re.search(r'BASELINE-WARM CYCLES (\d+) INSTRS (\d+)', n.uart)
    ok['native BASELINE-WARM'] = bool(m)
    if m: r['native_warm_cycles'], r['native_instrs'] = int(m.group(1)), int(m.group(2))
    m = re.search(r'TOTAL\.+\s+([\d.]+)s', n.uart); r['native_total_s'] = m.group(1) if m else None
if len(arms) > dom_i:
    d = arms[dom_i]
    ok['domain returned'] = d.returned is True
    ok['domain oracle hash'] = f'Verification Hash: {ORA}' in d.uart
    ok['domain arena 128 MiB'] = 'arena_bytes=134217728' in d.uart
    m = re.search(r'SPEEDTEST1-CYCLES (\d+) HIGHWATER \S+ HEAP 134217728 DROPPED 0 RC 0', d.uart)
    ok['domain HEAP 128 MiB DROPPED 0 RC 0'] = bool(m)
    if m: r['domain_cycles'] = int(m.group(1))
    ok['domain no share-trap / NOMEM'] = 'share-trap' not in d.uart and 'NOMEM' not in d.uart
    m = re.search(r'TOTAL\.+\s+([\d.]+)s', d.uart); r['domain_total_s'] = m.group(1) if m else None
for c, v in ok.items(): print(f"  {c:36} {'ok' if v else 'FAIL'}")
for kk, v in r.items(): print(f"  {kk:20} {v}")
if 'native_cycles' in r and 'domain_cycles' in r:
    ratio = r['domain_cycles'] / r['native_cycles']
    inband = 1.14 <= ratio <= 1.24
    print(f"  ratio {ratio:.4f} (predicted ~1.18, band 1.14-1.24: {'INSIDE' if inband else 'OUTSIDE'})")
    print(f"  native {r['native_cycles']/1e9:.2f} G cycles vs ~337.2 G predicted; domain {r['domain_cycles']/1e9:.2f} G vs ~398.6 G")
    ok['ratio in band'] = inband
sys.exit(0 if all(ok.values()) else 1)
