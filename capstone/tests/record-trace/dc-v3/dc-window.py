#!/usr/bin/env python3
"""dc-window.py BOOTLOG SNAP DCPLUGIN (nested lane, 2026-11-27): the D$ model's totals over each SPEC run window.
SNAP has one line per store to the UART transmit register, carrying the model's totals at that store; the k-th line
is the k-th console byte. The console is boot.log after boot.py's own first line ('# <command>'). A window runs from
the byte that ends '@@SPEC-BEGIN ...' (its newline) to the first byte of the matching '@@SPEC-END ...'.
Alignment check: the transmit-register stores must equal the console's characters to within D (printed); each
boundary is then uncertain by D bytes, and the model's change over D bytes either side is printed as the error.
Exits 1 when there is no window, no snap line, or D is above 64 (no data is an error, never a zero)."""
import re, sys
log, snapf, dcp = sys.argv[1:4]
txt = open(log, encoding='utf-8', errors='replace', newline='').read()
first = txt.index('\n') + 1
# The host pty maps each guest '\n' to '\r\n' (onlcr), so the log has one '\r' per newline that the UART never sent:
# the guest's '\r\n' reads '\r\r\n'. Undo it before counting bytes.
con = txt[first:].replace('\r\n', '\n')
S = []
for l in open(snapf):
    f = l.split()
    if f[0] != 'u':
        continue
    S.append((int(f[1]), int(f[3]), [int(x) for x in f[5:9]]))
if not S:
    print('NO-SNAP', snapf); sys.exit(1)
fin = S[-1]; S = S[:-1]
thr = fin[0]
D = abs(thr - len(con))
print('console_chars %d thr_writes %d D %d' % (len(con), thr, D))
for l in open(dcp):
    if l.startswith(('window', 'src', 'insns', 'iostore')):
        print('dcplugin', l.strip())
def at(k):
    k = max(0, min(k, len(S) - 1)); return S[k]
wins = 0
for mb in re.finditer(r'@@SPEC-BEGIN ([^\r\n]*)\r*\n', con):
    tag = mb.group(1)
    me = re.compile(r'@@SPEC-END ' + re.escape(tag)).search(con, mb.end())
    if not me:
        print('window', tag, 'NO-END'); continue
    kb, ke = mb.end() - 1, me.start()
    b, e = at(kb), at(ke)
    reads = e[1] - b[1]; miss = [e[2][i] - b[2][i] for i in range(4)]
    err_r = max(at(kb + D)[1] - at(kb - D)[1], 0) + max(at(ke + D)[1] - at(ke - D)[1], 0)
    err_m = [max(at(kb + D)[2][i] - at(kb - D)[2][i], 0) + max(at(ke + D)[2][i] - at(ke - D)[2][i], 0) for i in range(4)]
    print('window %s | start byte %d (end of "@@SPEC-BEGIN %s") end byte %d (start of "@@SPEC-END %s")' % (tag, kb, tag, ke, tag))
    print('  reads %d misses %s miss_fraction %s' % (reads, ' '.join(map(str, miss)), ' '.join('%.6f' % (x / reads if reads else -1) for x in miss)))
    print('  alignment error at +-%d bytes: reads %d misses %s' % (D, err_r, ' '.join(map(str, err_m))))
    wins += 1
if wins == 0 or D > 64:
    print('WINDOW-FAIL wins %d D %d' % (wins, D)); sys.exit(1)
print('WINDOW-OK', wins)
