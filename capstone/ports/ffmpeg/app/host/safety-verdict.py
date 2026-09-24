#!/usr/bin/env python3
"""Judge the safety fixtures of one boot against the pre-registered predictions.

    safety-verdict.py <serial.log> <safety-expect.txt> <arm> <fixture>...

Each fixture's OWN section of the log (the whole lines __FFAPP_BEGIN_FX<n>__ ..
__FFAPP_END_FX<n>__; a fault ends the emulator, so a faulting section runs to the end of the
log) is classified as exactly one of:

  RETURN <mark>   the host's DONE line reports capstone_main = <mark>, and no fault line
  FAULT <kind>    a capability-fault line, AFTER the fixture's "FFAPP-FIX <n> touch" line,
                  with no "FFAPP-FIX <n> returned" line; kind is oob / temporal / other
  POOLFAIL <code> the pool arms' refusal line "FFAPP-POOL fail <code>", after the touch line
  NOTHING         neither -- the image did not run, or died some other way. Always a failure.

A FAULT is attributed to the touch only if the touch line precedes it. An oob FAULT must
also name the target address the fixture printed ("FFAPP-FIX <n> target=<hex>"): a bounds
fault somewhere else is a different fault, not this fixture's result.

Exit 0 only if every requested fixture matches its prediction. A fixture with no prediction
for this arm is an ERROR, as is a log with no section at all for a requested fixture.
"""
import re
import sys

OOB = re.compile(r'Cap mem access OOB:.*?addr = ([0-9a-f]+), size = (\d+), bounds = \(([0-9a-f]+), ([0-9a-f]+)\)')
TEMPORAL = re.compile(r'Cap mem access requires capability(?:.*?value = ([0-9a-f]+))?')
# A revoked capability still held in a REGISTER (never reloaded, so still tagged) faults at the
# access with this line instead; its cursor is the address the access used (imm is 0 at every
# fixture's touch). Added with the pool fixtures, before any pool run.
REVOKED = re.compile(r'Cap mem access on revoked capability:.*?imm = (-?\d+)(?:, cursor = ([0-9a-f]+))?')
# The pool arms' refusals: buffer-pool's ff2_fail, printed by the app's glue before it exits.
POOLFAIL = re.compile(r'^FFAPP-POOL fail (\d+)$')
# capstone-qemu reloads a revoked capability UNTAGGED, so a touch that does pointer arithmetic
# first faults there, before any memory access, with this line instead of a "Cap mem access"
# one. Added 2026-09-23 after the first sublet boot: fixture 4 faulted in ffapp_fix_touch at
# `cincoffset a0, a0, a1` with the freed object's address as the untagged value, and the
# classifier, not knowing the line, called it "other". Counted as temporal ONLY when the
# untagged value is the fixture's printed target -- an unrelated untagged operand is not.
UNTAGGED_OP = re.compile(r'cincoffset with an UNTAGGED rs1 -- pc=0x([0-9a-f]+) .*?val=0x([0-9a-f]+)')
HALT = re.compile(r'domain halted by capability fault: cause = (\d+), pc = 0x([0-9a-f]+)')
DONE = re.compile(r'ffapp-host: DONE, serviced \d+ request\(s\), capstone_main = (-?\d+)')


def section(lines, n):
    begin, end = f'__FFAPP_BEGIN_FX{n}__', f'__FFAPP_END_FX{n}__'
    out, cur, seen = [], False, False
    for line in lines:
        t = line.strip()
        if t == begin:
            cur, seen = True, True
            continue
        if cur and t == end:
            break
        if cur:
            out.append(line.rstrip('\r\n'))
    return out if seen else None


def classify(sec, n):
    touch = returned = None
    target = length = None
    fault = None
    for i, line in enumerate(sec):
        if line.strip() == f'FFAPP-FIX {n} touch' and touch is None:
            touch = i
        if f'FFAPP-FIX {n} returned' in line:
            returned = i
        m = re.search(rf'FFAPP-FIX {n} target=([0-9a-f]+)', line)
        if m:
            target = int(m.group(1), 16)
        m = re.search(rf'FFAPP-FIX {n} p cursor=\S+ bounds=\S+ len-from-cursor=(\d+)', line)
        if m and length is None:
            length = int(m.group(1))
        if fault is None:
            m = OOB.search(line)
            if m:
                fault = (i, 'oob', int(m.group(1), 16), line.strip())
            elif TEMPORAL.search(line):
                v = TEMPORAL.search(line).group(1)
                fault = (i, 'temporal', int(v, 16) if v else None, line.strip())
            elif REVOKED.search(line):
                m = REVOKED.search(line)
                addr = int(m.group(2), 16) + int(m.group(1)) if m.group(2) else None
                fault = (i, 'temporal', addr, line.strip())
            elif POOLFAIL.match(line.strip()):
                fault = (i, 'poolfail', int(POOLFAIL.match(line.strip()).group(1)), line.strip())
            elif UNTAGGED_OP.search(line):
                fault = (i, 'temporal-untagged-op', int(UNTAGGED_OP.search(line).group(2), 16), line.strip())
            elif HALT.search(line):
                fault = (i, 'other', None, line.strip())
    done = next((int(m.group(1)) for m in map(DONE.search, sec) if m), None)
    info = {'touch': touch, 'target': target, 'len': length}
    if fault is not None:
        i, kind, addr, text = fault
        if touch is None or i < touch:
            return ('FAULT-BEFORE-TOUCH', kind, text), info
        if kind == 'poolfail':
            # a refusal, not a fault: the pool code rejected what it was given and the program
            # exited with the code. After the touch, and with no return line, like a fault --
            # AND the domain must then have ended CLEANLY, the host reporting capstone_main equal
            # to the code. The first pool run printed the refusal and then crashed in the exit
            # path (an undefined weak hook, see ffapp_pool.h), and a refusal line alone passed.
            if returned is not None:
                return ('POOLFAIL-AFTER-RETURN', str(addr), text), info
            if any(HALT.search(l) for l in sec[i:]):
                return ('POOLFAIL-THEN-HALT', str(addr), text), info
            if done != addr:
                return ('POOLFAIL-NO-CLEAN-EXIT', str(addr), f'{text}; capstone_main = {done}'), info
            return ('POOLFAIL', str(addr), text), info
        if returned is not None:
            return ('FAULT-AFTER-RETURN', kind, text), info
        if kind == 'oob' and addr != target:
            return ('FAULT-ELSEWHERE', f'oob at {addr:x}, target {target and f"{target:x}"}', text), info
        if kind == 'temporal-untagged-op':
            if addr != target:
                return ('FAULT-ELSEWHERE', f'untagged operand {addr:x}, target {target and f"{target:x}"}', text), info
            kind = 'temporal'
        elif kind == 'temporal':
            # The value is what makes a temporal fault THIS fixture's: without a printed target,
            # or without the faulting value on the line, a NULL dereference would read the same
            # (audit, 2026-09-23: fixture 7 printed no target and a doctored `value = 0` passed).
            if target is None or addr is None:
                return ('FAULT-UNATTRIBUTED', 'temporal fault with no target or no value to compare', text), info
            if addr != target:
                return ('FAULT-ELSEWHERE', f'untagged access {addr:x}, target {target:x}', text), info
        # the halt line names the cause; carry it
        halt = next((HALT.search(l) for l in sec[i:] if HALT.search(l)), None)
        cause = halt.group(1) if halt else '?'
        return ('FAULT', kind, f'cause {cause}: {text}'), info
    if done is not None:
        return ('RETURN', f'{done:x}', None), info
    return ('NOTHING', None, None), info


def main(argv):
    if len(argv) < 5:
        sys.exit(__doc__)
    log, expect_path, arm, fixtures = argv[1], argv[2], argv[3], argv[4:]
    lines = open(log, errors='replace').readlines()
    expect = {}
    for raw in open(expect_path):
        f = raw.split('#', 1)[0].split()
        if len(f) == 4 and f[0] == arm:
            expect.setdefault(int(f[1]), []).append((f[2], f[3]))
    ok = True
    for n in map(int, fixtures):
        want = expect.get(n)
        if not want:
            print(f'fx{n}: ERROR no prediction registered for arm {arm}')
            ok = False
            continue
        sec = section(lines, n)
        if sec is None:
            print(f'fx{n}: ERROR no section in the log -- the image never started')
            ok = False
            continue
        (got, detail, text), info = classify(sec, n)
        verdicts = []
        for kind, val in want:
            if kind == 'RETURN':
                # a mark is 0x100000 * fixture + value: its hex length is that of the fixture's
                # base (6 digits for 1-15, 7 for 16 and up)
                base = f'{0x100000 * n:x}'
                hit = got == 'RETURN' and (val == '*' and detail.startswith(base[:len(base) - 5])
                                           and len(detail) == len(base) or val == detail)
            elif kind == 'FAULT':
                hit = got == 'FAULT' and (val == 'any' or val == detail)
            elif kind == 'POOLFAIL':
                hit = got == 'POOLFAIL' and val == detail
            elif kind == 'LEN':
                hit = info['len'] is not None and (info['len'] > 65536 if val == 'arena' else info['len'] == int(val))
            else:
                hit = False
            verdicts.append((kind, val, hit))
        good = all(h for _, _, h in verdicts)
        ok = ok and good
        pred = ', '.join(f'{k} {v}' for k, v, _ in verdicts)
        shown = f'{got} {detail}' + (f'  [{text}]' if text else '')
        extra = f'  len={info["len"]}' if info['len'] is not None else ''
        print(f'fx{n}: {"AS PREDICTED" if good else "DIFFERS"}  predicted: {pred}  got: {shown}{extra}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main(sys.argv)
