#!/usr/bin/env python3
"""Judge the tshark safety fixtures of one boot against the pre-registered predictions.

    safety-verdict.py <serial.log> <safety-expect.txt> <arm> <fixture>...
    safety-verdict.py --self-test

The FFmpeg port's host/safety-verdict.py, carried over to this port's host and markers. What
differs:
- the sections are __TSAPP_BEGIN_FX<n>__ .. __TSAPP_END_FX<n>__ and the lines TSAPP-FIX (a fault
  ends the emulator, so a faulting section runs to the end of the log);
- the mark is the exit status on libc_test_host's `LT-RESULT tsapp_fx<n>.dom status=<n>` line.
  A negative status is the host's own failure report (HALTED, DOMAIN-ERROR, NO-DONE, ...), never
  a mark;
- no POOLFAIL (this port has no pool arms).

Each fixture's section is classified as exactly one of:

  RETURN <mark>   the LT-RESULT line for this fixture's image, with a status >= 0, and no fault
  FAULT <kind>    a capability-fault line AFTER the fixture's "TSAPP-FIX <n> touch" line, with no
                  "TSAPP-FIX <n> returned" line; kind is oob / temporal / other
  NOTHING         neither: the image did not run, or died some other way. Always a failure.

A FAULT counts only if the touch line precedes it. An oob FAULT must also name the target the
fixture printed ("TSAPP-FIX <n> target=<hex>"): a bounds fault anywhere else is a different fault.

Exit 0 only if every requested fixture matches its prediction. A fixture with no prediction for
this arm is an ERROR, as is a log with no section for a requested fixture.

--self-test runs the classifier on doctored sections, one per outcome it must be able to report,
and exits non-zero if any comes back wrong: a detector that has never fired is not a passing one.
"""
import re
import sys

OOB = re.compile(r'Cap mem access OOB:.*?addr = ([0-9a-f]+), size = (\d+), bounds = \(([0-9a-f]+), ([0-9a-f]+)\)')
TEMPORAL = re.compile(r'Cap mem access requires capability(?:.*?value = ([0-9a-f]+))?')
REVOKED = re.compile(r'Cap mem access on revoked capability:.*?imm = (-?\d+)(?:, cursor = ([0-9a-f]+))?')
UNTAGGED_OP = re.compile(r'cincoffset with an UNTAGGED rs1 -- pc=0x([0-9a-f]+) .*?val=0x([0-9a-f]+)')
HALT = re.compile(r'domain halted by capability fault: cause = (\d+), pc = 0x([0-9a-f]+)')


def done_re(n):
    return re.compile(rf'LT-RESULT tsapp_fx{n}\.dom status=(-?\d+) ')


def section(lines, n):
    begin, end = f'__TSAPP_BEGIN_FX{n}__', f'__TSAPP_END_FX{n}__'
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
        if line.strip() == f'TSAPP-FIX {n} touch' and touch is None:
            touch = i
        if f'TSAPP-FIX {n} returned' in line:
            returned = i
        m = re.search(rf'TSAPP-FIX {n} target=([0-9a-f]+)', line)
        if m:
            target = int(m.group(1), 16)
        m = re.search(rf'TSAPP-FIX {n} p cursor=\S+ bounds=\S+ len-from-cursor=(\d+)', line)
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
            elif UNTAGGED_OP.search(line):
                fault = (i, 'temporal-untagged-op', int(UNTAGGED_OP.search(line).group(2), 16), line.strip())
            elif HALT.search(line):
                fault = (i, 'other', None, line.strip())
    done = next((int(m.group(1)) for m in map(done_re(n).search, sec) if m), None)
    info = {'touch': touch, 'target': target, 'len': length}
    if fault is not None:
        i, kind, addr, text = fault
        if touch is None or i < touch:
            return ('FAULT-BEFORE-TOUCH', kind, text), info
        if returned is not None:
            return ('FAULT-AFTER-RETURN', kind, text), info
        if kind == 'oob' and addr != target:
            return ('FAULT-ELSEWHERE', f'oob at {addr:x}, target {target and f"{target:x}"}', text), info
        if kind == 'temporal-untagged-op':
            if addr != target:
                return ('FAULT-ELSEWHERE', f'untagged operand {addr:x}, target {target and f"{target:x}"}', text), info
            kind = 'temporal'
        elif kind == 'temporal':
            if target is None or addr is None:
                return ('FAULT-UNATTRIBUTED', 'temporal fault with no target or no value to compare', text), info
            if addr != target:
                return ('FAULT-ELSEWHERE', f'untagged access {addr:x}, target {target:x}', text), info
        halt = next((HALT.search(l) for l in sec[i:] if HALT.search(l)), None)
        cause = halt.group(1) if halt else '?'
        return ('FAULT', kind, f'cause {cause}: {text}'), info
    if done is not None and done >= 0:
        return ('RETURN', f'{done:x}', None), info
    if done is not None:
        return ('NOTHING', f'host reported status {done}', None), info
    return ('NOTHING', None, None), info


def judge(got, detail, info, want, n):
    verdicts = []
    for kind, val in want:
        if kind == 'RETURN':
            # a mark is 0x100000 * fixture + value: '*' accepts any value under this fixture's prefix
            base = f'{0x100000 * n:x}'
            hit = got == 'RETURN' and (val == detail or val == '*' and len(detail) == len(base)
                                       and detail.startswith(base[:len(base) - 5]))
        elif kind == 'FAULT':
            hit = got == 'FAULT' and (val == 'any' or val == detail)
        elif kind == 'LEN':
            hit = info['len'] is not None and (info['len'] > 65536 if val == 'arena' else info['len'] == int(val))
        else:
            hit = False
        verdicts.append((kind, val, hit))
    return verdicts


def self_test():
    oob = ('[CAPSTONE] Cap mem access OOB: insn = 00054503, pc = 1, pcc_base = 0, va = 0, rs1 = x10, '
           'cursor = {a}, imm = 0, addr = {a}, size = 1, bounds = (100, {a})')
    halt = '[CAPSTONE] domain halted by capability fault: cause = 5, pc = 0x1'

    def run(n, body):
        return classify(section([f'__TSAPP_BEGIN_FX{n}__'] + body, n), n)
    cases = [
        ('return', 1, ['TSAPP-FIX 1 p cursor=10 bounds=[10,50) len-from-cursor=64',
                       'LT-RESULT tsapp_fx1.dom status=1048577 rounds=9 FAIL'], ('RETURN', '100001')),
        ('fault at the target', 3, ['TSAPP-FIX 3 target=140', 'TSAPP-FIX 3 touch', oob.format(a='140'), halt],
         ('FAULT', 'oob')),
        ('fault elsewhere', 3, ['TSAPP-FIX 3 target=140', 'TSAPP-FIX 3 touch', oob.format(a='180'), halt],
         ('FAULT-ELSEWHERE', None)),
        ('fault before the touch', 3, ['TSAPP-FIX 3 target=140', oob.format(a='140'), 'TSAPP-FIX 3 touch', halt],
         ('FAULT-BEFORE-TOUCH', None)),
        ('fault after a return line', 3, ['TSAPP-FIX 3 target=140', 'TSAPP-FIX 3 touch', 'TSAPP-FIX 3 returned p[64]=00',
                                          oob.format(a='140'), halt], ('FAULT-AFTER-RETURN', None)),
        ('host failure is not a mark', 4, ['LT-RESULT tsapp_fx4.dom status=-1 rounds=3 FAIL HALTED'], ('NOTHING', None)),
        ('another image\'s result is not this one\'s', 4, ['LT-RESULT tsapp_fx5.dom status=5 rounds=3 FAIL'],
         ('NOTHING', None)),
        ('nothing at all', 4, ['TSAPP-FIX 4 begin'], ('NOTHING', None)),
        # the temporal kinds, added with the sublet arm's results (audit, 2026-09-25)
        ('temporal fault at the target', 4,
         ['TSAPP-FIX 4 target=f7000600', 'TSAPP-FIX 4 touch',
          '[CAPSTONE] Cap mem access requires capability: pc = 1, rs1 = x10, imm = 0, value = f7000600, value_hi = 0',
          halt], ('FAULT', 'temporal')),
        ('temporal fault on another value', 4,
         ['TSAPP-FIX 4 target=f7000600', 'TSAPP-FIX 4 touch',
          '[CAPSTONE] Cap mem access requires capability: pc = 1, rs1 = x10, imm = 0, value = f7000700, value_hi = 0',
          halt], ('FAULT-ELSEWHERE', None)),
        ('untagged operand at the target', 5,
         ['TSAPP-FIX 5 target=f7000600', 'TSAPP-FIX 5 touch',
          'capstone-qemu: cincoffset with an UNTAGGED rs1 -- pc=0x1 rd=x10 rs1=x10 val=0xf7000600 priv=3', halt],
         ('FAULT', 'temporal')),
    ]
    bad = 0
    for name, n, body, (want_got, want_detail) in cases:
        (got, detail, _), _info = run(n, body)
        ok = got == want_got and (want_detail is None or detail == want_detail)
        bad += not ok
        print(f'self-test {"ok  " if ok else "FAIL"} {name}: got {got} {detail}')
    # the judge: a RETURN * accepts only this fixture's prefix, LEN arena only a long capability
    checks = [
        (judge('RETURN', '300060', {'len': None}, [('RETURN', '*')], 3), True),
        (judge('RETURN', '400060', {'len': None}, [('RETURN', '*')], 3), False),
        (judge('RETURN', '100001', {'len': 64}, [('RETURN', '100001'), ('LEN', 'arena')], 1), False),
        (judge('RETURN', '100001', {'len': 64}, [('RETURN', '100001'), ('LEN', '64')], 1), True),
        (judge('FAULT', 'oob', {'len': None}, [('RETURN', '2000ee')], 2), False),
    ]
    for i, (v, want) in enumerate(checks):
        ok = all(h for _, _, h in v) == want
        bad += not ok
        print(f'self-test {"ok  " if ok else "FAIL"} judge case {i + 1}')
    sys.exit(1 if bad else 0)


def main(argv):
    if argv[1:] == ['--self-test']:
        self_test()
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
        verdicts = judge(got, detail, info, want, n)
        good = all(h for _, _, h in verdicts)
        ok = ok and good
        pred = ', '.join(f'{k} {v}' for k, v, _ in verdicts)
        shown = f'{got} {detail}' + (f'  [{text}]' if text else '')
        extra = f'  len={info["len"]}' if info['len'] is not None else ''
        print(f'fx{n}: {"AS PREDICTED" if good else "DIFFERS"}  predicted: {pred}  got: {shown}{extra}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main(sys.argv)
