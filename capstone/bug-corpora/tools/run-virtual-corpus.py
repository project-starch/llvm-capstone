#!/usr/bin/env python3
"""Run a bug corpus in the VIRTUAL address space: one Linux process per case.

    run-virtual-corpus.py --plan <plan.json> --work <new dir>
                          --qemu <qemu-system-riscv64> --platform-images <dir>
                          --adapter <virtual adapter dir> [--timeout S]

WHY THIS EXISTS, and why it is not the physical runner with a flag.

The physical arms of these corpora launch one BARE-METAL domain per case
through `capstone_vm`, because a capability fault ends the domain and a faulted
domain cannot report beside itself. In the virtual address space a case is an
ordinary Linux process under `capstone-vexec`: the fault ends that process and
the guest keeps running. So the whole corpus fits in ONE boot, and the shared
QEMU slot is held once instead of once per case. That is the only reason this
runner exists; the case sources, their arguments and their oracles are the
corpus's own and are not restated here.

WHAT IT REFUSES TO DO.

  * Score a case whose BEGIN/END pair is missing. The gate brackets every case,
    so a missing bracket means the case did not run -- not that it was silent.
    Such a row is `harness`, and `harness` rows are never a measurement.

  * Treat the plan's label as evidence of what ran. Every row carries the
    sha256 of the image that produced it, and the record carries the hashes of
    the launcher, the module, the kernel, the QEMU binary and the gate script.

  * Call every trap a capability fault. Only causes 24-30 are Capstone's
    (capstone-qemu target/riscv/cpu_bits.h: UNEXP_OP_TYPE, INVALID_CAP,
    UNEXP_CAP_TYPE, INSUF_CAP_PERMS, CAP_OOB, ILLEGAL_OP_VAL,
    INSUF_RESOURCES). An illegal instruction (2), an access fault (5, 7) or a
    page fault (12, 13, 15) stops the process too, and reading one as a catch
    is how a broken image turns into twenty detections: the pymalloc arm did
    exactly that on 2026-10-07, faulting with cause 2 before any case had
    printed a character. Those rows are `trap`, never `detected`.

  * Score a fault that arrived before the case announced itself. When the plan
    names the corpus's own pre-defect marker in `defect_marker` and the fault
    happened without it, the row is `harness`: the image did not reach the
    defect, so the stop belongs to the build and not to the mechanism.

  * Call a fault a detection of the case's defect. A fault is recorded as
    `detected` with its cause; whether it is ON the defect is the separate
    `attributed` field, decided two ways and never hardcoded here:

      - the case itself published an `expect_fault_in=<name>@<address>` line
        before the defect ran, so the comparison is against an address the run
        published; or
      - the plan names the corpus's labelled probe in `expect_symbol`, and
        `--nm` resolves that symbol IN THE IMAGE THAT RAN. The image is loaded
        at an address the launcher chooses and publishes as `code=`, so the
        symbol's run-time extent is its ELF extent shifted by that address
        minus the image's own lowest PT_LOAD vaddr. Both ends come from the
        run and from the image, never from a constant here, so a relink cannot
        turn the check into a tautology -- rule 2 of the corpus contract. The
        criterion is the one the CheriBSD arms settled on: the faulting pc lies
        INSIDE the labelled probe's extent, not at its first instruction.

  * Convert an infrastructure failure into a verdict. A driver that gives up on
    its own setup exits 75 or prints CONTROL-FAILED, and that row is
    `control-failure`.

THE PLAN. A per-corpus builder writes it; this runner does not build.

    {"corpus": "capstone/bug-corpora/postgres/c-repros",   # for the record only
     "arm": "virtual",                                     # a label, see above
     "case_timeout": 120,                                  # per case, guest-side
     "stage": {"<dest in /mnt/vm>": "<host path>"},         # extra resources
     "setup": ["cp -r files /tmp/"],                        # before any case
     "env": {"NAME": "value"},                              # exported in the gate
     "cases": [{"tag": "00_...", "image": "<host path>", "argv": ["0"],
                "defect_marker": "PG_DEFECT"}]}

A case may declare `"control": true`. A control is the arm's OWN check that the
program works at all -- the port's smoke script, a fixture's fixed arm -- and it
must complete. The record FAILS when one does not, because an arm whose control
did not hold cannot report a catch, and controls are counted separately so they
never inflate a detection count.

/mnt/vm is mounted READ-ONLY, so anything a case writes to belongs under /tmp;
`setup` runs before the first case for exactly that.

A case may add `pre`, shell lines run inside its own bracket before it starts,
and `shell`, which replaces the default `capstone-vexec <image> <argv>`
invocation entirely. Both exist for the PostgreSQL SQL corpus: a stand-alone
backend needs a fresh cluster per case, so that one case cannot inherit
another's damage, refuses to run as root, and takes its statements on stdin. A
row that overrides its invocation records the line it ran, because otherwise the
record would describe something the run did not do.

A case may add `image_name` when many cases share ONE image -- an interpreter
driven by a per-case script. The image is then staged once under that name
instead of once per row, which is what keeps a 3.5 MB interpreter from being
copied twenty-three times into the gate disk.

`defect_marker` is optional: when given, a silent row says whether the case's
own marker appeared, which is the difference between "the arm saw nothing" and
"the defect never ran".
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
VIRTUAL = HERE.parents[1] / 'runtime/virtual'

# NOT anchored to a line start: a program whose last write has no newline
# leaves its own text in front of the launcher's record, and PostgreSQL's
# stand-alone backend does exactly that with its `backend> ` prompt. Anchoring
# here read one real CAP_OOB detection as an unexplained SIGSEGV on 2026-10-07.
FAULT = re.compile(r'capstone-exec: domain fault cause=(\d+) pc=(0x[0-9a-f]+) '
                   r'address=(0x[0-9a-f]+) entry=(?:0x[0-9a-f]+|unknown)'
                   r'(?: code=(0x[0-9a-f]+)-0x[0-9a-f]+)?', re.M)
EXPECT = re.compile(r'expect_fault_in=(\S+)@(0x[0-9a-f]+)')

# capstone-qemu target/riscv/cpu_bits.h, "Capstone-specific exceptions". Named
# here so a row says what stopped it; the numbers are the authority.
CAPABILITY_CAUSE = {24: 'UNEXP_OP_TYPE', 25: 'INVALID_CAP', 26: 'UNEXP_CAP_TYPE',
                    27: 'INSUF_CAP_PERMS', 28: 'CAP_OOB', 29: 'ILLEGAL_OP_VAL',
                    30: 'INSUF_RESOURCES'}
OTHER_CAUSE = {1: 'INST_ACCESS_FAULT', 2: 'ILLEGAL_INST', 3: 'BREAKPOINT',
               4: 'LOAD_ADDR_MIS', 5: 'LOAD_ACCESS_FAULT', 6: 'STORE_ADDR_MIS',
               7: 'STORE_ACCESS_FAULT', 12: 'INST_PAGE_FAULT',
               13: 'LOAD_PAGE_FAULT', 15: 'STORE_PAGE_FAULT'}


def link_base(path):
    """The lowest PT_LOAD vaddr of a little-endian ELF64 image.

    The launcher reports the run-time code base, and a symbol's run-time
    address is that base plus the symbol's offset from THIS address -- not
    from `e_entry`, which the image's marker region leaves pointing at the
    start of the first segment rather than at the entry the launcher prints.
    Read directly so this tool needs no ELF dependency in the gate path.
    """
    with open(path, 'rb') as stream:
        header = stream.read(64)
        if header[:4] != b'\x7fELF' or header[4] != 2 or header[5] != 1:
            raise ValueError(f'{path} is not a little-endian ELF64 image')
        offset = int.from_bytes(header[32:40], 'little')
        size = int.from_bytes(header[54:56], 'little')
        count = int.from_bytes(header[56:58], 'little')
        stream.seek(offset)
        table = stream.read(size * count)
    bases = [int.from_bytes(table[i * size + 16:i * size + 24], 'little')
             for i in range(count)
             if int.from_bytes(table[i * size:i * size + 4], 'little') == 1]
    if not bases:
        raise ValueError(f'{path} has no PT_LOAD segment')
    return min(bases)


def symbols(nm, path):
    """{name: (address, size)} for the image's defined symbols."""
    text = subprocess.check_output([str(nm), '--print-size', '--defined-only',
                                    str(path)], text=True)
    table = {}
    for line in text.splitlines():
        words = line.split()
        if len(words) == 4:
            table[words[3]] = (int(words[0], 16), int(words[1], 16))
        elif len(words) == 3:
            table[words[2]] = (int(words[0], 16), 0)
    return table


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def gate_script(plan):
    """The guest side. Every case is bracketed, and every case is time-bounded.

    The rootfs is not required to carry timeout(1), so the bound is a shell
    watchdog: the case runs in the background and a sleeper kills it. Without
    one bound per case, a single non-terminating case takes the whole boot with
    it and the run reports nothing at all.
    """
    limit = int(plan.get('case_timeout', 120))
    lines = ['#!/bin/sh', 'cd /mnt/vm || exit 1', 'dmesg -n 1',
             'insmod capstone_vm.ko || exit 1',
             'export CAPSTONE_EXEC_DIAGNOSTICS=1']
    for name, value in sorted(plan.get('env', {}).items()):
        lines.append(f"export {name}='{value}'")
    lines.extend(plan.get('setup', []))
    lines += ['limited() {',
              '  "$@" > /tmp/case.out 2>&1 &',
              '  pid=$!',
              f'  ( sleep {limit}; kill -9 $pid 2>/dev/null ) &',
              '  killer=$!',
              '  wait $pid; st=$?',
              '  kill -9 $killer 2>/dev/null',
              '  wait $killer 2>/dev/null',
              '  cat /tmp/case.out',
              '  return $st',
              '}']
    for case in plan['cases']:
        tag = case['tag']
        argv = ' '.join(f"'{a}'" for a in case.get('argv', []))
        image = case.get('image_name', f'{tag}.dom')
        invocation = case.get('shell') or \
            f'./capstone-vexec ./images/{image} {argv}'.rstrip()
        lines += [f'echo CASE_BEGIN:{tag}', *case.get('pre', []),
                  f'limited {invocation}', f'echo CASE_END:{tag}:$?']
    lines += ['rmmod capstone_vm', 'echo CORPUS_CLEANUP:$?',
              'echo VIRTUAL_STAGED_DONE', '']
    return '\n'.join(lines)


def score(case, text, status, table=None, base=None):
    """(verdict, detail, extras). Controls first, then the mechanism."""
    extras = {}
    if status == 137:
        return 'timeout', 'killed after the per-case limit; not a measurement', extras
    if 'CONTROL-FAILED' in text:
        line = next(l for l in text.splitlines() if 'CONTROL-FAILED' in l)
        return 'control-failure', f'the case refused its own setup: {line.strip()}', extras
    if status == 75:
        return 'control-failure', 'exit 75: an infrastructure failure, never a verdict', extras

    fault = FAULT.search(text)
    if fault:
        cause, pc, address, code_base = fault.groups()
        cause = int(cause)
        name = CAPABILITY_CAUSE.get(cause) or OTHER_CAUSE.get(cause, 'cause %d' % cause)
        extras.update(cause=cause, cause_name=name, pc=pc, address=address)
        marker = case.get('defect_marker')
        if marker and marker not in text:
            return 'harness', (f'faulted with cause {cause} ({name}) before the case '
                               f'printed its own {marker} marker: the image did not '
                               f'reach the defect'), extras
        expect = EXPECT.search(text)
        if expect:
            extras['expect_fault_in'] = expect.group(1)
            extras['expect_address'] = expect.group(2)
            # The case published the address; the launcher published the pc.
            # The published address names a function, not an instruction, and
            # the fault may legitimately land in a callee (a libc memcpy), so
            # this is reported as an offset and never used to deny a fault.
            extras['pc_offset'] = int(pc, 16) - int(expect.group(2), 16)
        wanted = case.get('expect_symbol')
        wanted = [wanted] if isinstance(wanted, str) else wanted
        if wanted and table is not None and base is not None and code_base:
            # The launcher chose the load address and reported it as `code=`.
            shift = int(code_base, 16) - base
            known = {name: table[name] for name in wanted if name in table}
            if not known:
                extras['attribution'] = ('the image defines none of ' +
                                         ', '.join(wanted) + ': a static probe '
                                         'cannot be resolved from the image, so '
                                         'attribution is not established here')
            else:
                hit = [name for name, (start, size) in known.items()
                       if shift + start <= int(pc, 16) < shift + start + max(size, 1)]
                extras.update(
                    expect_symbol=sorted(known),
                    symbol_range={name: [hex(shift + start), hex(shift + start + size)]
                                  for name, (start, size) in sorted(known.items())},
                    attributed=bool(hit))
                if hit:
                    extras['attributed_to'] = hit[0]
        if cause not in CAPABILITY_CAUSE:
            return 'trap', (f'stopped by cause {cause} ({name}) at pc={pc}, which is '
                            f'not a Capstone capability cause'), extras
        return 'detected', f'capability fault cause={cause} ({name}) pc={pc}', extras

    if status == 139:
        return 'trap', 'terminated by SIGSEGV with no fault line', extras
    marker = case.get('defect_marker')
    if marker and marker not in text:
        # NOT a silence. A silence is a measurement -- the defect ran and the
        # mechanism said nothing -- and a case whose own marker never appeared
        # has not shown that it ran. Scoring it `silent` is what let six fts3
        # rows and six PostgreSQL rows read as measured misses on 2026-10-08
        # when the first had no reachability proof and the second could not
        # create their extensions. Rule 4 of the contract: an infrastructure
        # failure is not a measurement.
        return 'unexecuted', f'completed with status {status}, but the case\'s own ' \
                             f'{marker} marker never appeared: nothing shows the ' \
                             f'defect ran', extras
    return 'silent', f'completed with status {status}', extras


def verdicts(plan, log, images, tables, bases):
    """One row per planned case, from a transcript. Shared by run and rescore."""
    rows = []
    for case in plan['cases']:
        tag = case['tag']
        bracket = re.search(r'CASE_BEGIN:%s\n(.*?)CASE_END:%s:(\d+)\n'
                            % (re.escape(tag), re.escape(tag)), log, re.S)
        if not bracket:
            verdict, detail, extras, status = (
                'harness', 'no BEGIN/END bracket: the case did not run', {}, None)
        else:
            status = int(bracket.group(2))
            verdict, detail, extras = score(case, bracket.group(1), status,
                                            tables.get(tag), bases.get(tag))
        row = dict(tag=tag, control=bool(case.get('control')), verdict=verdict,
                   detail=detail, status=status, image_sha256=images.get(tag), **extras)
        if case.get('shell'):
            row['shell'] = case['shell']
        rows.append(row)
        label = 'CONTROL ' if case.get('control') else ''
        print(f'{label}{tag:<56} {verdict:<16} {detail[:70]}', flush=True)
    return rows


def summary(plan, rows, log, completed, extra):
    controls = [r for r in rows if r['control']]
    held = [r for r in controls if r['verdict'] == 'silent' and r['status'] == 0]
    # `harness` and `timeout` are explicitly not measurements -- one case did
    # not run, the other was cut off -- so a corpus carrying either is not
    # fully measured and its record says FAIL rather than quietly averaging
    # the missing row away.
    measured = [r for r in rows
                if not r['control']
                and r['verdict'] not in ('harness', 'timeout', 'unexecuted')]
    counts = {}
    for row in rows:
        if not row['control']:
            counts[row['verdict']] = counts.get(row['verdict'], 0) + 1
    good = (completed and len(held) == len(controls)
            and len(measured) == len(rows) - len(controls))
    return dict(status='PASS' if good else 'FAIL', corpus=plan.get('corpus'),
                arm=plan.get('arm'), address_space='virtual', completed=completed,
                cases=len(rows) - len(controls), measured=len(measured),
                controls=len(controls), controls_held=len(held),
                verdicts=counts, rows=rows, **extra)


def rescore(a, plan):
    """Score an existing run again, from its own serial log."""
    previous = json.loads((a.work / 'result.json').read_text())
    log = (a.work / 'guest/serial.log').read_text(errors='replace').replace('\r', '')
    images = {row['tag']: row.get('image_sha256') for row in previous['rows']}
    tables, bases = {}, {}
    for case in plan['cases']:
        if a.nm and case.get('expect_symbol'):
            image = Path(case['image'])
            if image.is_file():
                tables[case['tag']] = symbols(a.nm, image)
                bases[case['tag']] = link_base(image)
    rows = verdicts(plan, log, images, tables, bases)
    record = summary(plan, rows, log, previous['completed'], dict(
        rescored_from=previous['status'],
        gate_sha256=previous['gate_sha256'],
        staged_sha256=previous.get('staged_sha256', {}),
        platform_sha256=previous['platform_sha256']))
    (a.work / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'rows'}, indent=2))
    return 0 if record['status'] == 'PASS' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True, help='New output directory')
    p.add_argument('--qemu', type=Path, required=True)
    p.add_argument('--platform-images', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=1800, help='Whole-boot bound')
    p.add_argument('--disk-mib', type=int, default=1024)
    # The processor profile the runtime's allocator needs, ON by default because
    # without it EVERY case faults before its first line and the run still looks
    # like a result. It cost a whole pass on 2026-10-08: the musl mallocng
    # policy in runtime/virtual/heap-musl.c requires the opt-in exact-bound
    # capabilities of the pinned QEMU (x-capstone-exact-bounds=true), and with
    # the option absent every image stopped at `cincoffsetimm with an UNTAGGED
    # rs1 ... val=0x0` -- which this runner scored as `detected` for the defect
    # arms and `harness` elsewhere, so fifteen corpora reported numbers that
    # measured nothing. A FIXED arm that faults is the tell, and it is why the
    # controls are counted separately from the verdicts.
    p.add_argument('--exact-bounds', action='store_true', default=True,
                   help='Enable the native mallocng processor profile (default)')
    p.add_argument('--no-exact-bounds', dest='exact_bounds', action='store_false',
                   help='For a QEMU that does not know the option')
    p.add_argument('--nm', type=Path, help='llvm-nm, to attribute a fault to a '
                                           'case\'s labelled probe symbol')
    p.add_argument('--rescore', action='store_true',
                   help='Score an existing --work directory again from its own '
                        'serial log, without booting. For a correction to the '
                        'parser or the verdict rules: the observation does not '
                        'change, so re-running the guest would only risk '
                        'changing it. The rewritten record keeps the original '
                        'run\'s input hashes and says it was rescored.')
    a = p.parse_args()

    plan = json.loads(a.plan.read_text())
    if not plan.get('cases'):
        sys.exit('the plan selects no case; an empty corpus run is a harness failure')
    if a.rescore:
        return rescore(a, plan)
    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    (stage / 'images').mkdir(parents=True)
    shutil.copy2(a.adapter / 'capstone-vexec', stage / 'capstone-vexec')
    shutil.copy2(a.adapter / 'module/capstone_vm.ko', stage / 'capstone_vm.ko')
    staged = {}
    for dest, source in sorted(plan.get('stage', {}).items()):
        target = stage / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        if Path(source).is_dir():
            shutil.copytree(source, target)
            # A resource a case reads is an input: hash each file, so a record
            # names the scripts and fixtures that produced it and not only the
            # images. A directory's own name says nothing about its contents.
            for file in sorted(p for p in target.rglob('*') if p.is_file()):
                staged[str(file.relative_to(stage))] = sha256(file)
        else:
            shutil.copy2(source, target)
            staged[dest] = sha256(target)

    images, tables, bases = {}, {}, {}
    for case in plan['cases']:
        image = Path(case['image'])
        if not image.is_file():
            sys.exit(f"the plan names an image that does not exist: {image}")
        destination = stage / 'images' / case.get('image_name', f"{case['tag']}.dom")
        if not destination.exists():
            shutil.copy2(image, destination)
        images[case['tag']] = sha256(image)
        if a.nm and case.get('expect_symbol'):
            tables[case['tag']] = symbols(a.nm, image)
            bases[case['tag']] = link_base(image)

    script = gate_script(plan)
    (stage / 'gate.sh').write_text(script)
    run = subprocess.run([sys.executable, str(VIRTUAL / 'run-staged.py'),
                          '--qemu', str(a.qemu.resolve()),
                          '--images', str(a.platform_images.resolve()),
                          '--stage', str(stage), '--work', str(a.work / 'guest'),
                          '--disk-mib', str(a.disk_mib), '--timeout', str(a.timeout)]
                         + (['--exact-bounds'] if a.exact_bounds else []))
    log = (a.work / 'guest/serial.log').read_text(errors='replace').replace('\r', '')

    rows = verdicts(plan, log, images, tables, bases)
    completed = run.returncode == 0 and '\nCORPUS_CLEANUP:0\n' in log
    record = summary(plan, rows, log, completed, dict(
        gate_sha256=hashlib.sha256(script.encode()).hexdigest(),
        staged_sha256=staged,
        platform_sha256={name: sha256(path) for name, path in dict(
            qemu=a.qemu, launcher=a.adapter / 'capstone-vexec',
            module=a.adapter / 'module/capstone_vm.ko',
            Image=a.platform_images / 'Image').items()}))
    (a.work / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'rows'}, indent=2))
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
