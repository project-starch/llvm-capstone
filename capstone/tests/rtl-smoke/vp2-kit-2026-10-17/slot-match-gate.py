#!/usr/bin/env python3
"""slot-match-gate.py: refuse a boot whose module believes in more (or fewer) supervisor slots than the platform has.

capstone_vm's supervisor_slots parameter (default 1, capstone_vm.c) only TELLS the module; capstone-qemu's slot count is
the CPU property x-capstone-supervisor-slots (default 1, target/riscv/cpu.c). A module told 2 on a 1-slot QEMU lets an
unprivileged domain halt the guest by creating and entering a thread (the firmware path's null-capability store,
cause 24 at 0x8002522a; found 2026-10-18). vp2 has one slot.

  slot-match-gate.py --applied-log KIT/applied.log [--cpu STR] [--module-args "k=v ..."] [--params-file F]
  slot-match-gate.py --kit KIT [--module-args ...] [--params-file F]    (BEFORE the boot: the kit wrapper's append)
  slot-match-gate.py --board [--module-args ...] [--params-file F]      (the platform is vp2: 1 slot)

The QEMU side comes from the -cpu string actually used: the last line of a kit wrapper's applied.log (the evidence), or
--cpu. The module side from its insmod arguments and/or a kit params file (name=value lines). Exit 0 = equal,
1 = MISMATCH (refuse the boot), 2 = no data (no -cpu string found): never a pass."""
import argparse, re, sys
from pathlib import Path

def slots_from_args(text):
    found = re.findall(r'(?:^|[\s,])supervisor_slots=(\d+)\b', text)
    return int(found[-1]) if found else None

ap = argparse.ArgumentParser()
ap.add_argument('--applied-log', type=Path); ap.add_argument('--cpu'); ap.add_argument('--board', action='store_true')
ap.add_argument('--kit', type=Path, help='pre-boot: read what the kit wrapper appends to -cpu')
ap.add_argument('--module-args', default=''); ap.add_argument('--params-file', type=Path)
a = ap.parse_args()
module = slots_from_args(a.module_args)
if a.params_file and a.params_file.is_file():
    p = slots_from_args(' '.join(l.strip() for l in a.params_file.read_text().splitlines() if not l.startswith('#')))
    if p is not None: module = p
module_src = 'set' if module is not None else 'default'
module = 1 if module is None else module
if a.board:
    platform, where = 1, 'vp2 (one saved-context slot)'
else:
    cpu = a.cpu
    if cpu is None and a.kit:
        w = a.kit / 'qemu-system-riscv64'
        if not w.is_file(): print(f'SLOT-GATE NO-DATA: no kit wrapper at {w}; refusing'); sys.exit(2)
        appended = re.findall(r'x-capstone-supervisor-slots=(\d+)', w.read_text(errors='replace'))
        cpu = f'(kit wrapper appends) x-capstone-supervisor-slots={appended[-1]}' if appended else '(kit wrapper appends no slot count)'
    if cpu is None and a.applied_log and a.applied_log.is_file():
        lines = [l for l in a.applied_log.read_text().splitlines() if '-cpu' in l]
        cpu = lines[-1].split('-cpu', 1)[1].strip() if lines else None
    if not cpu:
        print(f'SLOT-GATE NO-DATA: no -cpu string ({a.applied_log or "--cpu"}); refusing'); sys.exit(2)
    m = re.findall(r'x-capstone-supervisor-slots=(\d+)', cpu)
    platform, where = (int(m[-1]) if m else 1), ('-cpu ' + cpu if m else '-cpu without x-capstone-supervisor-slots (QEMU default 1)')
ok = module == platform
print(f'SLOT-GATE {"OK" if ok else "MISMATCH, refuse the boot"}: module supervisor_slots={module} ({module_src}), platform {platform} [{where}]')
sys.exit(0 if ok else 1)
