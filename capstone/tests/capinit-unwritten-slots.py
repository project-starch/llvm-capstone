#!/usr/bin/env python3
"""Audit capability-initializer store coverage in Capstone ELF objects/archives.

Usage: capinit-unwritten-slots.py FILE... (tools from $CAPSTONE_LLVM_BIN).

Every allocated data R_Capstone_64 relocation is a candidate pointer slot. Track
addresses through the straight-line __capstone_cap_init, including stack spills,
and require an stc to that exact slot. A reference to the containing object, as
either a source or destination, does not establish another slot's coverage.

This checks store coverage, not the stored capability's tag, bounds or authority.
Integer address constants are candidates too and need separate review. Excluded:
the loader's .gct and the runtime's link-address init/fini arrays.
Exit 0: all candidates covered; 1: uncovered candidates; 2: incomplete analysis
or invalid input. Unsupported instructions/control flow cannot yield a clean scan.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

R_64, R_PCREL_HI20, R_PCREL_LO12_I = 2, 23, 24
SHT_SYMTAB, SHT_RELA = 2, 4
SHF_ALLOC, SHF_EXEC = 2, 4
INSN = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
MEM = re.compile(r"(-?(?:0x[0-9a-f]+|\d+))\((\w+)\)$")


def tool(name):
    return os.path.join(os.environ.get('CAPSTONE_LLVM_BIN', ''), name)


def parse(blob):
    if blob[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<HH', blob, 16) != (1, 0x103):
        raise ValueError('expected a little-endian ELF64 Capstone relocatable object')
    shoff, = struct.unpack_from('<Q', blob, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', blob, 0x3a)
    secs = [struct.unpack_from('<IIQQQQIIQQ', blob, shoff + i * shentsize) for i in range(shnum)]

    def name(strtab, off):
        s = secs[strtab]
        b = blob[s[4] + off:s[4] + s[5]]
        return b[:b.index(b'\0')].decode()

    names = [name(shstrndx, s[0]) for s in secs]
    st = next(s for s in secs if s[1] == SHT_SYMTAB)
    syms = []
    for off in range(st[4], st[4] + st[5], 24):
        n, info, other, shndx, value, size = struct.unpack_from('<IBBHQQ', blob, off)
        syms.append((name(st[6], n), info & 0xf, shndx, value, size))
    relas = []
    for s in secs:
        if s[1] == SHT_RELA:
            for off in range(s[4], s[4] + s[5], 24):
                r_off, info, addend = struct.unpack_from('<QQq', blob, off)
                relas.append((s[7], r_off, info & 0xffffffff, info >> 32, addend))
    return secs, names, syms, relas


@dataclass(frozen=True)
class Address:
    section: object
    offset: int


@dataclass(frozen=True)
class High:
    location: Address


def add(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return a + b
    if isinstance(b, Address) and isinstance(a, int):
        a, b = b, a
    if isinstance(a, Address) and isinstance(b, int):
        return Address(a.section, a.offset + b)
    return None


def disassemble(path):
    out = subprocess.run([tool('llvm-objdump'), '-d', '--no-show-raw-insn', str(path)],
                         check=True, capture_output=True, text=True).stdout
    section, code = None, {}
    for line in out.splitlines():
        if line.startswith('Disassembly of section '):
            section = line[len('Disassembly of section '):-1]
            code.setdefault(section, [])
        m = INSN.match(line)
        if section is not None and m:
            pc, op, args = m.groups()
            code[section].append((int(pc, 16), op,
                                  [x.strip() for x in args.split('#')[0].split(',')]))
    return code


def stores(path, names, syms, relas):
    """Exact stc destinations; reject an initializer we cannot model."""
    inits = [(sh, v, v + sz) for n, t, sh, v, sz in syms if n == '__capstone_cap_init']
    if not inits:
        return set()
    code = disassemble(path)
    reloc = {(sh, off): (typ, sym, delta) for sh, off, typ, sym, delta in relas
             if typ in (R_PCREL_HI20, R_PCREL_LO12_I)}
    covered = set()

    def symbol(sym, delta):
        _, _, sh, value, _ = syms[sym]
        return Address(sh if sh else f'extern:{sym}', value + delta)

    for sh, start, end in inits:
        regs = {'zero': 0, 'sp': Address('stack', 0), 'gp': Address('gp', 0)}
        memory = {}
        expected_pc = start
        body = [(pc, op, args) for pc, op, args in code.get(names[sh], []) if start <= pc < end]
        for pc, op, args in body:
            where = f'{names[sh]}+{pc:#x}'
            if pc != expected_pc:
                raise ValueError(f'{where}: incomplete disassembly')
            expected_pc = pc + 4  # Capstone instructions have fixed 32-bit encodings.
            rd = args[0]
            r = reloc.get((sh, pc))
            if r and r[0] == R_PCREL_HI20 and op == 'auipc':
                regs[rd] = High(Address(sh, pc))
            elif r and r[0] == R_PCREL_LO12_I and op in ('addi', 'mv'):
                high_pc = symbol(r[1], r[2])
                high = reloc.get((high_pc.section, high_pc.offset))
                if regs.get(args[1]) != High(high_pc) or not high or high[0] != R_PCREL_HI20:
                    raise ValueError(f'{where}: unmatched PC-relative address')
                regs[rd] = symbol(high[1], high[2])
            elif r:
                raise ValueError(f'{where}: unsupported relocated instruction {op}')
            elif op in ('stc', 'sd', 'sw', 'sh', 'sb', 'ldc', 'ld', 'lw', 'lwu'):
                m = MEM.fullmatch(args[1])
                dest = add(regs.get(m[2]), int(m[1], 0)) if m else None
                if not isinstance(dest, Address):
                    raise ValueError(f'{where}: unknown memory address in {op}')
                width = {'stc': 16, 'ldc': 16, 'sd': 8, 'ld': 8,
                         'sw': 4, 'lw': 4, 'lwu': 4, 'sh': 2, 'sb': 1}[op]
                if op.startswith('l'):
                    regs[rd] = memory.get((dest, width))
                else:
                    # A later scalar/overlapping store destroys earlier coverage.
                    def overlaps(a, size):
                        return (a.section == dest.section and a.offset < dest.offset + width
                                and dest.offset < a.offset + size)
                    covered.difference_update(a for a in list(covered) if overlaps(a, 16))
                    for key in list(memory):
                        if overlaps(*key):
                            del memory[key]
                    memory[dest, width] = regs.get(rd)
                    if op == 'stc' and isinstance(dest.section, int):
                        covered.add(dest)
            elif op in ('mv', 'movc'):
                regs[rd] = regs.get(args[1])
            elif op == 'auipc':
                # Same-section references can be resolved by the assembler.
                n = int(args[1], 0) << 12
                regs[rd] = Address(sh, pc + (n - (1 << 32) if n & (1 << 31) else n))
            elif op == 'li':
                regs[rd] = int(args[1], 0)
            elif op == 'lui':
                n = int(args[1], 0) << 12
                regs[rd] = n - (1 << 32) if n & (1 << 31) else n
            elif op in ('addi', 'cincoffsetimm'):
                regs[rd] = add(regs.get(args[1]), int(args[2], 0))
            elif op in ('add', 'cincoffset'):
                base, offset = regs.get(args[1]), regs.get(args[2])
                # Data materialization rebases the link address through gp.
                regs[rd] = offset if base == Address('gp', 0) and isinstance(offset, Address) else add(base, offset)
            elif op in ('sub', 'slli', 'srli', 'andi', 'ori', 'addiw'):
                a = regs.get(args[1])
                b = regs.get(args[2]) if op == 'sub' else int(args[2], 0)
                if isinstance(a, int) and isinstance(b, int):
                    regs[rd] = {'sub': lambda: a - b, 'slli': lambda: a << b,
                                'srli': lambda: (a % (1 << 64)) >> b,
                                'andi': lambda: a & b, 'ori': lambda: a | b,
                                'addiw': lambda: ((a + b + (1 << 31)) % (1 << 32)) - (1 << 31)}[op]()
                else:
                    regs[rd] = None
            elif op in ('delin', 'shrink', 'tighten'):
                pass  # Permissions/bounds change; the cursor does not.
            elif op in ('ret', 'cjalr'):
                if pc + 4 != end or (op == 'cjalr' and args != ['zero', '0x0(ra)']):
                    raise ValueError(f'{where}: unexpected control flow')
            elif op != 'nop':
                raise ValueError(f'{where}: unsupported initializer instruction {op}')
            regs['zero'] = 0
        if expected_pc != end or not body or body[-1][1] not in ('ret', 'cjalr'):
            raise ValueError(f'{names[sh]}+{start:#x}: incomplete initializer')
    return covered


def census(label, path):
    secs, names, syms, relas = parse(path.read_bytes())
    candidates = [(sec, off, sym) for sec, off, typ, sym, delta in relas
                  if typ == R_64 and secs[sec][2] & SHF_ALLOC and not secs[sec][2] & SHF_EXEC
                  and not names[sec].startswith(('.eh_frame', '.capstone_cap_init', '.gct',
                                                 '.init_array', '.fini_array'))]
    if not candidates:
        return []
    covered = stores(path, names, syms, relas)
    out = []
    for sec, off, sym in candidates:
        if Address(sec, off) in covered:
            continue
        box = next(((v, n) for n, t, sh, v, sz in syms
                    if sh == sec and t in (0, 1) and v <= off < v + sz), (0, '?'))
        tn, _, sh, _, _ = syms[sym]
        target = tn or (names[sh] if sh < len(names) else '?')
        out.append((label, names[sec], box[1], off - box[0], target))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()
    seen, uncovered, incomplete = 0, 0, 0

    def scan(label, path):
        nonlocal seen, uncovered, incomplete
        try:
            rows = census(label, path)
            seen += 1
            uncovered += len(rows)
            for row in rows:
                print('\t'.join(map(str, row)))
        except (ValueError, IndexError, StopIteration, struct.error, OSError,
                subprocess.CalledProcessError) as exc:
            incomplete += 1
            print(f'{label}: INCOMPLETE: {exc}', file=sys.stderr)

    for path in args.files:
        if path.suffix != '.a':
            scan(str(path), path)
            continue
        try:
            members = subprocess.run([tool('llvm-ar'), 't', str(path)], check=True,
                                     capture_output=True, text=True).stdout.splitlines()
            with tempfile.TemporaryDirectory() as d:
                duplicate = any(n > 1 for n in Counter(members).values())
                occurrence = Counter()
                if not duplicate:
                    subprocess.run([tool('llvm-ar'), 'x', str(path.resolve())], cwd=d,
                                   check=True, capture_output=True)
                for member in members:
                    occurrence[member] += 1
                    if duplicate:
                        # ar x alone silently overwrites earlier names. SDK archives
                        # contain duplicate basenames, and each occurrence matters.
                        subprocess.run([tool('llvm-ar'), 'xN', str(occurrence[member]),
                                        str(path.resolve()), member], cwd=d, check=True,
                                       capture_output=True)
                    scan(f'{path}({member}#{occurrence[member]})', Path(d) / member)
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            incomplete += 1
            print(f'{path}: INCOMPLETE: {exc}', file=sys.stderr)
    print(f'# objects analyzed: {seen}, uncovered address slots: {uncovered}, '
          f'incomplete objects/archives: {incomplete}', file=sys.stderr)
    return 2 if incomplete or not seen else 1 if uncovered else 0


if __name__ == '__main__':
    sys.exit(main())
