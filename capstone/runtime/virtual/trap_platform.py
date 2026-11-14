#!/usr/bin/env python3
"""The trap-route platform: one boot of Linux on QEMU that runs a list of programs.

On the trap route the kernel runs a protected program itself. Its binfmt loader
(kernel parameter sublet.dom_binfmt=1) starts a .dom image as an ordinary task,
and a Sublet fault ends the task with SIGSEGV and one kernel line,

  sublet: fault cause=C pc=0x.. address=0x.. pid=N comm=NAME

So a run needs no module, no launcher binary and no supervisor slots. This
module holds every platform fact the evaluation runners need:

  qemu_command()      the QEMU argv: machine virt, one hart, sstc on, the
                      trap-route CPU properties, firmware, kernel, initramfs,
                      and the read-only stage disk
  kernel_arguments()  the kernel command line
  build_initramfs()   busybox, the C library and an /init that mounts the
                      stage, runs its gate.sh and powers off
  make_kit()          a kit directory in the layout the corpus tools read
  boot()              one boot of a stage directory, serial log in WORK
  run_programs()      stage a list of programs, run them in as few boots as
                      the guest allows, return the status and fault line of each

The stage's gate.sh starts a protected program as `./capstone-vexec IMAGE`,
as on the deleted route. The kit's capstone-vexec is now a shell script
(trap-vexec.sh): it runs the image, ends the way the image ended, and prints
the launcher's fault line, which the corpus judges parse,

  capstone-exec: domain fault cause=C pc=0x.. address=0x.. ra=0x.. fp=0x..
                 entry=0x.. code=0x..-0x.. image=IMAGE

from the kernel's fault line for its child. Words the kernel line lacks read
0x0; the code base falls back to the loader's sublet.dom_last_base.

  trap_platform.py make-kit KIT [--kernel IMAGE] [--firmware FW] [--qemu QEMU]
  trap_platform.py run --kit KIT --work DIR [--step-timeout S] [--file DEST=SRC]... [--step NAME=CMD]... [PROGRAM...]

A fault or a watchdog kill (CAPSTONE_STEP_TIMEOUT, default 600 s per step)
ends one program and the boot goes on. Only a dead or hung guest ends a boot;
its current program then reads kind 'harness' and a fresh boot runs the rest.
"""
import argparse
import gzip
import json
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qemu_lock import qemu_lock  # noqa: E402

HOME = Path.home()
DEFAULT_QEMU = Path(os.environ.get('SUBLET_QEMU', HOME / 'asr-work/del-q/bin/qemu-s5'))
DEFAULT_FIRMWARE = Path(os.environ.get('SUBLET_FIRMWARE', HOME / 'asr-work/del-fw/out/f-keep/fw_jump.elf'))
DEFAULT_KERNEL = Path(os.environ.get('SUBLET_KERNEL', HOME / 'asr-work/del-fw/linux/Image'))
DEFAULT_BASE_INITRAMFS = Path(os.environ.get('SUBLET_BASE_INITRAMFS', HOME / 'asr-work/del-fw/linux/initramfs.cpio.gz'))
DEFAULT_LIBRARIES = Path(os.environ.get('SUBLET_LIBRARIES', HOME / 'caplifive-system/sw/buildroot/build/target/lib'))

# The kit's files, in the order virtualvm.platform() hashes them.
QEMU_FILE = 'qemu-system-riscv64'
IMAGE_FILES = ('Image', 'fw_jump.elf', 'initramfs.cpio.gz')
LAUNCHER = 'capstone-vexec'
LAUNCHER_SOURCE = HERE / 'trap-vexec.sh'

GUEST_MEMORY = '4G'
DONE = 'VIRTUAL_STAGED_DONE'
INIT_UP = 'SUBLET-INIT: up'
INIT_GATE = 'SUBLET-INIT: gate status '

# The trap route admits .dom images only with this parameter.
KERNEL_PARAMETERS = ('sublet.dom_binfmt=1',)
CONSOLE_ARGUMENTS = ('console=ttyS0', 'earlycon=sbi', 'rdinit=/init', 'panic=-1')
# A protected program may open files only beneath the directory it started in. This word also
# grants /tmp, /usr and the null, zero and urandom devices, which the corpus cases use.
WIDE_GRANTS = 'sublet.dom_wide_grants=1'
# The loader's parameters that survive from capstone_vm.*; every other capstone_vm word named
# a part of the deleted route and is dropped.
KEPT_PARAMETERS = ('dom_wide_grants', 'dom_local_tmp', 'dom_null', 'dom_shell', 'dom_binfmt')

CPU_MODEL = 'rv64'
CPU_PROPERTIES = (
    'sstc=true',
    'h=false',
    'sv48=false',
    'sv57=false',
    'x-capstone-trap-route=true',
    'x-capstone-supervisor-refs=true',
    'x-capstone-raw-moves=true',
)
# CPU properties of the deleted route. A caller's CAPSTONE_QEMU_CPU_EXTRA may still name them;
# QEMU would refuse the boot, so cpu_string() drops them and says so.
REMOVED_CPU_PROPERTIES = (
    'x-capstone-checked-copy', 'x-capstone-supervisor-slots', 'x-capstone-context-transport',
    'x-capstone-ids', 'x-capstone-u-mode', 'x-capstone-addr',
)

INIT_SCRIPT = f"""#!/bin/busybox sh
# The trap-route /init: mount the stage, run its gate.sh, power off.
/bin/busybox mkdir -p /proc /sys /dev /tmp /mnt/vm /sbin /usr/bin /usr/sbin
/bin/busybox mount -t proc proc /proc
/bin/busybox mount -t sysfs sys /sys
/bin/busybox mount -t devtmpfs dev /dev 2>/dev/null
/bin/busybox --install -s
mount -t tmpfs tmpfs /tmp
ifconfig lo 127.0.0.1 up 2>/dev/null || ip link set lo up
export PATH=/bin:/sbin:/usr/bin:/usr/sbin HOME=/root
echo "{INIT_UP}, kernel $(uname -r)"
if [ -b /dev/vda ]; then
    mount -t ext4 -o ro /dev/vda /mnt/vm || echo "SUBLET-INIT: stage mount failed"
fi
if [ -f /mnt/vm/gate.sh ]; then
    cd /mnt/vm && sh /mnt/vm/gate.sh
    echo "{INIT_GATE}$?"
else
    echo "SUBLET-INIT: no gate.sh"
fi
cd /
sync
poweroff -f
"""


@dataclass
class Program:
    """One step of a run: a host file staged under its own name and run with ARGS, or a shell COMMAND."""
    path: Path = None
    args: list = field(default_factory=list)
    name: str = ''
    protected: bool = None
    command: str = ''

    def __post_init__(self):
        if self.path is not None:
            self.path = Path(self.path)
            self.name = self.name or self.path.name
            if self.protected is None:
                self.protected = self.path.suffix == '.dom'
        elif not (self.name and self.command):
            raise ValueError('a step without a program file needs a name and a command')


@dataclass
class Result:
    """How one program ended: kind is exit, signal or none (it never ended)."""
    name: str
    kind: str
    value: int = None
    fault: str = None
    output: str = ''
    kernel_faults: list = field(default_factory=list)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    kit = commands.add_parser('make-kit', help='write a kit directory')
    kit.add_argument('kit', type=Path)
    kit.add_argument('--qemu', type=Path, default=DEFAULT_QEMU)
    kit.add_argument('--firmware', type=Path, default=DEFAULT_FIRMWARE)
    kit.add_argument('--kernel', type=Path, default=DEFAULT_KERNEL)
    kit.add_argument('--base-initramfs', type=Path, default=DEFAULT_BASE_INITRAMFS)
    kit.add_argument('--libraries', type=Path, default=DEFAULT_LIBRARIES)
    kit.add_argument('--param', action='append', default=[],
                     help='a loader parameter name=value for adapter/params (repeatable)')

    run = commands.add_parser('run', help='run programs in one boot and print how each ended')
    run.add_argument('--kit', type=Path, required=True)
    run.add_argument('--work', type=Path, required=True, help='new output directory')
    run.add_argument('--timeout', type=int, help='seconds for one whole boot (default: from the step limit)')
    run.add_argument('--step-timeout', type=int, default=STEP_LIMIT, help='seconds per program before the watchdog')
    run.add_argument('--kernel-arg', action='append', default=[])
    run.add_argument('--step', action='append', default=[], metavar='NAME=COMMAND',
                     help='a shell command run from /mnt/vm as one step (repeatable, after the programs)')
    run.add_argument('--file', action='append', default=[], metavar='DEST=SOURCE',
                     help='a host file or directory staged read-only at /mnt/vm/DEST (repeatable)')
    run.add_argument('programs', nargs='*', type=Path)

    a = parser.parse_args()
    if a.command == 'make-kit':
        make_kit(a.kit, a.qemu, a.firmware, a.kernel, a.base_initramfs, a.libraries, a.param)
        print(a.kit)
        return 0
    elif a.command == 'run':
        steps = [Program(p) for p in a.programs]
        steps += [Program(name=n, command=c) for n, _, c in (w.partition('=') for w in a.step)]
        files = dict((d, Path(s)) for d, _, s in (w.partition('=') for w in a.file))
        if not steps:
            parser.error('run needs a program or a --step')
        results, harness_ok = run_programs(a.kit, steps, a.work, a.timeout, a.kernel_arg, a.step_timeout, files)
        (a.work / 'results.json').write_text(json.dumps(
            [{'name': r.name, 'kind': r.kind, 'value': r.value, 'fault': r.fault,
              'kernel_faults': r.kernel_faults, 'output': r.output}
             for r in results], indent=1) + '\n')
        for r in results:
            print(f'{r.name} {r.kind} {"" if r.value is None else r.value}'.rstrip()
                  + (f' | {r.fault}' if r.fault else ''))
        print(f'harness_ok={int(harness_ok)} boots={len(list(a.work.glob("boot-*")))} work={a.work}')
        return 0 if harness_ok else 1
    else:
        raise ValueError(f'unknown command {a.command}')


# ---- the platform facts ------------------------------------------------------------------------

def kernel_arguments(extra=()):
    """The kernel command line. capstone_vm.X words become sublet.X when the loader still has X; a later word wins."""
    words = [*CONSOLE_ARGUMENTS, *KERNEL_PARAMETERS]
    for word in extra:
        words.extend(translate_kernel_word(word))
    return words


def cpu_string(extra=None):
    """QEMU's -cpu value. CAPSTONE_QEMU_CPU_EXTRA appends to it, minus the deleted route's properties."""
    if extra is None:
        extra = os.environ.get('CAPSTONE_QEMU_CPU_EXTRA', '')
    words = [CPU_MODEL, *CPU_PROPERTIES]
    for word in (w.strip() for w in extra.split(',')):
        if not word:
            continue
        name = word.partition('=')[0]
        if name in REMOVED_CPU_PROPERTIES:
            print(f'trap_platform: dropped CPU property {word}: not on the trap route', file=sys.stderr)
            continue
        words.append(word)
    return ','.join(words)


def qemu_command(qemu, firmware, kernel, initramfs, append, disk=None, memory=GUEST_MEMORY, cpu=None,
                 share=None, plugin=None, plugin_args=(), icount=None, dtb=None):
    """The QEMU argv for one boot. One hart: the trap route keeps one table root per hart."""
    cmd = [str(qemu), '-M', 'virt', '-m', memory, '-smp', '1', '-nographic', '-monitor', 'none',
           '-serial', 'stdio', '-cpu', cpu or cpu_string(), '-bios', str(firmware), '-kernel', str(kernel),
           '-initrd', str(initramfs), '-append', ' '.join(append)]
    if disk:
        cmd += ['-drive', f'file={disk},format=raw,id=gate,readonly=on', '-device', 'virtio-blk-device,drive=gate']
    if share:
        cmd += ['-virtfs', f'local,path={Path(share).resolve()},mount_tag=gate9p,security_model=none,id=gate9p']
    if plugin:
        cmd += ['-plugin', ','.join([str(plugin), *plugin_args])]
    if icount is not None:
        cmd += ['-icount', f'shift={icount}']
    if dtb:
        cmd += ['-dtb', str(dtb)]
    return cmd


# ---- kits and the initramfs --------------------------------------------------------------------

def make_kit(kit, qemu=DEFAULT_QEMU, firmware=DEFAULT_FIRMWARE, kernel=DEFAULT_KERNEL,
             base_initramfs=DEFAULT_BASE_INITRAMFS, libraries=DEFAULT_LIBRARIES, params=()):
    """A kit: qemu-system-riscv64, images/{Image,fw_jump.elf,initramfs.cpio.gz}, adapter/capstone-vexec[, params]."""
    kit = Path(kit)
    if kit.exists():
        raise FileExistsError(f'{kit} exists; a kit is written into a fresh directory')
    (kit / 'images').mkdir(parents=True)
    (kit / 'adapter').mkdir()

    shutil.copy2(qemu, kit / QEMU_FILE)
    shutil.copy2(kernel, kit / 'images/Image')
    shutil.copy2(firmware, kit / 'images/fw_jump.elf')
    build_initramfs(kit / 'images/initramfs.cpio.gz', base_initramfs, libraries)

    shutil.copy2(LAUNCHER_SOURCE, kit / 'adapter' / LAUNCHER)
    (kit / 'adapter' / LAUNCHER).chmod(0o755)
    if params:
        lines = [p if p.startswith('#') else p.removeprefix('sublet.').removeprefix('capstone_vm.') for p in params]
        (kit / 'adapter/params').write_text('\n'.join(lines) + '\n')

    sums = subprocess.run(['sha256sum', *sorted(str(p.relative_to(kit)) for p in kit.rglob('*') if p.is_file())],
                          cwd=kit, capture_output=True, text=True, check=True).stdout
    (kit / 'SHA256SUMS').write_text(sums)
    return kit


def build_initramfs(out, base=DEFAULT_BASE_INITRAMFS, libraries=DEFAULT_LIBRARIES, files=None):
    """The base busybox initramfs with the C library tree, this module's /init and FILES {dest: source} added."""
    with tempfile.TemporaryDirectory(prefix='trap-initramfs.') as tmp:
        root = Path(tmp) / 'root'
        root.mkdir()

        unpack = subprocess.run(['cpio', '-id', '--quiet'], cwd=root, input=gzip.decompress(Path(base).read_bytes()))
        if unpack.returncode != 0:
            raise RuntimeError(f'{base}: cpio could not unpack it')
        if not (root / 'bin/busybox').is_file():
            raise RuntimeError(f'{base}: no bin/busybox')

        if libraries and Path(libraries).is_dir():
            subprocess.run(['cp', '-a', '--no-preserve=ownership', f'{libraries}/.', str(root / 'lib')], check=True)
        for dest, source in (files or {}).items():
            target = root / dest.lstrip('/')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        # root and nobody: the postgres gates start the server with `su nobody`, memcached's job drops to 65534.
        (root / 'etc').mkdir(exist_ok=True)
        (root / 'etc/passwd').write_text('root:x:0:0:root:/root:/bin/sh\nnobody:x:65534:65534:nobody:/tmp:/bin/sh\n')
        (root / 'etc/group').write_text('root:x:0:\nnogroup:x:65534:\n')
        (root / 'root').mkdir(exist_ok=True)
        (root / 'init').write_text(INIT_SCRIPT)
        (root / 'init').chmod(0o755)

        names = subprocess.run(['find', '.'], cwd=root, capture_output=True, check=True).stdout
        archive = subprocess.run(['cpio', '-o', '-H', 'newc', '--quiet', '--owner=0:0'], cwd=root, input=names,
                                 capture_output=True, check=True).stdout
        Path(out).write_bytes(gzip.compress(archive, 6))
    return Path(out)


def kit_paths(kit):
    """{qemu, Image, fw_jump.elf, initramfs.cpio.gz, launcher} of a kit, each checked."""
    kit = Path(kit)
    paths = {'qemu': kit / QEMU_FILE, **{n: kit / 'images' / n for n in IMAGE_FILES},
             'launcher': kit / 'adapter' / LAUNCHER}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'kit {kit} lacks {", ".join(missing)}')
    return paths


# ---- one boot ------------------------------------------------------------------------------------

def boot(qemu, images, stage, work, timeout=600, kernel_args=(), disk_mib=None, memory=GUEST_MEMORY,
         firmware=None, done=DONE, share=None, plugin=None, plugin_args=(), icount=None, dtb=None, log_prefix='boot',
         step_limit=None):
    """Boot once: the stage directory becomes a read-only ext4 disk at /mnt/vm and /init runs its gate.sh.

    images is a directory holding Image, fw_jump.elf and initramfs.cpio.gz. Without initramfs.cpio.gz the
    boot builds one from the default base into WORK. Returns True when the serial log reached DONE. With
    STEP_LIMIT the host also stops a guest that prints no step marker for STEP_LIMIT + HANG_MARGIN seconds.
    """
    images, stage, work = Path(images), Path(stage), Path(work)
    work.mkdir(parents=True, exist_ok=False)
    if not (stage / 'gate.sh').is_file():
        raise FileNotFoundError(f'{stage}/gate.sh')

    initramfs = images / 'initramfs.cpio.gz'
    if not initramfs.is_file():
        initramfs = build_initramfs(work / 'initramfs.cpio.gz')

    size = sum(f.stat().st_size for f in stage.rglob('*') if f.is_file())
    disk = work / 'stage.ext4'
    with disk.open('wb') as f:
        f.truncate((disk_mib or max(64, (size >> 20) * 2 + 64)) << 20)
    subprocess.run(['/sbin/mkfs.ext4', '-F', '-q', '-d', str(stage), str(disk)], check=True)

    append = kernel_arguments(kernel_args)
    cmd = qemu_command(qemu, firmware or images / 'fw_jump.elf', images / 'Image', initramfs, append, disk,
                       memory, share=share, plugin=plugin, plugin_args=plugin_args, icount=icount, dtb=dtb)
    (work / 'cpu.txt').write_text(cmd[cmd.index('-cpu') + 1] + '\n')
    (work / 'cmdline.txt').write_text(' '.join(cmd) + '\n')
    print(f'{log_prefix}: -cpu {cmd[cmd.index("-cpu") + 1]}', flush=True)
    print(f'{log_prefix}: -append {" ".join(append)}', flush=True)

    return drive(cmd, work, timeout, done, step_limit)


def drive(cmd, work, timeout, done=DONE, step_limit=None):
    """Run QEMU under the shared lock until DONE, power-off, a panic, a hang or the timeout. The guest needs no
    input. A hang: with STEP_LIMIT, no step marker for STEP_LIMIT + HANG_MARGIN seconds (the guest watchdog
    ends every step within STEP_LIMIT, so only a dead or wedged kernel stays silent that long)."""
    completed = False
    progress = time.monotonic()
    window = b''
    marker = f'\n{done}\r\n'.encode()
    stops = (b'Kernel panic', b'Oops', b'reboot: Power down')
    with qemu_lock(), (Path(work) / 'serial.log').open('wb') as log:
        guest = subprocess.Popen(cmd, env=dict(os.environ, TMPDIR=str(Path(work).resolve())), stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if not select.select([guest.stdout], [], [], 1)[0]:
                    if guest.poll() is not None:
                        break
                    if step_limit and time.monotonic() - progress > step_limit + HANG_MARGIN:
                        log.write(b'\n[harness] no step marker for too long: the guest is hung\n')
                        break
                    continue
                data = os.read(guest.stdout.fileno(), 65536)
                if not data:
                    break
                log.write(data)
                log.flush()
                window = (window + data)[-65536:]
                if b'@@STEP-' in data:
                    progress = time.monotonic()
                if step_limit and time.monotonic() - progress > step_limit + HANG_MARGIN:
                    log.write(b'\n[harness] no step marker for too long: the guest is hung\n')
                    break
                if marker in window:
                    completed = True
                if any(s in window for s in stops):
                    break
        finally:
            if guest.poll() is None:
                guest.terminate()
            try:
                guest.wait(timeout=5)
            except subprocess.TimeoutExpired:
                guest.kill()
                guest.wait()
    return completed


# ---- a list of programs ------------------------------------------------------------------------

BEGIN, END = '@@STEP-BEGIN ', '@@STEP-END '
BOUND = '@@STEP-BOUND '   # the guest watchdog killed the step at its limit
RESTART = '@@HARNESS-RESTART '
NSIG = 64
STEP_LIMIT = int(os.environ.get('CAPSTONE_STEP_TIMEOUT', '600'))
HANG_MARGIN = 120   # seconds past a step's limit after which the host declares the guest hung


def run_programs(kit, programs, work, timeout=None, kernel_args=(), step_limit=STEP_LIMIT, files=None):
    """Run PROGRAMS on the kit, as many per boot as the guest survives. Returns ([Result], harness_ok).

    A .dom program runs through the kit's capstone-vexec, which prints the fault line; an ordinary
    program runs directly. A fault or a watchdog kill ends one program and the boot goes on. When the
    guest itself dies, the program it was running reads kind 'harness' and a fresh boot runs the rest.
    FILES {dest: host path} are staged beside the programs.
    """
    paths = kit_paths(kit)
    work = Path(work)
    work.mkdir(parents=True)
    names = [p.name for p in programs]
    if len(set(names)) != len(names) or not all(re.fullmatch(r'[A-Za-z0-9_.+-]+', n) for n in names):
        raise ValueError(f'program names {names} must be unique and shell-safe')

    def stage_for(remaining, boot_stage):
        boot_stage.mkdir(parents=True)
        shutil.copy2(paths['launcher'], boot_stage / LAUNCHER)
        (boot_stage / LAUNCHER).chmod(0o755)
        for dest, source in (files or {}).items():
            target = boot_stage / dest
            target.parent.mkdir(parents=True, exist_ok=True)
            if Path(source).is_dir():
                shutil.copytree(source, target, symlinks=False)
            else:
                shutil.copy2(source, target)
        steps = []
        for p in remaining:
            if p.command:
                steps.append((p.name, p.command, step_limit))
                continue
            shutil.copy2(p.path, boot_stage / p.name)
            (boot_stage / p.name).chmod(0o755)
            args = ' '.join(shell_quote(x) for x in p.args)
            start = f'./{LAUNCHER} ./{p.name}' if p.protected else f'./{p.name}'
            steps.append((p.name, f'{start} {args}'.rstrip(), step_limit))
        (boot_stage / 'gate.sh').write_text(gate_script(steps))

    def boot_once(remaining, n):
        boot_stage = work / f'stage-{n}'
        stage_for(remaining, boot_stage)
        budget = timeout or 120 + (step_limit + 10) * len(remaining)
        boot(paths['qemu'], paths['Image'].parent, boot_stage, work / f'boot-{n}', budget,
             [WIDE_GRANTS, *kernel_args], log_prefix='run', step_limit=step_limit)
        serial = work / f'boot-{n}/serial.log'
        return serial.read_text(errors='replace').replace('\r\n', '\n') if serial.exists() else ''

    results, harness_ok = run_with_restarts(programs, lambda p: p.name, boot_once)
    return [results[p.name] for p in programs], harness_ok


def run_with_restarts(items, name_of, boot_once):
    """Boot until every item ran. boot_once(remaining, n) returns boot n's serial text.

    The item a dead guest was running reads kind 'harness' (a harness failure, never a verdict) and
    the next boot starts after it. A boot that began no item ends the run: the rest read 'harness' too.
    """
    results = {}
    remaining = list(items)
    harness_ok = True
    n = 0
    while remaining:
        n += 1
        text = boot_once(remaining, n)
        steps = split_steps(text)
        finished = DONE in text

        ran = 0
        for item in remaining:
            r = steps.get(name_of(item))
            if r is None:
                break
            ran += 1
            if r.kind == 'none':
                r.kind = 'harness'
                r.output += f'\n[harness] the guest died during this step (boot {n}); a new boot runs the rest\n'
                harness_ok = False
                results[r.name] = r
                break
            results[r.name] = r

        if ran == 0:
            for item in remaining:
                results[name_of(item)] = Result(name_of(item), 'harness',
                                                output=f'[harness] boot {n} began no step\n')
            return results, False
        if not finished:
            harness_ok = False   # the guest died or hung: a harness failure even when no step was cut short
        remaining = remaining[ran:]
    return results, harness_ok


def gate_script(steps):
    """gate.sh for [(name, command, limit)]: each step under its own watchdog, then DONE."""
    lines = ['#!/bin/sh', 'cd /mnt/vm || exit 1', 'dmesg -n 1', 'export CAPSTONE_EXEC_DIAGNOSTICS=1']
    for name, command, limit in steps:
        lines += step_lines(name, command, limit)
    lines += [f'echo {DONE}', '']
    return '\n'.join(lines)


def step_lines(name, command, limit):
    """The guest lines for one step. The step runs in its own session; a watchdog kills that whole process group
    after LIMIT seconds and prints BOUND, so a hung or looping program ends and the boot goes on."""
    dog = (f'sleep {int(limit)}; kill -0 $1 2>/dev/null || exit 0; '
           f'echo "{BOUND}{name} {int(limit)}" > /tmp/step.bound; kill -KILL -$1 2>/dev/null')
    return [f"echo '{BEGIN}{name}'",
            'rm -f /tmp/step.bound',
            f'setsid sh -c {shell_quote(command)} > /tmp/step.out 2>&1 < /dev/null &',
            'pid=$!',
            f'setsid sh -c {shell_quote(dog)} watchdog $pid > /dev/null 2>&1 &',
            'dog=$!',
            'wait $pid',
            'rc=$?',
            'kill -KILL -$dog 2>/dev/null',
            'wait $dog 2>/dev/null',
            'cat /tmp/step.out',
            '[ -f /tmp/step.bound ] && cat /tmp/step.bound',
            f'echo "{END}{name} $rc"']


def split_steps(serial):
    """{name: Result} for every step that began, read from one boot's serial log."""
    results = {}
    pattern = rf'^{re.escape(BEGIN)}(\S+)\n(.*?)(?:{re.escape(END)}\1 (\d+)$|\Z)'
    for m in re.finditer(pattern, serial, re.S | re.M):
        name, text, rc = m.group(1), m.group(2), m.group(3)
        fault = next((line.strip() for line in text.splitlines() if 'domain fault cause=' in line), None)
        kernel = [line.strip() for line in text.splitlines() if 'capstone-exec: kernel fault ' in line]
        if rc is None:
            results[name] = Result(name, 'none', fault=fault, output=text, kernel_faults=kernel)
        elif f'{BOUND}{name} ' in text:
            results[name] = Result(name, 'timeout', None, fault, text, kernel)
        elif 128 < int(rc) <= 128 + NSIG:
            results[name] = Result(name, 'signal', int(rc) - 128, fault, text, kernel)
        else:
            results[name] = Result(name, 'exit', int(rc), fault, text, kernel)
    return results


# ---- details -------------------------------------------------------------------------------------

def translate_kernel_word(word):
    """A caller's kernel word: capstone_vm.X=V becomes sublet.X=V for a kept loader parameter, otherwise nothing."""
    name, _, value = word.partition('=')
    if not name.startswith('capstone_vm.'):
        return [word]
    parameter = name.removeprefix('capstone_vm.')
    if parameter in KEPT_PARAMETERS:
        return [f'sublet.{parameter}={value}']
    print(f'trap_platform: dropped kernel word {word}: not on the trap route', file=sys.stderr)
    return []


def shell_quote(word):
    """WORD as one sh word."""
    return "'" + str(word).replace("'", "'\\''") + "'"


if __name__ == '__main__':
    raise SystemExit(main())
