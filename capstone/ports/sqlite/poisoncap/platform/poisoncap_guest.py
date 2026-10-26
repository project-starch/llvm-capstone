"""The PoisonCap guest every runner here boots, made in one place.

    sys.path.insert(0, str(<this folder>)); import poisoncap_guest

A Guest (ports/common/host/cheribsd/guest.py) over the platform's SDK, rootfs (or another kernel root)
and disk image, single-user, with libc's revocation default set on the console before sshd starts:
0 unless asked otherwise, because the published kernel panics (share->excl at vm_map.c:6103) when
libc's automatic revocation runs in ordinary processes
(ports/ffmpeg/buffer-pool/results/measurements/20260919-poisoncap-pilot). Explicit revocation calls
are unaffected.
"""
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / 'capstone/ports/common/host/cheribsd'))
from guest import Guest  # noqa: E402

# Generous budgets for a guest under TCG on a shared host (Guest's own are 180 / 45 / 45 s).
GENEROUS_BUDGETS = {'CHERI_GUEST_BOOT_SECONDS': 1200, 'CHERI_GUEST_CMD_SECONDS': 600,
                    'CHERI_SSH_READY_SECONDS': 600}


def add_platform_arguments(parser, port):
    """--platform (POISONCAP_PLATFORM, default ~/poisoncap-platform) and --port (default PORT)."""
    parser.add_argument('--platform', type=Path,
                        default=Path(os.environ.get('POISONCAP_PLATFORM', Path.home() / 'poisoncap-platform')))
    parser.add_argument('--port', type=int, default=port)


def use_generous_budgets():
    """Raise the guest's boot and command budgets, unless the environment already sets them."""
    for name, seconds in GENEROUS_BUDGETS.items():
        os.environ.setdefault(name, str(seconds))


def new_guest(platform, vm_dir, port, kernel_root=None, revocation_default_off=True):
    """A Guest that boots PLATFORM's image with KERNEL_ROOT's kernel (the rootfs's by default); its QEMU
    command is recorded in VM_DIR/command.json. The guest is not started."""
    work = Path(platform) / 'work'
    guest = Guest(work / 'sdk', kernel_root or work / 'output/rootfs-riscv64-purecap',
                  work / 'output/cheribsd-riscv64-purecap.img', Path(vm_dir), port,
                  disable_default_revocation=revocation_default_off)
    (Path(vm_dir) / 'command.json').write_text(json.dumps(guest.argv) + '\n')
    return guest
