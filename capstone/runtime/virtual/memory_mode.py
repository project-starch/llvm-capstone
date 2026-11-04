"""The module's memory mode (pinned or unpinned) as one gate input and one logged fact.

Every gate runner loads capstone_vm.ko itself. It passes `unpinned=1` exactly when its
`--unpinned` flag is given, and echoes the mode the loaded module reports right after each
insmod, as a `<PREFIX>MEMORY_MODE:<Y|N|A>` line in the guest console. `A` is a module that has no
`unpinned` parameter (a pinned-only build): it is pinned, and the only value the pinned mode
accepts besides `N`. gates/compare.py reads the same lines back, so a
run whose log disagrees with the mode it was asked for is refused rather than compared.

The unpinned fault path is a second, independent option (module parameter `unpinned_fast_fault`, default
OFF, valid only with unpinned=1). The gate environment asks for it with GATE_UNPINNED_FAST_FAULT=1 (run-arm.sh
exports it from FAST_FAULT=on); args() then adds `unpinned_fast_fault=1` to every unpinned insmod, and echo()
also logs `<PREFIX>FAST_FAULT:<Y|N|A>` (A: a module without the parameter) so that the guest log of every
module load says which path that load ran. gates/compare.py reads those lines back against OUT/fast-fault.
"""
import os
import re

PARAMETER = '/sys/module/capstone_vm/parameters/unpinned'
FAST_PARAMETER = '/sys/module/capstone_vm/parameters/unpinned_fast_fault'
FAST_ENV = 'GATE_UNPINNED_FAST_FAULT'
LINE = re.compile(r'^(\w*MEMORY_MODE):(\w*)\s*$', re.M)
FAST_LINE = re.compile(r'^(\w*FAST_FAULT):(\w*)\s*$', re.M)


def fast_requested():
    return os.environ.get(FAST_ENV) == '1'


def args(unpinned):
    """The insmod argument for the requested mode (empty when pinned: the default)."""
    if not unpinned:
        return ''
    return ' unpinned=1 unpinned_fast_fault=1' if fast_requested() else ' unpinned=1'


def echo(prefix=''):
    """The guest shell line that logs the loaded module's mode and fault path; put it after each insmod."""
    return (f'echo "{prefix}MEMORY_MODE:$(cat {PARAMETER} 2>/dev/null || echo A)"; '
            f'echo "{prefix}FAST_FAULT:$(cat {FAST_PARAMETER} 2>/dev/null || echo A)"')


def modes(log):
    return [m[2] for m in LINE.finditer(log)]


def fast_modes(log):
    return [m[2] for m in FAST_LINE.finditer(log)]


def verified(log, unpinned, count=None):
    """Every mode line the log holds is the requested one, and there is at least one
    (exactly `count` when the caller knows how many insmods it made)."""
    found = modes(log)
    want = ('Y',) if unpinned else ('N', 'A')
    return bool(found) and all(m in want for m in found) and (count is None or len(found) == count)
