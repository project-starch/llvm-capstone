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

The module's context capacity (`supervisor_slots`, default 1, what silicon holds) is a third input. The gate
environment names it with GATE_SUPERVISOR_SLOTS=<n> (run-arm.sh exports the x-capstone-supervisor-slots value of the
QEMU CPU it boots, so the module and the CPU hold the same count); args() then adds `supervisor_slots=<n>` to every
insmod, pinned or not, and echo() logs `<PREFIX>SUPERVISOR_SLOTS:<n|A>` read back from the loaded module. Unset, both
return exactly what they did before, so a runner's guest script is byte-identical outside the gate.
"""
import os
import re

PARAMETER = '/sys/module/capstone_vm/parameters/unpinned'
FAST_PARAMETER = '/sys/module/capstone_vm/parameters/unpinned_fast_fault'
FAST_ENV = 'GATE_UNPINNED_FAST_FAULT'
SLOTS_PARAMETER = '/sys/module/capstone_vm/parameters/supervisor_slots'
SLOTS_ENV = 'GATE_SUPERVISOR_SLOTS'
LINE = re.compile(r'^(\w*MEMORY_MODE):(\w*)\s*$', re.M)
FAST_LINE = re.compile(r'^(\w*FAST_FAULT):(\w*)\s*$', re.M)


def fast_requested():
    return os.environ.get(FAST_ENV) == '1'


def slots_requested():
    """The gate's supervisor slot count, or None outside a gate. A value that is not a positive integer is an error."""
    value = os.environ.get(SLOTS_ENV, '').strip()
    if not value:
        return None
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f'{SLOTS_ENV}={value!r}: not a positive slot count')
    return int(value)


def slots_arg():
    """` supervisor_slots=<n>` inside a gate, empty outside: for an insmod that does not go through args()."""
    slots = slots_requested()
    return f' supervisor_slots={slots}' if slots is not None else ''


def args(unpinned):
    """The insmod arguments for the requested mode (empty when pinned: the default), and the gate's slot count."""
    extra = slots_arg()
    if not unpinned:
        return extra
    return (' unpinned=1 unpinned_fast_fault=1' if fast_requested() else ' unpinned=1') + extra


def echo(prefix=''):
    """The guest shell line that logs the loaded module's mode and fault path; put it after each insmod."""
    line = (f'echo "{prefix}MEMORY_MODE:$(cat {PARAMETER} 2>/dev/null || echo A)"; '
            f'echo "{prefix}FAST_FAULT:$(cat {FAST_PARAMETER} 2>/dev/null || echo A)"')
    if slots_requested() is not None:
        line += f'; echo "{prefix}SUPERVISOR_SLOTS:$(cat {SLOTS_PARAMETER} 2>/dev/null || echo A)"'
    return line


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
