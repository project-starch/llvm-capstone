"""The module's memory mode (pinned or unpinned) as one gate input and one logged fact.

Every gate runner loads capstone_vm.ko itself. It passes `unpinned=1` exactly when its
`--unpinned` flag is given, and echoes the mode the loaded module reports right after each
insmod, as a `<PREFIX>MEMORY_MODE:<Y|N|A>` line in the guest console. `A` is a module that has no
`unpinned` parameter (dev before the option existed, a pinned-only build): it is pinned, and the
only value the pinned mode accepts besides `N`. gates/compare.py reads the same lines back, so a
run whose log disagrees with the mode it was asked for is refused rather than compared.
"""
import re

PARAMETER = '/sys/module/capstone_vm/parameters/unpinned'
LINE = re.compile(r'^(\w*MEMORY_MODE):(\w*)\s*$', re.M)


def args(unpinned):
    """The insmod argument for the requested mode (empty when pinned: the default)."""
    return ' unpinned=1' if unpinned else ''


def echo(prefix=''):
    """The guest shell line that logs the loaded module's mode; put it after each insmod."""
    return f'echo "{prefix}MEMORY_MODE:$(cat {PARAMETER} 2>/dev/null || echo A)"'


def modes(log):
    return [m[2] for m in LINE.finditer(log)]


def verified(log, unpinned, count=None):
    """Every mode line the log holds is the requested one, and there is at least one
    (exactly `count` when the caller knows how many insmods it made)."""
    found = modes(log)
    want = ('Y',) if unpinned else ('N', 'A')
    return bool(found) and all(m in want for m in found) and (count is None or len(found) == count)
