"""The platform words the runners read, now those of the trap route (trap_platform.py).

The kernel runs a protected program as an ordinary Linux task through its .dom
loader (sublet.dom_binfmt=1). A runner loads no module, passes no supervisor
slot count and starts every image through the kit's `adapter/capstone-vexec`,
which is now the shell script trap-vexec.sh. It prints the launcher's
`domain fault` line from the kernel's fault line.

This module keeps the names the runners import, so they need no change:

  kernel_arguments()   the words for the kernel command line
  cpu_properties()     the words for QEMU's -cpu string, without the model
  GUEST_MEMORY         QEMU's -m value
  admit()              fails the gate script unless the kernel runs protected programs
  leave()              reports that no loadable Capstone module is present

The supervisor slot count of the deleted route is gone. The functions still
take it, so old callers work, and ignore it.
"""
import trap_platform

KERNEL_PARAMETERS = trap_platform.KERNEL_PARAMETERS
DEFAULT_SUPERVISOR_SLOTS = 1
CPU_PROPERTIES = trap_platform.CPU_PROPERTIES
GUEST_MEMORY = trap_platform.GUEST_MEMORY
WIDE_GRANTS = trap_platform.WIDE_GRANTS
LAUNCHER = trap_platform.LAUNCHER
ADAPTER_FILES = (LAUNCHER,)


def kernel_arguments(supervisor_slots=DEFAULT_SUPERVISOR_SLOTS, extra=()):
    """The trap route's kernel words. capstone_vm.X words in EXTRA become sublet.X or drop out."""
    return trap_platform.kernel_arguments(extra)


def cpu_properties(supervisor_slots=DEFAULT_SUPERVISOR_SLOTS):
    """The trap route's words for QEMU's -cpu string, without the model name."""
    return list(CPU_PROPERTIES)


def supervisor_slots_from_environment():
    """The deleted route's slot count. The trap route has none; the default keeps old callers working."""
    return DEFAULT_SUPERVISOR_SLOTS


def admit():
    """The guest shell line that stands where `insmod capstone_vm.ko` stood."""
    return f'./{LAUNCHER} --platform || exit 1'


def leave(prefix=''):
    """The guest shell lines that stand where `rmmod capstone_vm` stood.

    Nothing is unloaded. The line `<PREFIX>CLEANUP:0` says that no loadable Capstone module is
    present, which is what a trap-route kit requires.
    """
    return ("! grep -q '^capstone_vm ' /proc/modules\n"
            f'echo {prefix}CLEANUP:$?')
