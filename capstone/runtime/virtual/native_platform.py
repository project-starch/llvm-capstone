"""The native platform as one runner input.

The Capstone adapter is built into the kernel and runs an image as an ordinary
Linux task. A runner therefore loads no module. It boots the kernel with the
adapter's admission parameters, gives QEMU the matching CPU properties, and
starts every image through the kit's `adapter/capstone-vexec`
(../linux/native-exec.c), which prints the launcher's `domain fault` line from
the kernel's fault record.

Every runner takes the three platform facts from here, so a change of the
platform's parameters is one edit:

  kernel_arguments()   the words for the kernel command line
  cpu_properties()     the words for QEMU's -cpu string
  GUEST_MEMORY         QEMU's -m value

and the two guest shell lines that stand where `insmod` and `rmmod` stood:

  admit()      fails the gate script unless the kernel runs native tasks
  leave()      reports that no loadable Capstone module is present

The adapter requires unpinned memory, so the pinned mode of the module route
does not exist here. `memory_mode.echo()` still reads the mode from sysfs, and
it reads `Y`.
"""
import os

# Each admission parameter is required: the adapter refuses to run an image with one of them off.
KERNEL_PARAMETERS = (
    'native_task=1',
    'dom_binfmt=1',
    'native_dispatch=1',
    'checked_copy=1',
    'native_slot_transport=1',
    'unpinned=1',
)
DEFAULT_SUPERVISOR_SLOTS = 1

CPU_PROPERTIES = (
    'x-pcrel=false',
    'x-capstone-raw-moves=true',
    'x-capstone-test-csrs=false',
    'x-capstone-checked-copy=true',
    'x-capstone-context-transport=true',
)

GUEST_MEMORY = '4G'

# A native task may open files only beneath the directory it started in. The gates start an image
# in the read-only stage and let it write to /tmp, as the launcher of the module route allowed.
# This kernel word also grants /tmp, /usr and the null, zero and urandom devices. A runner whose
# application needs them adds it to the kernel command line. It is not a default, because the
# adapter refuses it together with dom_local_tmp.
WIDE_GRANTS = 'capstone_vm.dom_wide_grants=1'
LAUNCHER = 'capstone-vexec'
ADAPTER_FILES = (LAUNCHER, 'capstone-job')


def kernel_arguments(supervisor_slots=DEFAULT_SUPERVISOR_SLOTS, extra=()):
    """The adapter's words for the kernel command line. The kernel and QEMU must name the same slot count."""
    words = [f'capstone_vm.{parameter}' for parameter in KERNEL_PARAMETERS]
    words.append(f'capstone_vm.supervisor_slots={supervisor_slots}')
    words.extend(extra)
    return words


def cpu_properties(supervisor_slots=DEFAULT_SUPERVISOR_SLOTS):
    """The matching words for QEMU's -cpu string, without the model name."""
    return [f'x-capstone-supervisor-slots={supervisor_slots}', *CPU_PROPERTIES]


def supervisor_slots_from_environment():
    """The slot count a caller asked for in CAPSTONE_QEMU_CPU_EXTRA, or the default.

    run-arm.sh and the corpus tools pass CPU properties in that variable. The kernel needs the same
    count on its command line, so the runner reads it back instead of taking it twice.
    """
    extra = os.environ.get('CAPSTONE_QEMU_CPU_EXTRA', '')
    for word in extra.split(','):
        name, _, value = word.strip().partition('=')
        if name == 'x-capstone-supervisor-slots' and value.isdigit():
            return int(value)
    return DEFAULT_SUPERVISOR_SLOTS


def admit():
    """The guest shell line that stands where `insmod capstone_vm.ko` stood."""
    return f'./{LAUNCHER} --platform || exit 1'


def leave(prefix=''):
    """The guest shell lines that stand where `rmmod capstone_vm` stood.

    Nothing is unloaded. The line `<PREFIX>CLEANUP:0` now says that no loadable Capstone module is
    present, which is what a native kit requires.
    """
    return ("! grep -q '^capstone_vm ' /proc/modules\n"
            f'echo {prefix}CLEANUP:$?')
