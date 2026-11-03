# Virtual runtime gate arms

`run-arm.sh OUT` builds the contract and application images, then runs the nine
landing gates. Its header lists the required tree, platform, hooks, compiler and
cross-compiler inputs. `PHASE=build` builds only; `PHASE=gates` runs an existing
`OUT/build`. The build must finish before sharing its immutable images between
runs. Every run needs its own OUT. `compare.py CONTROL ARM --all-rows` checks
execution coverage before comparing outcomes; INCOMPLETE is never a match.

## Snapshot slots

Parallel gate dispatch is not qualified. Keep using the serial queue until the
full same-tree comparison has succeeded three times, including every memcached
row. The initial trial found guest accounting differences in pin-ahead and
step-hooks, despite complete row coverage. The dispatcher remains a candidate;
host process tests alone do not qualify guest timing or test outcomes. Preserve
these gate assertions when investigating differences.

`GATE_SLOT=0` or `GATE_SLOT=1` opts into resource isolation. The resource wrapper
owns a nonblocking lease for the entire arm, before building or launching gates.
`PHASE=build` takes only the output lease and does not occupy a snapshot slot.
A second arm in that slot exits 75 with `REFUSED: resource busy`, including when
the arms use different trees and outputs. Invalid slots, unavailable CPUs, a
busy native port, and a conflicting QEMU lock override also fail before running
a gate. The wrapper pins its process and descendants and records the selection,
start/end and exit code in `OUT/resources-PHASE.json`. It terminates its own child
process group before releasing leases, including a server left by a failed
harness. Do not SIGKILL the wrapper; use TERM so it can clean up descendants.

| Resource | Slot 0 | Slot 1 |
|---|---|---|
| QEMU snapshot domain | A | B |
| CPUs (the existing `capstone_qemu_domain` sets) | 28-29,60-61 | 30-31,62-63 |
| Native memcached TCP loopback port | 21300 | 21301 |
| Whole-arm lease | `~/.capstone-locks/gate-slot-0.lock` | `~/.capstone-locks/gate-slot-1.lock` |
| Temporary files, caches, crash dumps | `OUT/slot-0/` | `OUT/slot-1/` |

Two is the existing qualified QEMU snapshot concurrency ceiling. Memory or CPU
headroom alone does not authorize more snapshot domains. All slots use the same
`~/.capstone-locks/qemu.lock` with the existing turnstile and domain locks. Each
guest holds the global lock shared and its domain exclusively; exclusive QEMU
users still exclude both. Do not set a private `QEMU_LOCK` on a slotted arm.
Unslotted arms use the global lock by default too, but still need the gate queue
because the native memcached reference runs outside the QEMU lock.

## Resource audit

This audit is for the nine calls in this driver, not every runner in the repo.

| Resource | Ownership and handling |
|---|---|
| Host TCP/UDP | Only the native memcached reference listens on the host: TCP loopback, default 21299 for an unslotted arm, or the table above. The server and harness receive the same port. UDP is disabled with `-U 0`. Reserve these ports for gate use; a bind preflight catches existing outsiders but cannot reserve a socket on behalf of a server that does its own bind. |
| Guest network | memcached uses guest loopback. `run.py` and `run-staged.py` specify no host forwarding or network backend. There is no SSH port to allocate. |
| Persistent VM state | This path never calls the persistent VM CLI and never accesses `~/.capstone-spec-vm/<name>`. No VM name, pidfile, QMP or Unix socket is shared. Adding such a caller requires a new audit; it is not snapshot-qualified. |
| 9p | `run-staged.py` supports `--share`, but none of the nine calls passes it. Stage files travel through private read-only ext4 disks and the serial pipe. No shared host 9p export is writable. |
| Rootfs and stage images | Both low-level runners use `-snapshot`; the stage disk is read-only and under the run's work directory. QEMU's temporary overlays use its private work directory. The platform is read-only input and must contain real files, as enforced by the driver. Never bypass QEMU image locking. |
| Host temporary files | `TMPDIR`, `TMP`, `TEMP`, XDG cache/state and clang crash directories are private to OUT and the slot. Runtime's `virtual-app.*` directories are under `OUT/gates/tmp`; other stage disks and logs are under their gate's work directory. |
| Guest paths and module names | `/tmp/one.out`, `/tmp/virtual.db`, `/tmp/mc-*`, `/dev/capstone-vm`, `capstone_vm`, `/mnt/vm` and the stage filenames are private to each fresh guest kernel/rootfs snapshot. They must remain unchanged in the checks. |
| Source tree and kit | Python bytecode writes are disabled. Contract and app builds write beneath `OUT/build`; native harness compilation writes beneath `OUT/gates/memcached`. SDKs, compiler binaries, platform files and already finished build images are read-only inputs. No gate writes a kit image. Build recipes retain their own per-build locks. |
| Native dependency build tests | libevent's `make verify` uses kernel-chosen TCP/UDP ports, socketpairs, and a high-port bind search for its bind test. Its `regress_make_tmpfile` uses `/tmp/eventtmp.XXXXXX` with `mkstemp` and immediate unlink; that upstream helper ignores TMPDIR but shares no filename. Test binaries and logs are under the private memcached build. The parallel gate proof reuses one finished image build; it does not qualify overlapping full application builds. |
| Output directories | `OUT/.arm.lock` excludes simultaneous writers even across distinct slots. Existing phase outputs are refused. A gates-only repeat may symlink `OUT/build` to the same finished build; never rebuild that shared input while gates run. |
| Coordination files | Slot leases, QEMU/domain locks, queue locks and the queue log are intentionally shared. Never unlink a live lock file: replacing its inode defeats exclusion. |

No gate assertion, fixture, guest script, timeout or comparison rule changes for
slots. The only memcached argument change is its port, which is already a runner
option. This changes host scheduling and naming, not capability rights, module
semantics or the application's protection configuration.

## Queue dispatcher

The repository's `gate-queue.sh` is the candidate replacement for
`~/retire-gate/gate-queue.sh`. It retains `.gate-queue.lock`, so an already running
old queue process still excludes new jobs. Ordinary callers stay exclusive:

```sh
~/retire-gate/gate-queue.sh bash "$TREE/capstone/runtime/virtual/gates/run-arm.sh" "$OUT"
```

After qualification, opt in with `GATE_PARALLEL=1 GATE_SLOTS=2`. The dispatcher
accepts only the slot-aware driver invocation, assigns `GATE_SLOT` itself and
holds a dispatch permit until completion. `GATE_SLOTS` defaults to 1 and is
limited to 1 or 2. An admission lock orders slot waiters; a turnstile prevents
new shared holders from overtaking an exclusive queue caller. Pending slot
jobs do not hold the legacy queue lock while waiting for a permit. A direct
slotted caller racing a dispatcher assignment is safely refused by the arm
lease; retry it through the dispatcher. All callers must use the same HOME and
queue installation. Nonqueued legacy arms remain incompatible with parallel
qualification.

For qualification before deployment, hold the existing serial gate queue around
the complete experiment driver, and let that driver launch the two experimental
slots. This reserves the host against other queued arms while testing overlap.
Check for nonqueued arms before and during the experiment. Never wrap a queue
wait in a timeout; guest runners already bound guest execution.

Run `test_resources.py` on a host admitting both CPU sets. It uses a private HOME
for its test lock files and real process/port collisions; it does not launch
QEMU. `test_compare.py` separately exercises the unchanged comparison rules.
These controls are necessary but do not substitute for the real gate proof.
