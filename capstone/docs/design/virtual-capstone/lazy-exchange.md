# Lazy exchange admission

The initial exchange grant (register 11) follows the ordinary lazy arena
policy. It does not need all 64 pages pinned at launch. A trusted Linux
writer must admit each destination before it can populate a page: Linux
faults do not enter the capability fault resolver, and process teardown can
release an unpinned page before the module clears its tags.

The delegate dispatcher validates the request, then calls `prepare_write`
for each output or in/out span before issuing the syscall. `readv` and
`recvmsg` also admit their snapshotted nested destinations; message lengths,
name, control buffer and header use the same snapshot for admission and
copy-back. Admission failure returns an error before consuming input or
receiving descriptors. Thread-name and signal-mask outputs use this path.

The virtual main transport implements admission with `CV_TOUCH`. Its cache
records only successful page admissions. Domain MAP/UNMAP/REMAP services
cannot retire or replace the initial exchange, so those entries remain
valid for the process lifetime. This cache must not be extended to a
retirable transport without invalidation tied to mapping identity.

Worker transports retain whole-span admission at every attachment,
including attachment after address reuse. Their existing mapping-ID holds
prevent retirement while a worker is attached or blocked in delegated I/O.
The signal metadata grant (register 10) remains eager: asynchronous signal
publication cannot issue a sleeping admission ioctl. Worker metadata is
admitted before signal-context initialization.

Admission protects frame lifetime and teardown, not object lifetime. The
domain still copies results through its original destination capability;
the change does not grant trusted copy-back access to a retired object.

`delegate-admission-test` exercises cross-page scalar/vector output,
message destinations and special output requests. A denied read leaves
the pipe data available for the succeeding request. Removing admission
from the dispatcher makes the protected-output test fail. Runtime,
pthread, signal and pin/teardown gates remain required for qualification.
This change is independent of internal NODES dispatch.
