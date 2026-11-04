# Physical-page lifecycle double, version 1

`python3 page_double.py` accepts one JSON object per line and emits one result
per line. This is an executable integration double, not an authority-granting
API, kernel implementation, or performance model. `python3 test_page_double.py`
checks it, including implementations with clearing disabled and scalar copying
substituted. Both broken implementations must fail their named oracle.

Every request carries `"version": 1`. Operations:

| op | Inputs | Result |
|---|---|---|
| allocate | frame: aligned physical byte address | page handle `[frame, epoch]` |
| query | page; optional ack | sticky flag, or nonempty result after verified-empty acknowledgement |
| copy | to, from: distinct page handles | exact bytes and opaque live/stale tag records copied |
| clear | page | every byte and tag zeroed; summary stays sticky until ack |
| release | page | query, clear only when flagged, verify empty, invalidate handle before reuse |
| write | page, offset, hex; optional tag record | fixture publication; scalar writes clear overlapping tags |
| dma | page, begin: boolean | fixture writable-device ownership transition |

Responses carry `ok` and `result`, or `ok: false` and `cause`. Unknown versions,
operations, stale handles, overlapping page copies and out-of-bounds writes
fail closed. Fixtures may publish arbitrary tag records because this is a
trusted physical-memory double; their fields never become usable authority.

Clear/copy/release require exclusive page ownership and no writable DMA
mapping. The allocator or migration owner supplies that exclusion. Copy
preserves the source and the destination's conservative summary state. Raw
copy must preserve stale tags too: namespace/liveness decisions belong to the
lifetime owner, and may not be inferred from the representation in this layer.
The model does not implement mapping transactions, namespace fork, revocation,
signals, user-pointer validation, SMP, swap, KSM or huge pages. Requests for
those operations are refused. These boundaries let the fault and lifecycle
lanes exercise their own doubles against physical page copy/clear/query.

Kernel counterparts are `copy_highpage`, `copy_user_highpage`, the zeroing
primitive, `arch_free_page`, and the physical CPTAG query/acknowledge operation
in `../linux-6.1-unpinned.patch`. A clear must complete before any allocator
reuse, including reuse for a kernel object. Anonymous-page initialization
alone cannot provide that ordering. Ordinary scalar and DMA writes clear
actual tags but do not clear the sticky CPTAG flag.
