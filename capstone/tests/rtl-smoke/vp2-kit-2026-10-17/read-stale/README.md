# read-stale: a candidate mechanism for cpython's board-only `cause=12 pc=0x0`, refuted from source

**Context.** On tree A, cpython's allocator-boundary steps 17-control, 23 and 23-control each take one
`domain fault cause=12 pc=0x0 address=0x0`. Each comes right after the delegate served a read (last=0x3f), and none
appears in any QEMU record. cpyp-DZ (srevroot-zero module) removed the fetch and data retries and left pc=0
unchanged, so the retries are not its cause.

**The candidate.**
1. A domain load from a fresh anonymous page maps Linux's shared zero page.
2. The delegated read's copy_to_user then copy-on-writes it.
3. On tree A, without S-29's satp flush, the domain keeps reading the stale zero page.
4. A function pointer read as 0 would end in exactly this correct cause-12 jump to 0.

**The instrument.** `read-stale.c` converts that into a wrong-answer count: map a fresh region, touch it per arm, read()
a known pattern through the delegate, then check every byte from the domain.
- arm 1: read-touched first, the trigger;
- arm 2: write-touched first;
- arm 3: untouched.

`capstone-vexec read-stale.dom FILE ARM ROUNDS PAGES`. The pattern is byte i = 0x5a ^ (i*131) ^ (i>>12), here
64 pages.

**QEMU** (kit qkit7-rev2, module a5dec6a32c49b3d3; image 21f63031829fd3c4 from t3-freeze-nb, pattern 0ba07deaed510442),
50 rounds x 64 pages per arm: all three arms read bad_rounds=0, bad_bytes=0, exit 0. Each arm took 3,216 domain faults,
one per page per round plus the program's own: the module resolves every page at the domain's first touch, a load in
arm 1.

**Refuted from source.** The module pins every domain fault with FOLL_WRITE:
- capstone_vm.c resolve(): `pin_user_pages_remote(..., FOLL_WRITE | ...)`;
- the image grant and arena paths: pin_user_pages_fast with FOLL_WRITE, pin_user_pages_remote with FOLL_WRITE.

A load fault is therefore served through write-fault semantics: a private writable page exists before any delegated
read, so the zero page is never mapped into a domain and no copy-on-write follows. Arm 1 cannot create the condition
it was built for, so it is not sent to the board as a discriminator for this mechanism. A clean board reading from it
would say nothing.

**What is left.** cpyp-DB (tree B, pre-registered with no predicted value) tests the bitstream as one variable. The
attribution of cpython's pc=0 stays open.
