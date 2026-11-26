# The protected-layout msghdr loader (runtime lane, an add-on to the trap-route series at 6d4504f8b073)

`sublet-msghdr-loader.patch` admits `sendmsg` and `recvmsg` for a protected (Sublet) task and reads its `msghdr`
in the protected layout, where pointers are 16-byte references: name 0x00, namelen 0x10, iov 0x20, iovlen 0x30,
control 0x40, controllen 0x50, flags 0x58, size 0x60 (iovec 0x20, mmsghdr 0x70). The three pointers load through
`sublet_load_user_pointer`, so each one is checked as a reference; the integer fields are read as u32. Files:
`asm/sublet-uaccess.h` (offsets, prototype), `kernel/sublet/uaccess.c` (`sublet_copy_msghdr_from_user`),
`kernel/sublet/loader.c` (admits the two calls), `net/socket.c` (`copy_msghdr_from_user` and `____sys_recvmsg`
branch on TIF_CAPSTONE). The patch applies with `patch -p1` on a tree that has the series, and reproduces the twin
kernel's four files byte for byte (checked against the 749fed26 tree).

**Directed test** (`msgt.c`, `init-msgt`): one twin boot on Image 749fed2606e3ebca, msgt.dom 2c6c30f79567f050,
msgt.native cba6b00ab1cb68b4. Native and protected print the same lines and both end rc 0, with no `sublet:` fault
or call line:

    MSGT sendmsg 11
    MSGT recvmsg 11 x=hell y=o world flags=0 controllen=24 namelen=0
    MSGT passed-fd write 1 read 1 z=z
    MSGT trunc 3 flags=0x20
    MSGT done fails=0

The two-element iov split (4 + 7 bytes), the SCM_RIGHTS descriptor and MSG_TRUNC each depend on a field read at its
protected offset. A layout error would garble the split or the control length. **Not run:** a negative control
on a loader-free kernel, `sendmmsg`/`recvmmsg`, and msgt itself on silicon.

**First silicon reading** (nested lane, boot nested-cpymem, board 2 on the full seal b42c8c644, 2026-11-26; result
at nested-workloads 098cae0f858b):
- The payload is `ccecad78bfcb78b0…`, with board kernel Image `f97b98f1…`.
- The nested pre-registration records that Image as relinked from `trapk/fpga/linux-6.1`, the tree that carries this
  patch. That provenance is the registration's: the Image is stripped and has no string unique to the patch.
- memcached ran as a server and sends its replies with `sendmsg` (memcached.c:137, 2844). Both arms are protected
  processes (`STAT pointer_size 128`; "stock" names the allocator), so both go through this loader. Three
  repetitions per arm each gave a whole 1,931,245 B transcript, all six byte-identical. None ended in `<EOF>` or
  `<TIMEOUT>`, and no `sublet:` fault line appeared.
- The negative control is the nested lane's, taken on the twin at the workload: the same images on a loader-free
  Image (`4fc06dde…`, run twin-all) read 0 of 6, every response `<EOF>`, 4,905 B transcripts.
- The two lines in which the transcripts differ from native's come from the board's clock, which starts at the
  epoch (no RTC). The twin reproduces this with `-rtc base=1970-01-01`, and protection plays no part in it.
- What this reading exercises is the plain iov path of sendmsg and recvmsg. SCM_RIGHTS and MSG_TRUNC are read on
  the twin only.

**Kernels that carry it:** the twin kernel 749fed26, and the board tree `trapk/fpga/linux-6.1` from 2026-11-25
13:33Z (the nested corpus kit, board Image e959b80d). The SPEC images built before then, and the pinned tree
`trapk/fpga-v1`, do not.
