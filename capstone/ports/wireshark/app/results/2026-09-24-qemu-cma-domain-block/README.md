# M-infra: a domain block of 64–128 MiB from CMA, and the monitor's split (QEMU)

**Why this exists.** The tshark domain comes to about 57–64 MiB (the plan, results 4b), and the
kernel module allocated a domain block with `__get_free_pages`, whose ceiling is 4 MiB.

**Where the change stands:**
- The CMA allocation itself was already written by the external collaborator
  (caplifive-buildroot `2b8ad05`) and merged there as `7440cfc` (PR #4). The parent repository
  still points at `d04bd83`.
- This folder tests `7440cfc` at tshark's sizes. It also tests one change on top, `a74a856`
  (branch `modcapstone/cma-domain-block`, pushed for review).

**Verdict (QEMU, kernel booted with `cma=1G`).**

1. **Domains past 4 MiB run.** A domain declaring 100 MiB of data gets a 128 MiB block from CMA
   (paddr `0xc0200000`, below 4 GiB) and runs: its stack is touched at both ends. That holds
   under `7440cfc` and under `a74a856`.
2. **`7440cfc` short-changes a declared domain at the edge of a power of two, and `a74a856`
   fixes it.**
   - **The test domain:** `edge64` declares 67,095,968 bytes of data, so code + 8 KiB + data
     lands 4 KiB under 64 MiB.
   - **Under `7440cfc`:** it gets a 64 MiB block. Its `dom_data` capability is
     `[0xc8220000, 0xcc200000)`, 66,977,792 bytes, which is exactly what `model.py` predicts from
     the monitor's split. The domain faults on a stack store 101,968 bytes below that base.
   - **Why:** the monitor rounds its split to a granule that is 64 KiB at this size, and the
     module's 8 KiB slack assumes at most 4 KiB.
   - **Under `a74a856`:** the module prints `declared dom_data 67095968 does not survive the
     monitor's split of order 14; doubling`, allocates 128 MiB, and the domain returns 2841
     (both touches landed).
3. **A corrupt declaration is refused, not looped on.** `a74a856` refuses:
   - a declaration of `0xffff…ffff` ("overflows");
   - one of 2^40 ("needs a block past order 20").

   A domain created after both still runs. Without the bound, which an audit asked for, an
   all-ones declaration made the doubling loop spin forever in the kernel.
4. **Blocks of 4 MiB or less are unchanged.** A small domain and one declaring 256 KiB get the
   same block, at the same address, under every module.
   - `model.py` sweeps 204,010 declared domains: the old rule is short in 833, none of them 4 MiB
     or less, and the new rule is short in none.
   - It also proves that no block of 4 MiB or less changes size.
5. **Without `cma=`,** the 128 MiB domain is refused cleanly ("Failed to allocate memory for
   domain (order 15)").

## Method

- **Build (`build.sh`):**
  - the gate domains: `touch.c`, my_first_domain ABI, `.capstone_domreq`;
  - a guest loader, `gateload.c`;
  - the three modules, out of tree from caplifive-buildroot commits: `d04bd83` as shipped,
    `7440cfc`, `a74a856`.

  Control: the `d04bd83` module built this way is byte-identical to the shipped
  `build/target/capstone.ko`, and a rebuild from these scripts reproduces every tested module and
  domain byte for byte.
- **How a domain tests `dom_data`:** my_first_domain's `start.S` takes the stack from the top of
  `dom_data`, so a frame touched at both ends tests `dom_data`'s size. It is blind to shortfalls
  under 16 KiB, the frame's margin.
- **Boot (`run.sh`):**
  - one module per boot, from a PRIVATE copy of the guest rootfs with `/capstone.ko` written in
    and read back identical; the shared `rootfs.ext2` is never written;
  - each guest prints its module's md5, and every one matches the module under test;
  - a predicted fault goes last in its boot.
- **Predictions (`predictions.txt`)** were written before the first run. Two attempts before the
  counted boots are recorded there:
  1. **The stock loader** (`/capstone-test.user`) called domain −1 after a refused creation, and
     the monitor faulted on its own table: ISSUES-ARCHIVE Q-01's shape, `capstone-test.c:31-38`.
     Replaced by `gateload.c`, which stops on −1.
  2. **A live `rmmod capstone`,** to swap modules within one boot, hung the guest. Replaced by
     one module per boot.
- **The matched pair (`edge64`, cma vs fix)** differs only in `/capstone.ko`: an audit compared
  the two rootfs images block by block. Same kernel command line, same CMA reservation, same
  paddr.
- **The counted run is `run.sh`'s,** on `build.sh`'s artifacts (`result-lines.txt`,
  `SHA256SUMS`):
  - boot B: `7440cfc`, `cma=1G`;
  - boot E: `a74a856`, `cma=1G`;
  - boot D: `a74a856`, no `cma=`.

  Its first attempt stalled before login in B and E. That is the QEMU stall class the FFmpeg rounds
  report. It happened while a heavy native build ran beside it, and a rerun on a quiet host did not
  stall. That is not evidence that load caused it: the load average was only 7–10 on 64 cores.

  **CORRECTED 2026-09-24:** the better-evidenced candidate is that the shared guest rootfs
  (`caplifive-buildroot/build/images/rootfs.ext2`) was ext4-corrupt from 13:17 today.
  - `e2fsck -n` exits 4, and every serial log since then shows `EXT4-fs error ... block bitmap
    corrupt` at boot.
  - Every boot here ran on a private copy of that image, the counted ones included.
  - The outcomes do not depend on it beyond booting. The exploratory boots that came first, including one of the fix before the audit's
  amendments, are recorded in `predictions.txt`. They agree with the counted run.

## What this does not establish

- **The board.** There the domain block would come out of the same 256 MiB reserved area as every
  region. Only the QEMU kernel (6.1) was built, not the board's 6.4.
- **gp-free (globals) images.** Every gate domain has globals offset 0. By the declaration's
  contract, `domreq.S` counts the blob the monitor copies, so a correctly declared globals image
  is covered; none was run.
- **Exactness at more than one point.** `dom_data` was measured at code_len 608, at 64 MiB.
- **`domdata-budget.py`** (llvm-capstone) still models `dom_data` as `tot − code − 1536`, without
  the granule. Above 4 MiB it will now predict a different block than the module does.

## Incidental findings

- **Buddy blocks already land above 4 GiB** on the 8 GiB QEMU guest (paddr `0x101860000`) and work.
  The worry was the monitor's arithmetic above 4 GiB, and its `unsigned` is 64-bit in the compiled
  monitor (audit).
- **A module refusal is not reported to the loader:** the failure path returns without writing
  `dom_id`. So a loader that does not check for −1 calls domain −1, and the monitor faults (Q-01).
- **The minimal domain writes a 32-bit result into a 64-bit slot,** so after earlier runs the
  loader can print `0x1_0000002A` for 42: the low half is the result.
