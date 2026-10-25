# tshark: memory behaviour of protected wmem (2026-10-11)

Pre-registered in `capstone/docs/plans/2026-10-11-protected-memory-behaviour.md` (82c1607119ec,
pushed before the runs). Wireshark row: the capture set; no allocator statistics exposed;
predicted difference "small; each block header gains two lifetimes".

The images are the two from `../2026-10-11-same-answers` (stock `9fd099f2...ec2e`, nested=none;
protected `be27d346...6d36`, nested=wireshark), on the same platform: QEMU af37cc32, launcher
3975314a, module a1c6cb6b, exact bounds, one hart.

The run used three repetitions per arm. Each repetition was one boot running all ten captures,
and all six boots ran at the same time on p12, each with its own `CAPSTONE_QEMU_LOCK`. All 60
runs exit 0, print the same stdout as native stock tshark (the ntp output as in the same-answers
run) and stderr equal to the native minimal build's, and none faults.

## What was measured, per run

1. **Allocator statistics:** wmem exposes none, so these are **not exposed** (not measured).
2. **Global counters:** `capstone-vexec --stats` immediately before and after the run (the
   module's global counters). Every field's delta is in `memory.json`.
   - After every one of the 60 runs: `nodes_live` 0, `nodes_retired` 0, `live_domains` 0 and
     `live_bytes` 0.
   - So every lifetime a run allocated had ended when its process exited.
3. **Per-process counters:** the launcher's per-process `CAPSTONE_VM_STATS` line
   (`CAPSTONE_VM_STATS=1`): lifetimes (`nodes`), lifetime-table capacity and bytes, collections,
   reclaimed, VM heap pages and peak pages, faults and steps.
4. **Peak resident memory:** busybox 1.36.1 `/usr/bin/time -f '%M'` around `capstone-vexec`.
   - busybox prints 4 x `ru_maxrss` on 4 KiB pages. The coordinator reported that the printed
     value is exactly 4 x the sampled `VmHWM` in 12 runs of another row; this run did not check
     that itself. The values below are the printed value / 4, and `memory.json` keeps both.
   - It is the kernel's `ru_maxrss` for the waited child.
   - A `/proc/<pid>/status` `VmHWM` sample of the launcher's pid would see only its small spawner,
     so it was not used.

## Results

Lifetimes per run, and which arm grows the lifetime table further. Stock never ran a collection,
and protected ran four in total.

| capture | stock lifetimes (3 reps) | protected lifetimes (3 reps) | table capacity, stock / protected | protected collections, reclaimed |
|---|---|---|---|---|
| dhcp | 14713, 14972, 14888 | 18288, 17834, 17053 | 17408 / 21504, 21504, 17408 | 0, 0, 1 (15784) |
| dns_port | 16241, 16500, 16445 | 19521, 19752, 19346 | 17408 / 21504 | 0 |
| http | 12366, 12530, 12530 | 13986, 13986, 13739 | 13312 / 17408 | 0 |
| arp | 12132, 11951, 12224 | 12628, 12720, 12786 | 13312 / 13312 | 0 |
| dhcp.flip | 15563, 14787, 15603 | 17243, 17277, 17087 | 17408 / 17408 | 1, 1, 1 (14728, 14072, 15800) |
| dns_port.flip | 15125, 15385, 15494 | 18847, 18849, 18792 | 17408 / 21504 | 0 |
| http.flip | 12501, 12705, 12524 | 13788, 13811, 13759 | 13312 / 17408 | 0 |
| arp.flip | 11945, 11877, 11862 | 12815, 12633, 12803 | 13312 / 13312 | 0 |
| ntp | 12192, 12192, 11947 | 13217, 13217, 13200 | 13312 / 17408 | 0 |
| dns-ooo | 14626, 15794, 15765 | 18044, 16905, 16747 | 17408 / 21504, 17408, 17408 | 0 |

**Lifetimes per boot** (all ten captures):
- stock 137,404, 138,693 and 139,282;
- protected 158,377, 156,984 and 155,312;
- that is +18,431 on the means (+13.3 %).

**Memory footprint:**
- **VM heap peak pages:** paired by repetition, protected minus stock is -1, 0 or +1 page
  (4 KiB) on every capture, with no consistent sign. The ranges are 11,136 to 11,184 pages in
  both arms.
- **Peak resident memory:** stock 53,928 to 54,000 KB and protected 53,932 to 54,004 KB, over all
  30 runs of each. That is the same band, with no consistent difference.
- **Lifetime table:** this is the one footprint difference that shows. Protected grows the table
  one step further, to 17,408 or 21,504 entries against 13,312 or 17,408:
  - on 5 captures in every repetition (http, http.flip, ntp, dns_port, dns_port.flip);
  - on dhcp in 2 of 3 repetitions and on dns-ooo in 1 of 3.

  Each step is +65,536 bytes of `node_bytes`: 225,280 to 290,816, or 290,816 to 356,352.

## Against the predictions

- **"Small footprint difference (each block header gains two lifetimes)": as predicted.**
  - Patch 0001 adds `{void *life; void *generation;}`, two 16-byte capabilities, to every block
    header of `block` and `block_fast`. That is 32 bytes per block; the size comes from the
    source and was not measured.
  - At the level these instruments resolve (4 KiB VM pages; RSS to its run-to-run band of
    76 KB), no footprint difference is visible.
  - The difference that is visible is the processor's lifetime table: one more growth step
    (+64 KiB of node storage) on 5 captures always and on 2 more sometimes.
- **"Protected allocates more lifetimes": as predicted.** Protected allocated more on every
  capture in every repetition, between +5.0 % (arp) and +22.8 % (dns_port.flip) over the stock
  means. Lifetimes per wmem hand-out cannot be computed here because wmem exposes no counts; the
  reuse-distance row counts hand-outs.
- **"...and reclaims them": partly.**
  - All extra lifetimes end, and no lifetime is live after any run.
  - A collection ran in only 4 of 30 protected runs (all three dhcp.flip and one dhcp), each
    reclaiming 14,072 to 15,800 nodes. Stock ran none.
  - In the other runs, the lifetimes ended with the process, not through a collection.
- **Identical across repetitions: no, for lifetimes; yes, for output.**
  - Lifetime counts vary across repetitions in both arms. Stock dhcp, for example, reads 14,713,
    14,972 and 14,888, with the same image and input.
  - Peak pages vary by one page, and RSS by up to 76 KB.
  - Stdout and stderr are identical in every repetition.
  - Where the run-to-run variation in stock comes from was not investigated.

## Commands

```sh
# per arm A in {stock, protected}, repetition R in {1,2,3}; all six at once
LD_LIBRARY_PATH=$K/toolchain/hostlib CAPSTONE_QEMU_LOCK=~/sa-ws/qemu-mem-$A-$R.lock \
python3 run-memory.py --kit $K/kit-sublet --repo ~/sa-ws/repo --image ~/sa-ws/build-$A/image \
  --captures <pristine 4.6.8>/test/captures --work ~/sa-ws/mem/$A-$R
```

- Runner: `run-memory.py` in this directory. Its sha256, the gate script's and all per-run
  numbers are in `memory.json`.
- Wall time: about 470 s per boot, with the six boots running in parallel.

No serial log is kept.
