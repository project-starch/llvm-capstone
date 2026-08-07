#!/usr/bin/env python3
"""Parse the CHERI binary-trees serial log into calibrated overhead tables.

Each config block emits n CAL runs (empty `lua`, startup+linking only) and n RUN
runs (the benchmark). Two axes:

  instructions : workload = mean(RUN instrs) - mean(CAL instrs); overhead = cfg/spatial
  memory       : peak RSS (KB) of the benchmark child (getrusage RUSAGE_CHILDREN in
                 runbench). We report both the RAW RUN peak and the calibrated
                 workload delta (RUN - CAL, i.e. above the empty-interpreter peak);
                 the temporal-safety memory cost is temporal - spatial.

  parse-bench.py <serial.log>
"""
import re, sys


def main():
    if len(sys.argv) != 2:
        sys.stderr.write("usage: parse-bench.py <serial.log>\n")
        return 2
    log = open(sys.argv[1], encoding="latin-1", errors="replace").read()
    blocks = re.split(r"==CFG (\w+)", log)
    mean = lambda x: sum(x) / len(x)
    wl, rss_run, rss_wl = {}, {}, {}
    print(f"  {'config':9} {'CAL instr':>13} {'RUN instr':>14} {'workload':>13}  "
          f"{'RUN RSS KB':>11} {'RSS-CAL KB':>11}  reps")
    for i in range(1, len(blocks), 2):
        cfg = blocks[i]
        body = blocks[i + 1].split("END cfg=")[0]
        cal_i = [int(m) for m in re.findall(r"CAL BENCH instrs=(\d+)", body)]
        all_i = [int(m) for m in re.findall(r"BENCH instrs=(\d+)", body)]
        run_i = all_i[len(cal_i):]
        cal_r = [int(m) for m in re.findall(r"CAL BENCH instrs=\d+ rc=\S+ maxrss_kb=(\d+)", body)]
        all_r = [int(m) for m in re.findall(r"maxrss_kb=(\d+)", body)]
        run_r = all_r[len(cal_r):]
        if not (cal_i and run_i):
            print(f"  {cfg:9} (incomplete)"); continue
        wl[cfg] = mean(run_i) - mean(cal_i)
        line = f"  {cfg:9} {mean(cal_i):>13,.0f} {mean(run_i):>14,.0f} {wl[cfg]:>13,.0f}"
        if cal_r and run_r:
            rss_run[cfg] = mean(run_r)
            rss_wl[cfg] = mean(run_r) - mean(cal_r)
            line += f"  {rss_run[cfg]:>11,.0f} {rss_wl[cfg]:>11,.0f}"
        else:
            line += f"  {'n/a':>11} {'n/a':>11}"
        print(f"{line}   n={len(run_i)}")

    if "spatial" in wl:
        b = wl["spatial"]
        print("\n  instruction overhead vs spatial:")
        for cfg in ("temporal", "eager"):
            if cfg in wl:
                print(f"    {cfg:9} = {wl[cfg] / b:.3f}x   (+{wl[cfg]-b:,.0f} workload instr)")
    if "spatial" in rss_run:
        b = rss_run["spatial"]
        print("\n  memory (peak RSS) overhead vs spatial:")
        for cfg in ("temporal", "eager"):
            if cfg in rss_run:
                print(f"    {cfg:9} = {rss_run[cfg]/b:.3f}x   (+{rss_run[cfg]-b:,.0f} KB peak RSS = the temporal-safety memory cost)")
    print("\n  Note: rdinstret counts ALL hart activity in the bracket; peak RSS is the")
    print("  whole child process (interpreter + libc + revocation quarantine/bitmap).")
    print("  spatial safety is ALWAYS on; configs differ only in the temporal layer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
