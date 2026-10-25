import json, re, sys, hashlib, statistics
M, tag, ref = sys.argv[1], sys.argv[2], sys.argv[3]
NATIVE_SMOKE = "a4863fc7bf3ca18217711d9df5fea63f1851bb96d6f79e1a487680ae9268450e"
frames = lambda t: re.findall(r"^0,.*[0-9a-f]{32}$", t, re.M)
nat = frames(open(f"{ref}/native.out").read())
out = {}
for prog in ("perl", "ffmpeg"):
    for arm in ("stock", "protected"):
        for rep in (1, 2, 3):
            id = f"{prog}-{arm}-{rep}"
            log = open(f"{M}/guest-{tag}-{id}/serial.log", errors="replace").read().replace("\r", "")
            sec = lambda name: re.search(rf"^MB_{name}_BEGIN\n(.*?)^MB_{name}_END$", log, re.M | re.S).group(1)
            before = json.loads(re.search(r"^MB_STATS_BEFORE:(.*)$", log, re.M).group(1))
            after = json.loads(re.search(r"^MB_STATS_AFTER:(.*)$", log, re.M).group(1))
            run = dict(kv.split("=", 1) for kv in re.search(r"^MB_RUN:(.*)$", log, re.M).group(1).split(":"))
            samples = [tuple(float(x) for x in l.split()) for l in sec("SAMPLES").splitlines() if l.strip()]
            tm = sec("TIME")
            bb_maxrss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", tm).group(1))
            elapsed = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\d+)m ([\d.]+)s", tm)
            err = sec("STDERR"); stdout = sec("STDOUT")
            vm = re.search(r"^CAPSTONE_VM_STATS (.*)$", err, re.M)
            vmstats = {k: int(v) for k, v in (kv.split("=") for kv in vm.group(1).split())} if vm else None
            ts = [s[0] for s in samples]
            gaps = [b - a for a, b in zip(ts, ts[1:])]
            r = {
                "image_sha256_guest": re.search(r"^MB_IMAGE:(\w+)$", log, re.M).group(1),
                "exit": int(run["rc"]),
                "guest_wall_s": round(float(run["end"]) - float(run["start"]), 2),
                "time_elapsed_s": round(int(elapsed.group(1)) * 60 + float(elapsed.group(2)), 2),
                "stdout_sha256_guest": run["stdout_sha256"],
                "vexec_stats_before": before, "vexec_stats_after": after,
                "vexec_stats_delta": {k: after[k] - before[k] for k in after if isinstance(after[k], int)},
                "VmHWM_kB_sampled_last": max(s[2] for s in samples),
                "VmRSS_kB_sampled_max": max(s[1] for s in samples),
                "VmRSS_kB_sampled_last": samples[-1][1],
                "VmHWM_reached_s_after_start": round(next(t for t, _, h in samples if h == max(x[2] for x in samples)) - float(run["start"]), 2),
                "VmRSS_kB_max_after_first_drop": max((s2[1] for i, s2 in enumerate(samples) if any(samples[j][1] < 0.75 * samples[j-1][1] for j in range(1, i + 1))), default=None),
                "samples": len(samples),
                "sample_interval_s": {"median": round(statistics.median(gaps), 3), "max": round(max(gaps), 3)},
                "last_sample_before_end_s": round(float(run["end"]) - ts[-1], 2),
                "busybox_time_maxrss_printed": bb_maxrss,
                "ru_maxrss_kB": bb_maxrss // 4,
                "launcher_domain_stats": vmstats,
                "stderr_other_lines": [l for l in err.splitlines() if l and not l.startswith("CAPSTONE_VM_STATS")],
                "cleanup_rmmod_rc": int(re.search(r"^MB_CLEANUP:(\d+)$", log, re.M).group(1)),
            }
            if prog == "perl":
                host_sha = hashlib.sha256(stdout.encode()).hexdigest()
                r["stdout_sha256_host"] = host_sha
                r["stdout_equals_same_answers"] = run["stdout_sha256"] == NATIVE_SMOKE and host_sha == NATIVE_SMOKE
            else:
                got = frames(stdout)
                r["frames"] = len(got)
                r["frames_equal_native"] = len(got) == 30 and got == nat
                r["stdout_non_frame_lines"] = [l for l in stdout.splitlines() if l and not re.match(r"^0,", l)]
            out[id] = r
json.dump(out, sys.stdout, indent=1)
