#!/usr/bin/env python3
"""r1-bundle.py <out_dir> <invocations.txt> <boot_dir>...   (boot dirs: board-r1-b<k>, each with boot.txt, driver.log, fw.sha)

The R1 bundle in the paper's record shape (METHODS.md): one record per (invocation, point) = one
repetition of one case, run_id = the boot, evidence_type = fpga; metrics are the harness's exclusive
cycle intervals; the node-pool position of each repetition is the boot's cumulative minted nodes
before its invocation, from the harness's own `R1 end` lines (the controls and the readback host mint
a handful more, not counted: stated in the manifest).
"""
import sys, re, json, pathlib, hashlib, csv, statistics, datetime

OUT = pathlib.Path(sys.argv[1]); LIST = pathlib.Path(sys.argv[2]); BOOTS = [pathlib.Path(b) for b in sys.argv[3:]]
STUDY = "R1"; SEED = 20260914

inv = [l.split() for l in LIST.read_text().splitlines() if l.strip()]          # rep arm series pattern arena

def reassemble(t):
    """R1 lines split by UART chunking: a point line ends at ' ok=N'; other R1 lines are single."""
    flat = t.replace("\r", ""); lines = []; buf = None
    for l in flat.split("\n"):
        if buf is not None:
            buf += l
            if re.search(r" ok=[01]$", buf): lines.append(buf); buf = None
            continue
        if l.startswith("R1 s=") and not re.search(r" ok=[01]$", l): buf = l; continue
        if l.startswith("R1 "): lines.append(l)
    return lines

def kvs(line):
    d = {}
    for k, v in re.findall(r"(\w+)=(\S+)", line): d[k] = int(v) if v.isdigit() else v
    return d

records = []; boots_meta = []; warnings = []
for bd in BOOTS:
    k = int(re.search(r"b(\d+)$", bd.name).group(1))
    boot_id = f"sw8x-r1-b{k}"
    t = (bd / "boot.txt").read_bytes().decode("latin1") if (bd / "boot.txt").exists() else ""
    fw = (bd / "fw.sha").read_text().strip() if (bd / "fw.sha").exists() else None
    i = t.rfind("load_image"); t = t[i:] if i >= 0 else t
    lines = reassemble(t)
    # walk the transcript invocation by invocation: each starts at 'R1 start' and ends at 'R1 end'
    my = inv[12 * (k - 1): 12 * k]
    starts = [j for j, l in enumerate(lines) if l.startswith("R1 start")]
    if len(starts) != len(my): warnings.append(f"{boot_id}: {len(starts)} R1 start lines for {len(my)} scheduled invocations")
    minted_before = 0; controls = re.findall(r"RESULT k800 retval=(-?\d+)", t)
    ran = re.findall(r"speedtest1-ran=(\d+)", t); released = re.findall(r"released pool rc=(\d+)", t)
    for n, j in enumerate(starts):
        end = next((m for m in range(j + 1, len(lines)) if lines[m].startswith("R1 end") or lines[m].startswith("R1 start")), len(lines))
        block = lines[j:end + 1] if end < len(lines) and lines[end].startswith("R1 end") else lines[j:end]
        st = kvs(block[0]); rep, arm, ser, pat, arena = (my[n] if n < len(my) else ("?", st.get("arm", "?"), st.get("series", "?"), st.get("pattern", "?"), "?"))
        endl = kvs(block[-1]) if block[-1].startswith("R1 end") else {}
        pts = [kvs(l) for l in block if l.startswith("R1 s=")]
        refused = [l for l in block if l.startswith("R1 refused")]
        declared = endl.get("lines"); got = len(block) + (1 if j > 0 and lines[j - 1].startswith("R1 entry") else 0)   # the harness counts its `R1 entry` line too
        for p in pts:
            case = f"{ser}/{pat}/{arm}/n{p['n']}-B{p['B']}-U{p['U']}-S{p['S']}-c{p['chain']}"
            records.append(dict(
                study_id=STUDY, manifest_id=f"{STUDY}-fpga-2026-09-15", run_id=boot_id, target="fpga-caplifive_r30r31_1bfff7776",
                evidence_type="fpga", arm={"S": "custom-sublet", "P": "custom-spatial"}[p["a"]], series=ser, case_id=case,
                parameters=dict(pattern=pat, n=p["n"], B=p["B"], U=p["U"], S=p["S"], chain=p["chain"], arena=int(arena) if str(arena).isdigit() else arena,
                                touch_unrelated=st.get("touch", 1), image="6a569a7e5e34178b", host="2c9e82d101b48160"),
                repetition=int(rep) if str(rep).isdigit() else rep, boot_id=boot_id,
                status="completed" if p["ok"] == 1 and p["bad"] == 0 else "invalid-run",
                reason="" if p["ok"] == 1 and p["bad"] == 0 else f"ok={p['ok']} bad={p['bad']}",
                oracle=dict(emulator_sweep="e3/oracle2 (reps=2, every combination ok)", nodes_expected=(2 * p["n"] if pat != "individual" else None)),
                metrics=dict(nodes_minted=p["nd"], cycles_bookkeeping=p["bk"], cycles_revoke=p["rv"], cycles_fill=p["fl"], cycles_init=p["in"],
                             cycles_reissue=p["re"], cycles_total=p["tt"], instret_total=p["ir"], returned_type=p["ty"], init_ran=p["ini"],
                             fill_bytes=p["fb"], inner_free_cycles=p["inf"], inner_reissue_cycles=p["inr"], teardown_cycles=p["td"],
                             survivors_bad=p["bad"], node_pool_position=minted_before),
                raw_files=[f"raw/{boot_id}-boot.txt"]))
        minted_before += endl.get("split", 0) + endl.get("mrev", 0)
        if refused: warnings.append(f"{boot_id} invocation {n+1} ({arm} {ser} {pat}): {refused[0]}")
        if declared is not None and declared != got: warnings.append(f"{boot_id} invocation {n+1}: harness declared {declared} lines, transcript reassembled {got}")
    boots_meta.append(dict(boot_id=boot_id, fw_sha256=fw, controls=controls, invocations=len(starts), ran_codes=len(ran),
                           released_rc=sorted(set(released)), boot_banners=max(t.count("OpenSBI v"), t.count("Linux version"))))
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    if (bd / "boot.txt").exists(): (OUT / "raw" / f"{boot_id}-boot.txt").write_bytes((bd / "boot.txt").read_bytes())

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "runs.jsonl", "w") as f:
    for r in records: f.write(json.dumps(r) + "\n")
cols = ["study_id", "run_id", "arm", "series", "case_id", "repetition", "status", "n", "B", "U", "S", "chain", "pattern", "nodes_minted", "node_pool_position",
        "cycles_bookkeeping", "cycles_revoke", "cycles_fill", "cycles_init", "cycles_reissue", "cycles_total", "instret_total", "fill_bytes", "returned_type", "init_ran",
        "inner_free_cycles", "inner_reissue_cycles", "teardown_cycles", "survivors_bad"]
with open(OUT / "points.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(cols)
    for r in records:
        w.writerow([r["study_id"], r["run_id"], r["arm"], r["series"], r["case_id"], r["repetition"], r["status"], r["parameters"]["n"], r["parameters"]["B"],
                    r["parameters"]["U"], r["parameters"]["S"], r["parameters"]["chain"], r["parameters"]["pattern"]] + [r["metrics"][c] for c in cols[13:]])
manifest = dict(study_id=STUDY, manifest_id=f"{STUDY}-fpga-2026-09-15", generated=datetime.datetime.now().isoformat(timespec="seconds"), seed=SEED,
                target=dict(board="Genesys2 CVA6 Capstone", bitstream="caplifive_r30r31_1bfff7776", monitor="4274268", module="d04bd83", core_mhz=25, timebase_mhz=12.5),
                harness=dict(source="capstone/sublet/r1/r1_slots_pools.c", image_sha256_16="6a569a7e5e34178b", opt="-O1", entry="0x410000", host="sqlite_host_rr.user 2c9e82d101b48160",
                             protocol="sqlite_host --speedtest1 --arena <bytes> --tables 65536; one invocation = one fresh domain and one fresh arena grant"),
                schedule=dict(rule="METHODS: a repetition is a fresh domain; five repetitions per point over at least three boots",
                              layout="90 invocations = 5 repetitions x 18 (arm x series x pattern), repetition-major, twelve per boot, eight boots",
                              covariate="the revocation-node head is monotonic within a boot and never reclaimed; each record carries node_pool_position = the boot's cumulative nodes minted by earlier harness invocations (the two k800 controls and the readback probe mint a handful more, not counted)",
                              per_boot_limit="at most 12 region-bearing invocations per boot: the module's region slots are never reclaimed within a boot (ISSUES M-9)"),
                counters="mcycle and minstret read in the domain (PRV_C); exclusive intervals: bookkeeping | REVOKE | the fill an UNINIT region asks for | INIT | reissue up to a checked load and store; cycles_total is the outer bracket",
                boots=boots_meta, warnings=warnings, records=len(records))
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
# summary: per case, the median of cycles_total and the components over repetitions
by = {}
for r in records:
    if r["status"] != "completed": continue
    by.setdefault((r["arm"], r["series"], r["parameters"]["pattern"], r["case_id"]), []).append(r["metrics"])
lines = [f"# {STUDY} on the FPGA (summary, generated)", "", f"boots: {[b['boot_id'] for b in boots_meta]}", f"records: {len(records)}; warnings: {len(warnings)}", "",
         "| arm | series | pattern | case | reps | nd | bk | rv | fl | in | re | total (median) | fill bytes |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for (arm, ser, pat, case), ms in sorted(by.items()):
    med = lambda k: int(statistics.median(m[k] for m in ms))
    lines.append(f"| {arm} | {ser} | {pat} | {case} | {len(ms)} | {med('nodes_minted')} | {med('cycles_bookkeeping')} | {med('cycles_revoke')} | {med('cycles_fill')} | {med('cycles_init')} | {med('cycles_reissue')} | {med('cycles_total')} | {med('fill_bytes')} |")
lines += ["", "warnings:"] + [f"- {w}" for w in warnings]
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
print(f"{len(records)} records, {len(boots_meta)} boots, {len(warnings)} warnings -> {OUT}")
for w in warnings: print("  warn:", w)
