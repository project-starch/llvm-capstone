#!/usr/bin/env python3
"""m2-bundle.py <out_dir> <invocations.txt> <boot_dir>...   (boot dirs: board-r1f5-b<k>, each with driver.log; or an
emulator run directory with boot.log)

The M2 (access path) bundle in the paper's record shape (METHODS.md): one record per (invocation, working-set
point) = one fresh-domain run of one case; run_id = the boot; evidence_type = fpga (or qemu for an emulator
directory). The harness's `--series chase` prints, per point, the cold first traversal and the timed accesses:
    R1 chase-cold arm=S N=256 seed=1 cyc=... per=... cycle_ok=1
    R1 chase arm=S N=256 seed=1 rep=1 loads=100000 cyc=... per100=... instret=... touched=256 minted=516 chk=... chk_ok=1
Arms: P = custom-spatial (one alias offset per record: one touched node), S = custom-sublet (a leaf and alias per
record: N touched nodes), D = data-only (no lookup ldc: the labelled non-protecting ablation). The invocation list
gives (run, arm, series, pattern, arena, extra...) per line, twelve lines per boot, and the harness's own
`R1 start` lines are matched to them in order. Every line-based read goes through the transcript module
(ISSUES M-11). The summary states the pre-registered shape test (flat-then-inflect in TOUCHED nodes).
"""
import sys, re, json, pathlib, hashlib, csv, statistics, datetime, collections
import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "tests" / "rtl-smoke"))
from fpga_driver import transcript as T

OUT = pathlib.Path(sys.argv[1]); LIST = pathlib.Path(sys.argv[2]); BOOTS = [pathlib.Path(b) for b in sys.argv[3:]]
STUDY = "M2"; SEED = 20260914; PER_BOOT = 12
ARM = {"S": "custom-sublet", "P": "custom-spatial", "D": "data-only"}
inv = [l.split() for l in LIST.read_text().splitlines() if l.strip()]

def transcript(bd):
    """The domain's lines: a framed driver.log through the module; an emulator boot.log is raw UART."""
    if (bd / "driver.log").exists():
        return T.strip_markers(T.uart_text(T.scope_to_run(T.read(bd / "driver.log"))), sq=True)
    t = T.read(bd / "boot.log").replace("\r", "")
    return T.strip_markers(t, sq=True)

def kvs(line):
    d = {}
    for m in re.finditer(r"(\w+)=(\S+)", line):
        v = m.group(2); d[m.group(1)] = int(v) if re.fullmatch(r"-?\d+", v) else v
    return d

records = []; boots_meta = []; warnings = []
for bd in BOOTS:
    m = re.search(r"b(\d+)$", bd.name); k = int(m.group(1)) if m else 1
    lines_for_boot = inv[PER_BOOT * (k - 1): PER_BOOT * k] if (bd / "driver.log").exists() else None
    text = transcript(bd)
    starts = [mm.start() for mm in re.finditer(r"^R1 start ", text, re.M)] + [len(text)]
    blocks = [text[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
    fw = (bd / "fw.sha").read_text().strip() if (bd / "fw.sha").exists() else None
    control = re.findall(r"RESULT k800 retval=(-?\d+)", text)
    boot_id = bd.name.replace("board-", "sw8x-")
    boots_meta.append(dict(boot_id=boot_id, fw_sha256=fw, control=control, invocations_seen=len(blocks),
                           invocations_planned=len(lines_for_boot) if lines_for_boot else None))
    if lines_for_boot is not None and len(blocks) != len(lines_for_boot):
        warnings.append(f"{boot_id}: {len(blocks)} harness invocations in the transcript, {len(lines_for_boot)} planned")
    for i, blk in enumerate(blocks):
        st = kvs(blk.split("\n", 1)[0])
        arm_code = st.get("arm", "?"); planned = lines_for_boot[i] if lines_for_boot and i < len(lines_for_boot) else None
        run = int(planned[0]) if planned else 1
        cold = {kvs(l)["N"]: kvs(l) for l in re.findall(r"^R1 chase-cold [^\n]*$", blk, re.M)}
        for l in re.findall(r"^R1 chase arm=[^\n]*$", blk, re.M):
            d = kvs(l)
            if not re.search(r" chk_ok=[01]$", l):
                warnings.append(f"{boot_id} invocation {i + 1}: a chase line cut before its terminal chk_ok field, {l[:80]}"); continue
            N = d["N"]; c = cold.get(N, {})
            status = "completed" if d.get("chk_ok") == 1 and c.get("cycle_ok") == 1 else "invalid-run"
            reason = None if status == "completed" else "checksum or cycle-closure check failed"
            records.append(dict(
                study_id=STUDY, manifest_id=f"{STUDY}-fpga-2026-09-15", run_id=boot_id,
                target="fpga-caplifive_r30r31_1bfff7776" if (bd / "driver.log").exists() else "qemu",
                evidence_type="fpga" if (bd / "driver.log").exists() else "qemu",
                arm=ARM.get(arm_code, arm_code), series="working-set",
                case_id=f"working-set/{ARM.get(arm_code, arm_code)}/N{N}-seed{d['seed']}",
                parameters=dict(records=N, record_bytes=64, seed=d["seed"], accesses=d["loads"], warmup_traversals=2,
                                arena=int(planned[4]) if planned else None, invocation=i + 1),
                repetition=run, boot_id=boot_id, status=status, reason=reason,
                oracle=dict(expected="chk_ok=1 cycle_ok=1", observed=f"chk_ok={d.get('chk_ok')} cycle_ok={c.get('cycle_ok')}"),
                metrics=dict(cycles=d["cyc"], cycles_per_100_accesses=d["per100"], cycles_per_access=d["cyc"] / d["loads"],
                             instret=d["instret"], instructions_per_access=d["instret"] / d["loads"],
                             nodes_touched_label=d["touched"], nodes_queried_per_access=(1 if arm_code in ("S", "P") else 0),
                             nodes_minted=d["minted"], checksum=d["chk"],
                             cold_first_traversal_cycles=c.get("cyc"), cold_first_traversal_cycles_per_access=c.get("per")),
                raw_files=[f"raw/{boot_id}-boot.txt"]))

OUT.mkdir(parents=True, exist_ok=True); (OUT / "raw").mkdir(exist_ok=True)
for bd in BOOTS:
    for name in ("boot.txt", "boot.log"):
        if (bd / name).exists():
            (OUT / "raw" / f"{bd.name.replace('board-', 'sw8x-')}-boot.txt").write_bytes((bd / name).read_bytes())
with open(OUT / "runs.jsonl", "w") as f:
    for r in records: f.write(json.dumps(r) + "\n")
cols = ["study_id", "run_id", "arm", "series", "case_id", "repetition", "status", "records", "seed", "accesses",
        "cycles", "cycles_per_access", "instructions_per_access", "nodes_touched_label", "nodes_queried_per_access", "nodes_minted", "cold_first_traversal_cycles_per_access"]
with open(OUT / "points.csv", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n"); w.writerow(cols)
    for r in sorted(records, key=lambda r: (r["arm"], r["parameters"]["records"], r["parameters"]["seed"], r["repetition"])):
        p, m = r["parameters"], r["metrics"]
        w.writerow([r["study_id"], r["run_id"], r["arm"], r["series"], r["case_id"], r["repetition"], r["status"], p["records"], p["seed"], p["accesses"],
                    m["cycles"], f"{m['cycles_per_access']:.4f}", f"{m['instructions_per_access']:.4f}", m["nodes_touched_label"], m["nodes_queried_per_access"], m["nodes_minted"],
                    m["cold_first_traversal_cycles_per_access"]])
# summary: medians per (arm, N) over seeds and runs; the pre-registered shape
by = collections.defaultdict(list)
for r in records:
    if r["status"] == "completed": by[(r["arm"], r["parameters"]["records"])].append(r["metrics"]["cycles_per_access"])
lines = [f"# {STUDY} on the FPGA (summary, generated)", "", f"boots: {[b['boot_id'] for b in boots_meta]}", f"records: {len(records)}; warnings: {len(warnings)}", "",
         "Pre-registered (before the boots): cycles per access flat in N for custom-spatial (one touched node) and data-only (no lookup",
         "ldc), and for custom-sublet below ~2,048 TOUCHED nodes (16..1024 records), higher at 4096 -- flat-then-inflect, not a slope",
         "from the first point; custom-sublet and custom-spatial carry the same instruction count per access, so their difference is",
         "the node table's footprint; data-only sits below both by the lookup ldc's own cost.", "",
         "CORRECTION (2026-09-15, written before any board number was read as a finding): the pre-registration mis-specified the",
         "quantity. The timed access is `cur = *lookup[cur]`: an LDC whose ADDRESS capability is the lookup array's (one node), then a",
         "plain load through the loaded record capability, and on this RTL the DYN unit's node-validity query is on the LDC's address",
         "capability only (Q-11: a loaded capability's node is not queried; the plain load's LSU check is gated and R-34). So the access",
         "path queries ONE node per access in both custom-spatial and custom-sublet; the harness's `touched` field is a label (N for",
         "the sublet arm), not a count of queried nodes. What the pair measures is whether the query cost depends on the number of LIVE",
         "nodes (about 2N+4 minted in the sublet arm against 6): flat means an indexed read, as the RTL says. The footprint of a growing",
         "QUERIED set is not measured by this design (it needs the record to hold the next capability, which the study excludes).", "",
         "| arm | records | queried nodes / access | live nodes (minted) | runs | median cycles/access | min | max |", "|---|---|---|---|---|---|---|---|"]
minted_by = collections.defaultdict(list)
for r in records:
    if r["status"] == "completed": minted_by[(r["arm"], r["parameters"]["records"])].append(r["metrics"]["nodes_minted"])
for (arm, N), v in sorted(by.items()):
    queried = {"custom-sublet": 1, "custom-spatial": 1, "data-only": 0}.get(arm, "?")
    mm = sorted(set(minted_by[(arm, N)])); minted = mm[0] if len(mm) == 1 else f"{mm[0]}..{mm[-1]}"
    lines.append(f"| {arm} | {N} | {queried} | {minted} | {len(v)} | {statistics.median(v):.2f} | {min(v):.2f} | {max(v):.2f} |")
if warnings: lines += ["", "warnings:"] + [f"- {w}" for w in warnings]
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
manifest = dict(study_id=STUDY, manifest_id=f"{STUDY}-fpga-2026-09-15", generated=datetime.datetime.now().isoformat(timespec="seconds"), seed=SEED,
    target=dict(board="Digilent Genesys2", core="CVA6 Capstone fork", bitstream="caplifive_r30r31_1bfff7776", clock="25 MHz constraint (timing not closed)",
                dcache="32 KiB, 8-way, 128-bit lines: 2,048 nodes of 16 bytes if nothing competes", node_table="65,536 entries, 65,532 usable, no reclamation"),
    harness=dict(source="capstone/sublet/r1/r1_slots_pools.c --series chase", image_sha256_16=hashlib.sha256((BOOTS[0] / "driver.log").read_bytes()).hexdigest()[:16] if False else None,
                 design="N 64-byte records holding the next INDEX in a seeded Sattolo cycle; capabilities in a separately counted lookup array; one access = ld of the index through the record's capability, then ldc of the next capability from the lookup array; two warm-up traversals then 100,000 timed accesses; the cold first traversal reported apart; checksum and cycle closure verified",
                 arms=dict(P="custom-spatial: every lookup entry is one alias of the region offset to its record (about 6 live nodes; one node queried per access)",
                           S="custom-sublet: every record its own object with its own alias (about 2N+4 live nodes minted; the access path still queries ONE node per access, the lookup array's -- see claim_scope)",
                           D="data-only: the same chase by integer arithmetic on one base, no lookup ldc (labelled non-protecting ablation)")),
    schedule=dict(points="working-set 16 / 64 / 256 / 1024 / 4096 records (1 KiB..256 KiB)", seeds=[1, 2, 3], runs_per_point=5, fresh_domain_per_invocation=True,
                  invocations_per_boot=PER_BOOT, sublet_invocations_per_boot_max=4, node_budget="<= 80 % of 65,532 nodes cumulatively per boot",
                  active_nodes_series="DEFERRED to M1 (repeated turnover)", memory_latency_series="simulator-only per the protocol",
                  order=f"seeded permutation, seed {SEED}, sublet invocations capped per boot"),
    claim_scope=("the record's index load carries no temporal query on this bitstream (the LSU's check for plain loads through a capability base is "
                 "M-mode-gated, and its exceptions are lost on an immediately granted access at every privilege, R-34: E1 measured stale loads retiring in a domain); the lookup's LDC runs the DYN unit's node-validity query at every "
                 "privilege level, one indexed 16-byte read with no walk -- on the LDC's ADDRESS capability, which is the lookup array's capability in "
                 "every access of both arms (Q-11: the loaded capability's node is not queried), so the series measures that one query "
                 "under a growing number of LIVE nodes (the sublet arm mints about 2N+4, the spatial arm 6), not under a growing QUERIED set; "
                 "the harness's touched field is a label, corrected 2026-09-15 before the numbers were read; "
                 "stale-data enforcement on the plain load is what it does not measure; no node-cache knee or node-size inference is made"),
    calibration=dict(dependent_load_cycles_L1=9.00, dependent_load_cycles_DRAM=48.2, timer_mcycle_read_to_read=2, source="H1 manifest, boot sw8x-e4"),
    boots=boots_meta, warnings=warnings, records=len(records))
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"{len(records)} records, {len(BOOTS)} boots, {len(warnings)} warnings -> {OUT}")
