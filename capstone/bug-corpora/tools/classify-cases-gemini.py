#!/usr/bin/env python3
"""Classify all declared corpus cases with independent Gemini reviews."""
import argparse, asyncio, json, os, re
from collections import Counter
from datetime import date
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
CORPORA = TOOLS.parent
TMP = Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone")) / "gemini-corpus-classification"
OUT = CORPORA / "classification.json"
AXES = {"spatial", "temporal", "both", "other-memory-safety", "not-memory-safety", "uncertain"}
CWE_RE = re.compile(r"^CWE-[1-9][0-9]*$")
KEEP = ("case", "upstream_fix", "title", "consumer", "class", "taxonomy_class", "object",
        "lifetime_ender", "shape", "allocator_layer", "fidelity", "live_in_pin", "live_proof",
        "status", "note", "distinguishing", "memory_error_is_the_defect",
        "memory_error_is_the_defect_why", "withdrawn", "withdrawn_why", "duplicate_of")
MANUAL = {
    "cpython/allocator-boundary-repros/03_gh-140492_heap-buffer-overflow-in-ast-copy":
        ("spatial", "CWE-843", ["CWE-125"],
         "Type confusion is the root cause, while the independently measured heap-buffer over-read is spatial."),
    "cpython/allocator-boundary-repros/04_gh-140492_heap-buffer-overflow-in-ast-copy-large-buffer":
        ("spatial", "CWE-843", ["CWE-125"],
         "Type confusion is the root cause, while the independently measured heap-buffer over-read is spatial."),
    "mruby/release-differential/17_bef45e223_pat-values-stale-ptr-and-len":
        ("both", "CWE-416", ["CWE-125"],
         "The record independently identifies a freed-buffer read and a stale length indexing past the shorter replacement array."),
    "perl/release-differential/06_40727c420c_code-blocks-count-reused":
        ("other-memory-safety", "CWE-457", [],
         "The dated case audit retracts the old temporal label: cleanup walks never-written entries and dereferences allocator-fill garbage."),
    "sqlite/capi-repros/row8_cpython_backup_closed":
        ("temporal", "CWE-672", ["CWE-476"],
         "Use-after-close is a resource-lifetime violation; the faithful null dereference is its manifestation."),
}

SYSTEM = r'''You review real software defects for a research corpus. Return ONLY a JSON array,
one object per supplied case, in the same order. Never use tools. Each object must be exactly:
{"id":string,"memory_axis":"spatial|temporal|both|other-memory-safety|not-memory-safety|uncertain",
"root_cwe":"CWE-NNN" or null,"manifestation_cwes":["CWE-NNN",...],
"confidence":"high|medium|low","evidence_facts":[string,...],"rationale":string}

spatial = access outside the intended live object's extent. temporal = operation/dereference
outside logical lifetime, including UAF/UAR/use-after-close, double/invalid free, stale alias,
and reuse without allocator free. both requires two independently evidenced violations; a UAF
that later goes OOB is temporal, not both. other-memory-safety covers uninitialized values, type
confusion, null dereference, stack exhaustion, etc. not-memory-safety means the defect itself has
no memory-safety violation. uncertain means evidence is insufficient.

Classify the upstream root cause, not the harness, an arm named spatial, or the crash symptom.
root_cwe is causal; manifestation_cwes are resulting accesses. Integer overflow causing OOB:
root CWE-190, manifestation CWE-125/787. Common mappings: UAF CWE-416; double free CWE-415;
stack escape/UAR CWE-562; invalid non-heap free CWE-590; expired resource/use-after-close CWE-672;
OOB read CWE-125; OOB write CWE-787; uninitialized CWE-457; null dereference CWE-476;
type confusion CWE-843; resource exhaustion CWE-400; recursion CWE-674. Use null rather than
inventing. Evidence priority: upstream provenance/code > sanitizer oracle > structured fields > title.'''

ADJ = '''You are the final adjudicator of a software-defect classification. Return ONLY one JSON
object with exactly these keys: id, memory_axis, root_cwe, manifestation_cwes, confidence,
evidence_facts, rationale. memory_axis is one of spatial, temporal, both, other-memory-safety,
not-memory-safety, uncertain. root_cwe is CWE-NNN or null; manifestation_cwes is a list of CWE-NNN;
confidence is high, medium, or low. Resolve the two supplied votes from the supplied case evidence.
Spatial means outside a live object's extent. Temporal means outside logical lifetime. Both needs
two independent violations, not a UAF whose symptom is OOB. Classify root cause, not crash symptom.
Prefer provenance/code, then sanitizer oracle, then structured fields. Do not compromise between
votes; choose the supported answer or uncertain/null. Never use tools. Copy the id byte-for-byte.'''

def load_dotenv(path, env):
    if not path.is_file(): return
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#") or "=" not in s: continue
        k, v = s.split("=", 1); k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'": v = v[1:-1]
        env.setdefault(k, v)

def declared_cases():
    rows = []
    for cp in sorted(CORPORA.rglob("corpus.json")):
        decl = json.loads(cp.read_text()); corpus = cp.parent
        pattern = "row[0-9]*" if decl["case_schema"] == "sqlite-row" else "[0-9][0-9]_*"
        dirs = [p for p in sorted(corpus.glob(pattern)) if p.is_dir()]
        if len(dirs) != decl["cases"]: raise SystemExit(f"{corpus}: says {decl['cases']}, found {len(dirs)}")
        for d in dirs:
            case = json.loads((d / "case.json").read_text())
            rel = corpus.relative_to(CORPORA).as_posix()
            rows.append((f"{rel}/{d.name}", decl, d, case))
    if len(rows) != 425: raise SystemExit(f"expected 425 declared cases, got {len(rows)}")
    return rows

def clipped(path, limit):
    if not path.is_file(): return None
    s = path.read_text(errors="replace")
    return s if len(s) <= limit else s[:limit] + "\n[truncated]"

def payload(row):
    cid, decl, d, case = row
    oracle = None
    if (d / "oracle.json").is_file():
        raw = json.loads((d / "oracle.json").read_text())
        oracle = {k: raw[k] for k in ("class", "category", "cwe", "population") if k in raw}
    source = d / case.get("trigger", "case.c")
    return {"id": cid, "program": decl["program"], "corpus_boundary": decl["boundary"],
            "schema": decl["case_schema"], "case": {k: case[k] for k in KEEP if k in case},
            "sanitizer_oracle": oracle, "provenance": clipped(d / "PROVENANCE.md", 6500),
            "trigger_or_reduction": clipped(source, 4500)}

def validate(v, cid):
    req = {"id","memory_axis","root_cwe","manifestation_cwes","confidence","evidence_facts","rationale"}
    if not isinstance(v, dict) or set(v) != req or v.get("id") != cid: raise ValueError(f"bad shape/id {cid}")
    if v["memory_axis"] not in AXES or v["confidence"] not in {"high","medium","low"}: raise ValueError(f"bad enum {cid}")
    if v["root_cwe"] is not None and not CWE_RE.fullmatch(v["root_cwe"]): raise ValueError(f"bad CWE {cid}")
    if not isinstance(v["manifestation_cwes"], list) or any(not CWE_RE.fullmatch(x) for x in v["manifestation_cwes"]): raise ValueError(f"bad manifestations {cid}")
    if not isinstance(v["evidence_facts"], list): raise ValueError(f"bad evidence {cid}")
    return v

def parse_output(raw):
    outer = json.loads(raw); response = outer.get("response", outer) if isinstance(outer, dict) else outer
    if isinstance(response, str):
        s = response.strip()
        if s.startswith("```"): s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
        return json.loads(s)
    return response

async def call(prompt, stdin, env, sem):
    error = ""
    async with sem:
        for attempt in range(4):
            p = await asyncio.create_subprocess_exec("gemini", "--skip-trust", "--approval-mode", "plan",
                "--output-format", "json", "-p", prompt, cwd=TMP, env=env,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, err = await p.communicate(stdin.encode())
            try:
                if p.returncode: raise ValueError(f"exit {p.returncode}")
                return parse_output(out)
            except Exception as exc:
                error = f"{exc}; stderr tail: {err.decode(errors='replace')[-300:]}"
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(error)

async def pass_batches(rows, name, env, concurrency, batch_size):
    outdir = TMP / name; outdir.mkdir(parents=True, exist_ok=True); sem = asyncio.Semaphore(concurrency)
    async def one(n, batch):
        checkpoint = outdir / f"{n:04d}.json"
        if checkpoint.is_file(): votes = json.loads(checkpoint.read_text())
        else:
            suffix = "\nBe conservative; prefer uncertain to unsupported specificity." if name == "pass-b" else "\nDistinguish root cause from manifestation."
            error = None
            for retry in range(3):
                try:
                    votes = await call(SYSTEM + suffix +
                        "\nCopy every id byte-for-byte; do not shorten or normalize it.",
                        json.dumps([payload(r) for r in batch]), env, sem)
                    if not isinstance(votes, list) or len(votes) != len(batch):
                        raise ValueError(f"batch {n} length")
                    votes = [validate(v, r[0]) for v, r in zip(votes, batch)]
                    break
                except (ValueError, json.JSONDecodeError) as exc:
                    error = exc
            else:
                raise ValueError(f"batch {n} invalid after retries: {error}")
            checkpoint.write_text(json.dumps(votes, indent=2) + "\n")
        return votes
    groups = await asyncio.gather(*(one(n, rows[i:i+batch_size]) for n,i in enumerate(range(0,len(rows),batch_size))))
    return {v["id"]: v for g in groups for v in g}

def control(row):
    _, _, d, case = row; cls = str(case.get("class", "")).lower(); axis = None
    if cls.startswith("spatial"): axis = "spatial"
    elif cls in {"temporal","uaf","use-after-close","double-free"}: axis = "temporal"
    elif cls == "both": axis = "both"
    cwe = json.loads((d/"oracle.json").read_text()).get("cwe") if (d/"oracle.json").is_file() else None
    return axis, cwe

async def run(args):
    TMP.mkdir(parents=True, exist_ok=True); env = dict(os.environ); load_dotenv(Path.home()/".env", env)
    if not env.get("GOOGLE_API_KEY") and not env.get("GEMINI_API_KEY"): raise SystemExit("no Gemini key")
    rows = declared_cases(); duplicates = {}; withdrawn = {}; review = []
    for row in rows:
        cid, _, _, case = row
        if case.get("duplicate_of"): duplicates[cid] = f"{cid.rsplit('/',1)[0]}/{case['duplicate_of']}"
        elif case.get("withdrawn"): withdrawn[cid] = case.get("withdrawn_why", "withdrawn")
        else: review.append(row)
    a,b = await asyncio.gather(pass_batches(review,"pass-a",env,args.concurrency,args.batch_size), pass_batches(review,"pass-b",env,args.concurrency,args.batch_size))
    final={}; conflicts=[]
    for row in review:
        cid=row[0]; ca,cc=control(row); same=(a[cid]["memory_axis"],a[cid]["root_cwe"])==(b[cid]["memory_axis"],b[cid]["root_cwe"])
        if cc:
            chosen=dict(a[cid]); chosen["memory_axis"]=ca or chosen["memory_axis"]; chosen["root_cwe"]=cc; status="existing-authoritative"
        elif same and (not ca or a[cid]["memory_axis"]==ca): chosen=dict(a[cid]); status="model-agreement"
        else: conflicts.append(row); continue
        final[cid]={**chosen,"review_status":status}
    sem=asyncio.Semaphore(args.concurrency)
    async def adjudicate(row):
        cid=row[0]
        adjdir=TMP/"adjudication"; adjdir.mkdir(parents=True,exist_ok=True)
        checkpoint=adjdir/(re.sub(r"[^A-Za-z0-9_.-]+","__",cid)+".json")
        if checkpoint.is_file():
            v=validate(json.loads(checkpoint.read_text()),cid)
        else:
            error=None
            for retry in range(3):
                try:
                    v=await call(ADJ,
                        json.dumps({"case":payload(row),"vote_a":a[cid],"vote_b":b[cid],"existing_control":control(row)}),env,sem)
                    v=validate(v,cid); break
                except (ValueError,json.JSONDecodeError) as exc: error=exc
            else: raise ValueError(f"adjudication invalid after retries for {cid}: {error}")
            checkpoint.write_text(json.dumps(v,indent=2)+"\n")
        ca,_=control(row); status="needs-review" if (v["confidence"]=="low" or (ca and v["memory_axis"]!=ca)) else "adjudicated"
        final[cid]={**v,"review_status":status}
    await asyncio.gather(*(adjudicate(r) for r in conflicts))
    for cid,(axis,cwe,manifestations,reason) in MANUAL.items():
        if cid not in final: raise SystemExit(f"manual override names unknown case: {cid}")
        final[cid].update(memory_axis=axis,root_cwe=cwe,manifestation_cwes=manifestations,
                          confidence="high",rationale=reason,review_status="manual-adjudicated")
    for cid,why in withdrawn.items(): final[cid]={"id":cid,"memory_axis":"uncertain","root_cwe":None,"manifestation_cwes":[],"confidence":"high","evidence_facts":[why],"rationale":"Withdrawn; no defect label asserted.","review_status":"existing-authoritative"}
    unresolved=dict(duplicates)
    while unresolved:
        progress=False
        for cid,target in list(unresolved.items()):
            if target in final:
                x={k:v for k,v in final[target].items() if k != "id"}
                final[cid]={"id":cid,**x,"rationale":f"Duplicate of {target}; "+x["rationale"],"review_status":"existing-authoritative"}; del unresolved[cid]; progress=True
        if not progress: raise SystemExit(f"unresolved duplicates {unresolved}")
    records=[]
    for cid,decl,_,case in rows:
        disp="withdrawn" if cid in withdrawn else "duplicate" if cid in duplicates else "active"
        records.append({"id":cid,"program":decl["program"],"corpus":cid.rsplit('/',1)[0],"case":case["case"],"disposition":disp,"duplicate_of":duplicates.get(cid),"classification":final[cid]})
    counts=Counter(r["classification"]["memory_axis"] for r in records if r["disposition"]=="active")
    artifact={"schema_version":1,"generated":str(date.today()),"population":{"declared":len(records),"active_unique":sum(r["disposition"]=="active" for r in records),"duplicates":len(duplicates),"withdrawn":len(withdrawn)},"method":{"independent_model_passes":2,"disagreement_adjudication":True,"manual_adjudications":len(MANUAL),"existing_sanitizer_cwe_authoritative":True,"acceptance":"exact memory_axis and root_cwe agreement, otherwise adjudication"},"active_axis_counts":dict(sorted(counts.items())),"needs_review":sum(r["classification"]["review_status"]=="needs-review" for r in records),"records":records}
    dest=OUT if args.publish else TMP/"classification.json"; dest.write_text(json.dumps(artifact,indent=2,ensure_ascii=False)+"\n")
    print(f"wrote {dest}: {len(records)} declared, {artifact['population']['active_unique']} active, {artifact['needs_review']} need review")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--concurrency",type=int,default=6); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--publish",action="store_true"); a=p.parse_args()
    if not 1<=a.concurrency<=16 or not 1<=a.batch_size<=16: p.error("values must be 1..16")
    asyncio.run(run(a))
if __name__=="__main__": main()
