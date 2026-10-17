#!/usr/bin/env python3
"""Tier 3 of the virtual-profile acceptance: enumerate the bug corpus's virtual arms from the bundles.

    enumerate.py [--repo REPO] [--rev REV] [--survivors DIR] [--out DIR] [--no-write]
                 [--expect BUNDLES,CELLS,IMAGES] [--expect-survivors N]

THE RULE (plan 2026-10-16-virtual-profile-on-silicon.md, "What done means" 3.4): every verdicts-v1
bundle whose configuration's target is `capstone-virtual`. As implemented, read only from REV:

  1. every capstone/bug-corpora/**/corpus.json at REV;
  2. each entry of its `verdict_bundles` (arm -> bundle dir, relative to the corpus) -- the CURRENT
     bundle per arm, the one tools/derive-verdicts.py writes case.json verdicts from;
  3. kept when `arm_configurations[arm]` names a configuration whose `target` in
     capstone/bug-corpora/tools/arms.json is `capstone-virtual`;
  4. the bundle's inputs.json must say `"contract": "verdicts-v1"` and the same `configuration`,
     `arm` and `corpus` -- a mismatch is an ERROR, not a skip.

A CELL is one line of the bundle's verdicts.jsonl (one Observation of one case on that arm); an
IMAGE is a distinct `observation.image_sha256`. The verdict of a cell is the record's own `verdict`
and `reason`, verbatim; case.json's derived `verdict`/`verdict_reason`/`verdict_from` is reported
beside it as a cross-check, never substituted.

The plan's sentence read literally -- every inputs.json anywhere at REV with that contract, current
or superseded -- selects more bundles; that count is printed as a side line so the difference is
visible, and it is not the selection.

No data is an ERROR: no corpus.json, no arms.json, or zero selected bundles exits 2 and says where it
looked. A consistency failure exits 3. An --expect / --expect-survivors mismatch exits 4.
"""
import argparse
import getpass
import hashlib
import json
import os
import posixpath
import re
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path

DEFAULT_REPO = "/tmp/capstone/wt-land-b0"
DEFAULT_REV = "44d82a9eb3f3"
SURVIVORS_TEMPLATE = "/mnt/hdd/{user}/capstone/virtual-corpus-images-2026-10-16"
CORPORA_ROOT = "capstone/bug-corpora"
ARMS_JSON = "capstone/bug-corpora/tools/arms.json"
TARGET = "capstone-virtual"
CONTRACT = "verdicts-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
# The runner every case.c corpus shares, which corpus.json `runners` does not always list.
EXTRA_RUNNERS = ["capstone/bug-corpora/tools/run-virtual-cases.py"]
# Keys of inputs.json that are bookkeeping, not build inputs.
NON_BUILD_KEYS = {"arm", "cases", "configuration", "contract", "corpus", "tally", "platform", "predicted"}

# How each runner turns its inputs into the image it hashed, read from the runner itself. `anchor` is a
# regex located in the RECORDED runner blob so the citation names that revision's line, not today's.
RECIPES = {
    "capstone/bug-corpora/tools/run-virtual-cases.py": [
        dict(when="no build.prebuilt",
             anchor=r'"-O0", "-DCORPUS_VIRTUAL", f"-I\{shared\}"',
             recipe="<SDK>/capstone-cc -O0 -DCORPUS_VIRTUAL -I<corpus>/shared <case>/case.c <corpus>/shared/driver.c "
                    "<build.cc_args...> -o <case>.dom; one image per case, run as `fixed N` then `buggy N`; controls "
                    "image = capstone-cc -O0 postgres/c-repros/shared/controls.c (hash not recorded)"),
        dict(when="build.prebuilt",
             anchor=r'programs = \{d\.name: program\(a\.prebuilt, d\)',
             recipe="image = <build.prebuilt>/bin/defect-NN from the corpus's shared/build-cases.sh "
                    "capstone-application (port CMake, preset capstone-application) with the build.cache options "
                    "on a virtual SDK; controls from the same build of controls/ (hashes in build.controls)"),
    ],
    "capstone/bug-corpora/cpython/allocator-boundary-repros/runners/virtual/run-virtual.py": [
        dict(when="always", anchor=r'image = a\.build / "image/cpython\.dom"',
             recipe="ONE interpreter image <build>/image/cpython.dom from `build-virtual.sh cpython OUT` "
                    "(manifest nested none|cpython, profile virtual, runtime_revision/runtime_dirty); every case "
                    "is a trigger script run by that image; the stdlib is staged from the same build"),
    ],
    "capstone/bug-corpora/cpython/pymalloc-repros/runners/virtual/run-virtual.py": [
        dict(when="always", anchor=r'image = a\.build / "bin" / f"defect-\{number:02d\}"',
             recipe="image = <build>/bin/defect-NN from `build-cases.sh capstone-application OUT` with "
                    "PYMALLOC_SUBLET=OFF (virtual-malloc) or ON (virtual-nested-pools) on a virtual SDK; controls "
                    "from --controls-build (not recorded)"),
    ],
    "capstone/bug-corpora/postgres/c-repros/shared/run-arm.py": [
        dict(when="always", anchor=r'image = a\.bindir / f"\{d\.name\}\.dom"',
             recipe="image = <bindir>/<case>.dom from shared/build-domain.sh (build.json: heap, opt, sdk, "
                    "runtime_sha256, virtual); controls <bindir>/controls.dom (hash not recorded)"),
    ],
    "capstone/bug-corpora/postgres/mmgr-repros/shared/run-defects.py": [
        dict(when="always", anchor=r'image = program\(build, d\)',
             recipe="image = <hosted build>/bin/<NN-slug>, the memory-contexts port's hosted program per case (preset capstone-application, "
                    "virtual SDK), PG_SUBLET=OFF (virtual-malloc) or ON (virtual-pg-pools); controls from "
                    "--controls-hosted-build (not recorded)"),
    ],
    "capstone/bug-corpora/postgres/sql-repros/shared/run-arm.py": [
        dict(when="always", anchor=r'"image": \{"sha256": v\.sha256\(args\.image\)',
             recipe="ONE server image <out>/image/postgres.dom from `build-virtual.sh postgres` (nested none, or "
                    "sublet with PGSU_NESTED=sublet); every case is trigger.sql against a copy of the initdb'd "
                    "fixture (fixture.cluster_tree_sha256) with the recorded extensions"),
    ],
    "capstone/bug-corpora/sqlite/engine-repros/shared/run-virtual.py": [
        dict(when="always", anchor=r'image = a\.build / f"\{d\.name\}\.dom"',
             recipe="image = <build>/<case>.dom from ports/sqlite/repro322/build-virtual.py (build.json: arm "
                    "memsys5|memsys5-sublet, sdk, opt, per-group defines in build.groups; amalgamation_sha256); "
                    "controls controls/control_*_mem5.c built into the same tree (hash not recorded)"),
    ],
    "capstone/bug-corpora/wireshark/wmem-repros/shared/run-defects.py": [
        dict(when="always", anchor=r'image = program\(build, d\)',
             recipe="image = <hosted build>/bin/<NN-slug>, the wmem port's hosted program per case (preset capstone-application, virtual SDK), "
                    "WM_SUBLET=OFF (virtual-malloc) or ON (virtual-nested-pools); controls from "
                    "--controls-hosted-build (not recorded)"),
    ],
}


class NoData(Exception):
    pass


class Inconsistent(Exception):
    pass


def redact(text, user):
    """Never write the login into an output: the plan itself writes /mnt/hdd/<user>/..."""
    if not user or not isinstance(text, str):
        return text
    return text.replace(f"/{user}/", "/<user>/").replace(f"/{user}", "/<user>")


class Repo:
    def __init__(self, path, rev):
        self.path, self.rev = path, rev

    def git(self, *args, check=True):
        r = subprocess.run(["git", "-C", self.path, *args], capture_output=True, text=True)
        if check and r.returncode:
            raise NoData(f"git {' '.join(args)} failed in {self.path}: {r.stderr.strip()}")
        return r.stdout

    def show(self, path, rev=None):
        return self.git("show", f"{rev or self.rev}:{path}")

    def exists(self, path, rev=None):
        r = subprocess.run(["git", "-C", self.path, "cat-file", "-e", f"{rev or self.rev}:{path}"],
                           capture_output=True, text=True)
        return r.returncode == 0

    def ls(self, path, rev=None):
        return [l for l in self.git("ls-tree", "-r", "--name-only", rev or self.rev, path).splitlines() if l]

    def blobs(self, path, rev=None):
        """{relative name: blob id} for every file under path at rev."""
        out = {}
        for line in self.git("ls-tree", "-r", rev or self.rev, "--", path).splitlines():
            meta, name = line.split("\t", 1)
            out[name] = meta.split()[2]
        return out


def runner_blobs(repo, paths, scope):
    """sha256 of file content -> (path, blob, commit) for every blob any of `paths` ever had in `scope`."""
    raw = repo.git("log", *scope, "--raw", "--no-abbrev", "--format=C %H", "--", *paths)
    seen, commit = {}, None
    for line in raw.splitlines():
        if line.startswith("C "):
            commit = line[2:]
        elif line.startswith(":"):
            meta, name = line.split("\t", 1)
            blob = meta.split()[3]
            if blob != "0" * 40 and blob not in seen:
                seen[blob] = (name.split("\t")[-1], commit)
    found = {}
    if not seen:
        return found
    proc = subprocess.Popen(["git", "-C", repo.path, "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    for blob, (name, commit) in seen.items():
        proc.stdin.write((blob + "\n").encode())
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        data = proc.stdout.read(int(header[2]))
        proc.stdout.read(1)
        found.setdefault(hashlib.sha256(data).hexdigest(), (name, blob, commit))
    proc.stdin.close()
    proc.wait()
    return found


def anchor_line(repo, blob, regex):
    text = repo.git("cat-file", "-p", blob)
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(regex, line):
            return i
    return None


def tally_of(records):
    t = Counter()
    for r in records:
        t[r["verdict"] + (f":{r['reason']}" if r.get("reason") else "")] += 1
    return dict(sorted(t.items()))


def literal_reading(repo, arms):
    """The plan's sentence read literally: every inputs.json at REV with the contract, by its own configuration."""
    rows = []
    for f in repo.ls("capstone"):
        if not f.endswith("/inputs.json"):
            continue
        try:
            d = json.loads(repo.show(f))
        except (json.JSONDecodeError, NoData):
            continue
        if not isinstance(d, dict) or d.get("contract") != CONTRACT:
            continue
        cfg = d.get("configuration")
        rows.append((posixpath.dirname(f), cfg, arms[cfg]["target"] if cfg in arms else None))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--rev", default=DEFAULT_REV)
    p.add_argument("--survivors", help="default " + SURVIVORS_TEMPLATE + " with the current login")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--no-write", action="store_true", help="print the counts only; write no manifest")
    p.add_argument("--no-verify-survivor-files", action="store_true",
                   help="match the SHA256SUMS lines without re-hashing the files")
    p.add_argument("--expect", help="BUNDLES,CELLS,IMAGES the rule must reproduce (exit 4 otherwise)")
    p.add_argument("--expect-survivors", type=int)
    a = p.parse_args()
    user = os.environ.get("USER") or getpass.getuser()
    survivors_dir = Path(a.survivors or SURVIVORS_TEMPLATE.format(user=user))
    repo = Repo(a.repo, a.rev)

    # ---- where we look, said before anything is counted ----
    full_rev = repo.git("rev-parse", "--verify", f"{a.rev}^{{commit}}").strip()
    print(f"repo      {a.repo}")
    print(f"rev       {a.rev} ({full_rev})")
    corpus_files = [f for f in repo.ls(CORPORA_ROOT) if f.endswith("/corpus.json")]
    print(f"looked in {CORPORA_ROOT}/**/corpus.json at the rev: {len(corpus_files)} files")
    for f in corpus_files:
        print(f"            {f}")
    if not corpus_files:
        raise NoData(f"no corpus.json under {CORPORA_ROOT} at {a.rev}")
    if not repo.exists(ARMS_JSON):
        raise NoData(f"no {ARMS_JSON} at {a.rev}")
    print(f"arms      {ARMS_JSON}")
    arms = json.loads(repo.show(ARMS_JSON))["arms"]
    virtual_configs = sorted(k for k, v in arms.items() if v.get("target") == TARGET)
    print(f"          {len(arms)} configurations, {len(virtual_configs)} with target {TARGET}")

    problems = []
    bundles, cells = [], []
    runner_paths = set(EXTRA_RUNNERS)
    for f in corpus_files:
        decl = json.loads(repo.show(f))
        runner_paths.update(r for r in decl.get("runners") or [] if r.endswith((".py", ".sh")))
    corpora_without = []
    for f in corpus_files:
        root = posixpath.dirname(f)
        corpus = root[len(CORPORA_ROOT) + 1:]
        decl = json.loads(repo.show(f))
        vb, ac = decl.get("verdict_bundles") or {}, decl.get("arm_configurations") or {}
        if not vb:
            corpora_without.append(corpus)
        for arm in sorted(vb):
            cfg = ac.get(arm)
            if cfg is None:
                problems.append(f"{f}: verdict_bundles names arm {arm} with no arm_configurations entry")
                continue
            if cfg not in arms:
                problems.append(f"{f}: arm {arm} -> {cfg}, which {ARMS_JSON} does not define")
                continue
            if arms[cfg].get("target") != TARGET:
                continue
            bdir = f"{root}/{vb[arm]}"
            if not repo.exists(f"{bdir}/inputs.json") or not repo.exists(f"{bdir}/verdicts.jsonl"):
                problems.append(f"{bdir}: the bundle corpus.json names lacks inputs.json or verdicts.jsonl")
                continue
            inputs = json.loads(repo.show(f"{bdir}/inputs.json"))
            for key, want in (("contract", CONTRACT), ("configuration", cfg), ("arm", arm), ("corpus", corpus)):
                if inputs.get(key) != want:
                    problems.append(f"{bdir}/inputs.json: {key} is {inputs.get(key)!r}, the rule needs {want!r}")
            records = [json.loads(l) for l in repo.show(f"{bdir}/verdicts.jsonl").splitlines() if l.strip()]
            if inputs.get("cases") != len(records):
                problems.append(f"{bdir}: inputs.json cases={inputs.get('cases')} but verdicts.jsonl has {len(records)}")
            names = Counter(r["observation"]["case"] for r in records)
            dup = [n for n, c in names.items() if c > 1]
            if dup:
                problems.append(f"{bdir}: case(s) observed more than once: {dup}")
            recomputed = tally_of(records)
            if inputs.get("tally") != recomputed:
                problems.append(f"{bdir}: inputs.json tally {inputs.get('tally')} != the records' {recomputed}")
            shared_image = (inputs.get("build") or {}).get("image_sha256") or (inputs.get("image") or {}).get("sha256")
            # landing commit: the commit at or before REV that added this bundle's verdicts.jsonl
            added = repo.git("log", "--diff-filter=A", "--format=%H", a.rev, "--", f"{bdir}/verdicts.jsonl").split()
            landed = added[-1] if added else None
            build_record = OrderedDict((k, inputs[k]) for k in sorted(inputs) if k not in NON_BUILD_KEYS)
            b = OrderedDict(
                id=len(bundles), corpus=corpus, bundle=bdir, arm=arm, configuration=cfg,
                target=arms[cfg]["target"], contract=inputs.get("contract"), cases=len(records),
                tally=inputs.get("tally"), predicted=inputs.get("predicted"),
                controls_expected=arms[cfg].get("controls"),
                controls_for_missed=arms[cfg].get("controls_for_missed"),
                recorded_build=build_record, platform=inputs.get("platform"),
                shared_image_sha256=shared_image, landed=landed, case_schema=decl.get("case_schema"),
                upstream=decl.get("upstream"))
            bundles.append(b)
            for r in records:
                o = r["observation"]
                img = o.get("image_sha256")
                if not isinstance(img, str) or not HEX64.match(img):
                    problems.append(f"{bdir}: {o.get('case')} image_sha256 {img!r} is not 64 hex digits")
                if shared_image and img != shared_image:
                    problems.append(f"{bdir}: {o['case']} image {img} differs from the bundle's recorded {shared_image}")
                if o.get("arm") != arm:
                    problems.append(f"{bdir}: {o['case']} observation.arm {o.get('arm')!r} != {arm!r}")
                cells.append(OrderedDict(
                    bundle_id=b["id"], corpus=corpus, bundle=bdir, case=o["case"], arm=arm, configuration=cfg,
                    image_sha256=img, verdict=r.get("verdict"), reason=r.get("reason"), evidence=r.get("evidence"),
                    observation=o))

    # ---- the counts ----
    images = OrderedDict()
    for c in cells:
        images.setdefault(c["image_sha256"], []).append(c)
    print(f"\nRULE: corpus.json verdict_bundles -> arm_configurations -> arms.json target == {TARGET!r}, "
          f"inputs.json contract == {CONTRACT!r}")
    print(f"corpora with no verdict_bundles (none selected from them): {', '.join(corpora_without) or 'none'}")
    if not bundles:
        raise NoData(f"zero bundles selected at {a.rev}: looked in {len(corpus_files)} corpus.json files under "
                     f"{CORPORA_ROOT}, {len(virtual_configs)} {TARGET} configurations in {ARMS_JSON}")
    print(f"bundles   {len(bundles)}")
    print(f"cells     {len(cells)}")
    print(f"images    {len(images)} distinct observation.image_sha256")
    cross = Counter()
    for img, cs in images.items():
        cross[len({c['bundle'] for c in cs})] += 1
    print(f"          images by number of bundles using them: {dict(sorted(cross.items()))}")

    lit = literal_reading(repo, arms)
    lit_virtual = [r for r in lit if r[2] == TARGET]
    selected = {b["bundle"] for b in bundles}
    print(f"\nside line, the sentence read literally (every inputs.json at the rev with contract {CONTRACT}):")
    print(f"          {len(lit)} bundles; by target {dict(Counter(str(r[2]) for r in lit))}")
    print(f"          {len(lit_virtual)} with target {TARGET}: {len(selected & {r[0] for r in lit_virtual})} current "
          f"(selected), {len([r for r in lit_virtual if r[0] not in selected])} superseded or outside a corpus")
    unresolved_cfg = sorted({r[1] for r in lit if r[2] is None})
    print(f"          configurations absent from arms.json (no target): {unresolved_cfg}")

    # ---- case.json cross-check: what derive-verdicts.py wrote from the same bundle ----
    for c in cells:
        cj = f"{CORPORA_ROOT}/{c['corpus']}/{c['case']}/case.json"
        rel = c["bundle"][len(f"{CORPORA_ROOT}/{c['corpus']}") + 1:]
        if not repo.exists(cj):
            c["case_json"] = None
            problems.append(f"{cj}: absent, so {c['case']} has no case.json verdict to cross-check")
            continue
        arm_rec = (json.loads(repo.show(cj)).get("arms") or {}).get(c["arm"]) or {}
        c["case_json"] = OrderedDict((k, arm_rec.get(k)) for k in ("verdict", "verdict_reason", "verdict_from"))
        same = (arm_rec.get("verdict") == c["verdict"] and arm_rec.get("verdict_reason") == c["reason"]
                and arm_rec.get("verdict_from") == rel)
        c["case_json_agrees"] = same
        if not same:
            problems.append(f"{cj} arms.{c['arm']}: {c['case_json']} vs the bundle's {c['verdict']}/{c['reason']} "
                            f"from {rel}")
    agree = sum(1 for c in cells if c.get("case_json_agrees"))
    print(f"\ncase.json cross-check: {agree}/{len(cells)} cells carry the bundle's verdict, reason and verdict_from")

    # ---- runners: which revision of which script wrote each bundle ----
    by_hash = runner_blobs(repo, sorted(runner_paths), [a.rev])
    by_hash_all = None
    for b in bundles:
        h = (b["platform"] or {}).get("runner")
        hit, scope = by_hash.get(h), "ancestry of the rev"
        if not hit:
            if by_hash_all is None:
                by_hash_all = runner_blobs(repo, sorted(runner_paths), ["--all"])
            hit, scope = by_hash_all.get(h), "all refs"
        if not hit:
            b["runner"] = OrderedDict(sha256=h, path=None, note="UNRESOLVED: no blob of any listed runner has this sha256")
            problems.append(f"{b['bundle']}: runner sha256 {h} matches no runner blob")
            continue
        path, blob, commit = hit
        at_rev = repo.blobs(path).get(path)
        recipe = None
        for r in RECIPES.get(path, []):
            prebuilt = "prebuilt" in (b["recorded_build"].get("build") or {})
            if r["when"] == "always" or (r["when"] == "build.prebuilt") == prebuilt:
                recipe = OrderedDict(text=r["recipe"], cite=f"{path} blob {blob[:12]} line "
                                                             f"{anchor_line(repo, blob, r['anchor'])}")
                break
        ancestor = subprocess.run(["git", "-C", repo.path, "merge-base", "--is-ancestor", commit, a.rev],
                                  capture_output=True).returncode == 0
        refs = [] if ancestor else [r for r in repo.git("branch", "-a", "--contains", commit,
                                                        "--format=%(refname:short)").split() if r][:8]
        b["runner"] = OrderedDict(sha256=h, path=path, blob=blob, a_commit_containing_it=commit,
                                  found_in=scope, commit_is_ancestor_of_rev=ancestor, refs_containing_commit=refs,
                                  same_blob_at_rev=(at_rev == blob), recipe=recipe)

    # ---- case sources at the rev, and what changed in them since the bundle landed and since the runner's commit.
    # Neither commit is recorded as the build's source revision; both are proxies (README).
    def src_files(cdir, rev):
        return {k: v for k, v in repo.blobs(cdir, rev).items() if not k.endswith(("case.json", "PROVENANCE.md"))}

    def changed(now, then, cdir):
        return sorted(k[len(cdir) + 1:] for k in set(now) | set(then) if now.get(k) != then.get(k))

    for b in bundles:
        rc = (b.get("runner") or {}).get("a_commit_containing_it")
        b_cells = [c for c in cells if c["bundle_id"] == b["id"]]
        for c in b_cells:
            cdir = f"{CORPORA_ROOT}/{c['corpus']}/{c['case']}"
            now = src_files(cdir, a.rev)
            c["case_sources"] = OrderedDict((k[len(cdir) + 1:], v[:12]) for k, v in sorted(now.items()))
            c["case_sources_changed_since_landing"] = (changed(now, src_files(cdir, b["landed"]), cdir)
                                                       if b["landed"] else None)
            c["case_sources_changed_since_runner_commit"] = changed(now, src_files(cdir, rc), cdir) if rc else None
        shared = f"{CORPORA_ROOT}/{b['corpus']}/shared"
        port = (b["upstream"] or {}).get("port")
        paths = [shared] + ([port] if port else [])
        for key, base in (("landing", b["landed"]), ("runner_commit", rc)):
            b[f"shared_and_port_files_changed_since_{key}"] = (
                [l for l in repo.git("diff", "--name-only", f"{base}..{a.rev}", "--", *paths).splitlines() if l]
                if base else None)

    # ---- survivors ----
    surv = OrderedDict(dir=redact(str(survivors_dir), user), listed=0, matched=0, unmatched=[],
                       file_check=None, name_mismatches=[])
    sums = survivors_dir / "SHA256SUMS"
    survivor_by_hash = {}
    if sums.is_file():
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            h, rel = line.split(None, 1)
            rel = rel.lstrip("*")
            survivor_by_hash.setdefault(h, []).append(rel)
            surv["listed"] += 1
        if not a.no_verify_survivor_files:
            bad = []
            for h, rels in survivor_by_hash.items():
                for rel in rels:
                    fp = survivors_dir / rel
                    if not fp.is_file() or hashlib.sha256(fp.read_bytes()).hexdigest() != h:
                        bad.append(rel)
            surv["file_check"] = f"{surv['listed'] - len(bad)}/{surv['listed']} files hash to their SHA256SUMS line"
            if bad:
                problems.append(f"survivors: {len(bad)} files do not match SHA256SUMS: {bad[:5]}")
        for h, rels in survivor_by_hash.items():
            if h not in images:
                surv["unmatched"].extend(rels)
                continue
            surv["matched"] += 1
            for rel in rels:
                parts = rel.split("/")
                want = {(c["corpus"].replace("/", "__"), c["arm"], c["case"] + ".dom") for c in images[h]}
                if len(parts) != 3 or tuple(parts) not in want:
                    surv["name_mismatches"].append(f"{rel} vs cells {sorted(want)}")
        if surv["name_mismatches"]:
            problems.append(f"survivors: {len(surv['name_mismatches'])} files filed under a case other than the one "
                            "that recorded their hash")
    else:
        surv["note"] = f"no SHA256SUMS at {redact(str(survivors_dir), user)}"
        problems.append(surv["note"])
    print(f"\nsurvivors {surv['dir']}")
    print(f"          SHA256SUMS lines {surv['listed']}; {surv['file_check'] or 'files not re-hashed'}")
    print(f"          matched to recorded images: {surv['matched']}; listed but matching no cell: "
          f"{len(surv['unmatched'])}; name mismatches: {len(surv['name_mismatches'])}")

    image_rows = []
    for h, cs in images.items():
        rels = survivor_by_hash.get(h, [])
        image_rows.append(OrderedDict(
            image_sha256=h, survivor=bool(rels), survivor_paths=rels,
            bundles=sorted({c["bundle"] for c in cs}), cells=[f"{c['bundle']}::{c['case']}" for c in cs],
            n_cells=len(cs)))
    by_bundle_surv = Counter()
    for row in image_rows:
        if row["survivor"]:
            for bd in row["bundles"]:
                by_bundle_surv[bd] += 1
    for b in bundles:
        b["distinct_images"] = len({c["image_sha256"] for c in cells if c["bundle_id"] == b["id"]})
        b["surviving_images"] = by_bundle_surv.get(b["bundle"], 0)

    if problems:
        print("\nCONSISTENCY PROBLEMS:")
        for pr in problems:
            print(f"  {pr}")

    if not a.no_write:
        a.out.mkdir(parents=True, exist_ok=True)
        manifest = OrderedDict(
            rule=OrderedDict(
                text="every verdicts-v1 bundle whose configuration's target is capstone-virtual",
                as_implemented=[f"{CORPORA_ROOT}/**/corpus.json at the rev", "verdict_bundles[arm] (the current bundle per arm)",
                                "arm_configurations[arm] -> configuration", f"{ARMS_JSON} arms[configuration].target == {TARGET}",
                                f"inputs.json contract == {CONTRACT}, configuration/arm/corpus equal to the above (else error)"],
                cell="one line of the bundle's verdicts.jsonl", image="distinct observation.image_sha256",
                verdict_fields=["verdict", "reason"]),
            repo=a.repo, rev=a.rev, rev_full=full_rev,
            counts=OrderedDict(bundles=len(bundles), cells=len(cells), images=len(images),
                               survivors_listed=surv["listed"], survivors_matched=surv["matched"]),
            literal_reading=OrderedDict(
                bundles=len(lit), by_target=dict(Counter(str(r[2]) for r in lit)),
                virtual_not_selected=[OrderedDict(bundle=r[0], configuration=r[1]) for r in lit_virtual
                                      if r[0] not in selected],
                configurations_absent_from_arms=unresolved_cfg),
            survivors=surv, problems=problems, bundles=bundles,
            cells=[OrderedDict((k, v) for k, v in c.items() if k != "observation") | {"observation": c["observation"]}
                   for c in cells],
            images=image_rows)
        text = json.dumps(manifest, indent=1)
        (a.out / "manifest.json").write_text(redact(text, user) + "\n")
        cols = ["kind", "corpus", "bundle", "case", "arm", "configuration", "image_sha256", "verdict", "reason",
                "case_json_verdict", "case_json_verdict_reason", "case_json_agrees", "case_sources",
                "case_sources_changed_since_landing", "case_sources_changed_since_runner_commit", "survivor", "survivor_path", "n_cells_using_image",
                "cells_using_image", "runner", "recipe", "recorded_build"]
        lines = ["\t".join(cols)]
        bmap = {b["id"]: b for b in bundles}
        surv_of = {r["image_sha256"]: r for r in image_rows}

        def cell_text(v):
            if v is None:
                return ""
            if isinstance(v, (dict, list)):
                return json.dumps(v, separators=(",", ":"))
            return str(v)
        for c in cells:
            b = bmap[c["bundle_id"]]
            run = b.get("runner") or {}
            row = dict(kind="cell", corpus=c["corpus"], bundle=c["bundle"], case=c["case"], arm=c["arm"],
                       configuration=c["configuration"], image_sha256=c["image_sha256"],
                       verdict=f"verdict={c['verdict']}",
                       reason="reason=" + ("null" if c["reason"] is None else str(c["reason"])),
                       case_json_verdict=(c.get("case_json") or {}).get("verdict"),
                       case_json_verdict_reason=(c.get("case_json") or {}).get("verdict_reason"),
                       case_json_agrees=c.get("case_json_agrees"),
                       case_sources=" ".join(f"{k}@{v}" for k, v in (c.get("case_sources") or {}).items()),
                       case_sources_changed_since_landing=" ".join(c.get("case_sources_changed_since_landing") or []),
                       case_sources_changed_since_runner_commit=" ".join(
                           c.get("case_sources_changed_since_runner_commit") or []),
                       survivor="yes" if surv_of[c["image_sha256"]]["survivor"] else "no",
                       survivor_path=" ".join(surv_of[c["image_sha256"]]["survivor_paths"]),
                       n_cells_using_image=surv_of[c["image_sha256"]]["n_cells"], cells_using_image="",
                       runner=(f"{run.get('path')}@blob:{(run.get('blob') or '')[:12]} commit:"
                               f"{(run.get('a_commit_containing_it') or '')[:12]} "
                               f"{'on-rev-ancestry' if run.get('commit_is_ancestor_of_rev') else 'NOT-on-rev-ancestry:' + ','.join(run.get('refs_containing_commit') or [])}"),
                       recipe=(run.get("recipe") or {}).get("text"), recorded_build=b["recorded_build"])
            lines.append("\t".join(cell_text(row.get(k)).replace("\t", " ") for k in cols))
        for r in image_rows:
            b0 = r["bundles"][0]
            row = dict(kind="image", corpus=b0.split("/")[2] + "/" + b0.split("/")[3], bundle=" ".join(r["bundles"]),
                       case=" ".join(sorted({x.split("::")[1] for x in r["cells"]})), image_sha256=r["image_sha256"],
                       survivor="yes" if r["survivor"] else "no", survivor_path=" ".join(r["survivor_paths"]),
                       n_cells_using_image=r["n_cells"], cells_using_image=" ".join(r["cells"]))
            lines.append("\t".join(cell_text(row.get(k)).replace("\t", " ") for k in cols))
        (a.out / "manifest.tsv").write_text(redact("\n".join(lines), user) + "\n")
        print(f"\nwrote {a.out / 'manifest.json'} and {a.out / 'manifest.tsv'} "
              f"({len(cells)} cell rows, {len(image_rows)} image rows)")

    rc = 3 if problems else 0
    if a.expect:
        want = [int(x) for x in a.expect.split(",")]
        got = [len(bundles), len(cells), len(images)]
        print(f"\nexpect bundles,cells,images = {want}; got {got}: {'OK' if want == got else 'MISMATCH'}")
        if want != got:
            rc = rc or 4
    if a.expect_survivors is not None:
        ok = surv["matched"] == a.expect_survivors and not surv["unmatched"]
        print(f"expect survivors = {a.expect_survivors}; matched {surv['matched']}, unmatched "
              f"{len(surv['unmatched'])}: {'OK' if ok else 'MISMATCH'}")
        if not ok:
            rc = rc or 4
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main())
    except NoData as e:
        print(f"NO DATA: {e}", file=sys.stderr)
        sys.exit(2)
