#!/bin/bash
# Recompute the toolchain's integration tree from public llvm-capstone commits (the PR heads, fetched from
# the pull refs) and check it against manifest.lock.json; then, when the manifest has one, the runtime tree
# (a PR cherry-picked onto it) and the compiler subtrees it leaves unchanged. Merges run in memory (git merge-tree), in the manifest's
# order; a conflict is accepted only in the files the manifest lists for that merge, and is resolved
# to the base side there (documentation). Prints the tree; exits non-zero if it differs.
# usage: integration.sh KIT LLVM_CAPSTONE_GIT_DIR
set -euo pipefail
KIT=$(cd "$1" && pwd) G=$2
M=$KIT/manifest.lock.json
# The PR commits: fetched from the public repository's pull refs (each merge step names its `pr`); a bundle
# beside the manifest, when there is one, is fetched first so a head that has since moved is still there.
for b in $(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d['toolchain'].get('bundle',{}).get('file',''))" "$M"); do
  [ -f "$KIT/artifacts/sources/$b" ] && git -C "$G" -c fetch.writeCommitGraph=false fetch -q "$KIT/artifacts/sources/$b" "+refs/heads/*:refs/kit-bundle/*"
done
for n in $(python3 -c "import json,sys;print(' '.join(str(s['pr']) for s in json.load(open(sys.argv[1]))['toolchain']['merges'] if 'pr' in s))" "$M"); do
  git -C "$G" -c fetch.writeCommitGraph=false fetch -q origin "+refs/pull/$n/head:refs/kit-pr/$n" 2>/dev/null || echo "integration: could not fetch PR #$n from origin" >&2
done
python3 - "$M" "$G" <<'PY'
import json, os, re, subprocess, sys, tempfile
m = json.load(open(sys.argv[1]))['toolchain']; g = sys.argv[2]
# merge-tree writes the conflict markers the keep-both rule parses: the plain style, whatever the host's config.
git = lambda *a, **kw: subprocess.run(['git', '-C', g, '-c', 'merge.conflictStyle=merge', *a], capture_output=True, text=True, **kw)
def keep_both(conflicted_tree, paths):
    """The one resolution rule: in each named file, every conflict hunk keeps both sides, ours then theirs."""
    env = dict(os.environ, GIT_INDEX_FILE=tempfile.mktemp(prefix='kit-index-'))
    subprocess.run(['git', '-C', g, 'read-tree', conflicted_tree], check=True, env=env)
    for path in paths:
        blob = git('show', f'{conflicted_tree}:{path}').stdout
        fixed, n = re.subn(r'<<<<<<< [^\n]*\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n', lambda h: h.group(1) + h.group(2), blob, flags=re.S)
        if n < 1 or '<<<<<<<' in fixed: sys.exit(f"keep-both could not resolve {path}")
        oid = subprocess.run(['git', '-C', g, 'hash-object', '-w', '--stdin'], input=fixed, capture_output=True, text=True, check=True).stdout.strip()
        subprocess.run(['git', '-C', g, 'update-index', '--cacheinfo', f'100644,{oid},{path}'], check=True, env=env)
    tree = subprocess.run(['git', '-C', g, 'write-tree'], capture_output=True, text=True, check=True, env=env).stdout.strip()
    os.unlink(env['GIT_INDEX_FILE']); return tree
head = m['base']
for step in m['merges']:
    if 'pr' in step:
        ref = git('rev-parse', '--verify', '-q', f"refs/kit-pr/{step['pr']}").stdout.strip()
        if ref and ref != step['commit']:
            print(f"note: PR #{step['pr']}'s head is now {ref[:12]}; the manifest pins {step['commit'][:12]} and that is what is merged")
    if git('cat-file', '-e', step['commit'] + '^{commit}').returncode:
        sys.exit(f"commit {step['commit']} of PR #{step.get('pr', '?')} is not available: fetch failed, or the PR's head moved and no bundle holds it")
    r = git('merge-tree', '--write-tree', '--name-only', head, step['commit'])
    if r.returncode not in (0, 1): sys.exit(f"merge-tree failed: {r.stderr}")
    if r.returncode == 1:
        conflicts = [l for l in r.stdout.split('\n\n')[0].splitlines()[1:] if l]
        if step.get('resolution') == 'keep-both':
            if sorted(conflicts) != sorted(step.get('conflicts', [])):
                sys.exit(f"conflicts merging {step['commit']} are {conflicts}, the manifest expects {step.get('conflicts')}")
            conflicted = git('merge-tree', '--write-tree', head, step['commit']).stdout.splitlines()[0]
            tree = keep_both(conflicted, step['conflicts'])
            head = git('commit-tree', tree, '-p', head, '-p', step['commit'], '-m', f"kit integration: {step['what']}").stdout.strip()
            continue
        if sorted(conflicts) != sorted(step.get('docs_conflicts_to_base', [])):
            sys.exit(f"unexpected conflict merging {step['commit']}: {conflicts}")
        r = git('merge-tree', '--write-tree', '-X', 'ours', head, step['commit'])
        if r.returncode: sys.exit(f"merge-tree -X ours failed: {r.stdout}{r.stderr}")
    tree = r.stdout.splitlines()[0]
    head = git('commit-tree', tree, '-p', head, '-p', step['commit'], '-m', f"kit integration: {step['what']}").stdout.strip()
tree = git('rev-parse', head + '^{tree}').stdout.strip()
print(f"integration tree {tree}")
if tree != m['tree']: sys.exit(f"tree {tree} differs from the manifest's {m['tree']}")
# The runtime tree: the PR's commit cherry-picked onto the integration (merge-tree with the PR's parent as
# the merge base), and the compiler-side subtrees unchanged by it.
rt = json.load(open(sys.argv[1])).get('runtime')
if rt and rt.get('cherry_pick'):
    pick = rt['cherry_pick']['commit']
    r = git('merge-tree', '--write-tree', f"--merge-base={pick}^", head, pick)
    if r.returncode: sys.exit(f"cherry-pick of {pick} does not apply cleanly: {r.stdout}{r.stderr}")
    rtree = r.stdout.splitlines()[0]
    print(f"runtime tree {rtree}")
    if rtree != rt['tree']: sys.exit(f"runtime tree {rtree} differs from the manifest's {rt['tree']}")
    for path, want in rt['unchanged_subtrees'].items():
        a = git('rev-parse', f"{tree}:{path}").stdout.strip(); b = git('rev-parse', f"{rtree}:{path}").stdout.strip()
        if not (a == b == want): sys.exit(f"subtree {path}: toolchain {a}, runtime {b}, manifest {want}")
    print(f"unchanged subtrees {' '.join(rt['unchanged_subtrees'])}")
PY
