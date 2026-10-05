#!/usr/bin/env python3
"""Check aliasstat against a hand-built trace in capstone-qemu's record order.

    selftest_alias.py <aliasstat binary>

The trace builds 3 -> {4, 1} by mrev and split (cap_rev_tree.c's list order:
3 at depth 0, then 4 and 1 at depth 1), gives node 1 three memory copies (one
later overwritten), node 4 and node 3 one each, snapshots two registers, revokes
3 and frees 4 with its copy still recorded. Every expected number below is
derived by hand from that. Two broken traces -- a copy leaving a node that has
none, and a revoke whose recorded walk disagrees with the tree -- must exit 3.
"""
import json
import struct
import subprocess
import sys
import tempfile

READ, WRITE, ALLOC, FREE, RESET, REPEAT, INC, DEC, REG, SLOT = range(10)
LDST, LDC, MREV, SPLIT, REVOKE, DELIN, CREATE, SUP, GC, CAPSTORE, UNTAG, CLEAR, DROP = range(13)
NONE = 0xFFFFFFFF


def run(sim, recs):
    with tempfile.NamedTemporaryFile(suffix=".bin") as t:
        with open(t.name, "wb") as f:
            f.write(b"CRNTRC02")
            for node, kind, site in recs:
                f.write(struct.pack("<IBBH", node, kind, site, 0))
            f.write(struct.pack("<IBBH", 0xFFFFFFFF, 10, 0, 0))
        p = subprocess.run([sim, "--max-node", "1000", t.name], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.stdout else None)


def run_big(sim, recs):
    with tempfile.NamedTemporaryFile(suffix=".bin") as t:
        with open(t.name, "wb") as f:
            f.write(b"CRNTRC02")
            f.write(b"".join(struct.pack("<IBBH", node, kind, site, 0) for node, kind, site in recs))
            f.write(struct.pack("<IBBH", 0xFFFFFFFF, 10, 0, 0))
        p = subprocess.run([sim, "--max-node", "20000", t.name], capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.stdout else None)


def lone(n):
    return [(n, ALLOC, CREATE), (n, WRITE, CREATE)]


BASE = ([(NONE, RESET, CREATE)] + lone(0) + lone(1) + lone(2)
        # mrev of 1: R src, ALLOC new, W new, W src   (1 had no predecessor)
        + [(1, READ, MREV), (3, ALLOC, MREV), (3, WRITE, MREV), (1, WRITE, MREV)]
        # split of 1: R src, ALLOC new, W prev(=3), W new, W src
        + [(1, READ, SPLIT), (4, ALLOC, SPLIT), (3, WRITE, SPLIT), (4, WRITE, SPLIT), (1, WRITE, SPLIT)]
        # memory copies: 1 x3 (one as a repeat record), 4 x1, 3 x1; one copy of 1 overwritten
        + [(1, INC, CAPSTORE), (2, REPEAT, 0), (4, INC, CAPSTORE), (3, INC, CAPSTORE),
           (1, DEC, CAPSTORE), (5, LDST, 0)]
        # registers just before the revoke: one copy of 1, one of 3
        + [(1, REG, REVOKE), (3, REG, REVOKE)]
        # revoke 3: R root, (R n, W n) for 4 and 1, W root   (nothing follows the run)
        + [(3, READ, REVOKE), (4, READ, REVOKE), (4, WRITE, REVOKE), (1, READ, REVOKE),
           (1, WRITE, REVOKE), (3, WRITE, REVOKE)]
        # the collector frees 4 while its copy is still recorded
        + [(4, FREE, GC)])

failures = []


def check(cond, what):
    if not cond:
        failures.append(what)
        print("FAIL:", what)


def main():
    sim = sys.argv[1]
    rc, r = run(sim, BASE)
    check(rc == 0, f"base trace exit {rc}")
    a, t = r["aliases"], r["trees"]
    check((r["nodes_allocated"], r["mrev"], r["split"], r["lone"]) == (5, 1, 1, 3), "allocation counts")
    check(r["memory_alias_events"]["stored"] == 5, f"stored {r['memory_alias_events']['stored']}")
    check(r["memory_alias_events"]["left"]["mem_capstore"] == 1, "one copy overwritten")
    check(r["memory_alias_events"]["peak_live"] == 5 and r["memory_alias_events"]["live_at_end"] == 4, "peak/live")
    # incarnations 0,1,2,3,4: max copies 0,3,0,1,1
    check(a["max_memory_aliases_per_node"]["count"] == 5 and a["max_memory_aliases_per_node"]["max"] == 3
          and a["max_memory_aliases_per_node"]["buckets"] == {"0": 2, "1": 2, "3": 1},
          f"max aliases {a['max_memory_aliases_per_node']}")
    check(a["at_revoke_root_memory"]["sum" if False else "max"] == 1, "root memory copies 1")
    check(a["at_revoke_root_registers"]["max"] == 1, "root register copies 1")
    check(a["at_revoke_invalidated_memory"]["max"] == 3, "invalidated memory copies 2 (node 1) + 1 (node 4)")
    check(a["at_revoke_invalidated_registers"]["max"] == 1, "invalidated register copies 1")
    check(a["at_free_memory"]["buckets"] == {"1": 1}, f"free with a copy left {a['at_free_memory']}")
    check(t["revoke_run"]["max"] == 2 and t["revoke_root_children"]["max"] == 2
          and t["revoke_root_depth"]["max"] == 0, "revoke run 2, children 2, depth 0")
    check(r["errors"] == {"negative_alias_count": 0, "walk_mismatch": 0, "alloc_with_live_aliases": 0},
          f"errors {r['errors']}")
    # snapshots: a trace this short is never sampled
    check(t["snapshots"] == 0 and t["tree_size"]["count"] == 0, f"no snapshot expected {t['snapshots']}")
    # the 2^14th allocation takes a snapshot: BASE (5 allocations), an mrev of 0 making
    # node 5 its parent, then lone nodes 6..16383. Live valid then: 0, 2, 3, 5 and
    # 16378 lone nodes -- trees {5 -> 0}, {2}, {3} and the lone ones.
    many = BASE + [(0, READ, MREV), (5, ALLOC, MREV), (5, WRITE, MREV), (0, WRITE, MREV)]
    many += [x for i in range(6, 16384) for x in lone(i)]
    rc, r2 = run_big(sim, many)
    t2 = r2["trees"]
    check(rc == 0 and t2["snapshots"] == 1, f"one snapshot expected, got {t2['snapshots']} rc {rc}")
    check(t2["tree_size"]["count"] == 3 + 16378 and t2["tree_size"]["max"] == 2,
          f"trees in the snapshot {t2['tree_size']}")
    check(t2["live_node_depth"]["buckets"] == {"0": 2 + 1 + 16378, "1": 1},
          f"depths {t2['live_node_depth']['buckets']}")
    check(t2["live_node_children"]["buckets"] == {"0": 3 + 16378, "1": 1},
          f"children {t2['live_node_children']['buckets']}")

    # store classification, in capstone-qemu's record order (granule ids are not nodes,
    # and may exceed --max-node)
    st = ([(NONE, RESET, CREATE)] + lone(0) + lone(1)
          + [(50000, SLOT, CAPSTORE), (0, INC, CAPSTORE)]                       # into an empty slot
          + [(50000, SLOT, CAPSTORE), (0, DEC, CAPSTORE), (0, INC, CAPSTORE)]   # same node over it
          + [(50000, SLOT, CAPSTORE), (0, DEC, CAPSTORE), (1, INC, CAPSTORE)]   # another node over it
          + [(50000, SLOT, UNTAG), (1, DEC, UNTAG)]                             # a data store over it
          + [(50001, SLOT, CAPSTORE), (1, INC, CAPSTORE)])                      # empty slot again
    rc, r3 = run(sim, st)
    check(rc == 0 and r3["capability_stores"] == {"into_untagged": 2, "over_same_node": 1, "over_other_node": 1}
          and r3["data_stores_over_capability"] == 1, f"store classes {r3 and r3['capability_stores']}")

    rc, _ = run(sim, BASE + [(2, DEC, UNTAG)])
    check(rc == 3, f"a copy leaving a node with none must exit 3, got {rc}")
    broken = [x for x in BASE if x != (1, WRITE, REVOKE)]
    rc, _ = run(sim, broken)
    check(rc == 3, f"a recorded walk that disagrees with the tree must exit 3, got {rc}")

    print("selftest_alias:", "PASS" if not failures else f"FAIL ({len(failures)})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
