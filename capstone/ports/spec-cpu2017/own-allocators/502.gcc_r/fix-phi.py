#!/usr/bin/env python3
"""fix-phi.py DIR: a use after free in SPEC 502.gcc_r (gcc 4.5) tree-phinodes.c, found by the free-list port
(nested lane, 2026-11-20). remove_phi_node puts the phi node on gcc's free list (release_phi_node) and then reads its
result through the same reference (release_ssa_name (gimple_phi_result (phi))). Stock gcc survives because nothing
reuses the node in between. The fix reads the result first: the two calls swap, and release_ssa_name does not touch the
phi node, so what gcc computes is unchanged. The edit is asserted to match exactly once."""
import sys
from pathlib import Path
p = Path(sys.argv[1]) / 'tree-phinodes.c'; s = p.read_text()
old = """  release_phi_node (phi);
  if (release_lhs_p)
    release_ssa_name (gimple_phi_result (phi));
}"""
new = """  if (release_lhs_p)
    release_ssa_name (gimple_phi_result (phi));
  release_phi_node (phi);
}"""
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, new))
print('phi use after free fixed')
