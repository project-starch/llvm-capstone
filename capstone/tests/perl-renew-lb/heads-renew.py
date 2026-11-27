#!/usr/bin/env python3
"""heads-renew.py DIR (nested lane, 2026-11-26): an EXPERIMENT ARM on a perlbench 5.22 tree that port-sv-heads.py,
fix-lookup.py and fix-lookup-pages.py produced, never a port script. It is the 5.36 heads variant (oneop
fig10/renew/0003-sv-heads-renewed.patch) in the 5.22 port's shape, failing closed. Under
PERL_CAPSTONE_SUBLET_HEADS_RENEW:
  - S_sv_arena_retire renews the head's issued lifetime (PERL_CAP_RENEW, renew-shim.h) where the port revoked it and
    cleared the entry, and the arena's issued table keeps the renewed capability. If the renewal fails (an untagged 0:
    the old lifetime is still live), it revokes and clears the entry as the port does;
  - S_sv_arena_issue hands out a non-NULL issued entry as it is, with no CDERIVE; a head never issued, or one whose
    renewal failed, derives as in the port.
SVf_BREAK heads are untouched, as in the port. The issued table already tells a fresh head from a reissued one, so no
bounds test is added. Without the new define the preprocessed source is the port's. Every edit is asserted to match
once."""
import sys
from pathlib import Path
D = Path(sys.argv[1])


def edit(name, old, new):
    p = D / name; s = p.read_text()
    assert s.count(old) == 1, (name, old[:70], s.count(old))
    p.write_text(s.replace(old, new))


edit('sv.c', """    SV *const sva = S_sv_arena_of(sv);
    void *const life = SV_ARENA_LIFE(sva);
    SV *const child = (SV *)capstone_cap_derive(life,
        PTR2UV(sv) - __builtin_capstone_cap_get_base(life), sizeof(SV));
    SV_ARENA_ISSUED(sva)[sv - (sva + 1)] = child;
    return child;
}""", """    SV *const sva = S_sv_arena_of(sv);
#ifdef PERL_CAPSTONE_SUBLET_HEADS_RENEW
    SV **const slot = &SV_ARENA_ISSUED(sva)[sv - (sva + 1)];
    void *life;
    SV *child = *slot;
    if (child)
        return child;		/* renewed when it was freed: already its own lifetime */
    life = SV_ARENA_LIFE(sva);
    child = (SV *)capstone_cap_derive(life,
        PTR2UV(sv) - __builtin_capstone_cap_get_base(life), sizeof(SV));
    *slot = child;
    return child;
#else
    void *const life = SV_ARENA_LIFE(sva);
    SV *const child = (SV *)capstone_cap_derive(life,
        PTR2UV(sv) - __builtin_capstone_cap_get_base(life), sizeof(SV));
    SV_ARENA_ISSUED(sva)[sv - (sva + 1)] = child;
    return child;
#endif
}""")

edit('sv.c', """    if (*issued && !(SvFLAGS(head) & SVf_BREAK)) {
        capstone_cap_revoke_child(SV_ARENA_LIFE(sva), *issued);
        *issued = NULL;
    }
    return head;""", """    if (*issued && !(SvFLAGS(head) & SVf_BREAK)) {
#ifdef PERL_CAPSTONE_SUBLET_HEADS_RENEW
        SV *const renewed = (SV *)PERL_CAP_RENEW(*issued);
        if (renewed) {
            *issued = renewed;
            return head;
        }
        /* the renewal failed and the old lifetime is still live: end it as the port does */
#endif
        capstone_cap_revoke_child(SV_ARENA_LIFE(sva), *issued);
        *issued = NULL;
    }
    return head;""")
print('heads-renew: applied to', D)
