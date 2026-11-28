#!/usr/bin/env python3
"""he-renew-lb.py DIR (nested lane, 2026-11-26): an EXPERIMENT ARM on a perlbench 5.22 tree that port-he.py produced,
never a port script. It is oneop's he-renew.py (fig10/renew, 149231ca) failing closed, for Line B, where CRENEW can
fail. Under PERL_CAPSTONE_SUBLET_HE_RENEW (with PERL_CAPSTONE_SUBLET_HE), in capstone_sv_segs.h (hv.c's copy of the
segment helpers):
  - S_sv_seg_retire renews the entry (PERL_CAP_RENEW, renew-shim.h), and del_HE links the renewed capability into the
    free list in the HeNEXT word. If the renewal fails (an untagged 0: the old lifetime is still live), it revokes and
    returns the segment's pointer, as the port does;
  - S_sv_seg_issue hands a popped capability out as it is when its bounds are narrower than a segment (a renewed
    entry); an entry fresh from an arena, or one whose renewal failed, arrives as the segment's pointer and derives.
Without the new define the preprocessed source is the HE port's. Every edit is asserted to match once."""
import sys
from pathlib import Path
D = Path(sys.argv[1])


def edit(name, old, new):
    p = D / name; s = p.read_text()
    assert s.count(old) == 1, (name, old[:70], s.count(old))
    p.write_text(s.replace(old, new))


edit('capstone_sv_segs.h', """    struct sv_body_seg *const s = S_sv_seg_of(a);
    return s ? capstone_cap_derive(s->life, a - s->lo, size) : own;
}""", """    struct sv_body_seg *const s = S_sv_seg_of(a);
#ifdef PERL_CAPSTONE_SUBLET_HE_RENEW
    if (s && __builtin_capstone_cap_get_end(own) - __builtin_capstone_cap_get_base(own) < SV_BODY_SEG)
\treturn own;\t\t/* renewed when it was freed: already its own lifetime */
#endif
    return s ? capstone_cap_derive(s->life, a - s->lo, size) : own;
}""")
edit('capstone_sv_segs.h', """\treturn thing;
#ifndef PERL_CAPSTONE_SUBLET_BODIES_NOREVOKE""", """\treturn thing;
#ifdef PERL_CAPSTONE_SUBLET_HE_RENEW
    {
\tvoid *const renewed = PERL_CAP_RENEW(thing);
\tif (renewed)
\t    return renewed;
\t/* the renewal failed and the old lifetime is still live: end it as the port does */
    }
#endif
#ifndef PERL_CAPSTONE_SUBLET_BODIES_NOREVOKE""")
print('he-renew-lb: applied to', D)
