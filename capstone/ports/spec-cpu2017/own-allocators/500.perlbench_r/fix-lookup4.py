#!/usr/bin/env python3
"""fix-lookup4.py SV_C: on top of fix-lookup3.py, Perl_sv_arena_head inline inside sv.c (nested lane, 2026-11-21). The
exported function stays for the other files; after its definition sv.c's own uses (SvIS_FREED, SvTYPE_MAYBE_FREED:
two in every sv_setsv_flags) expand to a forced-inline copy. What is protected does not change. Every edit is asserted
to match exactly once."""
import sys
p = sys.argv[1]; s = open(p).read()
old = """SV *
Perl_sv_arena_head(const SV *sv)
{
    SV *const sva = S_sv_arena_of(sv);
    return sva ? sva + 1 + (PTR2UV(sv) - PTR2UV(sva + 1)) / sizeof(SV)
               : (SV *)sv;
}
"""
new = """static inline __attribute__((always_inline)) SV *
S_sv_arena_head_inline(const SV *sv)
{
    SV *const sva = S_sv_arena_of(sv);
    return sva ? sva + 1 + (PTR2UV(sv) - PTR2UV(sva + 1)) / sizeof(SV)
               : (SV *)sv;
}

SV *
Perl_sv_arena_head(const SV *sv)
{
    return S_sv_arena_head_inline(sv);
}
/* fix-lookup4.py: sv.c's own uses below this point take the inline copy. */
#define Perl_sv_arena_head(sv) S_sv_arena_head_inline(sv)
"""
assert s.count(old) == 1, s.count(old)
open(p, 'w').write(s.replace(old, new))
print('arena head inline in sv.c')
