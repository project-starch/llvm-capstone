#!/usr/bin/env python3
"""port-sv-heads.py DIR: apply the sv-heads-as-sublet-lifetimes patch (capstone/ports/perl/sublet/patches/5.36.3/0001,
blob 72db9236) to SPEC CPU2017 500.perlbench_r's Perl 5.22.1 sources in DIR, under PERL_CAPSTONE_SUBLET (nested lane,
2026-11-18). Without the define the preprocessed source is unchanged. Differences from 5.36.3: uproot_SV and more_sv
live in sv.c (no sv_inline.h; S_more_sv is static), and S_doeval is 5.22's name for the eval compiler. Every edit is
asserted to match exactly once."""
import sys
from pathlib import Path
D = Path(sys.argv[1])

def edit(name, old, new):
    p = D / name; s = p.read_text()
    assert s.count(old) == 1, (name, old[:70], s.count(old))
    p.write_text(s.replace(old, new))

# gv.h: GvIO's type test reads through the arena's own pointer
edit('gv.h', """         SvTYPE((const SV*)(gv)) == SVt_PVGV \\
      || SvTYPE((const SV*)(gv)) == SVt_PVLV  \\""",
"""         SvTYPE_MAYBE_FREED((const SV*)(gv)) == SVt_PVGV \\
      || SvTYPE_MAYBE_FREED((const SV*)(gv)) == SVt_PVLV  \\""")
# pp.c: pp_anonconst
edit('pp.c', "newCONSTSUB(SvTYPE(CopSTASH(PL_curcop))==SVt_PVHV", "newCONSTSUB(SvTYPE_MAYBE_FREED(CopSTASH(PL_curcop))==SVt_PVHV")
# pp_ctl.c: pp_caller and the eval compiler
edit('pp_ctl.c', "stash_hek = SvTYPE(CopSTASH(cx->blk_oldcop)) == SVt_PVHV", "stash_hek = SvTYPE_MAYBE_FREED(CopSTASH(cx->blk_oldcop)) == SVt_PVHV")
edit('pp_ctl.c', "if (SvTYPE(PL_curstash) != SVt_PVHV) PL_curstash = NULL;", "if (SvTYPE_MAYBE_FREED(PL_curstash) != SVt_PVHV) PL_curstash = NULL;")
# sv.h: SvIS_FREED and SvTYPE_MAYBE_FREED
edit('sv.h', "#define SvIS_FREED(sv)\tUNLIKELY(((sv)->sv_flags == SVTYPEMASK))\n",
"""#ifdef PERL_CAPSTONE_SUBLET
/* Capstone Sublet (sv.c): each head is a child lifetime of its arena, revoked
   when the head is freed. A pointer that may be stale is therefore not read
   through: the arena's own pointer to the same head is. */
#  if defined(MULTIPLICITY)
#    error "PERL_CAPSTONE_SUBLET supports one interpreter (no MULTIPLICITY)"
#  endif
extern SV *Perl_sv_arena_head(const SV *sv);
#  define SvIS_FREED(sv) \\
    UNLIKELY((Perl_sv_arena_head((const SV *)(sv))->sv_flags == SVTYPEMASK))
#  define SvTYPE_MAYBE_FREED(sv)\tSvTYPE(Perl_sv_arena_head((const SV *)(sv)))
#else
#define SvIS_FREED(sv)\tUNLIKELY(((sv)->sv_flags == SVTYPEMASK))
#  define SvTYPE_MAYBE_FREED(sv)\tSvTYPE(sv)
#endif
""")
# sv.c: the arena lifetime, issue on uproot, retire on plant
old_plant = """#define plant_SV(p) \\
    STMT_START {\t\t\t\t\t\\
\tconst U32 old_flags = SvFLAGS(p);\t\t\t\\
\tMEM_LOG_DEL_SV(p, __FILE__, __LINE__, FUNCTION__);  \\
\tDEBUG_SV_SERIAL(p);\t\t\t\t\\
\tFREE_SV_DEBUG_FILE(p);\t\t\t\t\\
\tPOISON_SV_HEAD(p);\t\t\t\t\\
\tSvFLAGS(p) = SVTYPEMASK;\t\t\t\\
\tif (!(old_flags & SVf_BREAK)) {\t\t\\
\t    SvARENA_CHAIN_SET(p, PL_sv_root);\t\\
\t    PL_sv_root = (p);\t\t\t\t\\
\t}\t\t\t\t\t\t\\
\t--PL_sv_count;\t\t\t\t\t\\
    } STMT_END

#define uproot_SV(p) \\
    STMT_START {\t\t\t\t\t\\
\t(p) = PL_sv_root;\t\t\t\t\\
\tPL_sv_root = MUTABLE_SV(SvARENA_CHAIN(p));\t\t\\
\t++PL_sv_count;\t\t\t\t\t\\
    } STMT_END
"""
new_plant = """#ifdef PERL_CAPSTONE_SUBLET
#include <capstone/capability.h>

/* Capstone Sublet: every SV head is a child lifetime of its arena, bounded to
   the head (CDERIVE), and revoked when the head is freed (CREVOKE), so every
   copy of a freed SV* is dead before the head is handed out again.

   An arena is one Newx block: upstream's PERL_ARENA_SIZE of heads, then the
   child each head was handed out as (NULL while free). The arena's own first
   slot, which is not an SV, keeps the arena's lifetime in its sv_u. The free
   list holds the arena's own pointers, as does everything below that works on
   a head by address. */
#define SV_ARENA_HEADS\t\t(PERL_ARENA_SIZE / sizeof(SV) - 1)
#define SV_ARENA_LIFE(sva)\t((void *)(sva)->sv_u.svu_rv)
#define SV_ARENA_ISSUED(sva)\t((SV **)((char *)(sva) + PERL_ARENA_SIZE))

static SV *sv_arena_last;\t/* the arena found last; most lookups repeat it */

/* The arena holding the head at sv's address, or NULL. sv is never read. */
static SV *
S_sv_arena_of(const SV *sv)
{
    SV *sva = sv_arena_last;
    if (sva && PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV))
        return sva;
    for (sva = PL_sv_arenaroot; sva; sva = MUTABLE_SV(SvANY(sva)))
        if (PTR2UV(sv) - PTR2UV(sva + 1) < SV_ARENA_HEADS * sizeof(SV))
            return sv_arena_last = sva;
    return NULL;
}

/* The arena's own pointer to the head at sv's address; sv itself for an SV
   that no arena holds (the immortals). */
SV *
Perl_sv_arena_head(const SV *sv)
{
    SV *const sva = S_sv_arena_of(sv);
    return sva ? sva + 1 + (PTR2UV(sv) - PTR2UV(sva + 1)) / sizeof(SV)
               : (SV *)sv;
}

/* Hand out the free head sv, the arena's own pointer, as its child. */
static SV *
S_sv_arena_issue(SV *sv)
{
    SV *const sva = S_sv_arena_of(sv);
    void *const life = SV_ARENA_LIFE(sva);
    SV *const child = (SV *)capstone_cap_derive(life,
        PTR2UV(sv) - __builtin_capstone_cap_get_base(life), sizeof(SV));
    SV_ARENA_ISSUED(sva)[sv - (sva + 1)] = child;
    return child;
}

/* End the lifetime of the head at p's address and return the arena's own
   pointer to it. An SVf_BREAK head is kept: upstream keeps it off the free
   list because pointers to it are expected to outlive it. */
static SV *
S_sv_arena_retire(SV *p)
{
    SV *const sva = S_sv_arena_of(p);
    SV *head;
    SV **issued;
    if (!sva)
        return p;
    head = sva + 1 + (PTR2UV(p) - PTR2UV(sva + 1)) / sizeof(SV);
    issued = &SV_ARENA_ISSUED(sva)[head - (sva + 1)];
    if (*issued && !(SvFLAGS(head) & SVf_BREAK)) {
        capstone_cap_revoke_child(SV_ARENA_LIFE(sva), *issued);
        *issued = NULL;
    }
    return head;
}

#define plant_SV(p0) \\
    STMT_START {\t\t\t\t\t\\
\tSV *const plant_p = S_sv_arena_retire(p0);\t\t\\
\tconst U32 old_flags = SvFLAGS(plant_p);\t\t\t\\
\tMEM_LOG_DEL_SV(plant_p, __FILE__, __LINE__, FUNCTION__);  \\
\tDEBUG_SV_SERIAL(plant_p);\t\t\t\t\\
\tFREE_SV_DEBUG_FILE(plant_p);\t\t\t\t\\
\tPOISON_SV_HEAD(plant_p);\t\t\t\t\\
\tSvFLAGS(plant_p) = SVTYPEMASK;\t\t\t\\
\tif (!(old_flags & SVf_BREAK)) {\t\t\\
\t    SvARENA_CHAIN_SET(plant_p, PL_sv_root);\t\\
\t    PL_sv_root = (plant_p);\t\t\t\t\\
\t}\t\t\t\t\t\t\\
\t--PL_sv_count;\t\t\t\t\t\\
    } STMT_END

/* The free list holds the arena's own pointers; the head is handed out as a
   child lifetime of its arena, bounded to it. */
#define uproot_SV(p) \\
    STMT_START {\t\t\t\t\t\\
\t(p) = PL_sv_root;\t\t\t\t\\
\tPL_sv_root = MUTABLE_SV(SvARENA_CHAIN(p));\t\t\\
\t++PL_sv_count;\t\t\t\t\t\\
\t(p) = S_sv_arena_issue(p);\t\t\t\\
    } STMT_END
#else
""" + old_plant + "#endif\n"
edit('sv.c', old_plant, new_plant)
edit('sv.c', """    Newx(chunk,PERL_ARENA_SIZE,char);  /* Safefree() in sv_free_arenas() */
    sv_add_arena(chunk, PERL_ARENA_SIZE, 0);
    uproot_SV(sv);
    return sv;""", """#ifdef PERL_CAPSTONE_SUBLET
    /* The heads, then the child each was handed out as (above plant_SV). */
    Newxz(chunk, PERL_ARENA_SIZE + SV_ARENA_HEADS * sizeof(SV *), char);
    sv_add_arena(chunk, PERL_ARENA_SIZE, 0);
    ((SV *)chunk)->sv_u.svu_rv = (SV *)capstone_cap_derive(chunk,
        __builtin_capstone_cap_get_cursor(chunk)
            - __builtin_capstone_cap_get_base(chunk), PERL_ARENA_SIZE);
#else
    Newx(chunk,PERL_ARENA_SIZE,char);  /* Safefree() in sv_free_arenas() */
    sv_add_arena(chunk, PERL_ARENA_SIZE, 0);
#endif
    uproot_SV(sv);
    return sv;""")
edit('sv.c', "    PL_sv_arenaroot = 0;\n    PL_sv_root = 0;\n", "    PL_sv_arenaroot = 0;\n    PL_sv_root = 0;\n#ifdef PERL_CAPSTONE_SUBLET\n    sv_arena_last = NULL;\n#endif\n")

# ---- further sites of Perl 5.22.1 that touch a head that may already be freed (found on SPEC's test.pl, 2026-11-18)
# pp.c S_rv2gv and pp_bless: a type test of the current stash, which may be freed ("freed package").
edit('pp.c', "\t\t\tif (SvTYPE(stash) != SVt_PVHV) stash = NULL;\n", "\t\t\tif (SvTYPE_MAYBE_FREED(stash) != SVt_PVHV) stash = NULL;\n")
edit('pp.c', """	stash = CopSTASH(PL_curcop);
	if (SvTYPE(stash) != SVt_PVHV)
	    Perl_croak(aTHX_ "Attempt to bless into a freed package");""",
"""	stash = CopSTASH(PL_curcop);
	if (SvTYPE_MAYBE_FREED(stash) != SVt_PVHV)
	    Perl_croak(aTHX_ "Attempt to bless into a freed package");""")
# sv.c sv_resetpvn: the same type test of a stash that may be freed.
edit('sv.c', "    if (!stash || SvTYPE(stash) != SVt_PVHV)\n", "    if (!stash || SvTYPE_MAYBE_FREED(stash) != SVt_PVHV)\n")
# sv.c S_glob_assign_glob: LEAVE may free sstr (*x = $x); upstream then reads SvTAINTED(sstr) and GvAV(sstr), a read
# of a freed head. Read both before LEAVE (the Capstone port of 5.36.3 does the same in its base patches).
edit('sv.c', """    GvGP_set(dstr, gp_ref(GvGP(sstr)));
    LEAVE;

    if (SvTAINTED(sstr))
	SvTAINT(dstr);
""", """    GvGP_set(dstr, gp_ref(GvGP(sstr)));
#ifdef PERL_CAPSTONE_SUBLET
    {
	const bool sstr_tainted = cBOOL(SvTAINTED(sstr));
	sstr_has_av = cBOOL(GvAV((const GV *)sstr));
	LEAVE;
	if (sstr_tainted)
	    SvTAINT(dstr);
    }
#else
    LEAVE;

    if (SvTAINTED(sstr))
	SvTAINT(dstr);
#endif
""")
edit('sv.c', "    I32 mro_changes = 0; /* 1 = method, 2 = isa, 3 = recursive isa */\n    HV *old_stash = NULL;\n",
 "    I32 mro_changes = 0; /* 1 = method, 2 = isa, 3 = recursive isa */\n    HV *old_stash = NULL;\n#ifdef PERL_CAPSTONE_SUBLET\n    bool sstr_has_av = FALSE;\n#endif\n")
edit('sv.c', """    if(mro_changes == 2) {
      if (GvAV((const GV *)sstr)) {
""", """    if(mro_changes == 2) {
#ifdef PERL_CAPSTONE_SUBLET
      if (sstr_has_av) {
#else
      if (GvAV((const GV *)sstr)) {
#endif
""")
# pad.c pad_free: the pad entry may be stale. pad_free runs when an op is freed, with whatever pad is current, so
# PL_curpad[po] can be a head that was freed and even handed out again (t/op/goto.t). Upstream tests and clears
# SVs_PADTMP in whatever is at that address; so does this, through the arena's own pointer to the same head.
edit('pad.c', """    if (sv && sv != &PL_sv_undef && !SvPADMY(sv))
	SvFLAGS(sv) &= ~SVs_PADTMP;
""", """#ifdef PERL_CAPSTONE_SUBLET
    if (sv && sv != &PL_sv_undef) {
	SV *const pad_head = Perl_sv_arena_head(sv);
	if (!SvPADMY(pad_head))
	    SvFLAGS(pad_head) &= ~SVs_PADTMP;
    }
#else
    if (sv && sv != &PL_sv_undef && !SvPADMY(sv))
	SvFLAGS(sv) &= ~SVs_PADTMP;
#endif
""")
# pp_hot.c pp_aassign: an element on the right may be a freed scalar (t/op/sort.t relies on the panic that
# sv_setsv raises for it); upstream tests its flags for get-magic first. A freed head has none.
edit('pp_hot.c', "\t\tif (LIKELY(*relem))\n\t\t    SvGETMAGIC(*relem); /* before newSV, in case it dies */\n",
 "#ifdef PERL_CAPSTONE_SUBLET\n\t\tif (LIKELY(*relem) && !SvIS_FREED(*relem))\n#else\n\t\tif (LIKELY(*relem))\n#endif\n\t\t    SvGETMAGIC(*relem); /* before newSV, in case it dies */\n")

print('ported')
