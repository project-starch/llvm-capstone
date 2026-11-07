#!/usr/bin/env python3
"""Apply the independent provenance adapter to an external Perl 5.22 tree.

The caller must provide a fresh copy. No benchmark inputs are read or changed.
The emitted source alternative replaces existing source files only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("tree", type=Path)
parser.add_argument("--record", type=Path,
                    help="Write source identity outside the licensed source tree")
args = parser.parse_args()
root = args.tree.resolve()
baseline = json.loads((HERE/'baseline.json').read_text())
before = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
          for name in baseline['files']}
if before != baseline['files']:
    raise SystemExit('Unsupported source baseline: ' + ', '.join(
        name for name in before if before[name] != baseline['files'][name]))
if args.record and args.record.exists():
    raise SystemExit('Refusing to replace source record: ' + str(args.record))
def inventory():
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


complete_before = inventory() if args.record else None
texts = {}


def read(name):
    if name not in texts:
        texts[name] = (root / name).read_text()
    return texts[name]


def replace(name, old, new, count=1):
    s = read(name)
    if s.count(old) != count:
        raise RuntimeError(f"{name}: expected {count} matches for {old[:90]!r}, got {s.count(old)}")
    texts[name] = s.replace(old, new)


def region(name, function):
    s = read(name)
    suffix = r"\s*\{" if function.startswith('PP(') else r"\([^;]*?\s*\{"
    m = re.search(r"\n" + re.escape(function) + suffix, s)
    if not m:
        raise RuntimeError(f"{name}: function not found: {function}")
    brace = m.end()-1
    # Skip strings and comments when matching the function body.
    tokens = re.finditer(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]', s[brace:], re.S)
    depth = 0
    for token in tokens:
        if token.group() == '{': depth += 1
        elif token.group() == '}':
            depth -= 1
            if not depth: return m.start()+1, brace, brace+token.end()
    raise RuntimeError("unclosed function")


def wrap(name, function, rettype, params, callargs, before, after="", result=True, cap_callargs=None):
    s = read(name)
    start, brace, end = region(name, function)
    original = 'capstone_original_' + function
    body = s[start:end].replace(function+'(', original+'(', 1)
    call = original + '(' + callargs + ')'
    wrapper = '\n\n' + rettype + '\n' + function + '(' + params + ')\n{\n'
    if before: wrapper += '#ifdef CAPSTONE_PERL_PROVENANCE\n' + before + '\n#endif\n'
    if cap_callargs is not None:
        if result: raise RuntimeError('conditional call currently only supports void wrappers')
        wrapper += '#ifdef CAPSTONE_PERL_PROVENANCE\n    '+original+'('+cap_callargs+');\n#else\n    '+call+';\n#endif\n'
    else:
        wrapper += ('    '+rettype+' result = '+call+';\n') if result else ('    '+call+';\n')
    if after: wrapper += '#ifdef CAPSTONE_PERL_PROVENANCE\n' + after + '\n#endif\n'
    if result: wrapper += '    return result;\n'
    wrapper += '}\n'
    texts[name] = s[:start]+body+wrapper+s[end:]


def insert_entry(name, function, code):
    s = read(name)
    _, brace, _ = region(name, function)
    texts[name] = s[:brace+1]+'\n#ifdef CAPSTONE_PERL_PROVENANCE\n'+code+'\n#endif\n'+s[brace+1:]


def in_function(name, function, old, new, count=1):
    s = read(name)
    start, _, end = region(name, function)
    body = s[start:end]
    if body.count(old) != count:
        raise RuntimeError(f'{function}: expected {count} matches for {old!r}, got {body.count(old)}')
    texts[name] = s[:start]+body.replace(old,new)+s[end:]


header = (HERE/'capstone-provenance.h').read_text()
replace('perl.h', '#endif /* Include guard */', header+'\n#endif /* Include guard */')
texts['sv.c'] = read('sv.c')+'\n'+(HERE/'capstone-provenance.inc').read_text()
replace('sv.c','#include "regcomp.h"','#include "regcomp.h"\n#ifdef CAPSTONE_PERL_PROVENANCE\nstatic bool cap_prov_offset(const U8 *, STRLEN, int, STRLEN *);\n#endif')
# These APIs hand out a writable string buffer. Explicitly authorized value
# transformations below retain a pre-write snapshot and publish its mapped
# fields only after the operation succeeds. Ordinary read APIs stay unchanged.
replace('sv.h','(SvPOK_pure_nogthink(sv) \\\n', '(CAP_PROV_WRITE(sv), SvPOK_pure_nogthink(sv) \\\n',3)
replace('sv.h','(SvLEN(sv) < (len) ? sv_grow(sv,len) : SvPVX_mutable(sv))',
        '(CAP_PROV_WRITE(sv), SvLEN(sv) < (len) ? sv_grow(sv,len) : SvPVX_mutable(sv))')
insert_entry('sv.c','Perl_sv_grow','    Perl_cap_prov_clear(aTHX_ sv);')

snap = '    SV *cap_old = Perl_cap_prov_snapshot(aTHX_ sv);\n    Perl_cap_prov_clear(aTHX_ sv);'
restore = '    if (cap_old) Perl_cap_prov_copy(aTHX_ sv, cap_old, 0, SvCUR(cap_old), 0, 0);'

wrap('sv.c','Perl_sv_setsv_flags','void', 'pTHX_ SV *dstr, SV *sstr, const I32 flags',
     'aTHX_ dstr, sstr, flags',
     '    SV *cap_source;\n    if (dstr == sstr) return;\n    if (SvIS_FREED(dstr) || (sstr && SvIS_FREED(sstr))) {\n        capstone_original_Perl_sv_setsv_flags(aTHX_ dstr, sstr, flags);\n        return;\n    }\n    SV_CHECK_THINKFIRST_COW_DROP(dstr);\n    if (sstr && (flags & SV_GMAGIC)) SvGETMAGIC(sstr);\n    cap_source = Perl_cap_prov_observed(aTHX_ sstr);\n    Perl_cap_prov_clear(aTHX_ dstr);',
     '    if (cap_source) Perl_cap_prov_copy(aTHX_ dstr, cap_source, 0, SvCUR(cap_source), 0, 0);',False,
     'aTHX_ dstr, sstr, flags & ~SV_GMAGIC')
wrap('sv.c','Perl_sv_setsv_cow','SV *','pTHX_ SV *dstr, SV *sstr','aTHX_ dstr, sstr',
     '    SV *cap_source = Perl_cap_prov_snapshot(aTHX_ sstr);\n    Perl_cap_prov_clear(aTHX_ dstr);',
     '    if (cap_source) Perl_cap_prov_copy(aTHX_ result, cap_source, 0, SvCUR(cap_source), 0, 0);')

for func in ('Perl_sv_setpvn','Perl_sv_setpv','Perl_sv_usepvn_flags','Perl_sv_gets',
             'Perl_sv_setiv','Perl_sv_setuv','Perl_sv_setnv','Perl_sv_inc_nomg','Perl_sv_dec_nomg',
             'Perl_sv_vsetpvfn'):
    insert_entry('sv.c',func,'    Perl_cap_prov_clear(aTHX_ sv);')

wrap('sv.c','Perl_sv_utf8_upgrade_flags_grow','STRLEN',
     'pTHX_ SV *const sv, const I32 flags, STRLEN extra','aTHX_ sv, flags, extra',snap,
     '    if (cap_old) Perl_cap_prov_copy(aTHX_ sv, cap_old, 0, SvCUR(cap_old), 0, SvUTF8(cap_old) ? 0 : 1);')
wrap('sv.c','Perl_sv_utf8_downgrade','bool','pTHX_ SV *const sv, const bool fail_ok',
     'aTHX_ sv, fail_ok',snap,
     '    if (cap_old) Perl_cap_prov_copy(aTHX_ sv, cap_old, 0, SvCUR(cap_old), 0, result && SvUTF8(cap_old) ? -1 : 0);')
for func in ('Perl_sv_utf8_encode','Perl_sv_utf8_decode'):
    insert_entry('sv.c',func,'    Perl_cap_prov_clear(aTHX_ sv);')

wrap('sv.c','Perl_sv_catpvn_flags','void',
     'pTHX_ SV *const dsv, const char *sstr, const STRLEN slen, const I32 flags',
     'aTHX_ dsv, sstr, slen, flags',
     '    SV *cap_old = Perl_cap_prov_snapshot(aTHX_ dsv);\n    Perl_cap_prov_clear(aTHX_ dsv);',
     '    if (cap_old) Perl_cap_prov_copy(aTHX_ dsv, cap_old, 0, SvCUR(cap_old), 0, !SvUTF8(cap_old) && SvUTF8(dsv) ? 1 : 0);',False)
in_function('sv.c','Perl_sv_catsv_flags', 'sv_catpvn_flags(dsv, spv, slen,',
            'CAP_PROV_CATPVN(dsv, ssv, spv, slen,')
in_function('mg.c','Perl_mg_get',
            '    if (PL_localizing == 1 && sv == DEFSV) return 0;',
            '    if (PL_localizing == 1 && sv == DEFSV) return 0;\n#ifdef CAPSTONE_PERL_PROVENANCE\n    Perl_cap_prov_prepare_get(aTHX_ sv);\n#endif')

wrap('sv.c','Perl_sv_insert_flags','void',
     'pTHX_ SV *const bigstr, const STRLEN offset, const STRLEN len, const char *const little, const STRLEN littlelen, const U32 flags',
     'aTHX_ bigstr, offset, len, little, littlelen, flags',
     '    SV *cap_old = Perl_cap_prov_snapshot(aTHX_ bigstr);\n    Perl_cap_prov_clear(aTHX_ bigstr);',
     '    if (cap_old) {\n        STRLEN n = SvCUR(cap_old);\n        if (offset <= n) Perl_cap_prov_copy(aTHX_ bigstr, cap_old, 0, offset, 0, 0);\n        if (offset <= n && len <= n-offset)\n            Perl_cap_prov_copy(aTHX_ bigstr, cap_old, offset+len, n-offset-len, offset+littlelen, 0);\n    }',False)
wrap('sv.c','Perl_sv_chop','void','pTHX_ SV *const sv, const char *const ptr','aTHX_ sv, ptr',
     snap+'\n    STRLEN cap_off = cap_old && ptr ? ptr-SvPVX_const(sv) : 0;',
     '    if (cap_old && cap_off <= SvCUR(cap_old)) Perl_cap_prov_copy(aTHX_ sv, cap_old, cap_off, SvCUR(cap_old)-cap_off, 0, 0);',False)
wrap('sv.c','Perl_sv_replace','void','pTHX_ SV *const sv, SV *const nsv','aTHX_ sv, nsv',
     '    SV *cap_source = Perl_cap_prov_snapshot(aTHX_ nsv);\n    Perl_cap_prov_clear(aTHX_ sv);\n    Perl_cap_prov_clear(aTHX_ nsv);',
     '    if (cap_source) Perl_cap_prov_copy(aTHX_ sv, cap_source, 0, SvCUR(cap_source), 0, 0);',False)

# Raw append helpers preserve the existing prefix, but cannot import authority
# from a char* without an SV source. Route them through the audited helper.
wrap('sv.c','Perl_sv_catpv','void','pTHX_ SV *const sv, const char *ptr','aTHX_ sv, ptr',snap,restore,False)

# Pack/unpack receives explicit provenance through its recursive parse context.
replace('pp_pack.c','typedef struct tempsym {','typedef struct tempsym {\n    SV *cap_source;\n    const char *cap_base;')
replace('pp_pack.c','(symptr)->patptr   = (p);', '(symptr)->cap_source = NULL; (symptr)->cap_base = NULL; \\\n\t(symptr)->patptr   = (p);')
replace('pp_pack.c','Perl_unpackstring(pTHX_ const char *pat, const char *patend, const char *s, const char *strend, U32 flags)',
        'capstone_unpackstring(pTHX_ const char *pat, const char *patend, const char *s, const char *strend, U32 flags, SV *cap_source)')
replace('pp_pack.c','    TEMPSYM_INIT(&sym, pat, patend, flags);\n\n    return unpack_rec',
        '    TEMPSYM_INIT(&sym, pat, patend, flags);\n    sym.cap_source = cap_source;\n    sym.cap_base = s;\n\n    return unpack_rec')
# Temporary UTF8 input owns a transformed copy of the same provenance.
replace('pp_pack.c','\tflags |= FLAG_DO_UTF8;\n    }', '''\tflags |= FLAG_DO_UTF8;
#ifdef CAPSTONE_PERL_PROVENANCE
        if (cap_source) {
            SV *converted = sv_2mortal(newSVpvn(s, len));
            SvUTF8_on(converted);
            Perl_cap_prov_copy(aTHX_ converted, cap_source, 0, SvCUR(cap_source), 0, 1);
            cap_source = converted;
        }
#endif
    }''')
start,brace,end=region('pp_pack.c','capstone_unpackstring')
s=read('pp_pack.c')
texts['pp_pack.c']=s[:end]+'''

I32
Perl_unpackstring(pTHX_ const char *pat, const char *patend, const char *s, const char *strend, U32 flags)
{
    return capstone_unpackstring(aTHX_ pat, patend, s, strend, flags, NULL);
}
'''+s[end:]
replace('pp_pack.c','    cnt = unpackstring(pat, patend, s, strend,','''#ifdef CAPSTONE_PERL_PROVENANCE
    {
        SV *snapshot = Perl_cap_prov_observed(aTHX_ right);
        if (snapshot) {
            right = snapshot;
            s = SvPVX_const(right);
            strend = s + SvCUR(right);
        }
    }
#endif
    cnt = capstone_unpackstring(aTHX_ pat, patend, s, strend,''')
replace('pp_pack.c','| (DO_UTF8(right) ? FLAG_DO_UTF8 : 0));','| (DO_UTF8(right) ? FLAG_DO_UTF8 : 0), right);')
replace('pp_pack.c','                SHIFT_VAR(utf8, s, strend, aptr, datumtype, needs_swap);', '''                const char *cap_begin = s;
                SHIFT_VAR(utf8, s, strend, aptr, datumtype, needs_swap);
#ifdef CAPSTONE_PERL_PROVENANCE
                aptr = (char *)Perl_cap_prov_pointer(aTHX_ symptr->cap_source,
                    cap_begin-symptr->cap_base, s-cap_begin, aptr);
#endif''',2)
replace('pp_pack.c','\t\tconst char *aptr;\n\n\t\tfromstr = NEXTFROM;', '\t\tconst char *aptr;\n                STRLEN cap_begin = cur-start;\n\n\t\tfromstr = NEXTFROM;')
replace('pp_pack.c','                PUSH_VAR(utf8, cur, aptr, needs_swap);', '''                PUSH_VAR(utf8, cur, aptr, needs_swap);
#ifdef CAPSTONE_PERL_PROVENANCE
                SvCUR_set(cat, cur-start);
                Perl_cap_prov_add(aTHX_ cat, cap_begin, cur-start-cap_begin, aptr);
#endif''')
replace('pp_pack.c','\tSvCUR_set(cat, cur - start);\n      no_change:', '''\tSvCUR_set(cat, cur - start);
#ifdef CAPSTONE_PERL_PROVENANCE
        Perl_cap_prov_trim(aTHX_ cat, cur-start);
#endif
      no_change:''')
wrap('pp_pack.c','marked_upgrade','void','pTHX_ SV *sv, tempsym_t *sym_ptr','aTHX_ sv, sym_ptr',snap,
     '    if (cap_old) Perl_cap_prov_copy(aTHX_ sv, cap_old, 0, SvCUR(cap_old), 0, SvUTF8(cap_old) ? 0 : 1);',False)
wrap('pp_pack.c','S_sv_exp_grow','char *','pTHX_ SV *sv, STRLEN needed',
     'aTHX_ sv, needed',snap,restore)

# The raw string fields are value slices. Padding has no provenance.
in_function('pp_pack.c','S_pack_rec','\t    const char *aptr;\n\n\t    fromstr = NEXTFROM;', '''\t    const char *aptr;
#ifdef CAPSTONE_PERL_PROVENANCE
            SV *cap_string;
            STRLEN cap_at, cap_written;
            int cap_encoding;
#endif
\t    fromstr = NEXTFROM;''')
in_function('pp_pack.c','S_pack_rec','\t    aptr = SvPV_const(fromstr, fromlen);\n\t    if (DO_UTF8(fromstr)) {', '''\t    aptr = SvPV_const(fromstr, fromlen);
#ifdef CAPSTONE_PERL_PROVENANCE
            cap_string = Perl_cap_prov_snapshot(aTHX_ fromstr);
            cap_at = cur-start;
            cap_encoding = 0;
#endif
\t    if (DO_UTF8(fromstr)) {''')
in_function('pp_pack.c','S_pack_rec','\t\t    cur = start + SvCUR(cat);\n\t\t}', '''\t\t    cur = start + SvCUR(cat);
#ifdef CAPSTONE_PERL_PROVENANCE
                    cap_at = cur-start;
#endif
\t\t}''')
in_function('pp_pack.c','S_pack_rec','\t\tGROWING(0, cat, start, cur, len);\n\t\tif (!S_utf8_to_bytes', '''#ifdef CAPSTONE_PERL_PROVENANCE
                cap_encoding = -1;
#endif
\t\tGROWING(0, cat, start, cur, len);
\t\tif (!S_utf8_to_bytes''')
in_function('pp_pack.c','S_pack_rec','\t    } else if (utf8) {\n\t\tif (howlen == e_star)', '''\t    } else if (utf8) {
#ifdef CAPSTONE_PERL_PROVENANCE
                cap_encoding = 1;
#endif
\t\tif (howlen == e_star)''')
in_function('pp_pack.c','S_pack_rec',"\t    memset(cur, datumtype == 'A' ? ' ' : '\\0', len);", '''#ifdef CAPSTONE_PERL_PROVENANCE
            cap_written = cur-start;
            SvCUR_set(cat, cap_written);
            if (cap_string) Perl_cap_prov_copy(aTHX_ cat, cap_string, 0, SvCUR(cap_string), cap_at, cap_encoding);
#endif
\t    memset(cur, datumtype == 'A' ? ' ' : '\\0', len);''')
in_function('pp_pack.c','S_unpack_rec','\t    if (utf8) {\n\t\tSvUTF8_on(sv);', '''#ifdef CAPSTONE_PERL_PROVENANCE
            if (symptr->cap_source) Perl_cap_prov_copy(aTHX_ sv, symptr->cap_source,
                s-symptr->cap_base, SvCUR(sv), 0, 0);
#endif
\t    if (utf8) {
\t\tSvUTF8_on(sv);''')

# Scalar concatenation must keep its alias snapshots as values, not byte
# reconstruction, and pass the source SV into the append helper.
in_function('pp_hot.c','PP(pp_concat)', 'newSVpvn_flags(rpv, rlen, SVs_TEMP)',
            'CAP_PROV_NEWSV(right, rpv, rlen, SVs_TEMP)',2)
in_function('pp_hot.c','PP(pp_concat)', '\tsv_setpvn(TARG, lpv, llen);', '''#ifdef CAPSTONE_PERL_PROVENANCE
        SV *cap_left = Perl_cap_prov_observed(aTHX_ left);
#endif
\tsv_setpvn(TARG, lpv, llen);
#ifdef CAPSTONE_PERL_PROVENANCE
        if (cap_left) Perl_cap_prov_copy(aTHX_ TARG, cap_left, 0, llen, 0, 0);
#endif''')
in_function('pp_hot.c','PP(pp_concat)', 'sv_catpvn_nomg(TARG, rpv, rlen);',
            'CAP_PROV_CATPVN(TARG, right, rpv, rlen, 0);')
in_function('pp_hot.c','PP(pp_concat)', '            SvPV_force_nomg_nolen(left);', '''#ifdef CAPSTONE_PERL_PROVENANCE
            SV *cap_left = Perl_cap_prov_observed(aTHX_ left);
#endif
            SvPV_force_nomg_nolen(left);
#ifdef CAPSTONE_PERL_PROVENANCE
            if (cap_left) Perl_cap_prov_copy(aTHX_ left, cap_left, 0, SvCUR(cap_left), 0, 0);
#endif''')
in_function('doop.c','Perl_do_join', 'sv_catpvn_flags(sv,delims,delimlen,delimflag);',
            'CAP_PROV_CATPVN(sv,delim,delims,delimlen,delimflag);')
in_function('doop.c','Perl_do_join', 'sv_catpvn_flags(sv,s,len,\n\t\t\t    DO_UTF8(*mark) ? SV_CATUTF8 : SV_CATBYTES);',
            'CAP_PROV_CATPVN(sv,*mark,s,len,DO_UTF8(*mark) ? SV_CATUTF8 : SV_CATBYTES);',2)

# Both rvalue and magical lvalue substrings transfer complete fields only.
in_function('pp.c','PP(pp_substr)', '    if (DO_UTF8(sv)) {\n        utf8_curlen', '''#ifdef CAPSTONE_PERL_PROVENANCE
    SV *cap_sub = Perl_cap_prov_observed(aTHX_ sv);
    SV *cap_repl = Perl_cap_prov_observed(aTHX_ repl_sv);
#endif
    if (DO_UTF8(sv)) {
        utf8_curlen''')
in_function('pp.c','PP(pp_substr)', '\ttmps = SvPV_force_nomg(sv, curlen);', '''#ifdef CAPSTONE_PERL_PROVENANCE
        SV *cap_prefix = Perl_cap_prov_observed(aTHX_ sv);
#endif
\ttmps = SvPV_force_nomg(sv, curlen);
#ifdef CAPSTONE_PERL_PROVENANCE
        if (cap_prefix) Perl_cap_prov_copy(aTHX_ sv, cap_prefix, 0, SvCUR(cap_prefix), 0, 0);
#endif''')
in_function('pp.c','PP(pp_substr)', '\t    if (utf8_curlen)\n\t\tSvUTF8_on(TARG);', '''\t    if (utf8_curlen)
\t\tSvUTF8_on(TARG);
#ifdef CAPSTONE_PERL_PROVENANCE
            if (cap_sub) Perl_cap_prov_copy(aTHX_ TARG, cap_sub, byte_pos, byte_len, 0, 0);
#endif''')
in_function('pp.c','PP(pp_substr)', '\t    sv_insert_flags(sv, byte_pos, byte_len, repl, repl_len, 0);', '''\t    sv_insert_flags(sv, byte_pos, byte_len, repl, repl_len, 0);
#ifdef CAPSTONE_PERL_PROVENANCE
            if (cap_repl) Perl_cap_prov_copy(aTHX_ sv, cap_repl, 0, SvCUR(cap_repl), byte_pos,
                !SvUTF8(cap_repl) && SvUTF8(sv) ? 1 : 0);
#endif''')
in_function('mg.c','Perl_magic_getsubstr', '    sv_setpvn(sv, tmps + offs, rem);', '''#ifdef CAPSTONE_PERL_PROVENANCE
    SV *cap_sub = Perl_cap_prov_observed(aTHX_ lsv);
#endif
    sv_setpvn(sv, tmps + offs, rem);
#ifdef CAPSTONE_PERL_PROVENANCE
    if (cap_sub) Perl_cap_prov_copy(aTHX_ sv, cap_sub, offs, rem, 0, 0);
#endif''')
in_function('mg.c','Perl_magic_setsubstr', '    SvGETMAGIC(lsv);', '''#ifdef CAPSTONE_PERL_PROVENANCE
    SV *cap_sub = Perl_cap_prov_observed(aTHX_ sv);
#endif
    SvGETMAGIC(lsv);''')
in_function('mg.c','Perl_magic_setsubstr', '    SvPV_force_nomg(lsv,lsv_len);', '''#ifdef CAPSTONE_PERL_PROVENANCE
    SV *cap_prefix = Perl_cap_prov_observed(aTHX_ lsv);
#endif
    SvPV_force_nomg(lsv,lsv_len);
#ifdef CAPSTONE_PERL_PROVENANCE
    if (cap_prefix) Perl_cap_prov_copy(aTHX_ lsv, cap_prefix, 0, SvCUR(cap_prefix), 0, 0);
#endif''')
in_function('mg.c','Perl_magic_setsubstr', '    if (!neglen) LvTARGLEN(sv) = newtarglen;', '''#ifdef CAPSTONE_PERL_PROVENANCE
    if (cap_sub) Perl_cap_prov_copy(aTHX_ lsv, cap_sub, 0, SvCUR(cap_sub), lvoff,
        !SvUTF8(cap_sub) && SvUTF8(lsv) ? 1 : 0);
#endif
    if (!neglen) LvTARGLEN(sv) = newtarglen;''')
in_function('pp.c','PP(pp_repeat)', '\tif (TARG != tmpstr)\n\t    sv_setsv_nomg(TARG, tmpstr);', '''#ifdef CAPSTONE_PERL_PROVENANCE
        SV *cap_repeat = Perl_cap_prov_snapshot(aTHX_ tmpstr);
#endif
\tif (TARG != tmpstr)
\t    sv_setsv_nomg(TARG, tmpstr);''')
in_function('pp.c','PP(pp_repeat)', '\n\tPUSHTARG;\n    }', '''
#ifdef CAPSTONE_PERL_PROVENANCE
        if (cap_repeat) {
            IV cap_i;
            for (cap_i = 0; cap_i < count; cap_i++)
                Perl_cap_prov_copy(aTHX_ TARG, cap_repeat, 0, SvCUR(cap_repeat), (STRLEN)cap_i*len, 0);
        }
#endif
\tPUSHTARG;
    }''')
insert_entry('pp.c','S_do_chomp','    Perl_cap_prov_clear(aTHX_ sv);')

# Core mutators which do not preserve the supported value algebra revoke before
# obtaining writable bytes, including error paths and same-byte overwrites.
for name,func,var in [
    ('doop.c','Perl_do_trans','sv'), ('doop.c','Perl_do_vecset','LvTARG(sv)'),
    ('doop.c','Perl_do_vop','sv'), ('sv.c','Perl_sv_catpv_flags','dstr')]:
    insert_entry(name,func,f'    Perl_cap_prov_clear(aTHX_ {var});')
in_function('pp_hot.c','PP(pp_subst)','    SvGETMAGIC(TARG); /* must come before cow check */', '''    SvGETMAGIC(TARG); /* must come before cow check */
#ifdef CAPSTONE_PERL_PROVENANCE
    Perl_cap_prov_clear(aTHX_ TARG);
#endif''')
in_function('pp_ctl.c','PP(pp_substcont)','    REGEXP *old = PM_GETRE(pm);', '''    REGEXP *old = PM_GETRE(pm);
#ifdef CAPSTONE_PERL_PROVENANCE
    Perl_cap_prov_clear(aTHX_ cx->sb_targ);
    Perl_cap_prov_clear(aTHX_ dstr);
#endif''')
in_function('pp_sys.c','PP(pp_sysread)','    bufsv = *++MARK;', '''    bufsv = *++MARK;
#ifdef CAPSTONE_PERL_PROVENANCE
    Perl_cap_prov_clear(aTHX_ bufsv);
#endif''')
in_function('pp_sys.c','PP(pp_sselect)','\tSvGETMAGIC(sv);', '''\tSvGETMAGIC(sv);
#ifdef CAPSTONE_PERL_PROVENANCE
        Perl_cap_prov_clear(aTHX_ sv);
#endif''')
in_function('pp_sys.c','PP(pp_ioctl)','    if (!IoIFP(io)) {', '''#ifdef CAPSTONE_PERL_PROVENANCE
    Perl_cap_prov_clear(aTHX_ argsv);
#endif
    if (!IoIFP(io)) {''')

for name, text in texts.items():
    (root/name).write_text(text)
if args.record:
    after = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
             for name in before}
    args.record.write_text(json.dumps({
        'baseline': baseline,
        'adapter': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(HERE.iterdir()) if p.is_file()},
        'before': before, 'after': after,
        'changed': sorted(texts),
        'complete_before': complete_before,
        'complete_after': inventory(),
    }, indent=2)+'\n')
print('Updated:', ', '.join(sorted(texts)))
