/* Private Perl string provenance. Embedded in perl.h by apply.py. */
#if defined(__CAPSTONE_PURECAP__) || defined(CAPSTONE_PERL_PROVENANCE_TEST)
#define CAPSTONE_PERL_PROVENANCE 1
#define CAP_PROV_WRITE(sv) Perl_cap_prov_clear(aTHX_ (sv))
#define CAP_PROV_NEWSV(sv,p,n,f) Perl_cap_prov_newsv(aTHX_ sv,p,n,f)
#define CAP_PROV_CATPVN(d,s,p,n,f) Perl_cap_prov_catpvn(aTHX_ d,s,p,n,f)
void Perl_cap_prov_clear(pTHX_ SV *sv);
SV *Perl_cap_prov_snapshot(pTHX_ SV *sv);
SV *Perl_cap_prov_observed(pTHX_ SV *sv);
void Perl_cap_prov_prepare_get(pTHX_ SV *sv);
void Perl_cap_prov_copy(pTHX_ SV *dst, SV *snapshot, STRLEN off,
                        STRLEN len, STRLEN to, int encoding);
void Perl_cap_prov_add(pTHX_ SV *sv, STRLEN off, STRLEN len, const char *cap);
void Perl_cap_prov_trim(pTHX_ SV *sv, STRLEN len);
const char *Perl_cap_prov_pointer(pTHX_ SV *sv, STRLEN off, STRLEN len,
                                const char *candidate);
SV *Perl_cap_prov_newsv(pTHX_ SV *source, const char *pv, STRLEN len, U32 flags);
void Perl_cap_prov_catpvn(pTHX_ SV *dst, SV *source, const char *pv, STRLEN len, I32 flags);
#else
#define CAP_PROV_WRITE(sv) ((void)0)
#define CAP_PROV_NEWSV(sv,p,n,f) newSVpvn_flags(p,n,f)
#define CAP_PROV_CATPVN(d,s,p,n,f) sv_catpvn_flags(d,p,n,f)
#endif
