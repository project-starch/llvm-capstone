/* renew-shim.h (nested lane, 2026-11-26), force-included after oneop-shim.h into the Line B CRENEW arms only
   (perl-fullren, perl-fullren-ff). PERL_CAP_RENEW is CRENEW, or under PERL_CAPSTONE_RENEW_FORCE_FAIL a stand-in for
   CRENEW's failure result: an untagged 0 with the old lifetime left live, which is what the model and the RTL leave
   when a renewal cannot take a new identity in a table that cannot grow. The fixture arm uses it to drive every
   renew site down its fall-back path, the port's own revoke. */
#ifdef PERL_CAPSTONE_RENEW_FORCE_FAIL
#define PERL_CAP_RENEW(x) ((void)(x), (void *)0)
#else
#define PERL_CAP_RENEW(x) capstone_cap_renew(x)
#endif
