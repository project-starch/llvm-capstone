/* Only the LC_* category constants are needed, and only so that PHP_MINIT_FUNCTION(string)
 * COMPILES -- it registers them as PHP constants. The domain never runs a module init, and no
 * trigger path calls setlocale() or localeconv(), so neither is declared here: a use would fail at
 * the call site with a clear message rather than silently linking against nothing.
 * Values match glibc, so a constant read from PHP matches what the corpus build would report. */
#ifndef PHP_CAPSTONE_LOCALE_H
#define PHP_CAPSTONE_LOCALE_H

#define LC_CTYPE    0
#define LC_NUMERIC  1
#define LC_TIME     2
#define LC_COLLATE  3
#define LC_MONETARY 4
#define LC_MESSAGES 5
#define LC_ALL      6

/* `struct lconv` and localeconv() are needed because php_localeconv_r copies the whole struct
 * (ext/standard/string.c:148-150) on the number_format path. The layout is the standard one; the
 * implementation beside it returns the "C" locale, which is the only locale a domain has. */
struct lconv {
    char *decimal_point;
    char *thousands_sep;
    char *grouping;
    char *int_curr_symbol;
    char *currency_symbol;
    char *mon_decimal_point;
    char *mon_thousands_sep;
    char *mon_grouping;
    char *positive_sign;
    char *negative_sign;
    char  int_frac_digits;
    char  frac_digits;
    char  p_cs_precedes;
    char  p_sep_by_space;
    char  n_cs_precedes;
    char  n_sep_by_space;
    char  p_sign_posn;
    char  n_sign_posn;
};

struct lconv *localeconv(void);

/* PHP_FUNCTION(setlocale) must compile and link even though no trigger calls it. A domain has the
 * "C" locale and no way to acquire another, so the implementation reports that honestly rather
 * than pretending a change succeeded. */
char *setlocale(int category, const char *locale);

#endif
