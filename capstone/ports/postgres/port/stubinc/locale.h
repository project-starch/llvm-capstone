#ifndef A11_LOCALE_H
#define A11_LOCALE_H
#define LC_ALL 6
char *setlocale(int, const char *);
struct lconv { char *decimal_point; };
struct lconv *localeconv(void);
#endif
