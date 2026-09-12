#ifndef A11_LIBINTL_H
#define A11_LIBINTL_H
#define gettext(x) (x)
#define dgettext(d,x) (x)
#define dngettext(d,s,p,n) ((n)==1?(s):(p))
#endif
