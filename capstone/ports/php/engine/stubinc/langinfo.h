/* Stub: ext/standard includes <langinfo.h> behind a HAVE_* guard that this build happens to define, but
 * nothing on any trigger path references a symbol from it. The header only has to exist; a future
 * use fails at the use site with a clear message rather than here. */
#ifndef PHP_CAPSTONE_LANGINFO_H
#define PHP_CAPSTONE_LANGINFO_H
typedef int nl_item;

/* PHP_FUNCTION(nl_langinfo) must compile and link; a C-locale domain has no langinfo database. */
char *nl_langinfo(nl_item item);

#endif
