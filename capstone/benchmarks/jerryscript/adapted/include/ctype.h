/* Freestanding <ctype.h> for the Capstone domain.
 *
 * jerry-core includes it from jrt-libc-includes.h, so every file that includes
 * that header needs it -- six of them fail on it alone.
 *
 * ASCII only, and deliberately so: the domain has no locale, and a locale-aware
 * implementation would be pretending. Written as functions rather than the usual
 * table-driven macros because a table is a GLOBAL, and under -capstone-gp-captable
 * every global costs a carve out of dom_data. Seven comparisons are cheaper here
 * than 257 bytes of table plus its capability.
 */
#ifndef CAPSTONE_ADAPTED_CTYPE_H
#define CAPSTONE_ADAPTED_CTYPE_H

static inline int isdigit(int c)  { return c >= '0' && c <= '9'; }
static inline int isupper(int c)  { return c >= 'A' && c <= 'Z'; }
static inline int islower(int c)  { return c >= 'a' && c <= 'z'; }
static inline int isalpha(int c)  { return isupper(c) || islower(c); }
static inline int isalnum(int c)  { return isalpha(c) || isdigit(c); }
static inline int isspace(int c)  { return c == ' ' || (c >= '\t' && c <= '\r'); }
static inline int isxdigit(int c) { return isdigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
static inline int isprint(int c)  { return c >= 0x20 && c < 0x7f; }
static inline int iscntrl(int c)  { return (c >= 0 && c < 0x20) || c == 0x7f; }
static inline int ispunct(int c)  { return isprint(c) && !isalnum(c) && c != ' '; }
static inline int isgraph(int c)  { return isprint(c) && c != ' '; }
static inline int tolower(int c)  { return isupper(c) ? c - 'A' + 'a' : c; }
static inline int toupper(int c)  { return islower(c) ? c - 'a' + 'A' : c; }

#endif /* CAPSTONE_ADAPTED_CTYPE_H */
