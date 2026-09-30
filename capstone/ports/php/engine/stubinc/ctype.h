/* ASCII only: glibc's ctype goes through __ctype_b_loc, a locale table indirection that a
 * domain has no way to populate.
 *
 * THESE ARE FUNCTIONS, NOT MACROS, AND THAT IS LOAD-BEARING.
 *
 * They were macros, and each one mentioned its argument two to four times:
 *
 *     #define isupper(c)  ((c) >= 'A' && (c) <= 'Z')
 *     #define tolower(c)  (isupper(c) ? (c) + 32 : (c))
 *
 * so tolower(c) expanded to `((c) >= 'A' && (c) <= 'Z') ? (c) + 32 : (c)` -- THREE
 * evaluations. Zend/zend_operators.c:1755 is
 *
 *     *result++ = tolower((int)*str++);
 *
 * and `*str++` has a side effect, so str advanced THREE TIMES per iteration and the loop
 * read past the end of the string. Observed: zend_str_tolower_copy over "func_num_args"
 * (14 bytes incl. NUL, capability bounded to exactly that) faulted cause 5 reading offset
 * 14, one past the end, during zend_startup_builtin_functions. In the disassembly the loop
 * body held three `stc` of the incremented pointer for one `lbu` that mattered.
 *
 * On a normal libc these are real functions (C requires that they be callable), so the
 * corpus is written assuming single evaluation. Anything defined here that takes an
 * expression must evaluate it ONCE.
 *
 * Every predicate's argument is `int` and is read as a byte value the way the C library
 * specifies, so EOF (-1) and values above 0x7f answer false rather than indexing anything.
 */
#ifndef _CTYPE_H
#define _CTYPE_H 1

static __inline__ int isdigit(int c)  { return c >= '0' && c <= '9'; }
static __inline__ int isupper(int c)  { return c >= 'A' && c <= 'Z'; }
static __inline__ int islower(int c)  { return c >= 'a' && c <= 'z'; }
static __inline__ int isalpha(int c)  { return isupper(c) || islower(c); }
static __inline__ int isalnum(int c)  { return isalpha(c) || isdigit(c); }
static __inline__ int isspace(int c)  { return c == ' ' || c == '\t' || c == '\n'
                                           || c == '\v' || c == '\f' || c == '\r'; }
static __inline__ int isprint(int c)  { return c >= 0x20 && c < 0x7f; }
static __inline__ int isgraph(int c)  { return c > 0x20 && c < 0x7f; }
static __inline__ int iscntrl(int c)  { return (unsigned)c < 0x20u || c == 0x7f; }
static __inline__ int ispunct(int c)  { return isgraph(c) && !isalnum(c); }
static __inline__ int isxdigit(int c) { return isdigit(c)
                                           || (c >= 'a' && c <= 'f')
                                           || (c >= 'A' && c <= 'F'); }
static __inline__ int tolower(int c)  { return isupper(c) ? c + 32 : c; }
static __inline__ int toupper(int c)  { return islower(c) ? c - 32 : c; }

#endif
