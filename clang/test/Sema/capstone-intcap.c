// __intcap on capstone64: the type, its spellings, conversions and binary
// and unary arithmetic are accepted; pointer arithmetic with an __intcap operand,
// an __intcap subscript and _Atomic read-modify-write are refused for now.
// Other targets have no capabilities and reject the keyword.
//
// RUN: %clang_cc1 -triple capstone64-unknown-elf -ffreestanding -fsyntax-only -verify=cap %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ffreestanding -fsyntax-only -verify=x86 %s

#ifdef __SIZEOF_INTCAP__
__intcap_t a;
__uintcap_t b;
unsigned __intcap c;
signed __intcap d;

void *roundtrip(void *p) { return (void *)(__uintcap_t)p; } // no round-trip warning
unsigned long addr(__uintcap_t u) { return u; }
int cmp(__uintcap_t x, __uintcap_t y) { return x < y || !x; }

__uintcap_t add(__uintcap_t u) { return u + 1; }
__uintcap_t band(__uintcap_t u) { return u & 15; }
long mix(long n, __intcap x) { return n - x; }
void addeq(__uintcap_t *u) { *u += 2; }
__intcap neg(__intcap x) { return -x; }
__intcap bnot(__intcap x) { return ~x; }
void inc(__uintcap_t *u) { (*u)++; }
void ainc(_Atomic __uintcap_t *a) { (*a)++; }         // cap-error {{operator '++' on '_Atomic(__uintcap_t)' is not supported yet on Capstone}}
char *ptradd(char *p, __uintcap_t u) { return p + u; } // cap-error {{operator '+' on '__uintcap_t'}}
char sub(char *p, __uintcap_t u) { return p[u]; }    // cap-error {{operator '[]' on '__uintcap_t'}}
void atom(_Atomic __uintcap_t *a) { *a += 1; }       // cap-error {{operator '+=' on '_Atomic(__uintcap_t)'}}
#else
__intcap e; // x86-error {{__intcap is not supported on this target}}
#endif

#ifdef __SIZEOF_INTCAP__
// -Wcheri-provenance: two operands that both carry provenance.
__uintcap_t both(__uintcap_t u, __uintcap_t v) { return u + v; } // cap-warning {{binary expression on capability types '__uintcap_t' (aka 'unsigned __intcap') and '__uintcap_t'; it is not clear which should be used as the source of provenance}}
__uintcap_t one(__uintcap_t u, unsigned long n) { return u + n; }                 // no warning: n carries none
__uintcap_t cast(__uintcap_t u, unsigned long n) { return (__uintcap_t)n | u; }   // no warning: converted from an integer
__uintcap_t diff(__uintcap_t u, __uintcap_t v) { return u - v; }                  // no warning: not commutative, the left one
#endif
