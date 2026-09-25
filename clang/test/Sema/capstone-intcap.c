// __intcap on capstone64, Phase B.1: the type, its spellings and conversions are
// accepted; arithmetic is refused until the address-replacing lowering exists.
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

__uintcap_t add(__uintcap_t u) { return u + 1; }    // cap-error {{operator '+' on '__uintcap_t' (aka 'unsigned __intcap') is not supported yet on Capstone}}
__uintcap_t band(__uintcap_t u) { return u & 15; }  // cap-error {{operator '&' on '__uintcap_t'}}
long mix(long n, __intcap x) { return n - x; }       // cap-error {{operator '-' on '__intcap'}}
void inc(__uintcap_t *u) { (*u)++; }                 // cap-error {{operator '++' on '__uintcap_t'}}
void addeq(__uintcap_t *u) { *u += 2; }              // cap-error {{operator '+=' on '__uintcap_t'}}
__intcap neg(__intcap x) { return -x; }              // cap-error {{operator '-' on '__intcap'}}
#else
__intcap e; // x86-error {{__intcap is not supported on this target}}
#endif
