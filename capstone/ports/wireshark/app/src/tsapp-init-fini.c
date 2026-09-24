/* .init_array and .fini_array in the tshark domain.
 *
 * Three of tshark's libraries register functions there: GLib (glib_init_ctor, which sets up quarks,
 * the message prefixes and the debug flags), libgpg-error (gpg_err_init) and libxml2
 * (xmlDestructor). Nothing in a domain ran either array (my_first_domain/link.ld says so, "for the
 * day one appears"), and musl's exit walks .fini_array through uintptr_t:
 *
 *     uintptr_t a = (uintptr_t)&__fini_array_end;
 *     for (; a > (uintptr_t)&__fini_array_start; a -= sizeof(void(*)()))
 *         (*(void (**)())(a - sizeof(void(*)())))();
 *
 * so the first tshark boot halted in exit() with cause 24 (a load through an integer) on its one
 * .fini_array slot (2026-09-24).
 *
 * THE SLOTS ARE NOT CAPABILITIES. A static domain image carries no relocations, and the capability
 * initialisers (.capstone_cap_init) do not cover these arrays, so each 16-byte slot holds the
 * function's LINK address as a plain integer in its low 8 bytes. The domain runs at another base.
 * A callable capability is derived from the code capability of a function of this file, the
 * anchor, moved by the distance between the two link addresses; the anchor's own link address is
 * written into .rodata by the assembler (`.quad`), which the static link resolves like the slots.
 * A slot that does hold a tagged capability is called as it is.
 *
 * __capstone_run_init_array runs the constructors, in order; deps/domain_entry.c calls it before
 * main. __libc_exit_fini replaces musl's weak one (exit.c) and runs the destructors in reverse.
 * Linked only into the images host/build-domain.sh makes.
 */
#include <stddef.h>

typedef void (*tsapp_fn)(void);

/* Defined by my_first_domain/link.ld for every domain, so their addresses are real ones (an
 * undefined weak symbol's would not be: ISSUES C-56). */
extern const unsigned char __init_array_start[], __init_array_end[];
extern const unsigned char __fini_array_start[], __fini_array_end[];

void __capstone_init_fini_anchor(void);
void __capstone_init_fini_anchor(void) {}

extern const unsigned long __capstone_init_fini_anchor_link;
__asm__(".section .rodata\n"
        ".p2align 3\n"
        ".globl __capstone_init_fini_anchor_link\n"
        "__capstone_init_fini_anchor_link:\n"
        ".quad __capstone_init_fini_anchor\n"
        ".previous\n");

static void tsapp_call_slot(const unsigned char *slot)
{
	tsapp_fn f = *(const tsapp_fn *)slot;
	if (!__builtin_capstone_cap_get_tag(f)) {
		unsigned long link = *(const unsigned long *)slot;
		f = (tsapp_fn)((const char *)__capstone_init_fini_anchor +
		               (long)(link - __capstone_init_fini_anchor_link));
	}
	f();
}

void __capstone_run_init_array(void)
{
	for (const unsigned char *p = __init_array_start; p < __init_array_end; p += sizeof(tsapp_fn))
		tsapp_call_slot(p);
}

void __libc_exit_fini(void)
{
	for (const unsigned char *p = __fini_array_end; p > __fini_array_start; p -= sizeof(tsapp_fn))
		tsapp_call_slot(p - sizeof(tsapp_fn));
}
