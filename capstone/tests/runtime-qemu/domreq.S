/* A domain's DECLARED resource requirement.
 *
 * Without this section the kernel module sizes a domain's region by
 *     dom_min_free = max(2 * code_len, 512 KiB)
 * which is a proxy: the stack is whatever survives after the gp-captable carve,
 * the carve grows with the image, so the rule compensates through code size. It
 * is not causally linked to how deep the domain actually recurses, and it is
 * wildly loose in both directions -- a 10 KB control domain gets 512 KB of
 * headroom, and a 2.9 MB interpreter gets 12.4 MB of stack it never asked for.
 * That 3x-on-the-image rule is what pushed an interpreter image past the buddy
 * allocator's maximum order and made a Linux patch look necessary.
 *
 * So: let the BUILD state what the domain needs, since the build is the only
 * place that knows. Absent section => the old rule, unchanged, for every domain
 * that has not been converted.
 *
 * Assemble with:
 *     -DCAPSTONE_DOMREQ_DATA=<bytes dom_data must have>
 *     -DCAPSTONE_DOMREQ_STACK=<how much of that is stack>   (diagnostics only)
 *
 * DOMREQ_DATA is the whole dom_data requirement -- initializer blob + cap table
 * + per-global storage + stack -- because that is the one number the module can
 * use without understanding the carve. DOMREQ_STACK is carried alongside so a
 * failure can say "3 MB, of which 2 MB stack" instead of just a total.
 */

#ifndef CAPSTONE_DOMREQ_DATA
#error "domreq.S needs -DCAPSTONE_DOMREQ_DATA=<bytes>"
#endif
#ifndef CAPSTONE_DOMREQ_STACK
#define CAPSTONE_DOMREQ_STACK 0
#endif

/* Flags "" means no SHF_ALLOC: the loader reads this out of the FILE through the
   section header, exactly as it already does for .capstone_gp_initdesc, so the
   declaration costs the loaded image zero bytes and cannot perturb any layout.
   That matters here -- an image-perturbing diagnostic is how S01 happened. */
.section .capstone_domreq,"",@progbits
  .balign 8
  .quad 0x5145524d4f445043      /* 'C','P','D','O','M','R','E','Q' in memory */
  .quad CAPSTONE_DOMREQ_DATA
  .quad CAPSTONE_DOMREQ_STACK
