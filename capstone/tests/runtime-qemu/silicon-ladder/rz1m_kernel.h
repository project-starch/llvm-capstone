#ifndef RZ1M_H
#define RZ1M_H
/* R-16 MINIMISATION. Every ~10 KB ladder rung has ENTERED reliably (r14sl 6/6, clp*, cdif8,
 * cst8, ...), while every 1.6 MB SQLite-derived image has ENTRY-STALLED. Nobody has
 * bisected that. These two ladders separate the candidate axes, holding the other fixed.
 * rz1m: ONE global plus 1024 KB of inert .rodata padding => the image
 * grows to ~1024 KB while the carve count stays at 1. Isolates IMAGE SIZE.
 * The padding is emitted with .space so the source stays small and the bytes are never
 * referenced -- nothing to optimise away, nothing to execute. */
__asm__(".pushsection .rodata\n.balign 16\n.space 1048576, 0xAB\n.popsection");
static char z0[2] = { 7, 0 };
static unsigned rz1m_compute(void)
{
  return (unsigned)(unsigned char)z0[0];
}
#endif
