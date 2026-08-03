#ifndef RB1M_H
#define RB1M_H
/* R-16 dom_data-GEOMETRY probe: mirrors SQLite's layout class -- globals offset 0x150000
 * (link-gpfree-sq.ld) so the image lands in the same order-9 / 2 MB dom_data allocation that
 * every SQLite image uses, where every previously-passing rung sat at order 5 / 128 KB.
 * Image size and carve count are each ALREADY ruled out (rz1m 1087 KB passes, rc192 with
 * 192 carves / 3072 B table passes, rzc1m both together passes), so geometry is what is
 * left to test. .bss is trimmed to land INSIDE the dom_data budget (SQLite fits at 371
 * pages; 128 KB of .bss overruns at 385). */
static char rb1m_big[65536];
static char rb1m_g[2] = { 5, 0 };
static unsigned rb1m_compute(void)
{
  rb1m_big[0] = 1; rb1m_big[65535] = 1;
  return (unsigned)(unsigned char)rb1m_g[0] + (unsigned)(unsigned char)rb1m_big[0] - 1u;
}
#endif
