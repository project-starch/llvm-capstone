/* Modell des MicroPython-Heaps: ein grosses Array, Sub-Allokation in Software. */
static unsigned char heap[384 * 1024] __attribute__((aligned(32)));

unsigned char *sub_alloc(unsigned long off) {
    return &heap[off];          /* das macht gc_alloc im Prinzip */
}
void write_through(unsigned char *p, unsigned char v) { *p = v; }
unsigned char *whole(void) { return heap; }
