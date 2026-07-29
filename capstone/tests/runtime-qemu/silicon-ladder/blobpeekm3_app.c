/* QEMU half of the blob-peek probe. */
#include "blobpeek_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = bp_compute(); }
