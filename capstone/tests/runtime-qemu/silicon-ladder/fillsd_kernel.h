#ifndef FILLSD_KERNEL_H
#define FILLSD_KERNEL_H
/* Is the 23.6 cycles/store a CAPABILITY-store premium, or just what a store costs on this
 * write-through memory system? The AXI adapter's shadow-tag transaction fires for plain loads and
 * stores too (wt_axi_adapter.sv:153), so the premium cannot be assumed.
 *
 * The same loop with a plain 8-byte `sd` in place of the 16-byte `stc`: same 256 iterations, same
 * 16-byte walk, so the same 256 cache lines are touched. It writes 2 KiB where fillcost writes 4,
 * and that asymmetry is the point -- if the cost is per STORE (a write-through drain) the two arms
 * land together; if it is per BYTE, fillsd lands near half. Either reading is a fact we do not
 * currently have.
 *
 * It stores only the low 8 bytes of each granule, so slot 0 and slot 255 still read back zero
 * through the `ld` the check compiles to, and the +512/+1024 controls still mean what they say. */
#define FILL_PAYLOAD "sd x0, 0(%[p])\n"
#define FILL_TAG 4096
#include "fillcost_kernel.h"
#define fillsd_compute fillcost_compute
#endif
