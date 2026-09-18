#ifndef FFPOOL_NODE_SNAPSHOTS_H
#define FFPOOL_NODE_SNAPSHOTS_H

#include <stdint.h>

/* Associate the QEMU node counters with an explicit replay event index. */
void ff2_node_snapshot(uint64_t cursor);

#endif
