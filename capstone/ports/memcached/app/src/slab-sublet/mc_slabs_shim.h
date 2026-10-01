/* The component port's ledger (ports/memcached/allocators/src/shared/leases.c) includes
 * "mc_slabs_shim.h" for one thing: the item layout, to restore slabs_clsid after a renew. Inside the
 * application that layout is memcached.h's own, with patch 0001's capability alignment, so this
 * stands in for the component's shim and the ledger is compiled unchanged. */
#include "memcached.h"
