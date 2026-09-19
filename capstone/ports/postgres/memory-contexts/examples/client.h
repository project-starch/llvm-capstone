#ifndef PG_EXAMPLE_CLIENT_H
#define PG_EXAMPLE_CLIENT_H

#include "postgres.h"
#include "utils/memutils.h"

/* Called with a live parent context. Return zero on success. The client owns
 * its child contexts, but not parent. No Capstone/Sublet API is needed here. */
int client_run(MemoryContext parent);

#endif
