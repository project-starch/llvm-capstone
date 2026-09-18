#ifndef FFPOOL_REPLAY_ENGINE_H
#define FFPOOL_REPLAY_ENGINE_H

#include "trace.h"

/* Runs one trace in the caller's report and metadata regions. The platform
 * entry point owns payload setup and implements ff2_fail to leave execution.
 * The engine keeps its cursor, pool table and lease table private. */
void ff2_replay_run(const struct ff2_header *input, struct ff2_header *output,
                    void *metadata);

/* Diagnostic access remains valid before the engine starts. */
uint64_t ff2_replay_cursor(void);
uint64_t ff2_replay_operation(void);

#endif
