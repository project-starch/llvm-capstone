/* FIFO-like batches: Generation recycles a block once all its chunks are free.
 */
#include "client.h"

struct Message {
  unsigned sequence;
  unsigned char payload[60];
};

int client_run(MemoryContext parent) {
  MemoryContext queue = GenerationContextCreate(parent, "queue", 0, 2048, 8192);
  int failed = 0;

  for (unsigned batch = 0; batch < 3; ++batch) {
    struct Message *messages[64];
    for (unsigned i = 0; i < 64; ++i) {
      messages[i] = MemoryContextAlloc(queue, sizeof(*messages[i]));
      messages[i]->sequence = batch * 64 + i;
      memset(messages[i]->payload, i, sizeof(messages[i]->payload));
    }
    for (unsigned i = 0; i < 64; ++i) {
      failed |= messages[i]->sequence != batch * 64 + i;
      failed |= messages[i]->payload[59] != i;
      pfree(messages[i]); /* this object's lifetime ends immediately */
      messages[i] = NULL;
    }
    /* No assumption that an individual free immediately reuses its address. */
  }
  MemoryContextDelete(queue);
  return failed;
}
