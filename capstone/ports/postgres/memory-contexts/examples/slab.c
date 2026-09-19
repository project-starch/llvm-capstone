/* A fixed-size job table. Every Slab allocation must request sizeof(Job). */
#include "client.h"

struct Job {
  unsigned id;
  unsigned completed;
  unsigned char name[24];
};

int client_run(MemoryContext parent) {
  MemoryContext jobs =
      SlabContextCreate(parent, "jobs", 4096, sizeof(struct Job));
  struct Job *slots[32];
  int failed = 0;

  for (unsigned i = 0; i < 32; ++i) {
    slots[i] = MemoryContextAlloc(jobs, sizeof(*slots[i]));
    slots[i]->id = i;
    slots[i]->completed = 0;
    memset(slots[i]->name, i, sizeof(slots[i]->name));
  }
  for (unsigned i = 0; i < 32; i += 2) {
    pfree(slots[i]);
    slots[i] = NULL;
  }
  /* Replace finished jobs; the odd slots must remain live and unchanged. */
  for (unsigned i = 0; i < 32; i += 2) {
    slots[i] = MemoryContextAlloc(jobs, sizeof(*slots[i]));
    slots[i]->id = i + 100;
    slots[i]->completed = 1;
    memset(slots[i]->name, i, sizeof(slots[i]->name));
  }
  for (unsigned i = 0; i < 32; ++i) {
    failed |= slots[i]->id != (i % 2 ? i : i + 100);
    failed |= slots[i]->completed != (i % 2 ? 0U : 1U);
    failed |= slots[i]->name[23] != i;
    pfree(slots[i]);
    slots[i] = NULL;
  }
  MemoryContextDelete(jobs);
  return failed;
}
