#include "postgres.h"
#include "utils/memutils.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
unsigned long life_calls, life_fail;
static MemoryContext context(int kind)
{
  switch(kind) {
    case 0: return AllocSetContextCreate(TopMemoryContext,"test",ALLOCSET_DEFAULT_SIZES);
    case 1: return GenerationContextCreate(TopMemoryContext,"test",ALLOCSET_DEFAULT_SIZES);
    case 2: return SlabContextCreate(TopMemoryContext,"test",8192,64);
    default: return BumpContextCreate(TopMemoryContext,"test",ALLOCSET_DEFAULT_SIZES);
  }
}
int main(int argc, char **argv)
{
  MemoryContextInit();
  MemoryContext c=context(0);
  char *p=MemoryContextAlloc(c,64);
  strcpy(p,"keep");
  if (argc>1 && !strcmp(argv[1],"stale")) {
    pfree(p);
    puts("TRAPTEST: postgres stale armed");
    return *(volatile char *)p;
  }
  if (argc>1 && !strcmp(argv[1],"large")) {
    /* Sizes above 4 KiB, with and without granule-aligned chunk bounds,
       from AllocSet, Generation and Bump, without injection. */
    static const Size sizes[]={5000,65536,65552,300000,1000000};
    for (int kind=0;kind<4;kind++) {
      if (kind==2) continue;
      MemoryContext l=context(kind);
      for (int i=0;i<5;i++) {
        char *q=MemoryContextAllocExtended(l,sizes[i],MCXT_ALLOC_NO_OOM);
        if (!q) { printf("TRAPTEST: postgres large OOM kind %d size %zu\n",kind,sizes[i]); return 21; }
        q[0]=1; q[sizes[i]-1]=2;
        if (kind!=3) pfree(q);
      }
      MemoryContextDelete(l);
    }
    puts("TRAPTEST: postgres large success");
    return 0;
  }
  if (argc>1 && !strcmp(argv[1],"external")) {
    /* A failed large realloc keeps the old chunk live and unchanged; a
       successful one copies it. */
    p=MemoryContextAlloc(c,9000);
    strcpy(p,"keep");
    life_calls=0;life_fail=1;
    if (repalloc_extended(p,9500,MCXT_ALLOC_NO_OOM)) return 18;
    puts("TRAPTEST: postgres external failure old read armed");
    if (strcmp(p,"keep")) return 19;
    life_fail=0;
    p=repalloc(p,300000);
    if (strcmp(p,"keep")) return 20;
    p[299999]=1;
    pfree(p);
    return 0;
  }
  MemoryContextDelete(c);
  for (int kind=0;kind<4;kind++) {
    c=context(kind);
    for (int n=1;n<=2;n++) {
      MemoryContextReset(c);
      life_calls=0;life_fail=n;
      void *q=MemoryContextAllocExtended(c,64,MCXT_ALLOC_NO_OOM);
      if(q) return 11+kind*20;
      life_fail=0;
      q=MemoryContextAlloc(c,64);
      if(!q) return 12+kind*20;
      if(kind!=3) pfree(q);
    }
    if(kind!=3) {
      p=MemoryContextAlloc(c,64);
      strcpy(p,"keep");
      life_calls=0;life_fail=1;
      if(repalloc_extended(p,64,MCXT_ALLOC_NO_OOM)) return 13+kind*20;
      if(strcmp(p,"keep")) return 14+kind*20;
      if(kind!=2) {
        life_calls=0;
        if(repalloc_extended(p,4096,MCXT_ALLOC_NO_OOM)) return 15+kind*20;
        if(strcmp(p,"keep")) return 16+kind*20;
      }
      life_fail=0;
      pfree(p);
    }
    MemoryContextDelete(c);
  }
  puts("TRAPTEST: postgres injection PASS");
  return 0;
}
