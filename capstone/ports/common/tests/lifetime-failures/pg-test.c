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
  if (argc>1 && !strcmp(argv[1],"header")) {
    /* A live caller loads the capability-sized words in front of its chunk.
       A header inside the chunk's bounds that keeps the block's reference
       shows as a tagged word; a header outside them faults (cause 28) at the
       first load. */
    for (int off=16; off<=48; off+=16) {
      void *word;
      printf("TRAPTEST: postgres header load -%d armed\n", off);
      word=*(void * volatile *)(p-off);
      if (__builtin_capstone_cap_get_tag(word)) {
        printf("TRAPTEST: postgres header reference at -%d covers %lu bytes\n", off,
               (unsigned long)(__builtin_capstone_cap_get_end(word)-__builtin_capstone_cap_get_base(word)));
        return 23;
      }
    }
    puts("TRAPTEST: postgres header holds no reference");
    return 0;
  }
  if (argc>1 && !strcmp(argv[1],"table")) {
    /* Every header reader goes through the allocator's block table:
       pfree, GetMemoryChunkContext, GetMemoryChunkSpace, in-place and
       moving repalloc, aligned chunks, empty and large requests, resets,
       and more AllocSet contexts than the context freelist keeps. */
    static const Size sizes[]={0,1,16,100,1000,5000,9000,65552,300000,1000000};
    MemoryContext many[150];
    for (int kind=0;kind<4;kind++) {
      MemoryContext t=context(kind);
      for (int round=0;round<3;round++) {
        char *q[40];
        for (int i=0;i<40;i++) {
          Size n=kind==2 ? 64 : sizes[i%10];
          q[i]=MemoryContextAlloc(t,n);
          if (n) { q[i][0]=(char)i; q[i][n-1]=(char)i; }
          if (kind!=3 && GetMemoryChunkContext(q[i])!=t) return 41;
          if (kind!=3 && GetMemoryChunkSpace(q[i])<n) return 42;
        }
        if (kind<2) {
          for (int i=0;i<40;i+=2) {
            Size n=sizes[i%10];
            q[i]=repalloc(q[i],n+40000);
            if (n && q[i][0]!=(char)i) return 43;
            q[i]=repalloc(q[i],8);
            if (n && q[i][0]!=(char)i) return 44;
          }
        }
        if (kind!=3) {
          for (int i=0;i<40;i++) pfree(q[i]);
        }
        if (kind<2) {
          char *a=MemoryContextAllocAligned(t,100,4096,0);
          if ((uintptr_t)a%4096) return 45;
          a[99]=1;
          if (GetMemoryChunkContext(a)!=t) return 46;
          pfree(a);
        }
        MemoryContextReset(t);
      }
      MemoryContextDelete(t);
    }
    for (int round=0;round<2;round++) {
      for (int i=0;i<150;i++) {
        char *x;
        many[i]=AllocSetContextCreate(TopMemoryContext,"many",ALLOCSET_SMALL_SIZES);
        x=MemoryContextAlloc(many[i],32);
        x[31]=1;
        if (GetMemoryChunkContext(x)!=many[i]) return 47;
      }
      for (int i=0;i<150;i++) MemoryContextDelete(many[i]);
    }
    puts("TRAPTEST: postgres table PASS");
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
