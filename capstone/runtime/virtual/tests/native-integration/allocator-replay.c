/* Replay captured allocator lifecycles with ordinary musl.
 * Generate events.h with allocator-events.py; see README.md.
 * Modes: 0 plain, 1 Capstone requests, 2 plain with larger cohorts,
 * 3 Capstone requests with smaller cohorts. No capability runtime required.
 */
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
struct event { unsigned op,phase,id,old,cohort; size_t size,align; };
#include "events.h"
static void *objects[MAX_IDS+1];
static size_t sizes[MAX_IDS+1];
__attribute__((noinline)) void init_cgraph(void) { __asm__ volatile("addi x0,x0,0" ::: "memory"); }
__attribute__((noinline)) void statistics_fini(void) { __asm__ volatile("addi x0,x0,0" ::: "memory"); }
static void fail(size_t i, const char *why) { fprintf(stderr,"FAIL %zu %s\n",i,why); exit(1); }
int main(int argc,char **argv) {
 if(argc!=2 || argv[1][0]<'0'||argv[1][0]>'3'||argv[1][1])return 2;
 unsigned mode=argv[1][0]-'0', native=mode&1;
 const struct event *events=native?cap_events:plain_events;
 size_t count=native?CAP_COUNT:PLAIN_COUNT,live=0;unsigned phase=0;
 for(size_t i=0;i<count;i++) {
  const struct event *e=events+i;size_t n=e->size;
  if(e->phase&&!phase){init_cgraph();phase=1;}
  if(mode>=2&&e->cohort)n=native?n/2:n*2;
  if(e->id>MAX_IDS||e->old>MAX_IDS)fail(i,"id bounds");
  if(e->op=='f') {
   if(e->id){if(!objects[e->id])fail(i,"free dead");free(objects[e->id]);objects[e->id]=0;live--;}
   continue;
  }
  if(e->id&&objects[e->id])fail(i,"overwrite live");
  void *p;
  if(e->op=='r') {
   void *old=e->old?objects[e->old]:0;
   if(e->old&&!old)fail(i,"realloc dead");
   p=realloc(old,n);
   if(n&&!p)fail(i,"realloc failed");
   if(p&&old&&n&&sizes[e->old]&&*(unsigned char*)p!=(unsigned char)e->old)fail(i,"realloc content");
   if(e->old){objects[e->old]=0;live--;}
  } else if(e->op=='c') {
   p=calloc(1,n);
   if(p&&n&&(*(unsigned char*)p||((unsigned char*)p)[n-1]))fail(i,"calloc nonzero");
  } else if(e->op=='a') p=aligned_alloc(e->align,n);
  else p=malloc(n);
  if(e->id&&!p)fail(i,"allocation failed");
  if(!e->id&&p)fail(i,"unexpected nonnull");
  if(p){objects[e->id]=p;sizes[e->id]=n;live++;if(n)*(unsigned char*)p=(unsigned char)e->id;}
 }
 statistics_fini();
 if(live!=(native?CAP_LIVE:PLAIN_LIVE))fail(count,"live count");
 printf("PASS mode=%u events=%zu live=%zu\n",mode,count,live);
 return 0;
}
