#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "mem_alloc.h"
#define LIVE 256
#define GROW 12
static char store[2000];
static int ov(uint8_t *a,int na,uint8_t *b,int nb){return a&&b&&a<b+nb&&b<a+na;}
static void run(uint32_t fsize, int ask)
{
    memset(store,0,sizeof(store));
    mem_allocator_t a = mem_allocator_create(store, sizeof(store));
    uint8_t *p,*q,*r; int i,cl=0;
    p = mem_allocator_malloc(a, LIVE);
    p = mem_allocator_realloc(a, p, LIVE+GROW);
    if(!p){printf("  fsize=0x%02x ask=%3d realloc=NULL\n",fsize,ask);return;}
    *(uint32_t*)(p+LIVE) = (1u<<30)|fsize;
    *(uint32_t*)(p+LIVE+GROW-4) = GROW;
    memset(p,0xA5,LIVE);
    q = mem_allocator_malloc(a, 64);
    if(q) mem_allocator_free(a,q);
    r = mem_allocator_malloc(a, ask);
    if(r) memset(r,0x5A,ask);
    for(i=0;i<LIVE;i++) if(p[i]!=(uint8_t)0xA5){cl=1;break;}
    printf("  fsize=0x%02x ask=%3d r=%-5s innerhalb_p=%d clobber=%d\n",
           fsize, ask, r?"ok":"NULL", ov(r,ask,p,LIVE), cl);
}
int main(void){
    uint32_t sizes[]={0x20,0x40,0x60,0x80,0xc0,0x100};
    int asks[]={16,32,64,128};
    for(unsigned s=0;s<sizeof(sizes)/sizeof(*sizes);s++)
        for(unsigned k=0;k<sizeof(asks)/sizeof(*asks);k++) run(sizes[s],asks[k]);
    return 0;
}
