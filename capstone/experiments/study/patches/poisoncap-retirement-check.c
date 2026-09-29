#include <stdio.h>
#include <stdlib.h>
#include <cheri/cheric.h>
#include <cheri/revoke.h>
static void *slots[64];
int main(void) {
 for(int round=0;round<3;round++) {
  for(int i=0;i<64;i++) if(posix_memalign(&slots[i],16,512*1024)) return 2;
  for(int i=0;i<64;i++) {free(slots[i]);slots[i]=NULL;}
 }
 unsigned stale=0;
 for(int i=0;i<64;i++) {
  if(posix_memalign(&slots[i],16,7038)) return 2;
  void *word=*(void **)slots[i];
  if(cheri_gettag(word) && cheri_is_poison(word)) ++stale;
 }
 fprintf(stderr,"stale_poison_capabilities=%u\n",stale);
 struct cheri_revoke_syscall_info info={0};
 if(cheri_revoke(CHERI_REVOKE_LAST_PASS|CHERI_REVOKE_IGNORE_START,0,&info)) return 3;
 unsigned valid=0;
 for(int i=0;i<64;i++) valid+=cheri_gettag(slots[i]);
 fprintf(stderr,"valid_allocations_after_revoke=%u/64\n",valid);
 if(valid!=64) return 1;
 for(int i=0;i<64;i++) free(slots[i]);
 return 0;
}
