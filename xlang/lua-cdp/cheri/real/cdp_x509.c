/* luaossl #124 cross-domain double-free through real Lua on CheriBSD purecap.
 * Same reproduction as the Capstone LUA_CDP_X509 domain, as a normal process:
 * under revocation the 2nd owner's stale refcount access faults (CAUGHT); with
 * revocation off it completes and prints CDP-MISS. */
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define X509_STORE_BYTES 152
#define REFCOUNT_OFF 136
typedef struct { unsigned char *cert_store; } MockSSL_CTX;
static void x509_store_free(unsigned char *s){ if(!s)return; volatile unsigned int*rc=(volatile unsigned int*)(s+REFCOUNT_OFF); if(--(*rc)==0) free(s); }
static int xs__gc(lua_State*L){ unsigned char**pp=lua_touserdata(L,1); x509_store_free(*pp); return 0; }
static int sx__gc(lua_State*L){ MockSSL_CTX**pp=lua_touserdata(L,1); MockSSL_CTX*c=*pp; x509_store_free(c->cert_store); free(c); return 0; }
static int store_new(lua_State*L){ unsigned char*s=malloc(X509_STORE_BYTES); memset(s,0,X509_STORE_BYTES); *(volatile unsigned int*)(s+REFCOUNT_OFF)=1; unsigned char**ud=lua_newuserdatauv(L,sizeof(void*),0); *ud=s; luaL_setmetatable(L,"x509.store"); return 1; }
static int ctx_new(lua_State*L){ MockSSL_CTX*c=malloc(sizeof(MockSSL_CTX)); c->cert_store=0; MockSSL_CTX**ud=lua_newuserdatauv(L,sizeof(void*),0); *ud=c; luaL_setmetatable(L,"ssl.context"); return 1; }
static int setStore(lua_State*L){ MockSSL_CTX**c=lua_touserdata(L,1); unsigned char**s=lua_touserdata(L,2); (*c)->cert_store=*s; return 0; }
static const char CHUNK[]="local ctx=ctx_new() do local s=store_new() setStore(ctx,s) end collectgarbage('collect') ctx=nil collectgarbage('collect') return 1";
int main(void){
  lua_State*L=luaL_newstate();
  luaL_requiref(L,"_G",luaopen_base,1); lua_pop(L,1);
  luaL_newmetatable(L,"x509.store"); lua_pushcfunction(L,xs__gc); lua_setfield(L,-2,"__gc"); lua_pop(L,1);
  luaL_newmetatable(L,"ssl.context"); lua_pushcfunction(L,sx__gc); lua_setfield(L,-2,"__gc"); lua_pop(L,1);
  lua_pushcfunction(L,store_new); lua_setglobal(L,"store_new");
  lua_pushcfunction(L,ctx_new); lua_setglobal(L,"ctx_new");
  lua_pushcfunction(L,setStore); lua_setglobal(L,"setStore");
  if(luaL_dostring(L,CHUNK)!=LUA_OK){ printf("CDP-ERR: %s\n",lua_tostring(L,-1)); return 2; }
  printf("CDP-MISS: double-free survived\n");
  lua_close(L); return 0;
}
