#include <lua.hpp>
#include <LuaBridge/LuaBridge.h>
#include <iostream>
#include <cstring>
struct A { ~A(){ i=-1; } int i=0; A& fn(){ i=5; return *this; } int getI() const { return i; } };
static A getA(){ return A{}; }
int main(int argc,char**argv){
  bool control = argc>1 && !strcmp(argv[1],"control");
  lua_State* L = luaL_newstate(); luaL_openlibs(L);
  luabridge::getGlobalNamespace(L).addFunction("getA", getA)
    .beginClass<A>("A").addFunction("fn",&A::fn).addFunction("getI",&A::getI).endClass();
  const char* prog = control
    ? "local a=getA(); a:fn(); collectgarbage('collect'); print('got', a:getI())"
    : "local a1=getA():fn(); collectgarbage('collect'); print('got', a1:getI())";
  if (luaL_dostring(L, prog)) std::cerr << "lua err: " << lua_tostring(L,-1) << "\n";
  lua_close(L);
}
