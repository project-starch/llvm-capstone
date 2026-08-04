// sol2 #1373 — a Lua wrapper (bb) holds an INTERIOR pointer into a C++ struct
// (ComplexStructC) owned by a Lua-GC userdata; collecting the parent dangles bb.
//
// Verbatim upstream reproducer (godbolt z/Eb6zGaKjr), parameterised:
//   argv[1] == "control"  -> keep the parent referenced, so bb stays valid.
// Two distinct allocations: the Lua userdata for csc, and bb = an interior
// reference into it. Not a string-borrow.
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>
#include <iostream>
#include <string>
struct ComplexStructA { int a = 10; };
struct ComplexStructB { ComplexStructA a; int b = 20; };
struct ComplexStructC { ComplexStructB b; int c = 30; };
static ComplexStructC GetC(int a,int b,int c){ ComplexStructC x; x.b.a.a=a; x.b.b=b; x.c=c; return x; }
static ComplexStructC GetC(){ return GetC(10,20,30); }
int main(int argc,char**argv){
  bool control = argc>1 && std::string(argv[1])=="control";
  sol::state lua; lua.open_libraries(sol::lib::base, sol::lib::string, sol::lib::table);
  lua.new_usertype<ComplexStructA>("ComplexStructA", sol::call_constructor, sol::constructors<ComplexStructA()>(), "a", &ComplexStructA::a);
  lua.new_usertype<ComplexStructB>("ComplexStructB", sol::call_constructor, sol::constructors<ComplexStructB()>(), "a", &ComplexStructB::a, "b", &ComplexStructB::b);
  lua.new_usertype<ComplexStructC>("ComplexStructC", sol::call_constructor, sol::constructors<ComplexStructC()>(), "b", &ComplexStructC::b, "c", &ComplexStructC::c);
  lua.set("GetComplexStructC", sol::overload(static_cast<ComplexStructC(*)(int,int,int)>(&GetC), static_cast<ComplexStructC(*)()>(&GetC)));
  for (int i=0;i<20;++i){
    if (control)
      lua.safe_script("local csc=GetComplexStructC(); bb=csc.b; bb.b=100; keep=csc", sol::script_pass_on_error);
    else
      lua.safe_script("local csc=GetComplexStructC(); bb=csc.b; bb.b=100; csc=nil", sol::script_pass_on_error);
    lua.collect_garbage(); lua.collect_garbage();
    auto res = lua.safe_script("return bb.a.a", sol::script_pass_on_error);   // UAF read on the vuln path
    (void)res;
  }
  std::cout << "DONE\n";
}
