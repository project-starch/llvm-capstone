// sol2 #1080 — a sol POINTER-userdata (metatable "sol.Foo*") wraps a native C++
// Foo; the native Foo is destroyed while Lua keeps the userdata alive (stored in
// the global `test`), and a later Lua->C++ call (`test:Print()`) re-derefs the
// freed pointer. TWO distinct allocations: the Lua userdata, and the native heap
// Foo. Direction: NATIVE-frees (the C++ object dies first; Lua outlives it).
//
// Faithful to the upstream reproducer (godbolt aG3qax buggy / Yffz4o value-copy).
// The bug itself is verbatim: an lvalue push => "Foo*" pointer-userdata that
// outlives the object, vs an rvalue push => a Lua-owned "Foo" value copy. Two
// adaptations make ASan report a HEAP use-after-free deterministically, without
// changing the bug:
//   * the native Foo is heap-allocated (new/delete) instead of a stack local, so
//     the free is a heap free (godbolt's stack local yields stack-use-after-scope,
//     not the heap-use-after-free the case asserts);
//   * Foo::Print reads a data member (`val`) so the stale-pointer method call
//     actually dereferences the freed block (the upstream Print only printed
//     `this`, an address, which never touches Foo's storage — a logical UAF that
//     ASan cannot see).
//
//   argv[1] == "control" -> value-copy: sol copies into a Lua-owned "Foo"
//                           userdata, so the stored object survives the delete.
//   default (vuln)       -> pointer-wrap: sol stores &Foo ("Foo*"), which dangles
//                           after delete; re-deref = heap-use-after-free.
//
// The differential is tight: BOTH paths new a heap Foo and delete it; they differ
// ONLY in the push value category (lvalue `*m` vs rvalue copy `Foo(*m)`), so a UAF
// on one and not the other isolates the cause to pointer-wrap vs value-copy — the
// free itself is identical.
#define SOL_ALL_SAFETIES_ON 1
#include <sol/sol.hpp>
#include <iostream>
#include <string>

struct Foo {
    int val;
    Foo() : val(42) { std::cout << "Foo: " << this << "\n"; }
    Foo(const Foo& o) : val(o.val) { std::cout << "Foo(copy): " << this << "\n"; }
    ~Foo() { std::cout << "~Foo: " << this << "\n"; val = -1; }
    int Print() { std::cout << "Print: " << this << " val=" << val << "\n"; return val; } // reads this->val
};

int main(int argc, char** argv) {
    bool control = argc > 1 && std::string(argv[1]) == "control";
    sol::state lua;
    lua.open_libraries();

    lua.new_usertype<Foo>("Foo",
        sol::no_constructor,
        "Print", &Foo::Print
    );

    lua.script(R"(
function storeFoo(foo)
    print("__name=" .. tostring(debug.getmetatable(foo).__name))
    test = foo          -- keep the Lua userdata alive past the native C++ object
    test:Print()
end
)");

    sol::protected_function storeFoo = lua["storeFoo"];
    {
        Foo* m = new Foo();
        if (control)
            storeFoo(Foo(*m));  // rvalue COPY  -> Lua owns a "Foo" value userdata
        else
            storeFoo(*m);       // lvalue REF   -> Lua stores a "Foo*" pointer to m
        delete m;               // native free of the heap Foo (identical on both paths)
    }
    // Lua's `test` still references the userdata; re-deref the stored object:
    auto r = lua.safe_script("return test:Print()", sol::script_pass_on_error); // vuln: UAF read of freed Foo::val
    (void)r;
    std::cout << "DONE\n";
}
