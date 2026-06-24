// Minimal repro: a C++ `new <type>` expression crashes the Capstone backend.
//
//   clang -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
//     -ffreestanding -std=c++17 -fno-exceptions -fno-rtti -c cxx-new-expr-crash.cc
//
//   clang: llvm/lib/Support/APInt.cpp:1013: APInt llvm::APInt::zext(unsigned):
//   Assertion `width >= BitWidth && "Invalid APInt ZeroExtend request"' failed.
//
// Narrowing:
//   - a plain C++ class/method (no `new`) compiles fine;
//   - a raw `operator new(8)` *call* compiles fine;
//   - only the `new <type>` *expression* crashes.
// So the bug is in lowering the new-expression (its size / pointer-width
// computation) for Capstone's 128-bit capability pointers — an APInt zext to a
// width smaller than the value's bit width.
//
// This is one of two blockers for the RV8 `bigint` (C++) benchmark; the other is
// the absence of a C++ standard library (<vector>/<string>/<iostream>) for the
// capstone64 target. Both are part of a future "C++ on the domain" bring-up.
int *make() { return new int; }
