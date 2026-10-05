/* C-77: a pointer-typed common symbol crashes the capstone64 backend.
 *
 * With -fcommon (the default of every compiler before GCC 10, and what the
 * 1990s C of Olden/Ptrdist/MallocBench relies on) this tentative definition
 * is a common symbol. Emitting it reaches
 *   llvm_unreachable("Unknown section kind")
 * in getSectionPrefixForGlobal, TargetLoweringObjectFileImpl.cpp:631.
 * An integer common symbol (`int n;`) compiles; a pointer one does not. */
char *shared_name;

int main(void) { return shared_name != 0; }
