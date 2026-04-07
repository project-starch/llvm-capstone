# my_first_domain

This sample now uses the in-tree LLVM Capstone compiler for the actual code,
while following the same domain-entry ABI as the working examples in
`caplifive-buildroot/package/capstone-test-domains/src`.

## Why the old `start.S` + `main()` path failed

- the usable stack/root capability comes from `cscratch`, not a normal ELF `_start` contract,
- the qemu test harness calls a domain entry shaped like `(ra capability in x1, unsigned func in a0, unsigned *buf capability in a1)`,
- the result must be written through `buf`,
- the domain must exit with `domreturn`, not by returning from a freestanding `main()`.

So the previous LLVM-style `_start -> main -> return` flow was the wrong ABI for
this environment.

## Current sample

- `start.S` is a handwritten ABI wrapper that enters from `cscratch`, calls a normal C function, and returns with `domreturn`.
- `main.c` is plain C compiled by our in-tree `clang`.

## Build

```bash
cd /home/alexey/dev/llvm-capstone/capstone/my_first_domain
chmod +x build.sh
./build.sh
```

Optional overrides:

```bash
LLVM_BIN=/path/to/llvm/build/bin \
BUILDROOT_HOST_BIN=/path/to/caplifive-buildroot/build/host/bin \
./build.sh
```

The output domain ELF is written to `my_domain.dom`.

## Current compatibility note

The existing Buildroot linker and userspace loader still hard-code `EM_RISCV`.
So `build.sh` currently applies a temporary header rewrite on the intermediate
objects before linking. The machine code still comes from our LLVM Capstone
backend; this is only a compatibility shim until `ld.lld` and the runtime are
updated for native `EM_CAPSTONE`.

## VM-verified behavior

With the in-tree LLVM compiler, the existing Buildroot toolchain/runtime, and
`capstone-qemu`, the domain loads and executes successfully under `/capstone-test.user`.

In the current test harness, the observable printed value is:

```text
Called dom (1-th time) retval = 0
```

This is consistent with the in-tree reference `fib.dom`, which shows the same
observable `retval` in the same harness.

