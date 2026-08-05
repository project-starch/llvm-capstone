# Real Lua on the gp-captable ABI: descriptor delivery to result=400

Reference Lua 5.4.7 brought up as an interpreter-scale (`-capstone-gp-captable`)
Capstone domain. On QEMU the interpreter now runs a chunk **end to end**, WITH the
base library loaded and CALLED: `luaL_newstate` + `luaopen_base` +
`luaL_loadbufferx(LUA_OK)` + `lua_pcall` -> `print('Lua 5.4.7 on Capstone')` (real
output) -> `LUA-OK result=400` for
`... local t={} for i=1,20 do t[i]=i*i end return t[20]` (numeric for, table
array-part growth over the tag-preserving revoking realloc, integer arithmetic, a
base-library call, return). This note records what was non-obvious: how the
gp-captable init descriptor is delivered on QEMU, and the FIVE compiler/ABI walls
between "compiles a chunk" and "runs full Lua" -- the headline ones being Lua's
computed-goto dispatch table and lua_gettop's pointer-difference lowering.

Commits: parent `b2ff97a` (xlang Lua port, `capstone-bootstrap-xlang`); submodule
`8c7b973` on local branch `xlang-gp-captable-delivery` in `caplifive-buildroot`
(the module/libcapstone delivery; parent submodule pointer intentionally **not**
bumped, not pushed).

## The chain, in order

1. **cjalr ABI is a dead end for Lua.** The small-domain `.capstone_cap_init` walk
   under-tags Lua-scale static capability tables (base_funcs[], luaX_tokens[]) ->
   `luaopen_base` reads an untagged cap and faults (cause 24). Amalgamation did not
   fix it; the mechanism is wrong for the scale.
2. **gp-captable is the right ABI** (the SQLite-scale path): every global reached
   via `ldc gp[i]`, tagged at entry from a single `.capstone_gp_initdesc`
   descriptor the interp glue (`start-gp-captable-interp.S`) walks. The compiler
   emits ONE descriptor per module (per-module `{built_flag,count}` header), so
   exactly one module may own gp-captable globals -> everything-with-globals is
   folded into `lua_gp_amalg.c`.
3. **But the descriptor was never delivered.** The glue reads `count` from the
   front of `dom_data`; nothing in this tree copies the globals template there (it
   lives in the code image, which the domain cannot read -- it holds no capability
   over its own image; `cepc` is untagged at entry, `cscratch` covers only
   dom_data, there is no PCC/DDC-materialising instruction, and QEMU strictly
   asserts `lcc`/`scc` on untagged operands). Glue reads `count==0`, skips the
   cap-table build and the `scc` that sets sp's cursor to the stack top, and the
   `test:` prologue underflows sp (cause 7 at pc 0x...024c, `stc ra,48(sp)` with
   sp.cursor 96 below its own base). This is the C-13 family; the **SQLite
   gp-captable domain fails identically**, proving it is build-independent.

## The fix: deliver the descriptor from the kernel module (no firmware rebuild)

The monitor has no copy and cannot be changed without a firmware rebuild; the
domain cannot read its own image. The kernel module, however, has wide access to
the freshly-loaded image and knows the dom_data layout the monitor carves. So
(submodule commit `8c7b973`):

- `libcapstone` computes `gp_offset` = the `.capstone_gp_initdesc` section vaddr
  minus loadable_start, and passes it in a new `ioctl_dom_create_args` field.
  (The compute MUST run before libcapstone munmaps the ELF -- a use-after-unmap
  there SIGSEGVs the host with exit 139.)
- `ioctl_create_dom` copies the globals template `[gp_offset, code_len)` of the
  just-loaded image to `dom_data.base` = `align16(code_len) + MONITOR_SEAL_SIZE`
  (16*96 = 1536, the monitor's DOMAIN_DATA_SIZE, NOT the module's larger
  allocation slack). Gated on `gp_offset != 0`, so non-gp-captable domains are
  untouched.

Also required: `ROF_MAX_SLOTS` was cut 8192 -> 1024 (256 KB -> 32 KB of
`rof_slots`) so the carved globals fit the ~118 KB dom_data with stack to spare.

Deployment on QEMU without touching the shared rootfs: the module uses
`misc_register`, so `rmmod capstone && insmod /mnt/host/capstone_new.ko` in the
guest command hot-swaps it via the 9p share.

## Verified on QEMU (marker trace)

`create_dom ok` -> 3 region shares (each enters+returns) -> run entry ->
`luaL_newstate()` returns -> S2 base SKIPPED -> `luaL_loadbufferx` returns
`LUA_OK` (the parser compiled the chunk) -> `lua_pcall` -> QEMU assert
`helper_cslcc: rs1_v->tag`.

## The remaining wall: untagged-cap `lcc` in Lua's stack move (relstack)

`lua_pcall` grows the stack (`luaD_growstack` -> `luaD_reallocstack` ->
`relstack`). `relstack` converts stack pointers to offsets via
`savestack(o) = (char*)o - (char*)L->stack.p`, which the compiler lowers to
`lcc <cap>,2` (read cursor) on each operand; on QEMU `lcc` asserts the operand is
tagged. `correctstack` stores `.p` via `stc` and `restorestack` uses
`cincoffset(L->stack.p, off)`, so the fixup is tag-preserving -- the loss is at
setup, on the FIRST realloc (newstate does none). Lua 5.4's `StkIdRel` is a UNION
of `StkId p` (capability) and `ptrdiff_t offset` (scalar); writing the scalar
member clears the 16-byte slot's tag, which is the classic capability-in-union
trap and the leading suspect.

ROOT CAUSE (CONFIRMED, and it is the COMPILER, not relstack/the union): the
Capstone backend mis-lowers **variable-offset pointer arithmetic** under the
capability ABI. Minimal repro:

```
typedef struct { long a[4]; } SV;            /* 32-byte, like StackValue */
SV *f_var(SV *p, int n) { return p - (n+1); } /* BUG */
SV *f_const(SV *p)      { return p - 1;     } /* OK  */
```

At -O0 -mllvm -capstone-gp-captable:
  f_const:  cincoffsetimm a0, a0, -32           (cincoffset the CAPABILITY -- correct)
  f_var:    lcc a0 = cursor(p); lcc a1 = cursor(offset); sub; cincoffsetimm a0,a0,-32
            -- i.e. ptrtoint(p) via lcc, scalar arithmetic, then cincoffset a SCALAR.
            The result is UNTAGGED (no capability base survives).

`lua_pcall`'s `c.func = L->top.p - (nargs+1)` (lapi.c) is exactly `ptr - (var+const)`,
so `c.func` comes out untagged; the first `lcc`/`cincoffset` that touches it traps
(QEMU asserts `helper_cslcc`/`helper_cscincoffsetimm rs1_v->tag`). Diagnosed by
patching QEMU's `helper_cslcc` to log the pc + tolerate an untagged operand
(op_helper.c:667): the log showed `pc` in lua_pcallk, `imm=2`, `val=0x0` (the NULL
= nargs*32 for nargs=0), and after tolerating, a `cincoffsetimm`-on-untagged trap --
matching the disasm above. That QEMU patch is diagnostic only; revert it.

FIX (backend, committed `1fe2a76`): in `lowerSUB`/`isCapstoneIntegerOffset`
(CapstoneISelLowering.cpp) -- any `shl i128` is an integer offset (a capability is
never shifted), and `lowerSUB` truncates any remaining integer-offset shape to
XLen -> a single register `CIncOffset` on the base capability. Verified: minimal
repro `p-(n+1)` now `slli; neg; cincoffset` (no lcc); lit 45/45 + new
`cap-i128-ptr-arith-variable.ll`. `ptr + variable` (lowerADD) was already correct.
Same family as the two earlier i128 gp-captable fixes.

## RESULT=400 REACHED. The last wall: Lua's computed-goto dispatch table

After the pointer-arithmetic fix the untagged-pointer trap was gone, but execution
still failed one layer deeper. With the diagnostic QEMU (tolerant `helper_cslcc`) it
looked like a HANG; reverting to the STRICT assert turned it into a clean, precisely
located fault, and a chunk LADDER (trivial `return 1` up to the full chunk, each
returning a csdebugprint marker in one boot -- lua_domain.c `run_lua_ladder`, built
with `-DLUA_CHUNK_LADDER`) showed EVEN `return 1` faults inside `lua_pcall`, on the
FIRST opcode dispatch:

  domain halted by capability fault: cause = 1, pc = 0x39048

ROOT CAUSE (4th and final for this bringup): Lua 5.4 defaults (under `__GNUC__`, so
under clang) to `ljumptab.h`'s computed-goto VM dispatch:

  static const void *const disptab[NUM_OPCODES] = { &&L_OP_MOVE, ... };
  #define vmdispatch(x)  goto *disptab[x];

`disptab[]` is a static table of address-of-label (`&&label`) values -- CODE
capabilities pointing into the middle of `luaV_execute`. Under the gp-captable ABI
those label-address entries are NOT tagged by the `.capstone_gp_initdesc` mechanism
(data-pointer and function-pointer globals ARE), so the first `goto *disptab[op]`
jumps through an UNTAGGED capability and faults on the instruction fetch (cause = 1)
at the correct-address-but-untagged handler (0x39048). The pc in the fault is the
target label; disasm confirmed the dispatch shape immediately before it:
`lw` instruction, `andi ..,0x7f` (GET_OPCODE, 7-bit), `slli ..,0x4` (x16 = capability
stride), `ldc ..,0x210(gp)`, `cincoffset`, `ldc ..,0(..)` -> load a capability from
the opcode-indexed table, then jump through it.

FIX (build flag, NOT a compiler change): `-DLUA_USE_JUMPTABLE=0`, baked into the
build's COMMON flags (build-lua-gp-captable.sh). This is Lua's own supported portable
config -- it replaces the computed goto with a plain `switch`, removing the
code-capability table entirely. `-fno-jump-tables` already makes that switch a
compare/branch chain (no table). Verified: `disptab` appears 7x in luaV_execute's asm
with the default, 0x with `=0`; luaV_execute shrinks 13188 -> 11034 asm lines; ldc-gp
count drops 443 -> 375. This is the lazy-correct rung (a config the platform already
provides), not a symptom patch.

KNOWN COMPILER GAP (recorded, not fixed -- out of scope for result=400): under
`-capstone-gp-captable`, a static array of address-of-label (`&&label`) values is not
emitted with capability-init records, so its entries load untagged. Data-pointer and
function-pointer globals are tagged correctly; only the `&&label`-inside-a-function
case is missed. Any future C that uses computed goto (GCC `&&`/`goto *`) on this ABI
hits the same wall. Lua was the only computed-goto user in this domain (grep of all
domain sources: `ljumptab.h` only).

VERIFIED ON QEMU (base skipped, staged demo LUA_STAGE=5):

  S1 newstate ok -> S2 base SKIPPED -> S3 load(LUA_OK) -> S4 pcall(rc=0)
  -> LUA-OK result=400 expected=400   (host process exit 0)

i.e. real Lua 5.4.7 runs the chunk `local t={} for i=1,20 do t[i]=i*i end return
t[20]` end to end -- newstate + parse + execute (numeric for, table array-part growth
+ tag-preserving revoking realloc, integer arithmetic, return) -- as a Capstone
gp-captable domain, and returns 400. The three compiler/ABI fixes it took to get here
(two i128 pointer-arith lowerings + this jump-table config) are all in place; the
descriptor delivery (module memcpy) is unchanged.

## Full Lua: the base library, and the last codegen bug (lua_gettop)

With the jump-table fix, base-SKIPPED core runs (result=400). Enabling the base
library (`luaopen_base`) then worked immediately -- the old "base wedge" was a stale
pre-fix diagnosis; base_funcs[]'s function-pointer globals ARE tagged (only
`&&label` code-pointer tables were not). But CALLING a base function faulted:
`print('...')` reached `S2 base ok` + parse, then `helper_cscincoffsetimm rs1->tag`
inside `lua_pcall`.

Localised by a fresh-state chunk ladder (`return 1` / `local p=print` /
`print()` / ...): the global LOOKUP works, the CALL faults, and `print()` with no
args faults -- so it is the `OP_CALL` -> C-function path, which the pure-core chunk
never exercised (it has no calls; `luaL_requiref` uses the C-API `lua_call`, not the
`OP_CALL` bytecode). A QEMU pc-log (env->pc synced via cpu_restore_state, minus the
PCC base) mapped the fault to `lua_gettop + 0x40`.

ROOT CAUSE (5th codegen bug): `lua_gettop` is `L->top.p - (L->ci->func.p + 1)`, a
pointer DIFFERENCE with a constant element offset. DAGCombine correctly reassociates
`p - (q+1)` to `add(sub(p,q), -32)`, where `sub(p,q)` is a pointer difference -- a
SCALAR byte count (32-byte StackValue -> the `-32` is the +1 element, and `>>5` is
the /32). But `lowerADD` treated every i128 add as capability+offset and emitted
`CIncOffset(<scalar>, -32)` -- a `cincoffsetimm` on the untagged scalar difference,
which faults. `lua_gettop` is on the entry path of nearly every C-API/base function,
so this blocked all base calls (print, assert, ...).

FIX (backend, `lowerADD` in CapstoneISelLowering.cpp): when NEITHER operand of an
i128 add is a capability base -- the offset is an integer offset and the base is a
ptr-ptr SUB of two capability values (or an already-lowered sign-extended difference)
-- lower the add in the XLen domain (`sub; addi; sign_extend`), never `CIncOffset`.
Fires only when no capability is present, so it can never make real pointer
arithmetic scalar. Same family as the earlier i128 fixes. Verified: `p-(q+1)` now
`sub; addi -32; srli` (no cincoffsetimm); Capstone lit 47/47 + new
`cap-i128-ptr-diff-const.ll`.

VERIFIED ON QEMU (base ENABLED, print chunk): `S2 base ok -> parse -> pcall ->
"Lua 5.4.7 on Capstone" (real print output) -> LUA-OK result=400`, no fault. Real
Lua 5.4.7 with its base library runs a chunk that CALLS a base function and returns
400 on Capstone.

Note: the earlier opcode-ladder's multi-chunk plumbing (`lua_settop(L,0)` between
chunks, one shared state) tripped this same `cscincoffsetimm` -- `lua_settop` also
goes through the pointer-difference path -- which is why the ladder was switched to a
fresh lua_State per snippet. The single-chunk demo never needed it. That artifact is
now fixed by the same lowerADD change.

## Build / run

```
LUA_STAGE=5 LUA_DBG_STAGE=1 LUA_SKIP_BASE=1 \
  bash xlang/lua-cdp/capstone-lua/build-lua-gp-captable.sh
# rebuild the module once (source in the caplifive-buildroot submodule):
#   make -C <buildroot>/build/build/linux-custom M=<modcapstone build>/module \
#        ARCH=riscv CROSS_COMPILE=<buildroot host>/riscv64-buildroot-linux-gnu- modules
# run: run-domain-smoke.py with a second -append 'console=ttyS0 earlycon=sbi
#   root=/dev/vda ro' (this machine's kernel needs an explicit console=), and the
#   rmmod/insmod hot-swap guest command above. Boot is ~50% flaky (OpenSBI->kernel
#   stall); retry until 'buildroot login' appears.
```

Build knobs (all QEMU-only diagnostics, off by default): `LUA_DBG_STAGE`
(csdebugprint stage markers 7xx/8xx/9xx that survive a wedge), `LUA_SKIP_BASE`
(runtime-skip luaopen_base, kept LINKED so the image layout matches the
base-enabled build), `LUA_DBG_RELSTACK`, `LUA_CHUNK_LADDER` (run a cheap->complex
chunk ladder in one boot, markers 3xx, to localise a hang/fault to an opcode class;
implies LUA_DBG_STAGE). `-DLUA_USE_JUMPTABLE=0` is NOT a knob -- it is baked into
COMMON as a correctness requirement (see the jump-table section above).

QEMU run note: the ~50% boot flake plus an occasional create_dom stall means several
retries per successful run; the driver (`run-ladder.sh` in scratch) caps the login
timeout to an absolute 90s and retries both flake classes, breaking only once the
domain has actually run (`create_dom ok` present) or a real result appears.
