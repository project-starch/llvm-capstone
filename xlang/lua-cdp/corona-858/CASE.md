# Corona / Solar2D #858 — Lua joint-proxy ⟷ Box2D `b2Joint` use-after-free

**One line.** A Lua physics-joint proxy (a `UserdataWrapper` userdata) is coupled
to a Box2D `b2Joint`; `physics.stop()` deletes the `b2World`, whose block
allocator bulk-frees every `b2Joint` **without** firing Box2D's `SayGoodbye`
destruction callbacks, so the wrapper is never invalidated — and the later Lua GC
finalizer dereferences the freed `b2Joint`.

## Identity

| | |
|---|---|
| Library | Corona / Solar2D game engine (`librtt`), physics via Box2D |
| Language pair | **C++ ⟷ Lua** — the coupled resource is a real Box2D engine object |
| Upstream | https://github.com/coronalabs/corona/pull/858 ("Core: fix box2d joint use after free when world deleted") |
| Fix commit | `aa4e07b82fe80de80ac493596cfbfa37f21fa47b` (PR #858, merged `c8c3bd1`) |
| Vulnerable commit | **`3b738fa825f825e4f1be3e04b8f2cfd2ca0762d7`** (parent of the fix) |
| Native dep | full Solar2D/Corona native engine + Box2D |
| Detect | SIGSEGV in the GC finalizer (per the report) |

## The two coupled objects (why unambiguous CDP)

1. **Lua-GC handle:** the physics-joint proxy userdata (metatable
   `"physics.joint"`), a `UserdataWrapper**` created by `physics.newJoint(...)`
   (`librtt/Rtt_LuaLibPhysics.cpp`). Its lifetime is the Lua GC's.
2. **Separate native resource:** the Box2D `b2Joint`, allocated inside the
   `b2World` block allocator by `b2World::CreateJoint`.

Two distinct allocations, bidirectionally linked: the `UserdataWrapper` stores
the `b2Joint*`, and `b2Joint::GetUserData()` stores the wrapper. **Not** a
borrowed pointer into a single managed object.

**Direction:** native-frees. Destroying the `b2World` frees the `b2Joint`; the
Lua proxy outlives it and is dereferenced by the GC finalizer.

## Mechanism (confirmed from the real source)

- Normal teardown fires `PhysicsDestructionListener::SayGoodbye(b2Joint*)`
  (`Rtt_PhysicsWorld.cpp:45`) → `wrapper->Invalidate()`, after which
  `UserdataWrapper::Dereference()` returns NULL and the proxy is safe.
- **The bug:** `PhysicsWorld::StopWorld()` iterated bodies only to detach their
  *display objects*, then `Rtt_DELETE( fWorld )` (`Rtt_PhysicsWorld.cpp:209`)
  deleted the `b2World`. `~b2World` frees the block allocator that owns every
  `b2Body`/`b2Joint`/`b2Fixture` **without** calling `SayGoodbye`, so the joint
  wrappers are never `Invalidate()`d — they keep pointing at freed `b2Joint`s.
- **Use-site:** later, the Lua GC finalizes the joint proxy →
  `PhysicsJoint::Finalizer` → `GetJoint()` → `wrapper->Dereference()`
  (`Rtt_PhysicsJoint.cpp:46`) returns the **freed** `b2Joint*`, then
  `baseJoint->GetUserData()`/`SetUserData()` (`~:558/:567`) read/write freed
  Box2D memory → SIGSEGV.
- **Fix:** PR #858 adds `fWorld->DestroyBody( body )` (`Rtt_PhysicsWorld.cpp:206`)
  inside the `StopWorld` loop; `DestroyBody` runs Box2D's normal joint
  destruction, which fires `SayGoodbye` → `Invalidate()` for every attached joint
  before the world is torn down.

## Reproduction status

**BLOCKED — no prebuilt engine, from-source build out of scope.**

Precise reason: reproducing the crash requires the full Solar2D/Corona native
engine (`librtt` + the Lua physics binding + Box2D + a platform runtime to run a
`.lua` that builds a physics joint and calls `physics.stop()`), and there is **no
apt/prebuilt package** for the SDK (`apt-cache policy solar2d corona corona-sdk`
→ 0 candidates). A from-source Corona build is a very heavy mobile-engine build
(full SDK build system + platform backends, producing a Simulator/app rather than
a headless lib), which the task scopes out. A hand-rolled harness pairing real
Box2D with a stand-in for Corona's `UserdataWrapper`+Lua proxy would be a
re-implementation, not a ground-truth reproduction of #858, so it is not counted
(cf. the corpus "two distinct real allocations" discipline and the
mock-fidelity rule).

The free-site and use-site above are quoted from the real vulnerable source, so
`boundary.md` and this file are complete despite the BLOCKED status.

## PASS signature (would-be)

Had the engine been buildable: a physics scene that creates a joint, then
`physics.stop()`, then forces a GC — SIGSEGV in `PhysicsJoint::Finalizer` via
`UserdataWrapper::Dereference()` on the freed `b2Joint`, on the vulnerable
`3b738fa8` tree and **clean** on the fixed `aa4e07b8` tree (SayGoodbye invalidates
the wrapper first).
