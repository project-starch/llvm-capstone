# Boundary annotation — Corona/Solar2D #858

### The object that crosses the boundary

A raw `b2Joint*` (a Box2D joint, owned by the `b2World` block allocator), stored
inside the Lua physics-joint proxy via a C++ `UserdataWrapper`. The proxy
userdata (metatable `"physics.joint"`) is the Lua-visible handle; the `b2Joint`
pointer it wraps is what crosses and is retained past the joint's lifetime. The
link is bidirectional: `UserdataWrapper` holds the `b2Joint*` and
`b2Joint::GetUserData()` holds the wrapper.

### Owner vs. borrower

- **The C++ side (Box2D, via Corona's `PhysicsWorld`) owns the memory.**
  `b2World::CreateJoint` allocates the `b2Joint` in the world's block allocator;
  `~b2World` (reached from `Rtt_DELETE( fWorld )`) frees it.
- **Lua (managed) owns the handle.** The joint-proxy userdata's lifetime is the
  Lua GC's; it can easily outlive the `b2World`.
- The coupling is meant to be severed by Box2D's destruction listener: on normal
  joint/body destruction Box2D calls `SayGoodbye(b2Joint*)`, and Corona's
  listener `Invalidate()`s the wrapper so `Dereference()` returns NULL. The bug
  is a teardown path that frees the joints while bypassing that callback.

### Free site

`PhysicsWorld::StopWorld()` → `Rtt_DELETE( fWorld )`
(`librtt/Rtt_PhysicsWorld.cpp:209`). `~b2World` destroys the block allocator that
owns all `b2Joint`s **without** invoking
`PhysicsDestructionListener::SayGoodbye(b2Joint*)` (`Rtt_PhysicsWorld.cpp:45`), so
no `wrapper->Invalidate()` runs. Native-frees. (In the vulnerable
`3b738fa8` tree the `StopWorld` body loop detaches only *display objects*; the
fix `aa4e07b8` adds `fWorld->DestroyBody( body )` at `:206` so `SayGoodbye` fires
per joint first.)

### Stale-use site (one crossing later)

Lua GC finalizes the proxy → `PhysicsJoint::Finalizer` → `GetJoint( L, 1 )` →
`wrapper->Dereference()` (`librtt/Rtt_PhysicsJoint.cpp:46`). Because the wrapper
was never invalidated, `Dereference()` returns the **freed** `b2Joint*` (instead
of NULL); the finalizer then dereferences it via `baseJoint->GetUserData()` /
`baseJoint->SetUserData(...)` (`~Rtt_PhysicsJoint.cpp:558/567`) → read/write of
freed Box2D memory → SIGSEGV.

### The lifetime rule that is violated

A managed handle wrapping a native resource must have its stored pointer
invalidated the moment the resource is freed. Box2D provides exactly that hook
(`b2DestructionListener::SayGoodbye`), and Corona wires it to
`UserdataWrapper::Invalidate()` — but only the *normal* destruction path
(`DestroyBody`/`DestroyJoint`) fires it. Bulk-freeing via `~b2World` skips the
callback, leaving `Dereference()` handing out a dangling pointer. The fix routes
world teardown through `DestroyBody` so the invalidation always runs.

### Capability note (revoke-on-free)

On a revoke-on-free allocator, `~b2World` freeing the `b2Joint` **revokes** the
capability the `UserdataWrapper` holds. The finalizer's `Dereference()` then
yields a revoked capability, so `baseJoint->GetUserData()` faults at the contract
point — the delivered fault the model promises, in place of the report's SIGSEGV
(and in place of the silent corruption a non-checking allocator would allow when
the freed block is reused).
