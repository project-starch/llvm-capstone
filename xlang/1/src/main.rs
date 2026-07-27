// rlua #19 -- minimal Rust host for the Lua trigger in `trigger.lua`.
//
// The host does three things and nothing else: define a Rust type that owns
// heap memory, hand one instance to Lua as userdata, and run the script.
// Everything that drives the bug lives in trigger.lua.

extern crate rlua;
use rlua::*;

/// The value that crosses the FFI boundary.
///
/// `payload` matters: it is what makes this a *memory-safety* reproduction
/// rather than only a logical use-after-drop. A `String` owns a heap buffer, so
/// `Drop` actually calls `free` on it and AddressSanitizer can see the later
/// read. A payload of plain scalars would drop without freeing anything and the
/// stale access would be invisible to the sanitizer.
struct Userdata {
    id: u8,
    payload: String,
}

impl Drop for Userdata {
    fn drop(&mut self) {
        // Printed once per drop. Lua's __gc running on a resurrected object is
        // what lets the value be dropped while Lua still hands it out.
        println!("dropping {}", self.id);
    }
}

impl LuaUserDataType for Userdata {
    fn add_methods(methods: &mut LuaUserDataMethods<Self>) {
        methods.add_method("access", |_, this: &Userdata, _: LuaMultiValue| {
            // THE STALE USE. By the time trigger.lua calls this, `Drop` has
            // already freed `payload`'s buffer; reading it here is the UAF.
            println!("accessing userdata {} payload={}", this.id, this.payload);
            Ok(LuaMultiValue::new())
        });
    }
}

fn main() {
    let lua = Lua::new();
    {
        let globals = lua.globals();
        globals
            .set(
                "userdata",
                Userdata {
                    id: 123,
                    // Long enough to be a distinct heap allocation rather than
                    // anything the allocator might inline or share.
                    payload: "HEAP_PAYLOAD_STRING_LONG_ENOUGH_TO_ALLOCATE".to_string(),
                },
            )
            .unwrap();
    }

    // Baked in at compile time so the artifact stays hermetic: no runtime file
    // lookup, no dependence on the working directory.
    let script = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/trigger.lua"));
    lua.eval::<()>(script, None).unwrap();
}
