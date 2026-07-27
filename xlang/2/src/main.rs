// rlua #97 -- callback arguments escape their real lifetime, then are used after
// the Lua state is dropped.
//
// This is the upstream proof-of-concept. It is the compile-fail test the fix
// commit (0f5a9a3, "Fix terrible soundness issue... barely") added as
// `tests/compile-fail/static_callback_args_tls.rs`: on a FIXED rlua this file
// must not compile, and the fact that it *does* compile at the pinned vulnerable
// commit is itself half the bug.
//
// THE UNSOUNDNESS
// At the vulnerable commit `create_function` reads:
//
//     pub fn create_function<'lua, 'callback, A, R, F>(&'lua self, func: F) -> ...
//     where
//         A: FromLuaMulti<'callback>,
//         R: ToLuaMulti<'callback>,
//         F: 'static + Send + Fn(&'callback Lua, A) -> Result<R>,
//
// `'callback` appears only in the bounds and is otherwise unconstrained, so the
// CALLER chooses it -- including `'static`. The callback's arguments are handles
// into the Lua state (here a `Table`), but their lifetime no longer has to be
// tied to the `&Lua` borrow that produced them. So a callback can stash a
// `Table<'static>` somewhere that outlives the state.
//
// The fix removes `'callback` entirely and ties everything to `'lua`, which makes
// the stash below fail to compile.
//
// WHAT THIS PROGRAM DOES
//   1. Declares a thread-local that can hold a `Table<'static>`.
//   2. Registers a callback that moves its `Table` argument into that TLS slot,
//      laundering it out of the callback's real lifetime.
//   3. Drops the `Lua` state, freeing the Lua heap the table lived in.
//   4. Reads the escaped table -- a use-after-free of Lua-owned memory.

extern crate rlua;

use std::cell::RefCell;

use rlua::{Lua, Table};

fn main() {
    thread_local! {
        // Only expressible because `'callback` is unconstrained at this commit.
        static BAD_TIME: RefCell<Option<Table<'static>>> = RefCell::new(None);
    }

    let lua = Lua::new();

    lua.create_function(|_, table: Table| {
        // Launder the argument out of its real lifetime into thread-local storage.
        BAD_TIME.with(|bt| {
            *bt.borrow_mut() = Some(table);
        });
        Ok(())
    })
    .unwrap()
    .call::<_, ()>(lua.create_table().unwrap())
    .unwrap();

    // Free the Lua state -- and with it the heap the escaped table points into.
    drop(lua);

    // THE STALE USE. `len()` walks the freed Lua state.
    BAD_TIME.with(|bt| {
        println!(
            "you're gonna have a bad time: {}",
            bt.borrow().as_ref().unwrap().len().unwrap()
        );
    });
}
