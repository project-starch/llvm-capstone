extern crate rlua;
use rlua::*;

fn main() {
    let lua = Lua::new();
    // Placeholder trigger code for rlua #97
    lua.eval::<()>("print('rlua #97 trigger')", None).unwrap();
}
