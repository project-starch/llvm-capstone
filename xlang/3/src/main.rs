extern crate hlua;
use hlua::*;

fn main() {
    let mut lua = Lua::new();
    // Placeholder trigger code for hlua #144
    lua.execute::<()>("print('hlua #144 trigger')").unwrap();
}
