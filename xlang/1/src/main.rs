extern crate rlua;
use rlua::*;

struct Userdata {
    id: u8,
}

impl Drop for Userdata {
    fn drop(&mut self) {
        println!("dropping {}", self.id);
    }
}

impl LuaUserDataType for Userdata {
    fn add_methods(methods: &mut LuaUserDataMethods<Self>) {
        methods.add_method("access", |_, this: &Userdata, _: LuaMultiValue| {
            println!("accessing userdata {}", this.id);
            Ok(LuaMultiValue::new())
        });
    }
}

fn main() {
    let lua = Lua::new();
    {
        let globals = lua.globals();
        globals.set("userdata", Userdata { id: 123 }).unwrap();
    }

    lua.eval::<()>(r#"
        local tbl = setmetatable({
            userdata = userdata
        }, { __gc = function(self)
            -- resurrect userdata by assigning it to a global variable
            hatch = self.userdata
        end })

        print("collecting...")
        tbl = nil
        userdata = nil -- make table and userdata collectable
        collectgarbage("collect")

        print("hatch = ", hatch)
        hatch:access() -- This triggers the use-after-free
    "#, None).unwrap();
}
