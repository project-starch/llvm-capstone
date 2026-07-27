// GHSA-f56g-chqp-22m9 -- libpulse-binding `proplist::Iterator` use-after-free.
//
// A Rust->C temporal borrow violation: the iterator holds a raw copy of the C
// `pa_proplist*` with no lifetime tie to the `Proplist` that owns it, so the
// owner can be destroyed while the iterator is still live.
//
// THE UNSOUNDNESS (libpulse-binding 2.4.0, pulse-binding/src/proplist.rs)
//
//     pub struct Iterator {                  // <-- no lifetime parameter at all
//         ptr: *const ProplistInternal,      //     raw copy of the C pointer
//         state: *mut c_void,
//     }
//
//     pub fn iter(&self) -> Iterator {       // borrows &self ...
//         Iterator::new(self.ptr)            // ... but the result is unbound
//     }
//
//     impl IntoIterator for Proplist {
//         fn into_iter(self) -> Self::IntoIter {
//             self.iter()                    // `self` is DROPPED here
//         }
//     }
//
// `into_iter` takes `self` by value, copies the raw pointer out via `iter()`, and
// then drops `self` when it returns -- running `Proplist`'s Drop, which calls
// `pa_proplist_free()`. The Iterator it hands back is already dangling. Upstream's
// own advisory note says this was "trivially" reachable "simply by using the
// into_iter() function".
//
// The 2.5.0 fix adds `Iterator<'a>` with `PhantomData<&'a ProplistInner>` so the
// borrow checker ties the iterator to the list, and reworks `into_iter` to
// transfer ownership of the C object into the iterator instead of freeing it.
//
// No PulseAudio server is required: `pa_proplist` is a standalone data structure.

use libpulse_binding::proplist::Proplist;

fn main() {
    let mut pl = Proplist::new().expect("pa_proplist_new failed");
    pl.sets("application.name", "xlang-row3").unwrap();
    pl.sets("application.id", "org.example.xlang").unwrap();

    println!("proplist populated, len = {}", pl.len());

    // `into_iter()` consumes the Proplist. Inside it, the raw pa_proplist* is
    // copied into the Iterator and then `self` is dropped -> pa_proplist_free().
    let mut it = pl.into_iter();
    println!("iterator obtained; the C pa_proplist has already been freed");

    // THE STALE USE. next() calls pa_proplist_iterate() on the freed C object.
    while let Some(key) = it.next() {
        println!("key: {}", key);
    }
}
