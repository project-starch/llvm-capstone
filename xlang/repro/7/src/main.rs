// RUSTSEC-2022-0070 / GHSA-969w-q74q-9j8v — secp256k1 0.24.0..0.24.2
// preallocated_gen_new had incorrect lifetime bounds, so the returned context
// could outlive the buffer it borrows: a use-after-free reachable from entirely
// safe code, across the Rust <-> C (libsecp256k1) boundary.
use secp256k1::{AllPreallocated, Secp256k1, SecretKey, Message};

fn escaped() -> Secp256k1<AllPreallocated<'static>> {
    let size = Secp256k1::preallocate_size();
    let mut buf = vec![secp256k1::ffi::types::AlignedType::zeroed(); size];
    Secp256k1::preallocated_gen_new(&mut buf).unwrap()
    // `buf` is dropped here; the returned context still points into it.
}

fn main() {
    let ctx = escaped();
    let msg = Message::from_slice(&[1u8; 32]).unwrap();
    let sk = SecretKey::from_slice(&[2u8; 32]).unwrap();
    // PublicKey::from_secret_key uses the PREALLOCATED generator tables, i.e.
    // it actually reads the freed context. sign_ecdsa does not.
    let pk = secp256k1::PublicKey::from_secret_key(&ctx, &sk);
    let sig = ctx.sign_ecdsa(&msg, &sk);
    println!("NO-FAULT: pk={} sig={}", pk, sig);
}
