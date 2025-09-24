//==- BuiltinsCapstone.td - Capstone Builtin function database ---*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Capstone-specific builtin function database.
//
//===----------------------------------------------------------------------===//

include "clang/Basic/BuiltinsBase.td"

class CapstoneBuiltin<string prototype, string features = ""> : TargetBuiltin {
  let Spellings = ["__builtin_capstone_" # NAME];
  let Prototype = prototype;
  let Features = features;
}

let Attributes = [NoThrow, Const] in {
//===----------------------------------------------------------------------===//
// Zbb extension.
//===----------------------------------------------------------------------===//
def orc_b_32 : CapstoneBuiltin<"unsigned int(unsigned int)", "zbb">;
def orc_b_64 : CapstoneBuiltin<"uint64_t(uint64_t)", "zbb,64bit">;
def clz_32 : CapstoneBuiltin<"unsigned int(unsigned int)", "zbb|xtheadbb">;
def clz_64 : CapstoneBuiltin<"unsigned int(uint64_t)", "zbb|xtheadbb,64bit">;
def ctz_32 : CapstoneBuiltin<"unsigned int(unsigned int)", "zbb">;
def ctz_64 : CapstoneBuiltin<"unsigned int(uint64_t)", "zbb,64bit">;

//===----------------------------------------------------------------------===//
// Zbc or Zbkc extension.
//===----------------------------------------------------------------------===//
def clmul_32 : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int)", "zbc|zbkc">;
def clmul_64 : CapstoneBuiltin<"uint64_t(uint64_t, uint64_t)", "zbc|zbkc,64bit">;
def clmulh_32 : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int)", "zbc|zbkc,32bit">;
def clmulh_64 : CapstoneBuiltin<"uint64_t(uint64_t, uint64_t)", "zbc|zbkc,64bit">;
def clmulr_32 : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int)", "zbc,32bit">;
def clmulr_64 : CapstoneBuiltin<"uint64_t(uint64_t, uint64_t)", "zbc,64bit">;

//===----------------------------------------------------------------------===//
// Zbkx extension.
//===----------------------------------------------------------------------===//
let Features = "zbkx,32bit" in {
def xperm4_32 : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int)">;
def xperm8_32 : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int)">;
} // Features = "zbkx,32bit"

let Features = "zbkx,64bit" in {
def xperm4_64 : CapstoneBuiltin<"uint64_t(uint64_t, uint64_t)">;
def xperm8_64 : CapstoneBuiltin<"uint64_t(uint64_t, uint64_t)">;
} // Features = "zbkx,64bit"

//===----------------------------------------------------------------------===//
// Zbkb extension.
//===----------------------------------------------------------------------===//
def brev8_32 : CapstoneBuiltin<"unsigned int(unsigned int)", "zbkb">;
def brev8_64 : CapstoneBuiltin<"uint64_t(uint64_t)", "zbkb,64bit">;
def zip_32 : CapstoneBuiltin<"unsigned int(unsigned int)", "zbkb,32bit">;
def unzip_32 : CapstoneBuiltin<"unsigned int(unsigned int)", "zbkb,32bit">;

//===----------------------------------------------------------------------===//
// Zknd extension.
//===----------------------------------------------------------------------===//
let Features = "zknd,32bit" in {
def aes32dsi : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)">;
def aes32dsmi : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)">;
} // Features = "zknd,32bit"

let Features = "zknd,64bit" in {
def aes64ds : CapstoneBuiltin<"uint64_t(uint64_t, uint64_t)">;
def aes64dsm : CapstoneBuiltin<"uint64_t(uint64_t, uint64_t)">;
def aes64im : CapstoneBuiltin<"uint64_t(uint64_t)">;
} // Features = "zknd,64bit"

//===----------------------------------------------------------------------===//
// Zknd & Zkne extension.
//===----------------------------------------------------------------------===//
let Features = "zknd|zkne,64bit" in {
def aes64ks1i : CapstoneBuiltin<"uint64_t(uint64_t, _Constant unsigned int)">;
def aes64ks2 : CapstoneBuiltin<"uint64_t(uint64_t, uint64_t)">;
} // Features = "zknd|zkne,64bit"

//===----------------------------------------------------------------------===//
// Zkne extension.
//===----------------------------------------------------------------------===//
let Features = "zkne,32bit" in {
def aes32esi : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)">;
def aes32esmi : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)">;
} // Features = "zkne,32bit"

let Features = "zkne,64bit" in {
def aes64es : CapstoneBuiltin<"uint64_t(uint64_t, uint64_t)">;
def aes64esm : CapstoneBuiltin<"uint64_t(uint64_t, uint64_t)">;
} // Features = "zkne,64bit"

//===----------------------------------------------------------------------===//
// Zknh extension.
//===----------------------------------------------------------------------===//
let Features = "zknh" in {
def sha256sig0 : CapstoneBuiltin<"unsigned int(unsigned int)">;
def sha256sig1 : CapstoneBuiltin<"unsigned int(unsigned int)">;
def sha256sum0 : CapstoneBuiltin<"unsigned int(unsigned int)">;
def sha256sum1 : CapstoneBuiltin<"unsigned int(unsigned int)">;
} // Features = "zknh"

let Features = "zknh,32bit" in {
def sha512sig0h : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int)">;
def sha512sig0l : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int)">;
def sha512sig1h : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int)">;
def sha512sig1l : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int)">;
def sha512sum0r : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int)">;
def sha512sum1r : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int)">;
} // Features = "zknh,32bit"

let Features = "zknh,64bit" in {
def sha512sig0 : CapstoneBuiltin<"uint64_t(uint64_t)">;
def sha512sig1 : CapstoneBuiltin<"uint64_t(uint64_t)">;
def sha512sum0 : CapstoneBuiltin<"uint64_t(uint64_t)">;
def sha512sum1 : CapstoneBuiltin<"uint64_t(uint64_t)">;
} // Features = "zknh,64bit"

//===----------------------------------------------------------------------===//
// Zksed extension.
//===----------------------------------------------------------------------===//
let Features = "zksed" in {
def sm4ed : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int )">;
def sm4ks : CapstoneBuiltin<"unsigned int(unsigned int, unsigned int, _Constant unsigned int)">;
} // Features = "zksed"

//===----------------------------------------------------------------------===//
// Zksh extension.
//===----------------------------------------------------------------------===//
let Features = "zksh" in {
def sm3p0 : CapstoneBuiltin<"unsigned int(unsigned int)">;
def sm3p1 : CapstoneBuiltin<"unsigned int(unsigned int)">;
} // Features = "zksh"

} // Attributes = [Const, NoThrow]

//===----------------------------------------------------------------------===//
// Zihintntl extension.
//===----------------------------------------------------------------------===//
let Features = "zihintntl", Attributes = [CustomTypeChecking] in {
def ntl_load : CapstoneBuiltin<"void(...)">;
def ntl_store : CapstoneBuiltin<"void(...)">;
} // Features = "zihintntl", Attributes = [CustomTypeChecking]

//===----------------------------------------------------------------------===//
// Zihintpause extension.
//===----------------------------------------------------------------------===//
let Features = "zihintpause", Attributes = [NoThrow] in
def pause : CapstoneBuiltin<"void()">;

//===----------------------------------------------------------------------===//
// XCV extensions.
//===----------------------------------------------------------------------===//
include "clang/Basic/BuiltinsCapstoneXCV.td"

//===----------------------------------------------------------------------===//
// XAndes extensions.
//===----------------------------------------------------------------------===//
include "clang/Basic/BuiltinsCapstoneXAndes.td"
