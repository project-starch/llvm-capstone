//===-- CapstoneRepresentable.h - Exactly representable bounds -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The smallest length >= Size, and the base alignment, at which bounds
// [Base, Base + Length) survive the compact bounds encoding exactly (vp2's
// compress_bounds; capstone-qemu target/riscv/cap_compress.c cap_compress):
//
//  * Length < 4096 is exact at any base (iE = 0).
//  * Otherwise E = msb(Length) - 12 and the granule is g = 2^(E+3): bounds are
//    exact iff Base and Length are both multiples of g. Rounding Length up to
//    g can reach the next power of two, whose granule is twice as large, so the
//    length is iterated to a fixpoint and the alignment is taken from it.
//
// Dependency-free on purpose: tests/capstone-codec-padding compiles this header
// directly against QEMU's codec and the RTL's bounds model.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CAPSTONE_MCTARGETDESC_CAPSTONEREPRESENTABLE_H
#define LLVM_LIB_TARGET_CAPSTONE_MCTARGETDESC_CAPSTONEREPRESENTABLE_H

#include <cstdint>

namespace llvm {
namespace CapstoneRepresentable {

// The granule a region of length Length is rounded to (1 below 4096).
inline uint64_t granule(uint64_t Length) {
  if (Length < 4096)
    return 1;
  unsigned Msb = 63 - static_cast<unsigned>(__builtin_clzll(Length));
  return uint64_t(1) << (Msb - 9);
}

// The smallest exactly representable length >= Size (Size itself below 4096).
inline uint64_t length(uint64_t Size) {
  uint64_t Length = Size;
  for (;;) {
    uint64_t G = granule(Length);
    uint64_t Rounded = (Size + G - 1) & ~(G - 1);
    if (granule(Rounded) == G)
      return Rounded;
    Length = Rounded;
  }
}

// The base alignment that makes [Base, Base + length(Size)) exact.
inline uint64_t alignment(uint64_t Size) { return granule(length(Size)); }

} // namespace CapstoneRepresentable
} // namespace llvm

#endif
