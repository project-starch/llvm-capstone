//===-- CapstoneFeatures.td - Capstone Features and Extensions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Capstone subtarget features and instruction predicates.
//===----------------------------------------------------------------------===//

// Subclass of SubtargetFeature to be used when the feature is also a Capstone
// extension. Extensions have a version and may be experimental.
// NOTE: The extension name must start with
//       - "FeatureStdExt" for standard extensions
//       - "FeatureVendor" for vendor-specific extensions
//
// major     - Major version of extension.
// minor     - Minor version of extension.
// desc      - Description of extension.
// implies   - Extensions or features implied by this extension.
// fieldname - name of field to create in CapstoneSubtarget. By default replaces
//             uses the record name by replacing Feature with Has.
// value     - Value to assign to the field in CapstoneSubtarget when this
//             extension is enabled. Usually "true", but can be changed.
// IsExperimental
//           - Indicates whether the extension is experimental
class CapstoneExtension<int major, int minor, string desc,
                     list<SubtargetFeature> implies = [],
                     string fieldname = !subst("Feature", "Has", NAME),
                     string value = "true", bit IsExperimental = false>
    : SubtargetFeature<"", fieldname, value, "", implies> {
  defvar extName = !subst("FeatureVendor", "", !subst("FeatureStdExt", "", NAME));

  // Name - Name of the extension in lower case.
  let Name = !if(IsExperimental, "experimental-", "") # !tolower(extName);

  // Desc - Description of extension with its name.
  let Desc = "'" # extName # "' (" # desc # ")";

  // MajorVersion - The major version for this extension.
  int MajorVersion = major;

  // MinorVersion - The minor version for this extension.
  int MinorVersion = minor;

  // Experimental - Does extension require -menable-experimental-extensions.
  bit Experimental = false;
}

// The groupID/bitmask of CapstoneExtension is used to retrieve a specific bit value
// from __capstone_feature_bits based on the groupID and bitmask.
// groupID - groupID of extension
// bitPos  - bit position of extension bitmask
class CapstoneExtensionBitmask<bits<3> groupID, int bitPos> {
    int GroupID = groupID;
    int BitPos = bitPos;
}

// Version of CapstoneExtension to be used for Experimental extensions. This
// sets the Experimental flag and prepends experimental- to the -mattr name.
class CapstoneExperimentalExtension<int major, int minor, string desc,
                                 list<CapstoneExtension> implies = [],
                                 string fieldname = !subst("Feature", "Has", NAME),
                                 string value = "true">
    : CapstoneExtension<major, minor, desc, implies, fieldname, value, true> {
  let Experimental = true;
}

// Integer Extensions

def FeatureStdExtI
    : CapstoneExtension<2, 1, "Base Integer Instruction Set">,
      CapstoneExtensionBitmask<0, 8>;

def FeatureStdExtE
    : CapstoneExtension<2, 0, "Embedded Instruction Set with 16 GPRs">,
      CapstoneExtensionBitmask<0, 4>;

def FeatureStdExtZibi
    : CapstoneExperimentalExtension<0, 1, "Branch with Immediate">;
def HasStdExtZibi : Predicate<"Subtarget->hasStdExtZibi()">,
                    AssemblerPredicate<(all_of FeatureStdExtZibi),
                                       "'Zibi' (Branch with Immediate)">;

def FeatureStdExtZic64b
    : CapstoneExtension<1, 0, "Cache Block Size Is 64 Bytes">;

def FeatureStdExtZicbom
    : CapstoneExtension<1, 0, "Cache-Block Management Instructions">;
def HasStdExtZicbom : Predicate<"Subtarget->hasStdExtZicbom()">,
                      AssemblerPredicate<(all_of FeatureStdExtZicbom),
                          "'Zicbom' (Cache-Block Management Instructions)">;

def FeatureStdExtZicbop
    : CapstoneExtension<1, 0, "Cache-Block Prefetch Instructions">;
def HasStdExtZicbop : Predicate<"Subtarget->hasStdExtZicbop()">,
                      AssemblerPredicate<(all_of FeatureStdExtZicbop),
                          "'Zicbop' (Cache-Block Prefetch Instructions)">;

def FeatureStdExtZicboz
    : CapstoneExtension<1, 0, "Cache-Block Zero Instructions">,
      CapstoneExtensionBitmask<0, 37>;
def HasStdExtZicboz : Predicate<"Subtarget->hasStdExtZicboz()">,
                      AssemblerPredicate<(all_of FeatureStdExtZicboz),
                          "'Zicboz' (Cache-Block Zero Instructions)">;

def FeatureStdExtZiccamoa
    : CapstoneExtension<1, 0, "Main Memory Supports All Atomics in A">;

def FeatureStdExtZiccamoc
    : CapstoneExtension<1, 0, "Main Memory Supports Atomics in Zacas">;

def FeatureStdExtZiccif
    : CapstoneExtension<1, 0,
                     "Main Memory Supports Instruction Fetch with Atomicity Requirement">;

def FeatureStdExtZicclsm
    : CapstoneExtension<1, 0, "Main Memory Supports Misaligned Loads/Stores">;

def FeatureStdExtZiccrse
    : CapstoneExtension<1, 0,
                     "Main Memory Supports Forward Progress on LR/SC Sequences">;

def FeatureStdExtZicsr
    : CapstoneExtension<2, 0, "CSRs">;
def HasStdExtZicsr : Predicate<"Subtarget->hasStdExtZicsr()">,
                     AssemblerPredicate<(all_of FeatureStdExtZicsr),
                                        "'Zicsr' (CSRs)">;

def FeatureStdExtZicntr
    : CapstoneExtension<2, 0, "Base Counters and Timers",
                     [FeatureStdExtZicsr]>;

def FeatureStdExtZicond
    : CapstoneExtension<1, 0, "Integer Conditional Operations">,
      CapstoneExtensionBitmask<0, 38>;
def HasStdExtZicond : Predicate<"Subtarget->hasStdExtZicond()">,
                      AssemblerPredicate<(all_of FeatureStdExtZicond),
                          "(Integer Conditional Operations)">;

def FeatureStdExtZifencei
    : CapstoneExtension<2, 0, "fence.i">;
def HasStdExtZifencei : Predicate<"Subtarget->hasStdExtZifencei()">,
                        AssemblerPredicate<(all_of FeatureStdExtZifencei),
                                           "'Zifencei' (fence.i)">;

def FeatureStdExtZihintpause
    : CapstoneExtension<2, 0, "Pause Hint">,
      CapstoneExtensionBitmask<0, 40>;
def HasStdExtZihintpause : Predicate<"Subtarget->hasStdExtZihintpause()">,
                           AssemblerPredicate<(all_of FeatureStdExtZihintpause),
                                              "'Zihintpause' (Pause Hint)">;

def FeatureStdExtZihintntl
    : CapstoneExtension<1, 0, "Non-Temporal Locality Hints">,
      CapstoneExtensionBitmask<0, 39>;
def HasStdExtZihintntl : Predicate<"Subtarget->hasStdExtZihintntl()">,
                         AssemblerPredicate<(all_of FeatureStdExtZihintntl),
                             "'Zihintntl' (Non-Temporal Locality Hints)">;

def FeatureStdExtZihpm
    : CapstoneExtension<2, 0, "Hardware Performance Counters",
                     [FeatureStdExtZicsr]>;

def FeatureStdExtZimop : CapstoneExtension<1, 0, "May-Be-Operations">,
                         CapstoneExtensionBitmask<1, 1>;
def HasStdExtZimop : Predicate<"Subtarget->hasStdExtZimop()">,
                     AssemblerPredicate<(all_of FeatureStdExtZimop),
                                        "'Zimop' (May-Be-Operations)">;

def FeatureStdExtZicfilp
    : CapstoneExperimentalExtension<1, 0, "Landing pad",
                                 [FeatureStdExtZicsr]>;
def HasStdExtZicfilp : Predicate<"Subtarget->hasStdExtZicfilp()">,
                       AssemblerPredicate<(all_of FeatureStdExtZicfilp),
                                          "'Zicfilp' (Landing pad)">;
def NoStdExtZicfilp : Predicate<"!Subtarget->hasStdExtZicfilp()">,
                      AssemblerPredicate<(all_of (not FeatureStdExtZicfilp))>;

def FeatureStdExtZicfiss
    : CapstoneExperimentalExtension<1, 0, "Shadow stack",
                                 [FeatureStdExtZicsr, FeatureStdExtZimop]>;
def HasStdExtZicfiss : Predicate<"Subtarget->hasStdExtZicfiss()">,
                       AssemblerPredicate<(all_of FeatureStdExtZicfiss),
                                          "'Zicfiss' (Shadow stack)">;

def FeatureStdExtZilsd
    : CapstoneExtension<1, 0,
                     "Load/Store Pair Instructions">,
      CapstoneExtensionBitmask<1, 8>;
def HasStdExtZilsd : Predicate<"Subtarget->hasStdExtZilsd()">,
                       AssemblerPredicate<(all_of FeatureStdExtZilsd),
                                          "'Zilsd' (Load/Store pair instructions)">;

// Multiply Extensions

def FeatureStdExtZmmul
    : CapstoneExtension<1, 0, "Integer Multiplication">;
def HasStdExtZmmul : Predicate<"Subtarget->hasStdExtZmmul()">,
                     AssemblerPredicate<(all_of FeatureStdExtZmmul),
                     "'Zmmul' (Integer Multiplication)">;

def FeatureStdExtM
    : CapstoneExtension<2, 0, "Integer Multiplication and Division",
                     [FeatureStdExtZmmul]>,
      CapstoneExtensionBitmask<0, 12>;
def HasStdExtM : Predicate<"Subtarget->hasStdExtM()">,
                 AssemblerPredicate<(all_of FeatureStdExtM),
                     "'M' (Integer Multiplication and Division)">;

// Atomic Extensions

def FeatureStdExtZaamo
    : CapstoneExtension<1, 0, "Atomic Memory Operations">;
def HasStdExtZaamo
    : Predicate<"Subtarget->hasStdExtZaamo()">,
      AssemblerPredicate<(any_of FeatureStdExtZaamo),
                         "'Zaamo' (Atomic Memory Operations)">;

def FeatureStdExtZalrsc
    : CapstoneExtension<1, 0, "Load-Reserved/Store-Conditional">;
def HasStdExtZalrsc
    : Predicate<"Subtarget->hasStdExtZalrsc()">,
      AssemblerPredicate<(any_of FeatureStdExtZalrsc),
                         "'Zalrsc' (Load-Reserved/Store-Conditional)">;

def FeatureStdExtA
    : CapstoneExtension<2, 1, "Atomic Instructions",
                     [FeatureStdExtZaamo, FeatureStdExtZalrsc]>,
      CapstoneExtensionBitmask<0, 0>;
def HasStdExtA : Predicate<"Subtarget->hasStdExtA()">,
                 AssemblerPredicate<(all_of FeatureStdExtA),
                                    "'A' (Atomic Instructions)">;

def FeatureStdExtZtso
    : CapstoneExtension<1, 0, "Memory Model - Total Store Order">,
      CapstoneExtensionBitmask<0, 47>;
def HasStdExtZtso : Predicate<"Subtarget->hasStdExtZtso()">,
                    AssemblerPredicate<(all_of FeatureStdExtZtso),
                        "'Ztso' (Memory Model - Total Store Order)">;
def NoStdExtZtso : Predicate<"!Subtarget->hasStdExtZtso()">;

def FeatureStdExtZa64rs
    : CapstoneExtension<1, 0, "Reservation Set Size of at Most 64 Bytes">;

def FeatureStdExtZa128rs
    : CapstoneExtension<1, 0, "Reservation Set Size of at Most 128 Bytes">;

def FeatureStdExtZabha
    : CapstoneExtension<1, 0, "Byte and Halfword Atomic Memory Operations",
                     [FeatureStdExtZaamo]>;
def HasStdExtZabha : Predicate<"Subtarget->hasStdExtZabha()">,
                     AssemblerPredicate<(all_of FeatureStdExtZabha),
                         "'Zabha' (Byte and Halfword Atomic Memory Operations)">;

def FeatureStdExtZacas
    : CapstoneExtension<1, 0, "Atomic Compare-And-Swap Instructions",
                     [FeatureStdExtZaamo]>,
      CapstoneExtensionBitmask<0, 26>;
def HasStdExtZacas : Predicate<"Subtarget->hasStdExtZacas()">,
                     AssemblerPredicate<(all_of FeatureStdExtZacas),
                         "'Zacas' (Atomic Compare-And-Swap Instructions)">;
def NoStdExtZacas : Predicate<"!Subtarget->hasStdExtZacas()">;

def FeatureStdExtZalasr
    : CapstoneExperimentalExtension<0, 1, "Load-Acquire and Store-Release Instructions">;
def HasStdExtZalasr : Predicate<"Subtarget->hasStdExtZalasr()">,
                      AssemblerPredicate<(all_of FeatureStdExtZalasr),
                          "'Zalasr' (Load-Acquire and Store-Release Instructions)">;

def FeatureStdExtZama16b
    : CapstoneExtension<1, 0, "Atomic 16-byte misaligned loads, stores and AMOs">;

def FeatureStdExtZawrs : CapstoneExtension<1, 0, "Wait on Reservation Set">,
                         CapstoneExtensionBitmask<1, 7>;
def HasStdExtZawrs : Predicate<"Subtarget->hasStdExtZawrs()">,
                     AssemblerPredicate<(all_of FeatureStdExtZawrs),
                                        "'Zawrs' (Wait on Reservation Set)">;

// Floating Point Extensions

def FeatureStdExtF
    : CapstoneExtension<2, 2, "Single-Precision Floating-Point",
                     [FeatureStdExtZicsr]>,
      CapstoneExtensionBitmask<0, 5>;
def HasStdExtF : Predicate<"Subtarget->hasStdExtF()">,
                 AssemblerPredicate<(all_of FeatureStdExtF),
                                    "'F' (Single-Precision Floating-Point)">;

def FeatureStdExtD
    : CapstoneExtension<2, 2, "Double-Precision Floating-Point",
                     [FeatureStdExtF]>,
      CapstoneExtensionBitmask<0, 3>;
def HasStdExtD : Predicate<"Subtarget->hasStdExtD()">,
                 AssemblerPredicate<(all_of FeatureStdExtD),
                                    "'D' (Double-Precision Floating-Point)">;

def FeatureStdExtQ
    : CapstoneExtension<2, 2, "Quad-Precision Floating-Point", [FeatureStdExtD]>,
      CapstoneExtensionBitmask<0, 16>;
def HasStdExtQ : Predicate<"Subtarget->hasStdExtQ()">,
                 AssemblerPredicate<(all_of FeatureStdExtQ),
                                    "'Q' (Quad-Precision Floating-Point)">;

def FeatureStdExtZfhmin
    : CapstoneExtension<1, 0, "Half-Precision Floating-Point Minimal",
                     [FeatureStdExtF]>,
      CapstoneExtensionBitmask<0, 36>;
def HasStdExtZfhmin : Predicate<"Subtarget->hasStdExtZfhmin()">,
                      AssemblerPredicate<(all_of FeatureStdExtZfhmin),
                          "'Zfh' (Half-Precision Floating-Point) or "
                          "'Zfhmin' (Half-Precision Floating-Point Minimal)">;

def FeatureStdExtZfh
    : CapstoneExtension<1, 0, "Half-Precision Floating-Point",
                     [FeatureStdExtZfhmin]>,
      CapstoneExtensionBitmask<0, 35>;
def HasStdExtZfh : Predicate<"Subtarget->hasStdExtZfh()">,
                   AssemblerPredicate<(all_of FeatureStdExtZfh),
                       "'Zfh' (Half-Precision Floating-Point)">;
def NoStdExtZfh : Predicate<"!Subtarget->hasStdExtZfh()">;

def FeatureStdExtZfbfmin
    : CapstoneExtension<1, 0, "Scalar BF16 Converts", [FeatureStdExtF]>;
def HasStdExtZfbfmin : Predicate<"Subtarget->hasStdExtZfbfmin()">,
                       AssemblerPredicate<(all_of FeatureStdExtZfbfmin),
                                          "'Zfbfmin' (Scalar BF16 Converts)">;

def HasHalfFPLoadStoreMove
    : Predicate<"Subtarget->hasHalfFPLoadStoreMove()">,
      AssemblerPredicate<(any_of FeatureStdExtZfh, FeatureStdExtZfhmin,
                                 FeatureStdExtZfbfmin),
                         "'Zfh' (Half-Precision Floating-Point) or "
                                    "'Zfhmin' (Half-Precision Floating-Point Minimal) or "
                                    "'Zfbfmin' (Scalar BF16 Converts)">;

def FeatureStdExtZfa
    : CapstoneExtension<1, 0, "Additional Floating-Point", [FeatureStdExtF]>,
      CapstoneExtensionBitmask<0, 34>;
def HasStdExtZfa : Predicate<"Subtarget->hasStdExtZfa()">,
                   AssemblerPredicate<(all_of FeatureStdExtZfa),
                                      "'Zfa' (Additional Floating-Point)">;
def NoStdExtZfa : Predicate<"!Subtarget->hasStdExtZfa()">;

def FeatureStdExtZfinx
    : CapstoneExtension<1, 0, "Float in Integer", [FeatureStdExtZicsr]>;
def HasStdExtZfinx : Predicate<"Subtarget->hasStdExtZfinx()">,
                     AssemblerPredicate<(all_of FeatureStdExtZfinx),
                                        "'Zfinx' (Float in Integer)">;
def HasStdExtFOrZfinx : Predicate<"Subtarget->hasStdExtFOrZfinx()">,
                        AssemblerPredicate<(any_of FeatureStdExtF,
                                                   FeatureStdExtZfinx),
                            "'F' (Single-Precision Floating-Point) or "
                            "'Zfinx' (Float in Integer)">;

def FeatureStdExtZdinx
    : CapstoneExtension<1, 0, "Double in Integer", [FeatureStdExtZfinx]>;
def HasStdExtZdinx : Predicate<"Subtarget->hasStdExtZdinx()">,
                     AssemblerPredicate<(all_of FeatureStdExtZdinx),
                                        "'Zdinx' (Double in Integer)">;

def FeatureStdExtZhinxmin
    : CapstoneExtension<1, 0, "Half Float in Integer Minimal",
                     [FeatureStdExtZfinx]>;
def HasStdExtZhinxmin : Predicate<"Subtarget->hasStdExtZhinxmin()">,
                        AssemblerPredicate<(all_of FeatureStdExtZhinxmin),
                            "'Zhinx' (Half Float in Integer) or "
                            "'Zhinxmin' (Half Float in Integer Minimal)">;

def FeatureStdExtZhinx
    : CapstoneExtension<1, 0, "Half Float in Integer", [FeatureStdExtZhinxmin]>;
def HasStdExtZhinx : Predicate<"Subtarget->hasStdExtZhinx()">,
                     AssemblerPredicate<(all_of FeatureStdExtZhinx),
                                        "'Zhinx' (Half Float in Integer)">;
def NoStdExtZhinx : Predicate<"!Subtarget->hasStdExtZhinx()">;

// Compressed Extensions
def FeatureStdExtZca
    : CapstoneExtension<1, 0,
                     "part of the C extension, excluding compressed "
                     "floating point loads/stores">,
      CapstoneExtensionBitmask<1, 2>;
// FIXME: Update this message - Zca always implies C.
def HasStdExtZca
    : Predicate<"Subtarget->hasStdExtZca()">,
      AssemblerPredicate<(any_of FeatureStdExtZca),
                         "'C' (Compressed Instructions) or "
                         "'Zca' (part of the C extension, excluding "
                         "compressed floating point loads/stores)">;

def FeatureStdExtC
    : CapstoneExtension<2, 0, "Compressed Instructions", [FeatureStdExtZca]>,
      CapstoneExtensionBitmask<0, 2>;
def HasStdExtC : Predicate<"Subtarget->hasStdExtC()">,
                 AssemblerPredicate<(all_of FeatureStdExtC),
                                    "'C' (Compressed Instructions)">;

def FeatureStdExtZcb
    : CapstoneExtension<1, 0, "Compressed basic bit manipulation instructions",
                     [FeatureStdExtZca]>,
      CapstoneExtensionBitmask<1, 3>;
def HasStdExtZcb : Predicate<"Subtarget->hasStdExtZcb()">,
                   AssemblerPredicate<(all_of FeatureStdExtZcb),
                       "'Zcb' (Compressed basic bit manipulation instructions)">;

def FeatureStdExtZcd
    : CapstoneExtension<1, 0,
                     "Compressed Double-Precision Floating-Point Instructions",
                     [FeatureStdExtD, FeatureStdExtZca]>,
      CapstoneExtensionBitmask<1, 4>;

def HasStdExtCOrZcd
    : Predicate<"Subtarget->hasStdExtCOrZcd()">,
      AssemblerPredicate<(any_of FeatureStdExtC, FeatureStdExtZcd),
                         "'C' (Compressed Instructions) or "
                         "'Zcd' (Compressed Double-Precision Floating-Point Instructions)">;

def FeatureStdExtZcf
    : CapstoneExtension<1, 0,
                     "Compressed Single-Precision Floating-Point Instructions",
                     [FeatureStdExtF, FeatureStdExtZca]>,
      CapstoneExtensionBitmask<1, 5>;

def FeatureStdExtZclsd
    : CapstoneExtension<1, 0,
                     "Compressed Load/Store Pair Instructions",
                     [FeatureStdExtZilsd, FeatureStdExtZca]>,
      CapstoneExtensionBitmask<1, 9>;
def HasStdExtZclsd : Predicate<"Subtarget->hasStdExtZclsd()">,
                    AssemblerPredicate<(all_of FeatureStdExtZclsd),
                        "'Zclsd' (Compressed Load/Store pair instructions)">;

def FeatureStdExtZcmp
    : CapstoneExtension<1, 0,
                     "sequenced instructions for code-size reduction",
                     [FeatureStdExtZca]>,
      CapstoneExtensionBitmask<1, 10>;
def HasStdExtZcmp : Predicate<"Subtarget->hasStdExtZcmp() && !Subtarget->hasStdExtC()">,
                    AssemblerPredicate<(all_of FeatureStdExtZcmp),
                        "'Zcmp' (sequenced instructions for code-size reduction)">;

def FeatureStdExtZcmt
    : CapstoneExtension<1, 0,
                     "table jump instructions for code-size reduction",
                     [FeatureStdExtZca, FeatureStdExtZicsr]>;
def HasStdExtZcmt : Predicate<"Subtarget->hasStdExtZcmt()">,
                           AssemblerPredicate<(all_of FeatureStdExtZcmt),
                           "'Zcmt' (table jump instructions for code-size reduction)">;

def FeatureStdExtZce
    : CapstoneExtension<1, 0,
                     "Compressed extensions for microcontrollers",
                     [FeatureStdExtZca, FeatureStdExtZcb, FeatureStdExtZcmp,
                      FeatureStdExtZcmt]>;

def HasStdExtCOrZcfOrZce
    : Predicate<"Subtarget->hasStdExtC() || Subtarget->hasStdExtZcf() ||"
                "Subtarget->hasStdExtZce()">,
      AssemblerPredicate<(any_of FeatureStdExtC, FeatureStdExtZcf,
                                 FeatureStdExtZce),
                         "'C' (Compressed Instructions) or "
                         "'Zcf' (Compressed Single-Precision Floating-Point Instructions)">;

def FeatureStdExtZcmop
    : CapstoneExtension<1, 0, "Compressed May-Be-Operations",
                     [FeatureStdExtZca]>,
      CapstoneExtensionBitmask<1, 6>;
def HasStdExtZcmop : Predicate<"Subtarget->hasStdExtZcmop()">,
                     AssemblerPredicate<(all_of FeatureStdExtZcmop),
                         "'Zcmop' (Compressed May-Be-Operations)">;

// Bitmanip Extensions

def FeatureStdExtZba
    : CapstoneExtension<1, 0, "Address Generation Instructions">,
      CapstoneExtensionBitmask<0, 27>;
def HasStdExtZba : Predicate<"Subtarget->hasStdExtZba()">,
                   AssemblerPredicate<(all_of FeatureStdExtZba),
                                      "'Zba' (Address Generation Instructions)">;
def NoStdExtZba : Predicate<"!Subtarget->hasStdExtZba()">;

def FeatureStdExtZbb
    : CapstoneExtension<1, 0, "Basic Bit-Manipulation">,
      CapstoneExtensionBitmask<0, 28>;
def HasStdExtZbb : Predicate<"Subtarget->hasStdExtZbb()">,
                   AssemblerPredicate<(all_of FeatureStdExtZbb),
                                      "'Zbb' (Basic Bit-Manipulation)">;
def NoStdExtZbb : Predicate<"!Subtarget->hasStdExtZbb()">,
                  AssemblerPredicate<(all_of (not FeatureStdExtZbb))>;

def FeatureStdExtZbc
    : CapstoneExtension<1, 0, "Carry-Less Multiplication">,
      CapstoneExtensionBitmask<0, 29>;
def HasStdExtZbc : Predicate<"Subtarget->hasStdExtZbc()">,
                   AssemblerPredicate<(all_of FeatureStdExtZbc),
                                      "'Zbc' (Carry-Less Multiplication)">;

def FeatureStdExtZbs
    : CapstoneExtension<1, 0, "Single-Bit Instructions">,
      CapstoneExtensionBitmask<0, 33>;
def HasStdExtZbs : Predicate<"Subtarget->hasStdExtZbs()">,
                   AssemblerPredicate<(all_of FeatureStdExtZbs),
                                      "'Zbs' (Single-Bit Instructions)">;

// Bitmanip Extensions for Cryptography Extensions

def FeatureStdExtB
    : CapstoneExtension<1, 0, "the collection of the Zba, Zbb, Zbs extensions",
                     [FeatureStdExtZba, FeatureStdExtZbb, FeatureStdExtZbs]>,
      CapstoneExtensionBitmask<0, 1>;

def FeatureStdExtZbkb
    : CapstoneExtension<1, 0, "Bitmanip instructions for Cryptography">,
      CapstoneExtensionBitmask<0, 30>;
def HasStdExtZbkb : Predicate<"Subtarget->hasStdExtZbkb()">,
                    AssemblerPredicate<(all_of FeatureStdExtZbkb),
                        "'Zbkb' (Bitmanip instructions for Cryptography)">;
def NoStdExtZbkb : Predicate<"!Subtarget->hasStdExtZbkb()">,
                   AssemblerPredicate<(all_of (not FeatureStdExtZbkb))>;

def FeatureStdExtZbkx
    : CapstoneExtension<1, 0, "Crossbar permutation instructions">,
      CapstoneExtensionBitmask<0, 32>;
def HasStdExtZbkx : Predicate<"Subtarget->hasStdExtZbkx()">,
                    AssemblerPredicate<(all_of FeatureStdExtZbkx),
                        "'Zbkx' (Crossbar permutation instructions)">;

def HasStdExtZbbOrZbkb
    : Predicate<"Subtarget->hasStdExtZbb() || Subtarget->hasStdExtZbkb()">,
      AssemblerPredicate<(any_of FeatureStdExtZbb, FeatureStdExtZbkb),
                         "'Zbb' (Basic Bit-Manipulation) or "
                         "'Zbkb' (Bitmanip instructions for Cryptography)">;

// The Carry-less multiply subextension for cryptography is a subset of basic
// carry-less multiply subextension. The former should be enabled if the latter
// is enabled.
def FeatureStdExtZbkc
    : CapstoneExtension<1, 0,
                     "Carry-less multiply instructions for Cryptography">,
      CapstoneExtensionBitmask<0, 31>;
def HasStdExtZbkc
    : Predicate<"Subtarget->hasStdExtZbkc()">,
      AssemblerPredicate<(all_of FeatureStdExtZbkc),
          "'Zbkc' (Carry-less multiply instructions for Cryptography)">;

def HasStdExtZbcOrZbkc
    : Predicate<"Subtarget->hasStdExtZbc() || Subtarget->hasStdExtZbkc()">,
      AssemblerPredicate<(any_of FeatureStdExtZbc, FeatureStdExtZbkc),
                         "'Zbc' (Carry-Less Multiplication) or "
                         "'Zbkc' (Carry-less multiply instructions "
                         "for Cryptography)">;

// Cryptography Extensions

def FeatureStdExtZknd
    : CapstoneExtension<1, 0, "NIST Suite: AES Decryption">,
      CapstoneExtensionBitmask<0, 41>;
def HasStdExtZknd : Predicate<"Subtarget->hasStdExtZknd()">,
                    AssemblerPredicate<(all_of FeatureStdExtZknd),
                                       "'Zknd' (NIST Suite: AES Decryption)">;

def FeatureStdExtZkne
    : CapstoneExtension<1, 0, "NIST Suite: AES Encryption">,
      CapstoneExtensionBitmask<0, 42>;
def HasStdExtZkne : Predicate<"Subtarget->hasStdExtZkne()">,
                    AssemblerPredicate<(all_of FeatureStdExtZkne),
                                       "'Zkne' (NIST Suite: AES Encryption)">;

// Some instructions belong to both Zknd and Zkne subextensions.
// They should be enabled if either has been specified.
def HasStdExtZkndOrZkne
    : Predicate<"Subtarget->hasStdExtZknd() || Subtarget->hasStdExtZkne()">,
      AssemblerPredicate<(any_of FeatureStdExtZknd, FeatureStdExtZkne),
                         "'Zknd' (NIST Suite: AES Decryption) or "
                         "'Zkne' (NIST Suite: AES Encryption)">;

def FeatureStdExtZknh
    : CapstoneExtension<1, 0, "NIST Suite: Hash Function Instructions">,
      CapstoneExtensionBitmask<0, 43>;
def HasStdExtZknh : Predicate<"Subtarget->hasStdExtZknh()">,
                    AssemblerPredicate<(all_of FeatureStdExtZknh),
                        "'Zknh' (NIST Suite: Hash Function Instructions)">;

def FeatureStdExtZksed
    : CapstoneExtension<1, 0, "ShangMi Suite: SM4 Block Cipher Instructions">,
      CapstoneExtensionBitmask<0, 44>;
def HasStdExtZksed : Predicate<"Subtarget->hasStdExtZksed()">,
                     AssemblerPredicate<(all_of FeatureStdExtZksed),
                         "'Zksed' (ShangMi Suite: SM4 Block Cipher Instructions)">;

def FeatureStdExtZksh
    : CapstoneExtension<1, 0, "ShangMi Suite: SM3 Hash Function Instructions">,
      CapstoneExtensionBitmask<0, 45>;
def HasStdExtZksh : Predicate<"Subtarget->hasStdExtZksh()">,
                    AssemblerPredicate<(all_of FeatureStdExtZksh),
                        "'Zksh' (ShangMi Suite: SM3 Hash Function Instructions)">;

def FeatureStdExtZkr
    : CapstoneExtension<1, 0, "Entropy Source Extension">;
def HasStdExtZkr : Predicate<"Subtarget->hasStdExtZkr()">,
                   AssemblerPredicate<(all_of FeatureStdExtZkr),
                                      "'Zkr' (Entropy Source Extension)">;

def FeatureStdExtZkn
    : CapstoneExtension<1, 0, "NIST Algorithm Suite",
                     [FeatureStdExtZbkb,
                      FeatureStdExtZbkc,
                      FeatureStdExtZbkx,
                      FeatureStdExtZkne,
                      FeatureStdExtZknd,
                      FeatureStdExtZknh]>;

def FeatureStdExtZks
    : CapstoneExtension<1, 0, "ShangMi Algorithm Suite",
                     [FeatureStdExtZbkb,
                      FeatureStdExtZbkc,
                      FeatureStdExtZbkx,
                      FeatureStdExtZksed,
                      FeatureStdExtZksh]>;

def FeatureStdExtZkt
    : CapstoneExtension<1, 0, "Data Independent Execution Latency">,
      CapstoneExtensionBitmask<0, 46>;

def FeatureStdExtZk
    : CapstoneExtension<1, 0, "Standard scalar cryptography extension",
                     [FeatureStdExtZkn,
                      FeatureStdExtZkr,
                      FeatureStdExtZkt]>;

// Vector Extensions

def FeatureStdExtZvl32b : CapstoneExtension<1, 0,
                                         "Minimum Vector Length 32", [],
                                         "ZvlLen", "32">;

foreach i = { 6-16 } in {
  defvar I = !shl(1, i);
  def FeatureStdExtZvl#I#b :
      CapstoneExtension<1, 0, "Minimum Vector Length "#I,
                     [!cast<CapstoneExtension>("FeatureStdExtZvl"#!srl(I, 1)#"b")],
                     "ZvlLen", !cast<string>(I)>;
}

def FeatureStdExtZve32x
    : CapstoneExtension<1, 0,
                     "Vector Extensions for Embedded Processors "
                     "with maximal 32 EEW",
                     [FeatureStdExtZicsr, FeatureStdExtZvl32b]>,
      CapstoneExtensionBitmask<0, 60>;


def FeatureStdExtZve32f
    : CapstoneExtension<1, 0,
                     "Vector Extensions for Embedded Processors "
                     "with maximal 32 EEW and F extension",
                     [FeatureStdExtZve32x, FeatureStdExtF]>,
      CapstoneExtensionBitmask<0, 61>;

def FeatureStdExtZve64x
    : CapstoneExtension<1, 0,
                     "Vector Extensions for Embedded Processors "
                     "with maximal 64 EEW",
                     [FeatureStdExtZve32x, FeatureStdExtZvl64b]>,
      CapstoneExtensionBitmask<0, 62>;

def FeatureStdExtZve64f
    : CapstoneExtension<1, 0,
                     "Vector Extensions for Embedded Processors "
                     "with maximal 64 EEW and F extension",
                     [FeatureStdExtZve32f, FeatureStdExtZve64x]>,
      CapstoneExtensionBitmask<0, 63>;

def FeatureStdExtZve64d
    : CapstoneExtension<1, 0,
                     "Vector Extensions for Embedded Processors "
                     "with maximal 64 EEW, F and D extension",
                     [FeatureStdExtZve64f, FeatureStdExtD]>,
      CapstoneExtensionBitmask<1, 0>;

def FeatureStdExtV
    : CapstoneExtension<1, 0,
                     "Vector Extension for Application Processors",
                     [FeatureStdExtZvl128b, FeatureStdExtZve64d]>,
      CapstoneExtensionBitmask<0, 21>;

def FeatureStdExtZvfbfa
    : CapstoneExperimentalExtension<0, 1, "Additional BF16 vector compute support",
                                 [FeatureStdExtZve32f, FeatureStdExtZfbfmin]>;
def HasStdExtZvfbfa : Predicate<"Subtarget->hasStdExtZvfbfa()">,
                      AssemblerPredicate<(all_of FeatureStdExtZvfbfa),
                          "'Zvfbfa' (Additional BF16 vector compute support)">;

def FeatureStdExtZvfbfmin
    : CapstoneExtension<1, 0, "Vector BF16 Converts", [FeatureStdExtZve32f]>;
def HasStdExtZvfbfmin : Predicate<"Subtarget->hasStdExtZvfbfmin()">,
                        AssemblerPredicate<(all_of FeatureStdExtZvfbfmin),
                            "'Zvfbfmin' (Vector BF16 Converts)">;

def FeatureStdExtZvfbfwma
    : CapstoneExtension<1, 0, "Vector BF16 widening mul-add",
                     [FeatureStdExtZvfbfmin, FeatureStdExtZfbfmin]>;
def HasStdExtZvfbfwma : Predicate<"Subtarget->hasStdExtZvfbfwma()">,
                        AssemblerPredicate<(all_of FeatureStdExtZvfbfwma),
                            "'Zvfbfwma' (Vector BF16 widening mul-add)">;

def FeatureStdExtZvfhmin
    : CapstoneExtension<1, 0,
                     "Vector Half-Precision Floating-Point Minimal",
                     [FeatureStdExtZve32f]>,
      CapstoneExtensionBitmask<0, 51>;

def FeatureStdExtZvfh
    : CapstoneExtension<1, 0, "Vector Half-Precision Floating-Point",
                     [FeatureStdExtZvfhmin, FeatureStdExtZfhmin]>,
      CapstoneExtensionBitmask<0, 50>;

def HasStdExtZfhOrZvfh
    : Predicate<"Subtarget->hasStdExtZfh() || Subtarget->hasStdExtZvfh()">,
      AssemblerPredicate<(any_of FeatureStdExtZfh, FeatureStdExtZvfh),
                         "'Zfh' (Half-Precision Floating-Point) or "
                         "'Zvfh' (Vector Half-Precision Floating-Point)">;

// Vector Cryptography and Bitmanip Extensions

def FeatureStdExtZvkb
    : CapstoneExtension<1, 0, "Vector Bit-manipulation used in Cryptography",
                     [FeatureStdExtZve32x]>,
      CapstoneExtensionBitmask<0, 52>;
def HasStdExtZvkb : Predicate<"Subtarget->hasStdExtZvkb()">,
                    AssemblerPredicate<(all_of FeatureStdExtZvkb),
                        "'Zvkb' (Vector Bit-manipulation used in Cryptography)">;

def FeatureStdExtZvbb
    : CapstoneExtension<1, 0, "Vector basic bit-manipulation instructions",
                     [FeatureStdExtZvkb]>,
      CapstoneExtensionBitmask<0, 48>;
def HasStdExtZvbb : Predicate<"Subtarget->hasStdExtZvbb()">,
                    AssemblerPredicate<(all_of FeatureStdExtZvbb),
                        "'Zvbb' (Vector basic bit-manipulation instructions)">;

def FeatureStdExtZvbc
    : CapstoneExtension<1, 0, "Vector Carryless Multiplication",
                     [FeatureStdExtZve64x]>,
      CapstoneExtensionBitmask<0, 49>;
def HasStdExtZvbc : Predicate<"Subtarget->hasStdExtZvbc()">,
                    AssemblerPredicate<(all_of FeatureStdExtZvbc),
                        "'Zvbc' (Vector Carryless Multiplication)">;

def FeatureStdExtZvbc32e
    : CapstoneExperimentalExtension<0, 7,
                                 "Vector Carryless Multiplication with 32-bits elements",
                                 [FeatureStdExtZve32x]>;

def HasStdExtZvbcOrZvbc32e : Predicate<"Subtarget->hasStdExtZvbc() || Subtarget->hasStdExtZvbc32e()">,
                             AssemblerPredicate<(any_of FeatureStdExtZvbc, FeatureStdExtZvbc32e),
                             "'Zvbc' or 'Zvbc32e' (Vector Carryless Multiplication)">;

def FeatureStdExtZvkg
    : CapstoneExtension<1, 0, "Vector GCM instructions for Cryptography",
                     [FeatureStdExtZve32x]>,
      CapstoneExtensionBitmask<0, 53>;
def HasStdExtZvkg : Predicate<"Subtarget->hasStdExtZvkg()">,
                    AssemblerPredicate<(all_of FeatureStdExtZvkg),
                        "'Zvkg' (Vector GCM instructions for Cryptography)">;

def FeatureStdExtZvkgs
    : CapstoneExperimentalExtension<0, 7,
                                 "Vector-Scalar GCM instructions for Cryptography",
                                 [FeatureStdExtZvkg]>;
def HasStdExtZvkgs : Predicate<"Subtarget->hasStdExtZvkgs()">,
                     AssemblerPredicate<(all_of FeatureStdExtZvkgs),
                        "'Zvkgs' (Vector-Scalar GCM instructions for Cryptography)">;

def FeatureStdExtZvkned
    : CapstoneExtension<1, 0,
                     "Vector AES Encryption & Decryption (Single Round)",
                     [FeatureStdExtZve32x]>,
      CapstoneExtensionBitmask<0, 54>;
def HasStdExtZvkned : Predicate<"Subtarget->hasStdExtZvkned()">,
                      AssemblerPredicate<(all_of FeatureStdExtZvkned),
                          "'Zvkned' (Vector AES Encryption & Decryption (Single Round))">;

def FeatureStdExtZvknha
    : CapstoneExtension<1, 0, "Vector SHA-2 (SHA-256 only)",
                     [FeatureStdExtZve32x]>,
      CapstoneExtensionBitmask<0, 55>;
def HasStdExtZvknha : Predicate<"Subtarget->hasStdExtZvknha()">,
                      AssemblerPredicate<(all_of FeatureStdExtZvknha),
                          "'Zvknha' (Vector SHA-2 (SHA-256 only))">;

def FeatureStdExtZvknhb
    : CapstoneExtension<1, 0, "Vector SHA-2 (SHA-256 and SHA-512)",
                     [FeatureStdExtZve64x]>,
      CapstoneExtensionBitmask<0, 56>;
def HasStdExtZvknhb : Predicate<"Subtarget->hasStdExtZvknhb()">,
                      AssemblerPredicate<(all_of FeatureStdExtZvknhb),
                          "'Zvknhb' (Vector SHA-2 (SHA-256 and SHA-512))">;

def HasStdExtZvknhaOrZvknhb : Predicate<"Subtarget->hasStdExtZvknha() || Subtarget->hasStdExtZvknhb()">,
                              AssemblerPredicate<(any_of FeatureStdExtZvknha, FeatureStdExtZvknhb),
                                  "'Zvknha' or 'Zvknhb' (Vector SHA-2)">;

def FeatureStdExtZvksed
    : CapstoneExtension<1, 0, "SM4 Block Cipher Instructions",
                     [FeatureStdExtZve32x]>,
      CapstoneExtensionBitmask<0, 57>;
def HasStdExtZvksed : Predicate<"Subtarget->hasStdExtZvksed()">,
                      AssemblerPredicate<(all_of FeatureStdExtZvksed),
                          "'Zvksed' (SM4 Block Cipher Instructions)">;

def FeatureStdExtZvksh
    : CapstoneExtension<1, 0, "SM3 Hash Function Instructions",
                     [FeatureStdExtZve32x]>,
      CapstoneExtensionBitmask<0, 58>;
def HasStdExtZvksh : Predicate<"Subtarget->hasStdExtZvksh()">,
                     AssemblerPredicate<(all_of FeatureStdExtZvksh),
                         "'Zvksh' (SM3 Hash Function Instructions)">;

def FeatureStdExtZvkt
    : CapstoneExtension<1, 0, "Vector Data-Independent Execution Latency">,
      CapstoneExtensionBitmask<0, 59>;

// Zvk short-hand extensions

def FeatureStdExtZvkn
    : CapstoneExtension<1, 0,
                     "shorthand for 'Zvkned', 'Zvknhb', 'Zvkb', and 'Zvkt'",
                     [FeatureStdExtZvkned, FeatureStdExtZvknhb,
                      FeatureStdExtZvkb, FeatureStdExtZvkt]>;

def FeatureStdExtZvknc
    : CapstoneExtension<1, 0, "shorthand for 'Zvknc' and 'Zvbc'",
                     [FeatureStdExtZvkn, FeatureStdExtZvbc]>;

def FeatureStdExtZvkng
    : CapstoneExtension<1, 0, "shorthand for 'Zvkn' and 'Zvkg'",
                     [FeatureStdExtZvkn, FeatureStdExtZvkg]>;

def FeatureStdExtZvks
    : CapstoneExtension<1, 0,
                     "shorthand for 'Zvksed', 'Zvksh', 'Zvkb', and 'Zvkt'",
                     [FeatureStdExtZvksed, FeatureStdExtZvksh,
                      FeatureStdExtZvkb, FeatureStdExtZvkt]>;

def FeatureStdExtZvksc
    : CapstoneExtension<1, 0, "shorthand for 'Zvks' and 'Zvbc'",
                     [FeatureStdExtZvks, FeatureStdExtZvbc]>;

def FeatureStdExtZvksg
    : CapstoneExtension<1, 0, "shorthand for 'Zvks' and 'Zvkg'",
                     [FeatureStdExtZvks, FeatureStdExtZvkg]>;

// Vector quad widening dot product

def FeatureStdExtZvqdotq
    : CapstoneExperimentalExtension<0, 0, "Vector quad widening 4D Dot Product",
                                 [FeatureStdExtZve32x]>;
def HasStdExtZvqdotq : Predicate<"Subtarget->hasStdExtZvqdotq()">,
                       AssemblerPredicate<(all_of FeatureStdExtZvqdotq),
                           "'Zvqdotq' (Vector quad widening 4D Dot Product)">;

// Vector instruction predicates

def HasVInstructions    : Predicate<"Subtarget->hasVInstructions()">,
      AssemblerPredicate<
          (any_of FeatureStdExtZve32x),
          "'V' (Vector Extension for Application Processors), 'Zve32x' "
          "(Vector Extensions for Embedded Processors)">;
def HasVInstructionsI64 : Predicate<"Subtarget->hasVInstructionsI64()">,
      AssemblerPredicate<
          (any_of FeatureStdExtZve64x),
          "'V' (Vector Extension for Application Processors) or 'Zve64x' "
          "(Vector Extensions for Embedded Processors)">;
def HasVInstructionsAnyF : Predicate<"Subtarget->hasVInstructionsAnyF()">,
      AssemblerPredicate<
          (any_of FeatureStdExtZve32f),
          "'V' (Vector Extension for Application Processors), 'Zve32f' "
          "(Vector Extensions for Embedded Processors)">;

def HasVInstructionsF16Minimal : Predicate<"Subtarget->hasVInstructionsF16Minimal()">,
      AssemblerPredicate<(any_of FeatureStdExtZvfhmin, FeatureStdExtZvfh),
                         "'Zvfhmin' (Vector Half-Precision Floating-Point Minimal) or "
                         "'Zvfh' (Vector Half-Precision Floating-Point)">;

def HasVInstructionsBF16Minimal : Predicate<"Subtarget->hasVInstructionsBF16Minimal()">;
def HasVInstructionsF16 : Predicate<"Subtarget->hasVInstructionsF16()">;
def HasVInstructionsF64 : Predicate<"Subtarget->hasVInstructionsF64()">;

def HasVInstructionsFullMultiply : Predicate<"Subtarget->hasVInstructionsFullMultiply()">;

// Hypervisor Extensions

def FeatureStdExtH : CapstoneExtension<1, 0, "Hypervisor">,
                     CapstoneExtensionBitmask<0, 7>;
def HasStdExtH : Predicate<"Subtarget->hasStdExtH()">,
                 AssemblerPredicate<(all_of FeatureStdExtH),
                                    "'H' (Hypervisor)">;

// Supervisor extensions

def FeatureStdExtSdext : CapstoneExtension<1, 0, "External debugger">;

def FeatureStdExtSdtrig : CapstoneExtension<1, 0, "Debugger triggers">;

def FeatureStdExtShgatpa
    : CapstoneExtension<1, 0,
                     "SvNNx4 mode supported for all modes supported by satp, as well as Bare">;
def FeatureStdExtShvsatpa
    : CapstoneExtension<1, 0,
                     "vsatp supports all modes supported by satp">;

def FeatureStdExtShlcofideleg
    : CapstoneExtension<1, 0,
                     "Delegating LCOFI Interrupts to VS-mode">;

def FeatureStdExtSmaia
    : CapstoneExtension<1, 0,
                     "Advanced Interrupt Architecture Machine Level">;
def FeatureStdExtSsaia
    : CapstoneExtension<1, 0,
                     "Advanced Interrupt Architecture Supervisor Level">;

def FeatureStdExtSmcntrpmf
    : CapstoneExtension<1, 0, "Cycle and Instret Privilege Mode Filtering">;

def FeatureStdExtSmcsrind
    : CapstoneExtension<1, 0, "Indirect CSR Access Machine Level">;
def FeatureStdExtSscsrind
    : CapstoneExtension<1, 0, "Indirect CSR Access Supervisor Level">;

def FeatureStdExtSmdbltrp
    : CapstoneExtension<1, 0, "Double Trap Machine Level", [FeatureStdExtZicsr]>;
def FeatureStdExtSsdbltrp
    : CapstoneExtension<1, 0, "Double Trap Supervisor Level", [FeatureStdExtZicsr]>;

def FeatureStdExtSmepmp
    : CapstoneExtension<1, 0, "Enhanced Physical Memory Protection">;

def FeatureStdExtSmrnmi
    : CapstoneExtension<1, 0, "Resumable Non-Maskable Interrupts">;
def HasStdExtSmrnmi : Predicate<"Subtarget->hasStdExtSmrnmi()">,
                       AssemblerPredicate<(all_of FeatureStdExtSmrnmi),
                           "'Smrnmi' (Resumable Non-Maskable Interrupts)">;

def FeatureStdExtSmcdeleg
    : CapstoneExtension<1, 0, "Counter Delegation Machine Level">;
def FeatureStdExtSsccfg
    : CapstoneExtension<1, 0, "Counter Configuration Supervisor Level">;

def FeatureStdExtSsccptr
    : CapstoneExtension<1, 0, "Main memory supports page table reads">;

def FeatureStdExtSscofpmf
    : CapstoneExtension<1, 0, "Count Overflow and Mode-Based Filtering">;

def FeatureStdExtShcounterenw
    : CapstoneExtension<1, 0,
                     "Support writeable hcounteren enable "
                     "bit for any hpmcounter that is not read-only zero">;
def FeatureStdExtSscounterenw
    : CapstoneExtension<1, 0,
                     "Support writeable scounteren enable "
                     "bit for any hpmcounter that is not read-only zero">;

def FeatureStdExtSmstateen
    : CapstoneExtension<1, 0,
                     "Machine-mode view of the state-enable extension">;
def FeatureStdExtSsstateen
    : CapstoneExtension<1, 0,
                     "Supervisor-mode view of the state-enable extension">;

def FeatureStdExtSsstrict
    : CapstoneExtension<1, 0,
                     "No non-conforming extensions are present">;

def FeatureStdExtSstc
    : CapstoneExtension<1, 0, "Supervisor-mode timer interrupts">;

def FeatureStdExtSsqosid
    : CapstoneExtension<1, 0, "Quality-of-Service (QoS) Identifiers">;

def FeatureStdExtShtvala
    : CapstoneExtension<1, 0, "htval provides all needed values">;
def FeatureStdExtShvstvala
    : CapstoneExtension<1, 0, "vstval provides all needed values">;
def FeatureStdExtSstvala
    : CapstoneExtension<1, 0, "stval provides all needed values">;

def FeatureStdExtShvstvecd
    : CapstoneExtension<1, 0, "vstvec supports Direct mode">;
def FeatureStdExtSstvecd
    : CapstoneExtension<1, 0, "stvec supports Direct mode">;

def FeatureStdExtSsu64xl
    : CapstoneExtension<1, 0, "UXLEN=64 supported">;

def FeatureStdExtSvade
    : CapstoneExtension<1, 0, "Raise exceptions on improper A/D bits">;

def FeatureStdExtSvadu
    : CapstoneExtension<1, 0, "Hardware A/D updates">;

def FeatureStdExtSvbare
    : CapstoneExtension<1, 0, "satp mode Bare supported">;

def FeatureStdExtSvinval
    : CapstoneExtension<1, 0,
                     "Fine-Grained Address-Translation Cache Invalidation">;
def HasStdExtSvinval : Predicate<"Subtarget->hasStdExtSvinval()">,
                       AssemblerPredicate<(all_of FeatureStdExtSvinval),
                           "'Svinval' (Fine-Grained Address-Translation Cache Invalidation)">;

def FeatureStdExtSvnapot
    : CapstoneExtension<1, 0, "NAPOT Translation Contiguity">;

def FeatureStdExtSvpbmt
    : CapstoneExtension<1, 0, "Page-Based Memory Types">;

def FeatureStdExtSvvptc
    : CapstoneExtension<1, 0,
                     "Obviating Memory-Management Instructions after Marking PTEs Valid">;

def FeatureStdExtSha
    : CapstoneExtension<1, 0, "Augmented Hypervisor",
                     [FeatureStdExtH, FeatureStdExtSsstateen, FeatureStdExtShcounterenw,
                      FeatureStdExtShvstvala, FeatureStdExtShtvala, FeatureStdExtShvstvecd,
                      FeatureStdExtShvsatpa, FeatureStdExtShgatpa]>;

def FeatureStdExtSvukte
    : CapstoneExperimentalExtension<0, 3,
                                 "Address-Independent Latency of User-Mode Faults to Supervisor Addresses">;

// Pointer Masking extensions

// A supervisor-level extension that provides pointer masking for the next lower
// privilege mode (U-mode), and for VS- and VU-modes if the H extension is
// present.
def FeatureStdExtSsnpm
    : CapstoneExtension<1, 0,
                     "Supervisor-level Pointer Masking for next lower privilege mode">;

// A machine-level extension that provides pointer masking for the next lower
// privilege mode (S/HS if S-mode is implemented, or U-mode otherwise).
def FeatureStdExtSmnpm
    : CapstoneExtension<1, 0,
                     "Machine-level Pointer Masking for next lower privilege mode">;

// A machine-level extension that provides pointer masking for M-mode.
def FeatureStdExtSmmpm
    : CapstoneExtension<1, 0,
                     "Machine-level Pointer Masking for M-mode">;

// An extension that indicates that there is pointer-masking support available
// in supervisor mode, with some facility provided in the supervisor execution
// environment to control pointer masking.
def FeatureStdExtSspm
    : CapstoneExtension<1, 0,
                     "Indicates Supervisor-mode Pointer Masking">;

// An extension that indicates that there is pointer-masking support available
// in user mode, with some facility provided in the application execution
// environment to control pointer masking.
def FeatureStdExtSupm
    : CapstoneExtension<1, 0,
                     "Indicates User-mode Pointer Masking">;

def FeatureStdExtSmctr
    : CapstoneExtension<1, 0,
                     "Control Transfer Records Machine Level",
                     [FeatureStdExtSscsrind]>;
def FeatureStdExtSsctr
    : CapstoneExtension<1, 0,
                     "Control Transfer Records Supervisor Level",
                     [FeatureStdExtSscsrind]>;
def HasStdExtSmctrOrSsctr : Predicate<"Subtarget->hasStdExtSmctrOrSsctr()">,
                            AssemblerPredicate<(any_of FeatureStdExtSmctr, FeatureStdExtSsctr),
                               "'Smctr' (Control Transfer Records Machine Level) or "
                               "'Ssctr' (Control Transfer Records Supervisor Level)">;

// Packed SIMD Extensions
def FeatureStdExtP
    : CapstoneExperimentalExtension<0, 15,
                                 "'Base P' (Packed SIMD)">;
def HasStdExtP : Predicate<"Subtarget->hasStdExtP()">,
                 AssemblerPredicate<(all_of FeatureStdExtP),
                                    "'Base P' (Packed SIMD)">;

def HasStdExtZbaOrP
    : Predicate<"Subtarget->hasStdExtZba() || Subtarget->hasStdExtP()">,
      AssemblerPredicate<(any_of FeatureStdExtZba, FeatureStdExtP),
                         "'Zba' (Address Generation Instructions) or "
                         "'Base P' (Packed-SIMD)">;

def HasStdExtZbbOrP
    : Predicate<"Subtarget->hasStdExtZbb() || Subtarget->hasStdExtP()">,
      AssemblerPredicate<(any_of FeatureStdExtZbb, FeatureStdExtP),
                         "'Zbb' (Basic Bit-Manipulation) or "
                         "'Base P' (Packed-SIMD)">;

def HasStdExtZbkbOrP
    : Predicate<"Subtarget->hasStdExtZbkb() || Subtarget->hasStdExtP()">,
      AssemblerPredicate<(any_of FeatureStdExtZbkb, FeatureStdExtP),
                         "'Zbkb' (Bitmanip instructions for Cryptography) or "
                         "'Base P' (Packed-SIMD)">;

def HasStdExtZbbOrZbkbOrP
    : Predicate<"Subtarget->HasStdExtZbbOrZbkb()|| Subtarget->hasStdExtP()">,
      AssemblerPredicate<(any_of FeatureStdExtZbb, FeatureStdExtZbkb, FeatureStdExtP),
                         "'Zbb' (Basic Bit-Manipulation) or "
                         "'Zbkb' (Bitmanip instructions for Cryptography) or "
                         "'Base P' (Packed-SIMD)">;

//===----------------------------------------------------------------------===//
// Vendor extensions
//===----------------------------------------------------------------------===//

// Ventana Extensions

def FeatureVendorXVentanaCondOps
    : CapstoneExtension<1, 0, "Ventana Conditional Ops">;
def HasVendorXVentanaCondOps : Predicate<"Subtarget->hasVendorXVentanaCondOps()">,
                               AssemblerPredicate<(all_of FeatureVendorXVentanaCondOps),
                                   "'XVentanaCondOps' (Ventana Conditional Ops)">;

// T-Head Extensions

def FeatureVendorXTHeadBa
    : CapstoneExtension<1, 0,
                     "T-Head address calculation instructions">;
def HasVendorXTHeadBa : Predicate<"Subtarget->hasVendorXTHeadBa()">,
                        AssemblerPredicate<(all_of FeatureVendorXTHeadBa),
                            "'XTHeadBa' (T-Head address calculation instructions)">;

def FeatureVendorXTHeadBb
    : CapstoneExtension<1, 0,
                     "T-Head basic bit-manipulation instructions">;
def HasVendorXTHeadBb : Predicate<"Subtarget->hasVendorXTHeadBb()">,
                        AssemblerPredicate<(all_of FeatureVendorXTHeadBb),
                            "'XTHeadBb' (T-Head basic bit-manipulation instructions)">;

def FeatureVendorXTHeadBs
    : CapstoneExtension<1, 0,
                     "T-Head single-bit instructions">;
def HasVendorXTHeadBs : Predicate<"Subtarget->hasVendorXTHeadBs()">,
                        AssemblerPredicate<(all_of FeatureVendorXTHeadBs),
                            "'XTHeadBs' (T-Head single-bit instructions)">;

def FeatureVendorXTHeadCondMov
    : CapstoneExtension<1, 0,
                     "T-Head conditional move instructions">;
def HasVendorXTHeadCondMov : Predicate<"Subtarget->hasVendorXTHeadCondMov()">,
                             AssemblerPredicate<(all_of FeatureVendorXTHeadCondMov),
                                 "'XTHeadCondMov' (T-Head conditional move instructions)">;

def FeatureVendorXTHeadCmo
    : CapstoneExtension<1, 0,
                     "T-Head cache management instructions">;
def HasVendorXTHeadCmo : Predicate<"Subtarget->hasVendorXTHeadCmo()">,
                         AssemblerPredicate<(all_of FeatureVendorXTHeadCmo),
                             "'XTHeadCmo' (T-Head cache management instructions)">;

def FeatureVendorXTHeadFMemIdx
    : CapstoneExtension<1, 0,
                     "T-Head FP Indexed Memory Operations">;
def HasVendorXTHeadFMemIdx : Predicate<"Subtarget->hasVendorXTHeadFMemIdx()">,
                             AssemblerPredicate<(all_of FeatureVendorXTHeadFMemIdx),
                                 "'XTHeadFMemIdx' (T-Head FP Indexed Memory Operations)">;

def FeatureVendorXTHeadMac
    : CapstoneExtension<1, 0,
                     "T-Head Multiply-Accumulate Instructions">;
def HasVendorXTHeadMac : Predicate<"Subtarget->hasVendorXTHeadMac()">,
                         AssemblerPredicate<(all_of FeatureVendorXTHeadMac),
                             "'XTHeadMac' (T-Head Multiply-Accumulate Instructions)">;

def FeatureVendorXTHeadMemIdx
    : CapstoneExtension<1, 0,
                     "T-Head Indexed Memory Operations">;
def HasVendorXTHeadMemIdx : Predicate<"Subtarget->hasVendorXTHeadMemIdx()">,
                            AssemblerPredicate<(all_of FeatureVendorXTHeadMemIdx),
                                "'XTHeadMemIdx' (T-Head Indexed Memory Operations)">;

def FeatureVendorXTHeadMemPair
    : CapstoneExtension<1, 0,
                     "T-Head two-GPR Memory Operations">;
def HasVendorXTHeadMemPair : Predicate<"Subtarget->hasVendorXTHeadMemPair()">,
                             AssemblerPredicate<(all_of FeatureVendorXTHeadMemPair),
                                 "'XTHeadMemPair' (T-Head two-GPR Memory Operations)">;

def FeatureVendorXTHeadSync
    : CapstoneExtension<1, 0,
                     "T-Head multicore synchronization instructions">;
def HasVendorXTHeadSync : Predicate<"Subtarget->hasVendorXTHeadSync()">,
                          AssemblerPredicate<(all_of FeatureVendorXTHeadSync),
                              "'XTHeadSync' (T-Head multicore synchronization instructions)">;

def FeatureVendorXTHeadVdot
    : CapstoneExtension<1, 0,
                     "T-Head Vector Extensions for Dot",
                     [FeatureStdExtV]>;
def HasVendorXTHeadVdot : Predicate<"Subtarget->hasVendorXTHeadVdot()">,
                          AssemblerPredicate<(all_of FeatureVendorXTHeadVdot),
                              "'XTHeadVdot' (T-Head Vector Extensions for Dot)">;

// SiFive Extensions

def FeatureVendorXSfvcp
    : CapstoneExtension<1, 0,
                     "SiFive Custom Vector Coprocessor Interface Instructions",
                     [FeatureStdExtZve32x]>;
def HasVendorXSfvcp : Predicate<"Subtarget->hasVendorXSfvcp()">,
                      AssemblerPredicate<(all_of FeatureVendorXSfvcp),
                          "'XSfvcp' (SiFive Custom Vector Coprocessor Interface Instructions)">;

def FeatureVendorXSfmmbase
    : CapstoneExtension<0, 6,
                     "All non arithmetic instructions for all TEWs and sf.vtzero",
                     [FeatureStdExtZve32x]>;
def HasVendorXSfmmbase : Predicate<"Subtarget->hasVendorXSfmmbase()">,
                         AssemblerPredicate<(all_of FeatureVendorXSfmmbase),
                             "'XSfmmbase' (All non arithmetic instructions for all TEWs and sf.vtzero)">;

def FeatureVendorXSfmm32a8f
    : CapstoneExtension<0, 6,
                     "TEW=32-bit accumulation, operands - float: fp8",
                     [FeatureVendorXSfmmbase, FeatureStdExtZve32f]>;
def HasVendorXSfmm32a8f : Predicate<"Subtarget->hasVendorXSfmm32a8f()">,
                          AssemblerPredicate<(all_of FeatureVendorXSfmm32a8f),
                              "'XSfmm32a8f' (TEW=32-bit accumulation, operands - float: fp8)">;

def FeatureVendorXSfmm32a16f
    : CapstoneExtension<0, 6,
                     "TEW=32-bit accumulation, operands - float: 16b, widen=2 (IEEE, BF)",
                     [FeatureVendorXSfmmbase, FeatureStdExtZve32f]>;

def FeatureVendorXSfmm32a32f
    : CapstoneExtension<0, 6,
                     "TEW=32-bit accumulation, operands - float: 32b",
                     [FeatureVendorXSfmmbase, FeatureStdExtZve32f]>;

def FeatureVendorXSfmm32a8i
    : CapstoneExtension<0, 6,
                     "TEW=32-bit accumulation, operands - int: 8b",
                     [FeatureVendorXSfmmbase]>;
def HasVendorXSfmm32a8i : Predicate<"Subtarget->hasVendorXSfmm32a8i()">,
                          AssemblerPredicate<(all_of FeatureVendorXSfmm32a8i),
                              "'XSfmm32a8i' (TEW=32-bit accumulation, operands - int: 8b)">;

def FeatureVendorXSfmm64a64f
    : CapstoneExtension<0, 6,
                     "TEW=64-bit accumulation, operands - float: fp64",
                     [FeatureVendorXSfmmbase, FeatureStdExtZve64d]>;
def HasVendorXSfmm32a16fOrXSfmm32a32fOrXSfmm64a64f
    : Predicate<"Subtarget->hasVendorXSfmm32a16fOrXSfmm32a32fOrXSfmm64a64f()">,
                AssemblerPredicate<(any_of FeatureVendorXSfmm32a16f,
                                           FeatureVendorXSfmm32a32f,
                                           FeatureVendorXSfmm64a64f),
                    "'XSfmm32a16f' (TEW=32-bit accumulation, operands - float: 16b, widen=2 (IEEE, BF)), or "
                    "'XSfmm32a32f' (TEW=32-bit accumulation, operands - float: 32b), or "
                    "'XSfmm64a64f' (TEW=64-bit accumulation, operands - float: fp64)">;

def FeatureVendorXSfmm16t
    : CapstoneExtension<0, 6,
                     "TE=16 configuration",
                     [FeatureVendorXSfmmbase, FeatureStdExtZvl64b], "XSfmmTE", "16">;

def FeatureVendorXSfmm32t
    : CapstoneExtension<0, 6,
                     "TE=32 configuration",
                     [FeatureVendorXSfmmbase, FeatureStdExtZvl128b], "XSfmmTE", "32">;

def FeatureVendorXSfmm64t
    : CapstoneExtension<0, 6,
                     "TE=64 configuration",
                     [FeatureVendorXSfmmbase, FeatureStdExtZvl256b], "XSfmmTE", "64">;

def FeatureVendorXSfmm128t
    : CapstoneExtension<0, 6,
                     "TE=128 configuration",
                     [FeatureVendorXSfmmbase, FeatureStdExtZvl512b], "XSfmmTE", "128">;

def FeatureVendorXSfvqmaccdod
    : CapstoneExtension<1, 0,
                     "SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2)",
                     [FeatureStdExtZve32x, FeatureStdExtZvl128b]>;
def HasVendorXSfvqmaccdod
    : Predicate<"Subtarget->hasVendorXSfvqmaccdod()">,
      AssemblerPredicate<(all_of FeatureVendorXSfvqmaccdod),
                         "'XSfvqmaccdod' (SiFive Int8 Matrix Multiplication Instructions (2-by-8 and 8-by-2))">;

def FeatureVendorXSfvqmaccqoq
    : CapstoneExtension<1, 0,
                     "SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4)",
                     [FeatureStdExtZve32x, FeatureStdExtZvl256b]>;
def HasVendorXSfvqmaccqoq
    : Predicate<"Subtarget->hasVendorXSfvqmaccqoq()">,
      AssemblerPredicate<(all_of FeatureVendorXSfvqmaccqoq),
                         "'XSfvqmaccqoq' (SiFive Int8 Matrix Multiplication Instructions (4-by-8 and 8-by-4))">;

def FeatureVendorXSfvfwmaccqqq
    : CapstoneExtension<1, 0,
                     "SiFive Matrix Multiply Accumulate Instruction (4-by-4)",
                     [FeatureStdExtZvfbfmin, FeatureStdExtZvl128b]>;
def HasVendorXSfvfwmaccqqq
    : Predicate<"Subtarget->hasVendorXSfvfwmaccqqq()">,
      AssemblerPredicate<(all_of FeatureVendorXSfvfwmaccqqq),
                         "'XSfvfwmaccqqq' (SiFive Matrix Multiply Accumulate Instruction (4-by-4))">;

def FeatureVendorXSfvfnrclipxfqf
    : CapstoneExtension<1, 0,
                     "SiFive FP32-to-int8 Ranged Clip Instructions",
                     [FeatureStdExtZve32f]>;
def HasVendorXSfvfnrclipxfqf
    : Predicate<"Subtarget->hasVendorXSfvfnrclipxfqf()">,
      AssemblerPredicate<(all_of FeatureVendorXSfvfnrclipxfqf),
                         "'XSfvfnrclipxfqf' (SiFive FP32-to-int8 Ranged Clip Instructions)">;

def FeatureVendorXSiFivecdiscarddlone
    : CapstoneExtension<1, 0,
                     "SiFive sf.cdiscard.d.l1 Instruction", []>;
def HasVendorXSiFivecdiscarddlone
    : Predicate<"Subtarget->hasVendorXSiFivecdiscarddlone()">,
      AssemblerPredicate<(all_of FeatureVendorXSiFivecdiscarddlone),
                         "'XSiFivecdiscarddlone' (SiFive sf.cdiscard.d.l1 Instruction)">;

def FeatureVendorXSiFivecflushdlone
    : CapstoneExtension<1, 0,
                     "SiFive sf.cflush.d.l1 Instruction", []>;
def HasVendorXSiFivecflushdlone
    : Predicate<"Subtarget->hasVendorXSiFivecflushdlone()">,
      AssemblerPredicate<(all_of FeatureVendorXSiFivecflushdlone),
                         "'XSiFivecflushdlone' (SiFive sf.cflush.d.l1 Instruction)">;

def FeatureVendorXSfcease
    : CapstoneExtension<1, 0,
                     "SiFive sf.cease Instruction", []>;
def HasVendorXSfcease
    : Predicate<"Subtarget->hasVendorXSfcease()">,
      AssemblerPredicate<(all_of FeatureVendorXSfcease),
                         "'XSfcease' (SiFive sf.cease Instruction)">;

def FeatureVendorXSfmclic
    : CapstoneExperimentalExtension<0, 1,
                                 "SiFive CLIC Machine-mode CSRs">;

def FeatureVendorXSfsclic
    : CapstoneExperimentalExtension<0, 1,
                                 "SiFive CLIC Supervisor-mode CSRs">;

// Core-V Extensions

def FeatureVendorXCVelw
   : CapstoneExtension<1, 0, "CORE-V Event Load Word">;
def HasVendorXCVelw
   : Predicate<"Subtarget->hasVendorXCVelw()">,
     AssemblerPredicate<(any_of FeatureVendorXCVelw),
                        "'XCVelw' (CORE-V Event Load Word)">;

def FeatureVendorXCVbitmanip
    : CapstoneExtension<1, 0, "CORE-V Bit Manipulation">;
def HasVendorXCVbitmanip
    : Predicate<"Subtarget->hasVendorXCVbitmanip()">,
      AssemblerPredicate<(all_of FeatureVendorXCVbitmanip),
                         "'XCVbitmanip' (CORE-V Bit Manipulation)">;

def FeatureVendorXCVmac
    : CapstoneExtension<1, 0, "CORE-V Multiply-Accumulate">;
def HasVendorXCVmac
    : Predicate<"Subtarget->hasVendorXCVmac()">,
      AssemblerPredicate<(all_of FeatureVendorXCVmac),
                         "'XCVmac' (CORE-V Multiply-Accumulate)">;

def FeatureVendorXCVmem
    : CapstoneExtension<1, 0, "CORE-V Post-incrementing Load & Store">;
def HasVendorXCVmem
    : Predicate<"Subtarget->hasVendorXCVmem()">,
      AssemblerPredicate<(any_of FeatureVendorXCVmem),
                         "'XCVmem' (CORE-V Post-incrementing Load & Store)">;

def FeatureVendorXCValu
    : CapstoneExtension<1, 0, "CORE-V ALU Operations">;
def HasVendorXCValu
    : Predicate<"Subtarget->hasVendorXCValu()">,
      AssemblerPredicate<(all_of FeatureVendorXCValu),
                         "'XCValu' (CORE-V ALU Operations)">;

def FeatureVendorXCVsimd
    : CapstoneExtension<1, 0, "CORE-V SIMD ALU">;
def HasVendorXCVsimd
    : Predicate<"Subtarget->hasVendorXCVsimd()">,
      AssemblerPredicate<(any_of FeatureVendorXCVsimd),
                         "'XCVsimd' (CORE-V SIMD ALU)">;

def FeatureVendorXCVbi
    : CapstoneExtension<1, 0, "CORE-V Immediate Branching">;
def HasVendorXCVbi
    : Predicate<"Subtarget->hasVendorXCVbi()">,
      AssemblerPredicate<(all_of FeatureVendorXCVbi),
                         "'XCVbi' (CORE-V Immediate Branching)">;
// MIPS Extensions

def FeatureVendorXMIPSCMov
    : CapstoneExtension<1, 0, "MIPS conditional move instruction (mips.ccmov)">;
def HasVendorXMIPSCMov
    : Predicate<"Subtarget->hasVendorXMIPSCMov()">,
      AssemblerPredicate<(all_of FeatureVendorXMIPSCMov),
                         "'Xmipscmov' ('mips.ccmov' instruction)">;
def UseCCMovInsn : Predicate<"Subtarget->useCCMovInsn()">;

def FeatureVendorXMIPSLSP
    : CapstoneExtension<1, 0, "MIPS optimization for hardware load-store bonding">;
def HasVendorXMIPSLSP
    : Predicate<"Subtarget->hasVendorXMIPSLSP()">,
      AssemblerPredicate<(all_of FeatureVendorXMIPSLSP),
                         "'Xmipslsp' (load and store pair instructions)">;

def FeatureVendorXMIPSCBOP : CapstoneExtension<1, 0, "MIPS Software Prefetch">;
def HasVendorXMIPSCBOP
    : Predicate<"Subtarget->hasVendorXMIPSCBOP()">,
      AssemblerPredicate<(all_of FeatureVendorXMIPSCBOP),
                         "'Xmipscbop' (MIPS hardware prefetch)">;
def NoVendorXMIPSCBOP : Predicate<"!Subtarget->hasVendorXMIPSCBOP()">;

def FeatureVendorXMIPSEXECTL : CapstoneExtension<1, 0, "MIPS execution control">;
def HasVendorXMIPSEXECTL
    : Predicate<"Subtarget->hasVendorXMIPSEXT()">,
      AssemblerPredicate<(all_of FeatureVendorXMIPSEXECTL),
                         "'Xmipsexectl' (MIPS execution control)">;

// WCH / Nanjing Qinheng Microelectronics Extension(s)

def FeatureVendorXwchc
    : CapstoneExtension<2, 2,
                     "WCH/QingKe additional compressed opcodes",
                     [FeatureStdExtZca]>;
def HasVendorXwchc
    : Predicate<"Subtarget->hasVendorXwchc()">,
      AssemblerPredicate<(all_of FeatureVendorXwchc),
                         "'Xwchc' (WCH/QingKe additional compressed opcodes)">;

// Qualcomm Extensions

def FeatureVendorXqccmp
    : CapstoneExperimentalExtension<0, 3,
                                 "Qualcomm 16-bit Push/Pop and Double Moves",
                                 [FeatureStdExtZca]>;
def HasVendorXqccmp
    : Predicate<"Subtarget->hasVendorXqccmp()">,
      AssemblerPredicate<(all_of FeatureVendorXqccmp),
                         "'Xqccmp' (Qualcomm 16-bit Push/Pop and Double Moves)">;

def FeatureVendorXqcia
    : CapstoneExperimentalExtension<0, 7, "Qualcomm uC Arithmetic Extension">;
def HasVendorXqcia
    : Predicate<"Subtarget->hasVendorXqcia()">,
      AssemblerPredicate<(all_of FeatureVendorXqcia),
                         "'Xqcia' (Qualcomm uC Arithmetic Extension)">;

def FeatureVendorXqciac
    : CapstoneExperimentalExtension<0, 3, "Qualcomm uC Load-Store Address Calculation Extension",
                                 [FeatureStdExtZca]>;
def HasVendorXqciac
    : Predicate<"Subtarget->hasVendorXqciac()">,
      AssemblerPredicate<(all_of FeatureVendorXqciac),
                         "'Xqciac' (Qualcomm uC Load-Store Address Calculation Extension)">;

def FeatureVendorXqcibi
    : CapstoneExperimentalExtension<0, 2, "Qualcomm uC Branch Immediate Extension",
                                 [FeatureStdExtZca]>;
def HasVendorXqcibi
    : Predicate<"Subtarget->hasVendorXqcibi()">,
      AssemblerPredicate<(all_of FeatureVendorXqcibi),
                         "'Xqcibi' (Qualcomm uC Branch Immediate Extension)">;

def FeatureVendorXqcibm
    : CapstoneExperimentalExtension<0, 8, "Qualcomm uC Bit Manipulation Extension",
                                 [FeatureStdExtZca]>;
def HasVendorXqcibm
    : Predicate<"Subtarget->hasVendorXqcibm()">,
      AssemblerPredicate<(all_of FeatureVendorXqcibm),
                         "'Xqcibm' (Qualcomm uC Bit Manipulation Extension)">;

def FeatureVendorXqcicli
    : CapstoneExperimentalExtension<0, 3,
                                 "Qualcomm uC Conditional Load Immediate Extension">;
def HasVendorXqcicli
    : Predicate<"Subtarget->hasVendorXqcicli()">,
      AssemblerPredicate<(all_of FeatureVendorXqcicli),
                         "'Xqcicli' (Qualcomm uC Conditional Load Immediate Extension)">;

def FeatureVendorXqcicm
    : CapstoneExperimentalExtension<0, 2, "Qualcomm uC Conditional Move Extension",
                                 [FeatureStdExtZca]>;
def HasVendorXqcicm
    : Predicate<"Subtarget->hasVendorXqcicm()">,
      AssemblerPredicate<(all_of FeatureVendorXqcicm),
                         "'Xqcicm' (Qualcomm uC Conditional Move Extension)">;

def FeatureVendorXqcics
    : CapstoneExperimentalExtension<0, 2, "Qualcomm uC Conditional Select Extension">;
def HasVendorXqcics
    : Predicate<"Subtarget->hasVendorXqcics()">,
      AssemblerPredicate<(all_of FeatureVendorXqcics),
                         "'Xqcics' (Qualcomm uC Conditional Select Extension)">;
def NoVendorXqcics
    : Predicate<"!Subtarget->hasVendorXqcics()">;

def HasVendorXqcicsOrXqcicm
    : Predicate<"Subtarget->hasVendorXqcics() || Subtarget->hasVendorXqcicm()">;

def FeatureVendorXqcicsr
    : CapstoneExperimentalExtension<0, 4, "Qualcomm uC CSR Extension">;
def HasVendorXqcicsr
    : Predicate<"Subtarget->hasVendorXqcicsr()">,
      AssemblerPredicate<(all_of FeatureVendorXqcicsr),
                         "'Xqcicsr' (Qualcomm uC CSR Extension)">;

def FeatureVendorXqciint
    : CapstoneExperimentalExtension<0, 10, "Qualcomm uC Interrupts Extension",
                                 [FeatureStdExtZca]>;
def HasVendorXqciint
    : Predicate<"Subtarget->hasVendorXqciint()">,
      AssemblerPredicate<(all_of FeatureVendorXqciint),
                         "'Xqciint' (Qualcomm uC Interrupts Extension)">;

def FeatureVendorXqciio
    : CapstoneExperimentalExtension<0, 1, "Qualcomm uC External Input Output Extension">;
def HasVendorXqciio
    : Predicate<"Subtarget->hasVendorXqciio()">,
      AssemblerPredicate<(all_of FeatureVendorXqciio),
                         "'Xqciio' (Qualcomm uC External Input Output Extension)">;

def FeatureVendorXqcilb
    : CapstoneExperimentalExtension<0, 2, "Qualcomm uC Long Branch Extension",
                                 [FeatureStdExtZca]>;
def HasVendorXqcilb
    : Predicate<"Subtarget->hasVendorXqcilb()">,
      AssemblerPredicate<(all_of FeatureVendorXqcilb),
                         "'Xqcilb' (Qualcomm uC Long Branch Extension)">;

def FeatureVendorXqcili
    : CapstoneExperimentalExtension<0, 2, "Qualcomm uC Load Large Immediate Extension",
                                 [FeatureStdExtZca]>;
def HasVendorXqcili
    : Predicate<"Subtarget->hasVendorXqcili()">,
      AssemblerPredicate<(all_of FeatureVendorXqcili),
                         "'Xqcili' (Qualcomm uC Load Large Immediate Extension)">;

def FeatureVendorXqcilia
    : CapstoneExperimentalExtension<0, 2, "Qualcomm uC Large Immediate Arithmetic Extension",
                                 [FeatureStdExtZca]>;
def HasVendorXqcilia
    : Predicate<"Subtarget->hasVendorXqcilia()">,
      AssemblerPredicate<(all_of FeatureVendorXqcilia),
                         "'Xqcilia' (Qualcomm uC Large Immediate Arithmetic Extension)">;

def FeatureVendorXqcilo
    : CapstoneExperimentalExtension<0, 3, "Qualcomm uC Large Offset Load Store Extension",
                                 [FeatureStdExtZca]>;
def HasVendorXqcilo
    : Predicate<"Subtarget->hasVendorXqcilo()">,
      AssemblerPredicate<(all_of FeatureVendorXqcilo),
                         "'Xqcilo' (Qualcomm uC Large Offset Load Store Extension)">;

def FeatureVendorXqcilsm
    : CapstoneExperimentalExtension<0, 6,
                                 "Qualcomm uC Load Store Multiple Extension">;
def HasVendorXqcilsm
    : Predicate<"Subtarget->hasVendorXqcilsm()">,
      AssemblerPredicate<(all_of FeatureVendorXqcilsm),
                         "'Xqcilsm' (Qualcomm uC Load Store Multiple Extension)">;

def FeatureVendorXqcisim
    : CapstoneExperimentalExtension<0, 2, "Qualcomm uC Simulation Hint Extension",
                                 [FeatureStdExtZca]>;
def HasVendorXqcisim
    : Predicate<"Subtarget->hasVendorXqcisim()">,
      AssemblerPredicate<(all_of FeatureVendorXqcisim),
                         "'Xqcisim' (Qualcomm uC Simulation Hint Extension)">;

def FeatureVendorXqcisls
    : CapstoneExperimentalExtension<0, 2,
                                 "Qualcomm uC Scaled Load Store Extension">;
def HasVendorXqcisls
    : Predicate<"Subtarget->hasVendorXqcisls()">,
      AssemblerPredicate<(all_of FeatureVendorXqcisls),
                         "'Xqcisls' (Qualcomm uC Scaled Load Store Extension)">;

def FeatureVendorXqcisync
    : CapstoneExperimentalExtension<0, 3, "Qualcomm uC Sync Delay Extension",
                                 [FeatureStdExtZca]>;
def HasVendorXqcisync
    : Predicate<"Subtarget->hasVendorXqcisync()">,
      AssemblerPredicate<(all_of FeatureVendorXqcisync),
                         "'Xqcisync' (Qualcomm uC Sync Delay Extension)">;

// Rivos Extension(s)

def FeatureVendorXRivosVisni
    :  CapstoneExperimentalExtension<0, 1, "Rivos Vector Integer Small New">;
def HasVendorXRivosVisni
    : Predicate<"Subtarget->hasVendorXRivosVisni()">,
      AssemblerPredicate<(all_of FeatureVendorXRivosVisni),
                         "'XRivosVisni' (Rivos Vector Integer Small New)">;

def FeatureVendorXRivosVizip
    :  CapstoneExperimentalExtension<0, 1, "Rivos Vector Register Zips">;
def HasVendorXRivosVizip
    : Predicate<"Subtarget->hasVendorXRivosVizip()">,
      AssemblerPredicate<(all_of FeatureVendorXRivosVizip),
                         "'XRivosVizip' (Rivos Vector Register Zips)">;

// Andes Extension(s)

def FeatureVendorXAndesPerf
    : CapstoneExtension<5, 0, "Andes Performance Extension">;
def HasVendorXAndesPerf
    : Predicate<"Subtarget->hasVendorXAndesPerf()">,
      AssemblerPredicate<(all_of FeatureVendorXAndesPerf),
                         "'XAndesPerf' (Andes Performance Extension)">;

def FeatureVendorXAndesBFHCvt
    : CapstoneExtension<5, 0, "Andes Scalar BFLOAT16 Conversion Extension",
                     [FeatureStdExtF]>;
def HasVendorXAndesBFHCvt
    : Predicate<"Subtarget->hasVendorXAndesBFHCvt()">,
      AssemblerPredicate<(all_of FeatureVendorXAndesBFHCvt),
                         "'XAndesBFHCvt' (Andes Scalar BFLOAT16 Conversion Extension)">;

def FeatureVendorXAndesVBFHCvt
    : CapstoneExtension<5, 0, "Andes Vector BFLOAT16 Conversion Extension",
                     [FeatureStdExtZve32f]>;
def HasVendorXAndesVBFHCvt
    : Predicate<"Subtarget->hasVendorXAndesVBFHCvt()">,
      AssemblerPredicate<(all_of FeatureVendorXAndesVBFHCvt),
                         "'XAndesVBFHCvt' (Andes Vector BFLOAT16 Conversion Extension)">;

def FeatureVendorXAndesVSIntLoad
    : CapstoneExtension<5, 0, "Andes Vector INT4 Load Extension",
                     [FeatureStdExtZve32x]>;
def HasVendorXAndesVSIntLoad
    : Predicate<"Subtarget->hasVendorXAndesVSIntLoad()">,
      AssemblerPredicate<(all_of FeatureVendorXAndesVSIntLoad),
                         "'XAndesVSIntLoad' (Andes Vector INT4 Load Extension)">;

def FeatureVendorXAndesVPackFPH
    : CapstoneExtension<5, 0,
                     "Andes Vector Packed FP16 Extension", [FeatureStdExtF]>;
def HasVendorXAndesVPackFPH
    : Predicate<"Subtarget->hasVendorXAndesVPackFPH()">,
      AssemblerPredicate<(all_of FeatureVendorXAndesVPackFPH),
                         "'XAndesVPackFPH' (Andes Vector Packed FP16 Extension)">;

def FeatureVendorXAndesVDot
    : CapstoneExtension<5, 0, "Andes Vector Dot Product Extension",
                     [FeatureStdExtZve32x]>;
def HasVendorXAndesVDot
    : Predicate<"Subtarget->hasVendorXAndesVDot()">,
      AssemblerPredicate<(all_of FeatureVendorXAndesVDot),
                         "'XAndesVDot' (Andes Vector Dot Product Extension)">;

def FeatureVendorXSMTVDot
    : CapstoneExtension<1, 0, "SpacemiT Vector Dot Product Extension",
                     [FeatureStdExtZve32f]>;
def HasVendorXSMTVDot
    : Predicate<"Subtarget->hasVendorXSMTVDot()">,
      AssemblerPredicate<(all_of FeatureVendorXSMTVDot),
                         "'XSMTVDot' (SpacemiT Vector Dot Product Extension)">;

//===----------------------------------------------------------------------===//
// LLVM specific features and extensions
//===----------------------------------------------------------------------===//

// Feature32Bit exists to mark CPUs that support RV32 to distinguish them from
// tuning CPU names.
def Feature32Bit
    : SubtargetFeature<"32bit", "IsRV32", "true", "Implements RV32">;
def Feature64Bit
    : SubtargetFeature<"64bit", "IsRV64", "true", "Implements RV64">;
def IsRV64 : Predicate<"Subtarget->is64Bit()">,
             AssemblerPredicate<(all_of Feature64Bit),
                                "RV64I Base Instruction Set">;
def IsRV32 : Predicate<"!Subtarget->is64Bit()">,
             AssemblerPredicate<(all_of (not Feature64Bit)),
                                "RV32I Base Instruction Set">;

defvar RV32 = DefaultMode;
def RV64 : HwMode<[IsRV64]>;

def FeatureRelax
    : SubtargetFeature<"relax", "EnableLinkerRelax", "true",
                       "Enable Linker relaxation.">;

def FeatureExactAssembly
    : SubtargetFeature<"exact-asm", "EnableExactAssembly", "true",
                       "Enable Exact Assembly (Disables Compression and Relaxation)">;

foreach i = {1-31} in
  def FeatureReserveX#i :
      SubtargetFeature<"reserve-x"#i, "UserReservedRegister[Capstone::X"#i#"]",
                       "true", "Reserve X"#i>;

def FeatureSaveRestore : SubtargetFeature<"save-restore", "EnableSaveRestore",
                                          "true", "Enable save/restore.">;

def FeatureNoTrailingSeqCstFence : SubtargetFeature<"no-trailing-seq-cst-fence",
                                          "EnableTrailingSeqCstFence",
                                          "false",
                                          "Disable trailing fence for seq-cst store.">;

def FeatureUnalignedScalarMem
   : SubtargetFeature<"unaligned-scalar-mem", "EnableUnalignedScalarMem",
                      "true", "Has reasonably performant unaligned scalar "
                      "loads and stores">;

def FeatureUnalignedVectorMem
   : SubtargetFeature<"unaligned-vector-mem", "EnableUnalignedVectorMem",
                      "true", "Has reasonably performant unaligned vector "
                      "loads and stores">;

def TuneNLogNVRGather
   : SubtargetFeature<"log-vrgather", "CapstoneVRGatherCostModel", "NLog2N",
                      "Has vrgather.vv with LMUL*log2(LMUL) latency">;

def TunePostRAScheduler : SubtargetFeature<"use-postra-scheduler",
    "UsePostRAScheduler", "true", "Schedule again after register allocation">;

def TuneDisableMISchedLoadClustering : SubtargetFeature<"disable-misched-load-clustering",
    "EnableMISchedLoadClustering", "false", "Disable load clustering in the machine scheduler">;

def TuneDisableMISchedStoreClustering : SubtargetFeature<"disable-misched-store-clustering",
    "EnableMISchedStoreClustering", "false", "Disable store clustering in the machine scheduler">;

def TuneDisablePostMISchedLoadClustering : SubtargetFeature<"disable-postmisched-load-clustering",
    "EnablePostMISchedLoadClustering", "false", "Disable PostRA load clustering in the machine scheduler">;

def TuneDisablePostMISchedStoreClustering : SubtargetFeature<"disable-postmisched-store-clustering",
    "EnablePostMISchedStoreClustering", "false", "Disable PostRA store clustering in the machine scheduler">;

def TuneDisableLatencySchedHeuristic
    : SubtargetFeature<"disable-latency-sched-heuristic", "DisableLatencySchedHeuristic", "true",
                       "Disable latency scheduling heuristic">;

def TunePredictableSelectIsExpensive
    : SubtargetFeature<"predictable-select-expensive", "PredictableSelectIsExpensive", "true",
                       "Prefer likely predicted branches over selects">;

def TuneOptimizedZeroStrideLoad
   : SubtargetFeature<"optimized-zero-stride-load", "HasOptimizedZeroStrideLoad",
                      "true", "Optimized (perform fewer memory operations)"
                      "zero-stride vector load">;

foreach nf = {2-8} in
  def TuneOptimizedNF#nf#SegmentLoadStore :
      SubtargetFeature<"optimized-nf"#nf#"-segment-load-store",
                       "HasOptimizedNF"#nf#"SegmentLoadStore",
                       "true", "vlseg"#nf#"eN.v and vsseg"#nf#"eN.v are "
                       "implemented as a wide memory op and shuffle">;

def TuneVLDependentLatency
    : SubtargetFeature<"vl-dependent-latency", "HasVLDependentLatency", "true",
                       "Latency of vector instructions is dependent on the "
                       "dynamic value of vl">;

def Experimental
   : SubtargetFeature<"experimental", "HasExperimental",
                      "true", "Experimental intrinsics">;

// Some vector hardware implementations do not process all VLEN bits in parallel
// and instead split over multiple cycles. DLEN refers to the datapath width
// that can be done in parallel.
def TuneDLenFactor2
   : SubtargetFeature<"dlen-factor-2", "DLenFactor2", "true",
                      "Vector unit DLEN(data path width) is half of VLEN">;

def TuneNoDefaultUnroll
    : SubtargetFeature<"no-default-unroll", "EnableDefaultUnroll", "false",
                       "Disable default unroll preference.">;

// SiFive 7 is able to fuse integer ALU operations with a preceding branch
// instruction.
def TuneShortForwardBranchOpt
    : SubtargetFeature<"short-forward-branch-opt", "HasShortForwardBranchOpt",
                       "true", "Enable short forward branch optimization">;
def HasShortForwardBranchOpt : Predicate<"Subtarget->hasShortForwardBranchOpt()">;
def NoShortForwardBranchOpt : Predicate<"!Subtarget->hasShortForwardBranchOpt()">;

// Some subtargets require a S2V transfer buffer to move scalars into vectors.
// FIXME: Forming .vx/.vf/.wx/.wf can reduce register pressure.
def TuneNoSinkSplatOperands
    : SubtargetFeature<"no-sink-splat-operands", "SinkSplatOperands",
                       "false", "Disable sink splat operands to enable .vx, .vf,"
                       ".wx, and .wf instructions">;

def TunePreferWInst
    : SubtargetFeature<"prefer-w-inst", "PreferWInst", "true",
                       "Prefer instructions with W suffix">;

def TuneConditionalCompressedMoveFusion
    : SubtargetFeature<"conditional-cmv-fusion", "HasConditionalCompressedMoveFusion",
                       "true", "Enable branch+c.mv fusion">;
def HasConditionalMoveFusion : Predicate<"Subtarget->hasConditionalMoveFusion()">;
def NoConditionalMoveFusion  : Predicate<"!Subtarget->hasConditionalMoveFusion()">;

def TuneMIPSP8700
    : SubtargetFeature<"mips-p8700", "CapstoneProcFamily", "MIPSP8700",
                       "MIPS p8700 processor">;

def TuneSiFive7 : SubtargetFeature<"sifive7", "CapstoneProcFamily", "SiFive7",
                                   "SiFive 7-Series processors">;

def TuneVentanaVeyron : SubtargetFeature<"ventana-veyron", "CapstoneProcFamily", "VentanaVeyron",
                                         "Ventana Veyron-Series processors">;

def TuneAndes45 : SubtargetFeature<"andes45", "CapstoneProcFamily", "Andes45",
                                   "Andes 45-Series processors">;

def TuneVXRMPipelineFlush : SubtargetFeature<"vxrm-pipeline-flush", "HasVXRMPipelineFlush",
                                             "true", "VXRM writes causes pipeline flush">;

def TunePreferVsetvliOverReadVLENB
    : SubtargetFeature<"prefer-vsetvli-over-read-vlenb",
                       "PreferVsetvliOverReadVLENB",
                       "true",
                       "Prefer vsetvli over read vlenb CSR to calculate VLEN">;

// Assume that lock-free native-width atomics are available, even if the target
// and operating system combination would not usually provide them. The user
// is responsible for providing any necessary __sync implementations. Code
// built with this feature is not ABI-compatible with code built without this
// feature, if atomic variables are exposed across the ABI boundary.
def FeatureForcedAtomics : SubtargetFeature<
    "forced-atomics", "HasForcedAtomics", "true",
    "Assume that lock-free native-width atomics are available">;
def HasAtomicLdSt
    : Predicate<"Subtarget->hasStdExtA() || Subtarget->hasForcedAtomics()">;

def FeatureTaggedGlobals : SubtargetFeature<"tagged-globals",
    "AllowTaggedGlobals",
    "true", "Use an instruction sequence for taking the address of a global "
    "that allows a memory tag in the upper address bits">;
