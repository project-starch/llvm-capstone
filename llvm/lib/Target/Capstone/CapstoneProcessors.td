//===-- CapstoneProcessors.td - Capstone Processors -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Capstone processors supported.
//===----------------------------------------------------------------------===//

// Predefined scheduling direction.
defvar TopDown       = [{ MISched::TopDown }];
defvar BottomUp      = [{ MISched::BottomUp }];
defvar Bidirectional = [{ MISched::Bidirectional }];

class CapstoneTuneInfo {
  bits<8> PrefFunctionAlignment = 1;
  bits<8> PrefLoopAlignment = 1;

  // Information needed by LoopDataPrefetch.
  bits<16> CacheLineSize = 0;
  bits<16> PrefetchDistance = 0;
  bits<16> MinPrefetchStride = 1;
  bits<32> MaxPrefetchIterationsAhead = -1;

  bits<32> MinimumJumpTableEntries = 5;

  // Tail duplication threshold at -O3.
  bits<32> TailDupAggressiveThreshold = 6;

  bits<32> MaxStoresPerMemsetOptSize = 4;
  bits<32> MaxStoresPerMemset = 8;

  bits<32> MaxGluedStoresPerMemcpy = 0;
  bits<32> MaxStoresPerMemcpyOptSize = 4;
  bits<32> MaxStoresPerMemcpy = 8;

  bits<32> MaxStoresPerMemmoveOptSize = 4;
  bits<32> MaxStoresPerMemmove = 8;

  bits<32> MaxLoadsPerMemcmpOptSize = 4;
  bits<32> MaxLoadsPerMemcmp = 8;

  // The direction of PostRA scheduling.
  code PostRASchedDirection = TopDown;
}

def CapstoneTuneInfoTable : GenericTable {
  let FilterClass = "CapstoneTuneInfo";
  let CppTypeName = "CapstoneTuneInfo";
  let Fields = ["Name", "PrefFunctionAlignment", "PrefLoopAlignment",
                "CacheLineSize", "PrefetchDistance", "MinPrefetchStride",
                "MaxPrefetchIterationsAhead", "MinimumJumpTableEntries",
                "TailDupAggressiveThreshold", "MaxStoresPerMemsetOptSize",
                "MaxStoresPerMemset", "MaxGluedStoresPerMemcpy",
                "MaxStoresPerMemcpyOptSize", "MaxStoresPerMemcpy",
                "MaxStoresPerMemmoveOptSize", "MaxStoresPerMemmove",
                "MaxLoadsPerMemcmpOptSize", "MaxLoadsPerMemcmp",
                "PostRASchedDirection"];
}

def getCapstoneTuneInfo : SearchIndex {
  let Table = CapstoneTuneInfoTable;
  let Key = ["Name"];
}

class GenericTuneInfo: CapstoneTuneInfo;

class CapstoneProcessorModel<string n,
                          SchedMachineModel m,
                          list<SubtargetFeature> f,
                          list<SubtargetFeature> tunef = [],
                          string default_march = "">
    :  ProcessorModel<n, m, f, tunef> {
  string DefaultMarch = default_march;
  int MVendorID = 0;
  int MArchID = 0;
  int MImpID = 0;
}

class CapstoneTuneProcessorModel<string n,
                              SchedMachineModel m,
                              list<SubtargetFeature> tunef = [],
                              list<SubtargetFeature> f = []>
    : ProcessorModel<n, m, f,tunef>;

defvar GenericTuneFeatures = [TuneOptimizedNF2SegmentLoadStore];

def GENERIC_RV32 : CapstoneProcessorModel<"generic-rv32",
                                       NoSchedModel,
                                       [Feature32Bit,
                                        FeatureStdExtI],
                                       GenericTuneFeatures>,
                   GenericTuneInfo;
def GENERIC_RV64 : CapstoneProcessorModel<"generic-rv64",
                                       NoSchedModel,
                                       [Feature64Bit,
                                        FeatureStdExtI],
                                       GenericTuneFeatures>,
                   GenericTuneInfo;
// Support generic for compatibility with other targets. The triple will be used
// to change to the appropriate rv32/rv64 version.
def GENERIC : CapstoneTuneProcessorModel<"generic", NoSchedModel>, GenericTuneInfo;
def GENERIC_OOO : CapstoneTuneProcessorModel<"generic-ooo", GenericOOOModel>,
                  GenericTuneInfo;

def MIPS_P8700 : CapstoneProcessorModel<"mips-p8700",
                                     MIPSP8700Model,
                                     [Feature64Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC,
                                      FeatureStdExtZba,
                                      FeatureStdExtZbb,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtZicsr,
                                      FeatureVendorXMIPSCMov,
                                      FeatureVendorXMIPSLSP,
                                      FeatureVendorXMIPSCBOP,
                                      FeatureVendorXMIPSEXECTL],
                                     [TuneMIPSP8700]>;

def ROCKET_RV32 : CapstoneProcessorModel<"rocket-rv32",
                                      RocketModel,
                                      [Feature32Bit,
                                       FeatureStdExtI,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtZicsr]>;
def ROCKET_RV64 : CapstoneProcessorModel<"rocket-rv64",
                                      RocketModel,
                                      [Feature64Bit,
                                       FeatureStdExtI,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtZicsr]>;
def ROCKET : CapstoneTuneProcessorModel<"rocket",
                                     RocketModel>;

defvar SiFive7TuneFeatures = [TuneSiFive7, TuneNoDefaultUnroll,
                              TuneShortForwardBranchOpt,
                              TunePostRAScheduler];
def SIFIVE_7 : CapstoneTuneProcessorModel<"sifive-7-series",
                                       SiFive7Model, SiFive7TuneFeatures>;

def SIFIVE_E20 : CapstoneProcessorModel<"sifive-e20",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtC]>;

def SIFIVE_E21 : CapstoneProcessorModel<"sifive-e21",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_E24 : CapstoneProcessorModel<"sifive-e24",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtC]>;

def SIFIVE_E31 : CapstoneProcessorModel<"sifive-e31",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_E34 : CapstoneProcessorModel<"sifive-e34",
                                     RocketModel,
                                     [Feature32Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtC]>;

def SIFIVE_E76 : CapstoneProcessorModel<"sifive-e76",
                                     SiFive7Model,
                                     [Feature32Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtC],
                                     SiFive7TuneFeatures>;

def SIFIVE_S21 : CapstoneProcessorModel<"sifive-s21",
                                     RocketModel,
                                     [Feature64Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_S51 : CapstoneProcessorModel<"sifive-s51",
                                     RocketModel,
                                     [Feature64Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtC]>;

def SIFIVE_S54 : CapstoneProcessorModel<"sifive-s54",
                                      RocketModel,
                                      [Feature64Bit,
                                       FeatureStdExtI,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC]>;

def SIFIVE_S76 : CapstoneProcessorModel<"sifive-s76",
                                     SiFive7Model,
                                     [Feature64Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC,
                                      FeatureStdExtZihintpause],
                                     SiFive7TuneFeatures>;

def SIFIVE_U54 : CapstoneProcessorModel<"sifive-u54",
                                     RocketModel,
                                     [Feature64Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC]>;

def SIFIVE_U74 : CapstoneProcessorModel<"sifive-u74",
                                     SiFive7Model,
                                     [Feature64Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC],
                                     SiFive7TuneFeatures>;

defvar SiFiveIntelligenceTuneFeatures = !listconcat(SiFive7TuneFeatures,
                                                    [TuneDLenFactor2,
                                                     TuneOptimizedZeroStrideLoad,
                                                     TuneOptimizedNF2SegmentLoadStore,
                                                     TuneVLDependentLatency]);
def SIFIVE_X280 : CapstoneProcessorModel<"sifive-x280", SiFive7Model,
                                      [Feature64Bit,
                                       FeatureStdExtI,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtV,
                                       FeatureStdExtZvl512b,
                                       FeatureStdExtZfh,
                                       FeatureStdExtZvfh,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb],
                                      SiFiveIntelligenceTuneFeatures>;

def SIFIVE_X390 : CapstoneProcessorModel<"sifive-x390",
                                      SiFiveX390Model,
                                      [Feature64Bit,
                                       FeatureStdExtI,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtB,
                                       FeatureStdExtV,
                                       FeatureStdExtZic64b,
                                       FeatureStdExtZicbom,
                                       FeatureStdExtZicbop,
                                       FeatureStdExtZicboz,
                                       FeatureStdExtZiccamoa,
                                       FeatureStdExtZiccif,
                                       FeatureStdExtZiccrse,
                                       FeatureStdExtZicfilp,
                                       FeatureStdExtZicfiss,
                                       FeatureStdExtZicntr,
                                       FeatureStdExtZicond,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtZihintntl,
                                       FeatureStdExtZihintpause,
                                       FeatureStdExtZihpm,
                                       FeatureStdExtZimop,
                                       FeatureStdExtZa64rs,
                                       FeatureStdExtZawrs,
                                       FeatureStdExtZfa,
                                       FeatureStdExtZfh,
                                       FeatureStdExtZcb,
                                       FeatureStdExtZcmop,
                                       FeatureStdExtZkr,
                                       FeatureStdExtZkt,
                                       FeatureStdExtZvbb,
                                       FeatureStdExtZvfbfmin,
                                       FeatureStdExtZvfbfwma,
                                       FeatureStdExtZvfh,
                                       FeatureStdExtZvkt,
                                       FeatureStdExtZvl1024b,
                                       FeatureVendorXSiFivecdiscarddlone,
                                       FeatureVendorXSiFivecflushdlone],
                                      SiFiveIntelligenceTuneFeatures>;

defvar SiFiveP400TuneFeatures = [TuneNoDefaultUnroll,
                                 TuneConditionalCompressedMoveFusion,
                                 TuneLUIADDIFusion,
                                 TuneAUIPCADDIFusion,
                                 TunePostRAScheduler];

def SIFIVE_P450 : CapstoneProcessorModel<"sifive-p450", SiFiveP400Model,
                                      !listconcat(RVA22U64Features,
                                      [FeatureStdExtZifencei,
                                       FeatureStdExtZihintntl,
                                       FeatureUnalignedScalarMem,
                                       FeatureUnalignedVectorMem]),
                                      SiFiveP400TuneFeatures>;

def SIFIVE_P470 : CapstoneProcessorModel<"sifive-p470", SiFiveP400Model,
                                      !listconcat(RVA22U64Features,
                                      [FeatureStdExtV,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtZihintntl,
                                       FeatureStdExtZvl128b,
                                       FeatureStdExtZvbb,
                                       FeatureStdExtZvknc,
                                       FeatureStdExtZvkng,
                                       FeatureStdExtZvksc,
                                       FeatureStdExtZvksg,
                                       FeatureVendorXSiFivecdiscarddlone,
                                       FeatureVendorXSiFivecflushdlone,
                                       FeatureUnalignedScalarMem,
                                       FeatureUnalignedVectorMem]),
                                      !listconcat(SiFiveP400TuneFeatures,
                                                  [TuneNoSinkSplatOperands,
                                                   TuneVXRMPipelineFlush])>;

defvar SiFiveP500TuneFeatures = [TuneNoDefaultUnroll,
                                 TuneConditionalCompressedMoveFusion,
                                 TuneLUIADDIFusion,
                                 TuneAUIPCADDIFusion,
                                 TunePostRAScheduler];

def SIFIVE_P550 : CapstoneProcessorModel<"sifive-p550", SiFiveP500Model,
                                      [Feature64Bit,
                                       FeatureStdExtI,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtM,
                                       FeatureStdExtA,
                                       FeatureStdExtF,
                                       FeatureStdExtD,
                                       FeatureStdExtC,
                                       FeatureStdExtZba,
                                       FeatureStdExtZbb],
                                      SiFiveP500TuneFeatures>;

def SIFIVE_P670 : CapstoneProcessorModel<"sifive-p670", SiFiveP600Model,
                                      !listconcat(RVA22U64Features,
                                      [FeatureStdExtV,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtZihintntl,
                                       FeatureStdExtZvl128b,
                                       FeatureStdExtZvbb,
                                       FeatureStdExtZvknc,
                                       FeatureStdExtZvkng,
                                       FeatureStdExtZvksc,
                                       FeatureStdExtZvksg,
                                       FeatureUnalignedScalarMem,
                                       FeatureUnalignedVectorMem]),
                                      [TuneNoDefaultUnroll,
                                       TuneConditionalCompressedMoveFusion,
                                       TuneLUIADDIFusion,
                                       TuneAUIPCADDIFusion,
                                       TuneNoSinkSplatOperands,
                                       TuneVXRMPipelineFlush,
                                       TunePostRAScheduler]>;

def SIFIVE_P870 : CapstoneProcessorModel<"sifive-p870", SiFiveP800Model,
                                      !listconcat(RVA23U64Features,
                                      [FeatureStdExtZama16b,
                                       FeatureStdExtZfh,
                                       FeatureStdExtZifencei,
                                       FeatureStdExtZkr,
                                       FeatureStdExtZvfbfmin,
                                       FeatureStdExtZvfbfwma,
                                       FeatureStdExtZvfh,
                                       FeatureStdExtZvknc,
                                       FeatureStdExtZvkng,
                                       FeatureStdExtZvksc,
                                       FeatureStdExtZvksg,
                                       FeatureStdExtZvl128b,
                                       FeatureUnalignedScalarMem,
                                       FeatureUnalignedVectorMem]),
                                      [TuneNoDefaultUnroll,
                                       TuneConditionalCompressedMoveFusion,
                                       TuneLUIADDIFusion,
                                       TuneAUIPCADDIFusion,
                                       TuneNoSinkSplatOperands,
                                       TuneVXRMPipelineFlush,
                                       TunePostRAScheduler]>;

def SYNTACORE_SCR1_BASE : CapstoneProcessorModel<"syntacore-scr1-base",
                                              SyntacoreSCR1Model,
                                              [Feature32Bit,
                                               FeatureStdExtI,
                                               FeatureStdExtZicsr,
                                               FeatureStdExtZifencei,
                                               FeatureStdExtC],
                                              [TuneNoDefaultUnroll]>;

def SYNTACORE_SCR1_MAX : CapstoneProcessorModel<"syntacore-scr1-max",
                                             SyntacoreSCR1Model,
                                             [Feature32Bit,
                                              FeatureStdExtI,
                                              FeatureStdExtZicsr,
                                              FeatureStdExtZifencei,
                                              FeatureStdExtM,
                                              FeatureStdExtC],
                                             [TuneNoDefaultUnroll]>;

def SYNTACORE_SCR3_RV32 : CapstoneProcessorModel<"syntacore-scr3-rv32",
                                              SyntacoreSCR3RV32Model,
                                              [Feature32Bit,
                                               FeatureStdExtI,
                                               FeatureStdExtZicsr,
                                               FeatureStdExtZifencei,
                                               FeatureStdExtM,
                                               FeatureStdExtC],
                                              [TuneNoDefaultUnroll, TunePostRAScheduler]>;

def SYNTACORE_SCR3_RV64 : CapstoneProcessorModel<"syntacore-scr3-rv64",
                                              SyntacoreSCR3RV64Model,
                                              [Feature64Bit,
                                               FeatureStdExtI,
                                               FeatureStdExtZicsr,
                                               FeatureStdExtZifencei,
                                               FeatureStdExtM,
                                               FeatureStdExtA,
                                               FeatureStdExtC],
                                              [TuneNoDefaultUnroll, TunePostRAScheduler]>;

def SYNTACORE_SCR4_RV32 : CapstoneProcessorModel<"syntacore-scr4-rv32",
                                              SyntacoreSCR4RV32Model,
                                              [Feature32Bit,
                                               FeatureStdExtI,
                                               FeatureStdExtZicsr,
                                               FeatureStdExtZifencei,
                                               FeatureStdExtM,
                                               FeatureStdExtF,
                                               FeatureStdExtD,
                                               FeatureStdExtC],
                                              [TuneNoDefaultUnroll, TunePostRAScheduler]>;

def SYNTACORE_SCR4_RV64 : CapstoneProcessorModel<"syntacore-scr4-rv64",
                                              SyntacoreSCR4RV64Model,
                                              [Feature64Bit,
                                               FeatureStdExtI,
                                               FeatureStdExtZicsr,
                                               FeatureStdExtZifencei,
                                               FeatureStdExtM,
                                               FeatureStdExtA,
                                               FeatureStdExtF,
                                               FeatureStdExtD,
                                               FeatureStdExtC],
                                              [TuneNoDefaultUnroll, TunePostRAScheduler]>;

def SYNTACORE_SCR5_RV32 : CapstoneProcessorModel<"syntacore-scr5-rv32",
                                              SyntacoreSCR5RV32Model,
                                              [Feature32Bit,
                                               FeatureStdExtI,
                                               FeatureStdExtZicsr,
                                               FeatureStdExtZifencei,
                                               FeatureStdExtM,
                                               FeatureStdExtA,
                                               FeatureStdExtF,
                                               FeatureStdExtD,
                                               FeatureStdExtC],
                                              [TuneNoDefaultUnroll, TunePostRAScheduler]>;

def SYNTACORE_SCR5_RV64 : CapstoneProcessorModel<"syntacore-scr5-rv64",
                                              SyntacoreSCR5RV64Model,
                                              [Feature64Bit,
                                               FeatureStdExtI,
                                               FeatureStdExtZicsr,
                                               FeatureStdExtZifencei,
                                               FeatureStdExtM,
                                               FeatureStdExtA,
                                               FeatureStdExtF,
                                               FeatureStdExtD,
                                               FeatureStdExtC],
                                              [TuneNoDefaultUnroll, TunePostRAScheduler]>;

def SYNTACORE_SCR7 : CapstoneProcessorModel<"syntacore-scr7",
                                              SyntacoreSCR7Model,
                                              [Feature64Bit,
                                               FeatureStdExtI,
                                               FeatureStdExtZicsr,
                                               FeatureStdExtZifencei,
                                               FeatureStdExtM,
                                               FeatureStdExtA,
                                               FeatureStdExtF,
                                               FeatureStdExtD,
                                               FeatureStdExtC,
                                               FeatureStdExtV,
                                               FeatureStdExtZba,
                                               FeatureStdExtZbb,
                                               FeatureStdExtZbc,
                                               FeatureStdExtZbs,
                                               FeatureStdExtZkn],
                                              [TuneNoDefaultUnroll, TunePostRAScheduler]>;

def TENSTORRENT_ASCALON_D8 : CapstoneProcessorModel<"tt-ascalon-d8",
                                                 TTAscalonD8Model,
                                                 !listconcat(RVA23S64Features,
                                                 [FeatureStdExtSmaia,
                                                  FeatureStdExtSsaia,
                                                  FeatureStdExtSsstrict,
                                                  FeatureStdExtZfbfmin,
                                                  FeatureStdExtZfh,
                                                  FeatureStdExtZvbc,
                                                  FeatureStdExtZvfbfmin,
                                                  FeatureStdExtZvfbfwma,
                                                  FeatureStdExtZvfh,
                                                  FeatureStdExtZvkng,
                                                  FeatureStdExtZvl256b,
                                                  FeatureUnalignedScalarMem,
                                                  FeatureUnalignedVectorMem]),
                                                 [TuneNoDefaultUnroll,
                                                  TuneNLogNVRGather,
                                                  TuneOptimizedZeroStrideLoad,
                                                  TunePostRAScheduler]>;

def VENTANA_VEYRON_V1 : CapstoneProcessorModel<"veyron-v1",
                                            NoSchedModel,
                                            [Feature64Bit,
                                             FeatureStdExtI,
                                             FeatureStdExtZifencei,
                                             FeatureStdExtZicsr,
                                             FeatureStdExtZicntr,
                                             FeatureStdExtZihpm,
                                             FeatureStdExtZihintpause,
                                             FeatureStdExtM,
                                             FeatureStdExtA,
                                             FeatureStdExtF,
                                             FeatureStdExtD,
                                             FeatureStdExtC,
                                             FeatureStdExtZba,
                                             FeatureStdExtZbb,
                                             FeatureStdExtZbc,
                                             FeatureStdExtZbs,
                                             FeatureStdExtZicbom,
                                             FeatureStdExtZicbop,
                                             FeatureStdExtZicboz,
                                             FeatureVendorXVentanaCondOps],
                                             [TuneVentanaVeyron,
                                              TuneDisableMISchedLoadClustering,
                                              TuneDisablePostMISchedLoadClustering,
                                              TuneDisablePostMISchedStoreClustering,
                                              TuneLUIADDIFusion,
                                              TuneAUIPCADDIFusion,
                                              TuneZExtHFusion,
                                              TuneZExtWFusion,
                                              TuneShiftedZExtWFusion,
                                              TuneADDLoadFusion,
                                              TuneAUIPCLoadFusion,
                                              TuneLUILoadFusion]> {
  let MVendorID = 0x61f;
  let MArchID = 0x8000000000010000;
  let MImpID = 0x111;
}

def XIANGSHAN_NANHU : CapstoneProcessorModel<"xiangshan-nanhu",
                                          XiangShanNanHuModel,
                                          [Feature64Bit,
                                           FeatureStdExtI,
                                           FeatureStdExtZicsr,
                                           FeatureStdExtZifencei,
                                           FeatureStdExtM,
                                           FeatureStdExtA,
                                           FeatureStdExtF,
                                           FeatureStdExtD,
                                           FeatureStdExtC,
                                           FeatureStdExtZba,
                                           FeatureStdExtZbb,
                                           FeatureStdExtZbc,
                                           FeatureStdExtZbs,
                                           FeatureStdExtZkn,
                                           FeatureStdExtZksed,
                                           FeatureStdExtZksh,
                                           FeatureStdExtSvinval,
                                           FeatureStdExtZicbom,
                                           FeatureStdExtZicboz],
                                           [TuneNoDefaultUnroll,
                                            TuneZExtHFusion,
                                            TuneZExtWFusion,
                                            TuneShiftedZExtWFusion]>;

def XIANGSHAN_KUNMINGHU : CapstoneProcessorModel<"xiangshan-kunminghu",
                                              NoSchedModel,
                                              !listconcat(RVA23S64Features,
                                              [FeatureStdExtZacas,
                                               FeatureStdExtZbc,
                                               FeatureStdExtZfh,
                                               FeatureStdExtZkn,
                                               FeatureStdExtZks,
                                               FeatureStdExtZvfh,
                                               FeatureStdExtSmaia,
                                               FeatureStdExtSmcsrind,
                                               FeatureStdExtSmdbltrp,
                                               FeatureStdExtSmmpm,
                                               FeatureStdExtSmnpm,
                                               FeatureStdExtSmrnmi,
                                               FeatureStdExtSmstateen,
                                               FeatureStdExtSsaia,
                                               FeatureStdExtSscsrind,
                                               FeatureStdExtSsdbltrp,
                                               FeatureStdExtSspm,
                                               FeatureStdExtSsstrict,
                                               FeatureStdExtZvl128b]),
                                              [TuneNoDefaultUnroll,
                                               TuneZExtHFusion,
                                               TuneZExtWFusion,
                                               TuneShiftedZExtWFusion]>;

def SPACEMIT_X60 : CapstoneProcessorModel<"spacemit-x60",
                                       SpacemitX60Model,
                                       !listconcat(RVA22S64Features,
                                       [FeatureStdExtV,
                                        FeatureStdExtSscofpmf,
                                        FeatureStdExtSstc,
                                        FeatureStdExtSvnapot,
                                        FeatureStdExtZbc,
                                        FeatureStdExtZbkc,
                                        FeatureStdExtZfh,
                                        FeatureStdExtZicond,
                                        FeatureStdExtZvfh,
                                        FeatureStdExtZvkt,
                                        FeatureStdExtZvl256b,
                                        FeatureVendorXSMTVDot,
                                        FeatureUnalignedScalarMem]),
                                       [TuneDLenFactor2,
                                        TuneOptimizedNF2SegmentLoadStore,
                                        TuneOptimizedNF3SegmentLoadStore,
                                        TuneOptimizedNF4SegmentLoadStore,
                                        TuneVXRMPipelineFlush]> {
  let MVendorID = 0x710;
  let MArchID = 0x8000000058000001;
  let MImpID = 0x1000000049772200;
}

def RP2350_HAZARD3 : CapstoneProcessorModel<"rp2350-hazard3",
                                         NoSchedModel,
                                         [Feature32Bit,
                                          FeatureStdExtI,
                                          FeatureStdExtM,
                                          FeatureStdExtA,
                                          FeatureStdExtC,
                                          FeatureStdExtZicsr,
                                          FeatureStdExtZifencei,
                                          FeatureStdExtZba,
                                          FeatureStdExtZbb,
                                          FeatureStdExtZbs,
                                          FeatureStdExtZbkb,
                                          FeatureStdExtZcb,
                                          FeatureStdExtZcmp]>;

def ANDES_A25 : CapstoneProcessorModel<"andes-a25",
                                    NoSchedModel,
                                    [Feature32Bit,
                                     FeatureStdExtI,
                                     FeatureStdExtZicsr,
                                     FeatureStdExtZifencei,
                                     FeatureStdExtM,
                                     FeatureStdExtA,
                                     FeatureStdExtF,
                                     FeatureStdExtD,
                                     FeatureStdExtC,
                                     FeatureVendorXAndesPerf]>;

def ANDES_AX25 : CapstoneProcessorModel<"andes-ax25",
                                     NoSchedModel,
                                     [Feature64Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC,
                                      FeatureVendorXAndesPerf]>;

defvar Andes45TuneFeatures = [TuneAndes45,
                              TuneNoDefaultUnroll,
                              TuneShortForwardBranchOpt,
                              TunePostRAScheduler];

def ANDES_45 : CapstoneTuneProcessorModel<"andes-45-series",
                                       Andes45Model, Andes45TuneFeatures>;

def ANDES_N45 : CapstoneProcessorModel<"andes-n45",
                                    Andes45Model,
                                    [Feature32Bit,
                                     FeatureStdExtI,
                                     FeatureStdExtZicsr,
                                     FeatureStdExtZifencei,
                                     FeatureStdExtM,
                                     FeatureStdExtA,
                                     FeatureStdExtF,
                                     FeatureStdExtD,
                                     FeatureStdExtC,
                                     FeatureVendorXAndesPerf],
                                    Andes45TuneFeatures>;

def ANDES_NX45 : CapstoneProcessorModel<"andes-nx45",
                                     Andes45Model,
                                     [Feature64Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC,
                                      FeatureVendorXAndesPerf],
                                     Andes45TuneFeatures>;

def ANDES_A45 : CapstoneProcessorModel<"andes-a45",
                                    Andes45Model,
                                    [Feature32Bit,
                                     FeatureStdExtI,
                                     FeatureStdExtZicsr,
                                     FeatureStdExtZifencei,
                                     FeatureStdExtM,
                                     FeatureStdExtA,
                                     FeatureStdExtF,
                                     FeatureStdExtD,
                                     FeatureStdExtC,
                                     FeatureVendorXAndesPerf],
                                    Andes45TuneFeatures>;

def ANDES_AX45 : CapstoneProcessorModel<"andes-ax45",
                                     Andes45Model,
                                     [Feature64Bit,
                                      FeatureStdExtI,
                                      FeatureStdExtZicsr,
                                      FeatureStdExtZifencei,
                                      FeatureStdExtM,
                                      FeatureStdExtA,
                                      FeatureStdExtF,
                                      FeatureStdExtD,
                                      FeatureStdExtC,
                                      FeatureVendorXAndesPerf],
                                     Andes45TuneFeatures>;

def ANDES_AX45MPV : CapstoneProcessorModel<"andes-ax45mpv",
                                        Andes45Model,
                                        [Feature64Bit,
                                         FeatureStdExtI,
                                         FeatureStdExtZicsr,
                                         FeatureStdExtZifencei,
                                         FeatureStdExtM,
                                         FeatureStdExtA,
                                         FeatureStdExtF,
                                         FeatureStdExtD,
                                         FeatureStdExtC,
                                         FeatureStdExtV,
                                         FeatureVendorXAndesPerf],
                                        Andes45TuneFeatures>;
