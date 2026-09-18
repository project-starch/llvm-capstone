; RUN: llc -mtriple=capstone64 -mattr=+a -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -mattr=+a -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=capstone64 -mattr=+a -verify-machineinstrs -filetype=obj < %s -o /dev/null

; Atomic data are integers; AS200 addresses must retain capability authority.
; Check both the instruction and the address register class before allocation.

define i32 @add32(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: add32:
; CHECK: amoadd.w.aqrl
; MIR-LABEL: name: add32
; MIR: %[[ADDR_ADD32:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOADD_W_AQ_RL_CAP %[[ADDR_ADD32]],
  %old = atomicrmw add ptr addrspace(200) %p, i32 %value seq_cst
  ret i32 %old
}

define i32 @sub32(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: sub32:
; CHECK: amoadd.w.aqrl
; MIR-LABEL: name: sub32
; MIR: %[[ADDR_SUB32:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOADD_W_AQ_RL_CAP %[[ADDR_SUB32]],
  %old = atomicrmw sub ptr addrspace(200) %p, i32 %value seq_cst
  ret i32 %old
}

define i32 @and32(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: and32:
; CHECK: amoand.w.aqrl
; MIR-LABEL: name: and32
; MIR: %[[ADDR_AND32:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOAND_W_AQ_RL_CAP %[[ADDR_AND32]],
  %old = atomicrmw and ptr addrspace(200) %p, i32 %value seq_cst
  ret i32 %old
}

define i32 @or32(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: or32:
; CHECK: amoor.w.aqrl
; MIR-LABEL: name: or32
; MIR: %[[ADDR_OR32:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOOR_W_AQ_RL_CAP %[[ADDR_OR32]],
  %old = atomicrmw or ptr addrspace(200) %p, i32 %value seq_cst
  ret i32 %old
}

define i32 @xor32(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: xor32:
; CHECK: amoxor.w.aqrl
; MIR-LABEL: name: xor32
; MIR: %[[ADDR_XOR32:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOXOR_W_AQ_RL_CAP %[[ADDR_XOR32]],
  %old = atomicrmw xor ptr addrspace(200) %p, i32 %value seq_cst
  ret i32 %old
}

define i32 @xchg32(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: xchg32:
; CHECK: amoswap.w.aqrl
; MIR-LABEL: name: xchg32
; MIR: %[[ADDR_XCHG32:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOSWAP_W_AQ_RL_CAP %[[ADDR_XCHG32]],
  %old = atomicrmw xchg ptr addrspace(200) %p, i32 %value seq_cst
  ret i32 %old
}

define i32 @min32(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: min32:
; CHECK: amomin.w.aqrl
; MIR-LABEL: name: min32
; MIR: %[[ADDR_MIN32:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOMIN_W_AQ_RL_CAP %[[ADDR_MIN32]],
  %old = atomicrmw min ptr addrspace(200) %p, i32 %value seq_cst
  ret i32 %old
}

define i32 @max32(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: max32:
; CHECK: amomax.w.aqrl
; MIR-LABEL: name: max32
; MIR: %[[ADDR_MAX32:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOMAX_W_AQ_RL_CAP %[[ADDR_MAX32]],
  %old = atomicrmw max ptr addrspace(200) %p, i32 %value seq_cst
  ret i32 %old
}

define i32 @umin32(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: umin32:
; CHECK: amominu.w.aqrl
; MIR-LABEL: name: umin32
; MIR: %[[ADDR_UMIN32:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOMINU_W_AQ_RL_CAP %[[ADDR_UMIN32]],
  %old = atomicrmw umin ptr addrspace(200) %p, i32 %value seq_cst
  ret i32 %old
}

define i32 @umax32(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: umax32:
; CHECK: amomaxu.w.aqrl
; MIR-LABEL: name: umax32
; MIR: %[[ADDR_UMAX32:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOMAXU_W_AQ_RL_CAP %[[ADDR_UMAX32]],
  %old = atomicrmw umax ptr addrspace(200) %p, i32 %value seq_cst
  ret i32 %old
}

define i32 @nand32(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: nand32:
; CHECK: lr.w.aqrl
; CHECK: sc.w.rl
; MIR-LABEL: name: nand32
; MIR: %[[ADDR_NAND32:[0-9]+]]:gpcr = COPY $c10
; MIR: PseudoAtomicLoadNand32_CAP %[[ADDR_NAND32]],
  %old = atomicrmw nand ptr addrspace(200) %p, i32 %value seq_cst
  ret i32 %old
}

define i32 @cmpxchg32(ptr addrspace(200) %p, i32 %expected, i32 %desired) addrspace(200) {
; CHECK-LABEL: cmpxchg32:
; CHECK-NOT: mv
; CHECK: lr.w.aqrl
; CHECK: sc.w.rl
; MIR-LABEL: name: cmpxchg32
; MIR: %[[ADDR_CMPXCHG32:[0-9]+]]:gpcr = COPY $c10
; MIR: PseudoCmpXchg32_CAP %[[ADDR_CMPXCHG32]],
  %pair = cmpxchg ptr addrspace(200) %p, i32 %expected, i32 %desired seq_cst seq_cst
  %old = extractvalue { i32, i1 } %pair, 0
  ret i32 %old
}

define i64 @add64(ptr addrspace(200) %p, i64 %value) addrspace(200) {
; CHECK-LABEL: add64:
; CHECK: amoadd.d.aqrl
; MIR-LABEL: name: add64
; MIR: %[[ADDR_ADD64:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOADD_D_AQ_RL_CAP %[[ADDR_ADD64]],
  %old = atomicrmw add ptr addrspace(200) %p, i64 %value seq_cst
  ret i64 %old
}

define i64 @sub64(ptr addrspace(200) %p, i64 %value) addrspace(200) {
; CHECK-LABEL: sub64:
; CHECK: amoadd.d.aqrl
; MIR-LABEL: name: sub64
; MIR: %[[ADDR_SUB64:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOADD_D_AQ_RL_CAP %[[ADDR_SUB64]],
  %old = atomicrmw sub ptr addrspace(200) %p, i64 %value seq_cst
  ret i64 %old
}

define i64 @and64(ptr addrspace(200) %p, i64 %value) addrspace(200) {
; CHECK-LABEL: and64:
; CHECK: amoand.d.aqrl
; MIR-LABEL: name: and64
; MIR: %[[ADDR_AND64:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOAND_D_AQ_RL_CAP %[[ADDR_AND64]],
  %old = atomicrmw and ptr addrspace(200) %p, i64 %value seq_cst
  ret i64 %old
}

define i64 @or64(ptr addrspace(200) %p, i64 %value) addrspace(200) {
; CHECK-LABEL: or64:
; CHECK: amoor.d.aqrl
; MIR-LABEL: name: or64
; MIR: %[[ADDR_OR64:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOOR_D_AQ_RL_CAP %[[ADDR_OR64]],
  %old = atomicrmw or ptr addrspace(200) %p, i64 %value seq_cst
  ret i64 %old
}

define i64 @xor64(ptr addrspace(200) %p, i64 %value) addrspace(200) {
; CHECK-LABEL: xor64:
; CHECK: amoxor.d.aqrl
; MIR-LABEL: name: xor64
; MIR: %[[ADDR_XOR64:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOXOR_D_AQ_RL_CAP %[[ADDR_XOR64]],
  %old = atomicrmw xor ptr addrspace(200) %p, i64 %value seq_cst
  ret i64 %old
}

define i64 @xchg64(ptr addrspace(200) %p, i64 %value) addrspace(200) {
; CHECK-LABEL: xchg64:
; CHECK: amoswap.d.aqrl
; MIR-LABEL: name: xchg64
; MIR: %[[ADDR_XCHG64:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOSWAP_D_AQ_RL_CAP %[[ADDR_XCHG64]],
  %old = atomicrmw xchg ptr addrspace(200) %p, i64 %value seq_cst
  ret i64 %old
}

define i64 @min64(ptr addrspace(200) %p, i64 %value) addrspace(200) {
; CHECK-LABEL: min64:
; CHECK: amomin.d.aqrl
; MIR-LABEL: name: min64
; MIR: %[[ADDR_MIN64:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOMIN_D_AQ_RL_CAP %[[ADDR_MIN64]],
  %old = atomicrmw min ptr addrspace(200) %p, i64 %value seq_cst
  ret i64 %old
}

define i64 @max64(ptr addrspace(200) %p, i64 %value) addrspace(200) {
; CHECK-LABEL: max64:
; CHECK: amomax.d.aqrl
; MIR-LABEL: name: max64
; MIR: %[[ADDR_MAX64:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOMAX_D_AQ_RL_CAP %[[ADDR_MAX64]],
  %old = atomicrmw max ptr addrspace(200) %p, i64 %value seq_cst
  ret i64 %old
}

define i64 @umin64(ptr addrspace(200) %p, i64 %value) addrspace(200) {
; CHECK-LABEL: umin64:
; CHECK: amominu.d.aqrl
; MIR-LABEL: name: umin64
; MIR: %[[ADDR_UMIN64:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOMINU_D_AQ_RL_CAP %[[ADDR_UMIN64]],
  %old = atomicrmw umin ptr addrspace(200) %p, i64 %value seq_cst
  ret i64 %old
}

define i64 @umax64(ptr addrspace(200) %p, i64 %value) addrspace(200) {
; CHECK-LABEL: umax64:
; CHECK: amomaxu.d.aqrl
; MIR-LABEL: name: umax64
; MIR: %[[ADDR_UMAX64:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOMAXU_D_AQ_RL_CAP %[[ADDR_UMAX64]],
  %old = atomicrmw umax ptr addrspace(200) %p, i64 %value seq_cst
  ret i64 %old
}

define i64 @nand64(ptr addrspace(200) %p, i64 %value) addrspace(200) {
; CHECK-LABEL: nand64:
; CHECK: lr.d.aqrl
; CHECK: sc.d.rl
; MIR-LABEL: name: nand64
; MIR: %[[ADDR_NAND64:[0-9]+]]:gpcr = COPY $c10
; MIR: PseudoAtomicLoadNand64_CAP %[[ADDR_NAND64]],
  %old = atomicrmw nand ptr addrspace(200) %p, i64 %value seq_cst
  ret i64 %old
}

define i64 @cmpxchg64(ptr addrspace(200) %p, i64 %expected, i64 %desired) addrspace(200) {
; CHECK-LABEL: cmpxchg64:
; CHECK-NOT: mv
; CHECK: lr.d.aqrl
; CHECK: sc.d.rl
; MIR-LABEL: name: cmpxchg64
; MIR: %[[ADDR_CMPXCHG64:[0-9]+]]:gpcr = COPY $c10
; MIR: PseudoCmpXchg64_CAP %[[ADDR_CMPXCHG64]],
  %pair = cmpxchg ptr addrspace(200) %p, i64 %expected, i64 %desired seq_cst seq_cst
  %old = extractvalue { i64, i1 } %pair, 0
  ret i64 %old
}

define i32 @add_monotonic_as0(ptr %p, i32 %value) addrspace(200) {
; CHECK-LABEL: add_monotonic_as0:
; CHECK: amoadd.w{{[ \t]}}
; MIR-LABEL: name: add_monotonic_as0
; MIR: %[[ADDR_ADD_MONOTONIC_AS0:[0-9]+]]:gpr = COPY $x10
; MIR: AMOADD_W %[[ADDR_ADD_MONOTONIC_AS0]],
  %old = atomicrmw add ptr %p, i32 %value monotonic
  ret i32 %old
}

define i32 @add_acquire_as0(ptr %p, i32 %value) addrspace(200) {
; CHECK-LABEL: add_acquire_as0:
; CHECK: amoadd.w.aq{{[ \t]}}
; MIR-LABEL: name: add_acquire_as0
; MIR: %[[ADDR_ADD_ACQUIRE_AS0:[0-9]+]]:gpr = COPY $x10
; MIR: AMOADD_W_AQ %[[ADDR_ADD_ACQUIRE_AS0]],
  %old = atomicrmw add ptr %p, i32 %value acquire
  ret i32 %old
}

define i32 @add_release_as0(ptr %p, i32 %value) addrspace(200) {
; CHECK-LABEL: add_release_as0:
; CHECK: amoadd.w.rl{{[ \t]}}
; MIR-LABEL: name: add_release_as0
; MIR: %[[ADDR_ADD_RELEASE_AS0:[0-9]+]]:gpr = COPY $x10
; MIR: AMOADD_W_RL %[[ADDR_ADD_RELEASE_AS0]],
  %old = atomicrmw add ptr %p, i32 %value release
  ret i32 %old
}

define i32 @add_acq_rel_as0(ptr %p, i32 %value) addrspace(200) {
; CHECK-LABEL: add_acq_rel_as0:
; CHECK: amoadd.w.aqrl{{[ \t]}}
; MIR-LABEL: name: add_acq_rel_as0
; MIR: %[[ADDR_ADD_ACQ_REL_AS0:[0-9]+]]:gpr = COPY $x10
; MIR: AMOADD_W_AQ_RL %[[ADDR_ADD_ACQ_REL_AS0]],
  %old = atomicrmw add ptr %p, i32 %value acq_rel
  ret i32 %old
}

define i32 @add_monotonic_as200(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: add_monotonic_as200:
; CHECK: amoadd.w{{[ \t]}}
; MIR-LABEL: name: add_monotonic_as200
; MIR: %[[ADDR_ADD_MONOTONIC_AS200:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOADD_W_CAP %[[ADDR_ADD_MONOTONIC_AS200]],
  %old = atomicrmw add ptr addrspace(200) %p, i32 %value monotonic
  ret i32 %old
}

define i32 @add_acquire_as200(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: add_acquire_as200:
; CHECK: amoadd.w.aq{{[ \t]}}
; MIR-LABEL: name: add_acquire_as200
; MIR: %[[ADDR_ADD_ACQUIRE_AS200:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOADD_W_AQ_CAP %[[ADDR_ADD_ACQUIRE_AS200]],
  %old = atomicrmw add ptr addrspace(200) %p, i32 %value acquire
  ret i32 %old
}

define i32 @add_release_as200(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: add_release_as200:
; CHECK: amoadd.w.rl{{[ \t]}}
; MIR-LABEL: name: add_release_as200
; MIR: %[[ADDR_ADD_RELEASE_AS200:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOADD_W_RL_CAP %[[ADDR_ADD_RELEASE_AS200]],
  %old = atomicrmw add ptr addrspace(200) %p, i32 %value release
  ret i32 %old
}

define i32 @add_acq_rel_as200(ptr addrspace(200) %p, i32 %value) addrspace(200) {
; CHECK-LABEL: add_acq_rel_as200:
; CHECK: amoadd.w.aqrl{{[ \t]}}
; MIR-LABEL: name: add_acq_rel_as200
; MIR: %[[ADDR_ADD_ACQ_REL_AS200:[0-9]+]]:gpcr = COPY $c10
; MIR: AMOADD_W_AQ_RL_CAP %[[ADDR_ADD_ACQ_REL_AS200]],
  %old = atomicrmw add ptr addrspace(200) %p, i32 %value acq_rel
  ret i32 %old
}
