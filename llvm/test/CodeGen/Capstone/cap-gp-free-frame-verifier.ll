; The gp-free and gp-captable ABIs must produce machine code the VERIFIER accepts.
;
; They did not, and nothing noticed for a simple reason: of 58 tests here, 34 run
; -verify-machineinstrs and 2 use these ABIs, and the two sets did not intersect.
; The default ABI never takes the arm that was wrong, and the ABI every real
; domain builds with was never verified. Running the verifier over musl fired on
; 60 of 60 sampled files.
;
; What it caught: `ra` under gp-free is a plain integer return address -- calls
; are jal/jalr within PCC, so there is no tag -- and the spill was emitted as
;
;     frame-setup SD killed $c1, $c2, 0 :: (store (s128) into %stack.0)
;
; an SD naming a CAPABILITY register, with a memory operand claiming 16 bytes for
; an instruction that writes 8. It assembled and ran, because c1 and x1 print the
; same name and the reload matched, which is why only the verifier could see it.
;
; A THREE-LINE FUNCTION IS ENOUGH, and that is the point: it needs only a call,
; so that ra is saved. cap-gp-captable.ll exists and passes precisely because it
; has no call in it.
;
; RUN: llc -mtriple=capstone64 -verify-machineinstrs < %s | FileCheck %s --check-prefix=DEFAULT
; RUN: llc -mtriple=capstone64 -capstone-gp-captable -verify-machineinstrs < %s | FileCheck %s --check-prefix=GPFREE
; RUN: llc -mtriple=capstone64 -capstone-gp-free -verify-machineinstrs < %s | FileCheck %s --check-prefix=GPFREE

target datalayout = "e-m:e-pf200:128:128:128:64-p:64:64-i64:64-i128:128-n32:64-S128-A200-P200-G200"

declare void @sink()

; The default ABI keeps ra a capability, so it spills with stc and reloads with ldc.
; DEFAULT-LABEL: saves_ra:
; DEFAULT: stc ra,
; DEFAULT: ldc ra,
; DEFAULT: cjalr zero, 0(ra)

; gp-free keeps it an integer: an 8-byte sd/ld pair, and the spill comment says
; 8-byte rather than the 16 it used to claim.
; GPFREE-LABEL: saves_ra:
; GPFREE: sd ra, {{[0-9]+}}(sp){{.*}}8-byte Folded Spill
; GPFREE-NOT: stc ra,
; GPFREE: ld ra, {{[0-9]+}}(sp){{.*}}8-byte Folded Reload
define void @saves_ra() {
  call void @sink()
  ret void
}
