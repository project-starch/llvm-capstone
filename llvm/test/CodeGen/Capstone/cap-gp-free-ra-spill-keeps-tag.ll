; A capability the register allocator spills must come back as a capability, even
; when it lands in ra. That is issue S-14, and it was a live defect: two of seven
; MicroPython test-table sizes built a domain that died at startup with cause 24
; inside the synthesised __capstone_cap_init, and a third wrote an untagged pointer
; into a global that faulted three hundred tests later.
;
; WHY ra IS SPECIAL AND WHY THAT WAS NOT ENOUGH. gp-free keeps ra as a plain integer
; return address, because calls are jal/jalr within PCC and there is no tag to save,
; so the frame save of ra is an 8-byte SD and its restore an LD. That is correct for
; the RETURN ADDRESS. It is wrong for anything else, and c1 is an ordinary allocatable
; GPCR that sits last in the allocation order, so the allocator parks capabilities
; there exactly when a function is under pressure. Keying the 8-byte form on the
; physical register rather than on what it holds truncated such a spill to its address
; half and the tag died. The frame save carries FrameSetup and an allocator spill
; carries no flag, which separates the two without guessing.
;
; THE SHAPE OF THE TEST is the shape of __capstone_cap_init: a straight line of
; volatile capability stores, one per global, which is what the cap-global pass emits.
; Twenty globals is the smallest count that reaches ra here; below sixteen the frame
; needs no ra save at all and the defect cannot appear.
;
; BOTH FORMS MUST BE PRESENT. The 8-byte save of the return address proves the
; gp-free ABI is still taking its own arm, and the 16-byte spill proves an allocator
; spill of a capability is not taking it. A test that checked only the second would
; also pass if the ABI arm were removed altogether.
;
; WHAT THIS TEST DOES AND DOES NOT COVER, because a test that passes either way is
; worse than none. It pins BOTH arms of the decision: the 8-byte save proves the
; gp-free ABI still treats the return address as an integer, and the 16-byte spill
; proves an ordinary capability spill does not take that arm. Remove the arm and the
; first check fails; widen it to every spill of ra and the second fails.
;
; It does NOT reproduce S-14 itself. The defect needs the physical register to reach
; storeRegToStackSlot, which happens at -O0 through RegAllocFast and not at -O2, where
; the spiller passes a virtual register and the arm cannot fire. Three shapes were
; tried at -O0 and none put a spilled capability in ra: a flat array of pointer
; globals at up to 300 entries, a call with 24 capability arguments, and inline asm
; with 28 live capability inputs. The real __capstone_cap_init reaches it because its
; initialisers are nested aggregates the cap-global pass walks recursively, which
; keeps a holder capability live across many leaves. The gate that does catch the
; defect is capstone/tests/capinit-scan.py, run over a built domain image, which is
; also what issue S-14 asks for: gate on the pattern, not on a geometry number.
;
; WHICH ABI CARRIES THE CHECK. capstoneGpFreeAbiActive() is true under gp-free and
; under gp-captable, so both take the arm and one reproducer locks it for both. Only
; gp-free is checked, because this shape never reaches ra under gp-captable at any
; size: globals come from the gp table there, so fewer values are live at once.
; gp-captable is still compiled and verified below, so a change that breaks it is not
; silently allowed.
; RUN: llc -mtriple=capstone64 -capstone-gp-free -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=capstone64 -capstone-gp-captable -verify-machineinstrs < %s -o /dev/null

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

@t0 = addrspace(200) global i64 0
@t1 = addrspace(200) global i64 1
@t2 = addrspace(200) global i64 2
@t3 = addrspace(200) global i64 3
@t4 = addrspace(200) global i64 4
@t5 = addrspace(200) global i64 5
@t6 = addrspace(200) global i64 6
@t7 = addrspace(200) global i64 7
@t8 = addrspace(200) global i64 8
@t9 = addrspace(200) global i64 9
@t10 = addrspace(200) global i64 10
@t11 = addrspace(200) global i64 11
@t12 = addrspace(200) global i64 12
@t13 = addrspace(200) global i64 13
@t14 = addrspace(200) global i64 14
@t15 = addrspace(200) global i64 15
@t16 = addrspace(200) global i64 16
@t17 = addrspace(200) global i64 17
@t18 = addrspace(200) global i64 18
@t19 = addrspace(200) global i64 19
@slot0 = addrspace(200) global ptr addrspace(200) null
@slot1 = addrspace(200) global ptr addrspace(200) null
@slot2 = addrspace(200) global ptr addrspace(200) null
@slot3 = addrspace(200) global ptr addrspace(200) null
@slot4 = addrspace(200) global ptr addrspace(200) null
@slot5 = addrspace(200) global ptr addrspace(200) null
@slot6 = addrspace(200) global ptr addrspace(200) null
@slot7 = addrspace(200) global ptr addrspace(200) null
@slot8 = addrspace(200) global ptr addrspace(200) null
@slot9 = addrspace(200) global ptr addrspace(200) null
@slot10 = addrspace(200) global ptr addrspace(200) null
@slot11 = addrspace(200) global ptr addrspace(200) null
@slot12 = addrspace(200) global ptr addrspace(200) null
@slot13 = addrspace(200) global ptr addrspace(200) null
@slot14 = addrspace(200) global ptr addrspace(200) null
@slot15 = addrspace(200) global ptr addrspace(200) null
@slot16 = addrspace(200) global ptr addrspace(200) null
@slot17 = addrspace(200) global ptr addrspace(200) null
@slot18 = addrspace(200) global ptr addrspace(200) null
@slot19 = addrspace(200) global ptr addrspace(200) null

define void @capinit_shape() {
  store volatile ptr addrspace(200) @t0, ptr addrspace(200) @slot0
  store volatile ptr addrspace(200) @t1, ptr addrspace(200) @slot1
  store volatile ptr addrspace(200) @t2, ptr addrspace(200) @slot2
  store volatile ptr addrspace(200) @t3, ptr addrspace(200) @slot3
  store volatile ptr addrspace(200) @t4, ptr addrspace(200) @slot4
  store volatile ptr addrspace(200) @t5, ptr addrspace(200) @slot5
  store volatile ptr addrspace(200) @t6, ptr addrspace(200) @slot6
  store volatile ptr addrspace(200) @t7, ptr addrspace(200) @slot7
  store volatile ptr addrspace(200) @t8, ptr addrspace(200) @slot8
  store volatile ptr addrspace(200) @t9, ptr addrspace(200) @slot9
  store volatile ptr addrspace(200) @t10, ptr addrspace(200) @slot10
  store volatile ptr addrspace(200) @t11, ptr addrspace(200) @slot11
  store volatile ptr addrspace(200) @t12, ptr addrspace(200) @slot12
  store volatile ptr addrspace(200) @t13, ptr addrspace(200) @slot13
  store volatile ptr addrspace(200) @t14, ptr addrspace(200) @slot14
  store volatile ptr addrspace(200) @t15, ptr addrspace(200) @slot15
  store volatile ptr addrspace(200) @t16, ptr addrspace(200) @slot16
  store volatile ptr addrspace(200) @t17, ptr addrspace(200) @slot17
  store volatile ptr addrspace(200) @t18, ptr addrspace(200) @slot18
  store volatile ptr addrspace(200) @t19, ptr addrspace(200) @slot19
  ret void
}

; CHECK-LABEL: capinit_shape:
; The return address, saved as the integer it is.
; CHECK:      sd ra, {{.*}} 8-byte Folded Spill
; A capability the allocator put in ra, saved as the capability it is. An SD here
; keeps the address bits and drops the tag, and the next access through ra faults.
; CHECK:      stc ra, {{.*}} 16-byte Folded Spill
; CHECK:      ret
