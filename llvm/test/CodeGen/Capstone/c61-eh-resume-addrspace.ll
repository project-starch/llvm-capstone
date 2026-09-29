; C-61: `DwarfEHPrepare` built `_Unwind_Resume`'s parameter -- and the PHI that
; feeds it -- with `PointerType::getUnqual`, i.e. address space 0, while the
; exception object on this target is a capability in addrspace(200) (element 0
; of the `resume` aggregate).  The call's argument then disagreed with the
; callee's signature and clang ABORTED in CallInst::init, "Calling a function
; with a bad signature!".  Any C++ with exceptions enabled hit it, and
; exceptions are on by default -- a destructor live across a call is enough, no
; try/catch required.
;
; WHY THE CHECKS LOOK AT THE TYPE AND NOT ONLY AT SURVIVAL.  The pre-fix
; compiler ABORTS on this file, so a bare "it compiled" test would pass the
; moment the crash went away -- including if the exception object were silently
; put in the wrong address space.  These checks pin the address space at both
; sites the pass constructs a pointer type.
;
; THE SECOND FUNCTION IS NOT A DUPLICATE.  `DwarfEHPrepare` has two paths: with
; ONE resume it appends the call to that block (the shape clang emits for the
; C++ reproducer, since it merges landingpads itself), and with SEVERAL it
; creates a shared block with an `exn.obj` PHI.  The PHI is the second site and
; the single-resume path never reaches it.
;
; RUN: llc -mtriple=capstone64 -O0 -stop-after=dwarf-eh-prepare -o - %s \
; RUN:   | FileCheck %s
; RUN: llc -mtriple=capstone64 -O2 -stop-after=dwarf-eh-prepare -o - %s \
; RUN:   | FileCheck %s
;
; THERE IS DELIBERATELY NO FULL-CODEGEN RUN LINE, and that is a statement about
; the target rather than about this test.  This fix makes `DwarfEHPrepare`
; produce correctly typed IR, which is all it can do; running the result through
; instruction selection still aborts with
;   SelectionDAG.cpp: Assertion `VT.isInteger() && N1.getValueType().isInteger()
;   && "Invalid ZERO_EXTEND!"'
; on a `landingpad`, at -O0 and -O2 alike.  That is a SEPARATE, PRE-EXISTING
; defect: it reproduces on stock dev with a function that has a landingpad and
; NO resume at all, where this pass returns early and changes nothing.  So C++
; exceptions are not usable on this target yet, C-61 is the first of at least
; two layers, and a full-codegen line here would pin the wrong defect and fail
; for a reason this change cannot fix.

declare void @g() addrspace(200)
declare void @h() addrspace(200)
declare i32 @__gxx_personality_v0(...) addrspace(200)

; The single-resume path: the call is appended to the resume's own block.
; CHECK-LABEL: define void @single_resume
; CHECK: call addrspace(200) void @_Unwind_Resume(ptr addrspace(200) %exn.obj)
define void @single_resume() addrspace(200) personality ptr addrspace(200) @__gxx_personality_v0 {
entry:
  invoke addrspace(200) void @g()
          to label %cont unwind label %lpad

cont:
  ret void

lpad:
  %0 = landingpad { ptr addrspace(200), i32 }
          cleanup
  resume { ptr addrspace(200), i32 } %0
}

; The multi-resume path: a shared unwind block whose PHI is the second site.
; CHECK-LABEL: define void @two_resumes
; CHECK: %exn.obj = phi ptr addrspace(200)
; CHECK: call addrspace(200) void @_Unwind_Resume(ptr addrspace(200) %exn.obj)
define void @two_resumes(i1 %c) addrspace(200) personality ptr addrspace(200) @__gxx_personality_v0 {
entry:
  br i1 %c, label %a, label %b

a:
  invoke addrspace(200) void @g()
          to label %cont unwind label %lpad1

b:
  invoke addrspace(200) void @h()
          to label %cont unwind label %lpad2

cont:
  ret void

lpad1:
  %0 = landingpad { ptr addrspace(200), i32 }
          cleanup
  resume { ptr addrspace(200), i32 } %0

lpad2:
  %1 = landingpad { ptr addrspace(200), i32 }
          cleanup
  resume { ptr addrspace(200), i32 } %1
}

; The declaration the pass creates carries the same address space.
; CHECK: declare void @_Unwind_Resume(ptr addrspace(200))
