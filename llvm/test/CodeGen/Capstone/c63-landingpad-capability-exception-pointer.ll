; C-63: a landingpad whose exception pointer is a capability could not be selected.
; visitLandingPad copied the exception register at the address-space-0 pointer type
; (i64) and zero-extended it to the landingpad's c128 ("Invalid ZERO_EXTEND!"). The
; register contract is now: under a capability personality (address space 200) the
; exception pointer arrives in C10, the capability half of a0, and the selector in
; X11; the landing pad takes C10 live-in at the capability class and copies it at
; its own type. A domain's unwinder has to honour this contract. None exists yet, so
; C++ with exceptions compiles but does not link (ISSUES C-63).
; RUN: llc -mtriple=capstone64-unknown-elf -O0 -verify-machineinstrs < %s -o /dev/null
; RUN: llc -mtriple=capstone64-unknown-elf -O2 -verify-machineinstrs < %s -o /dev/null
; RUN: llc -mtriple=capstone64-unknown-elf -O2 -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=capstone64-unknown-elf -O2 < %s | FileCheck %s --check-prefix=ASM

; MIR-LABEL: name: catch_one
; MIR: (landing-pad):
; MIR-NEXT: liveins: $c10, $x11
; MIR: %{{[0-9]+}}:gpr = COPY {{(killed )?}}$x11
; MIR: %{{[0-9]+}}:gpcr = COPY {{(killed )?}}$c10

; The exception object reaches __cxa_begin_catch as a capability, unchanged.
; ASM-LABEL: catch_one:
; ASM: .cfi_lsda
; ASM-NOT: zext
; ASM: cjalr ra,

; MIR-LABEL: name: cleanup_only
; MIR: (landing-pad):
; MIR-NEXT: liveins: $c10, $x11

declare void @g() addrspace(200)
declare ptr addrspace(200) @__cxa_begin_catch(ptr addrspace(200)) addrspace(200)
declare void @__cxa_end_catch() addrspace(200)
declare i32 @__gxx_personality_v0(...) addrspace(200)

define i32 @catch_one() addrspace(200) personality ptr addrspace(200) @__gxx_personality_v0 {
entry:
  invoke addrspace(200) void @g() to label %c unwind label %l
c:
  ret i32 0
l:
  %lp = landingpad { ptr addrspace(200), i32 } catch ptr addrspace(200) null
  %exn = extractvalue { ptr addrspace(200), i32 } %lp, 0
  %sel = extractvalue { ptr addrspace(200), i32 } %lp, 1
  %obj = call addrspace(200) ptr addrspace(200) @__cxa_begin_catch(ptr addrspace(200) %exn)
  call addrspace(200) void @__cxa_end_catch()
  ret i32 %sel
}

; The registry's reproducer: a cleanup landingpad whose result is unused.
define i32 @cleanup_only() addrspace(200) personality ptr addrspace(200) @__gxx_personality_v0 {
entry:
  invoke addrspace(200) void @g() to label %c unwind label %l
c:
  ret i32 0
l:
  %0 = landingpad { ptr addrspace(200), i32 } cleanup
  ret i32 1
}
