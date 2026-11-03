; A dense switch has a different inline cost when jump tables are forbidden.
; Keep this independent of capability-pointer costs: changing only the switch
; policy must cross the threshold while the address-space-200 loads stay put.
; The SDK's no-jump-tables policy can therefore explain an inlining difference
; from a target whose driver permits tables, without a capability cost bug.
;
; RUN: opt -mtriple=capstone64 -passes='cgscc(inline)' -inline-threshold=100 -S %s | FileCheck %s --check-prefix=TABLE
; RUN: sed 's/"no-jump-tables"="false"/"no-jump-tables"="true"/' %s | opt -mtriple=capstone64 -passes='cgscc(inline)' -inline-threshold=100 -S | FileCheck %s --check-prefix=COMPARE
;
; RUN: opt -mtriple=capstone64 -capstone-gp-captable -passes='cgscc(inline)' -inline-threshold=100 -S %s | FileCheck %s --check-prefix=COMPARE
;
; TABLE-LABEL: define i64 @caller(
; TABLE-NOT: call{{.*}}@dispatch
; TABLE: switch i32
; TABLE: ret i64
; COMPARE-LABEL: define i64 @caller(
; COMPARE: call addrspace(200) i64 @dispatch
; COMPARE-NEXT: ret i64

define i64 @dispatch(i32 %x, ptr addrspace(200) %p) #0 {
entry:
  switch i32 %x, label %invalid [
    i32 0, label %c0
    i32 1, label %c1
    i32 2, label %c2
    i32 3, label %c3
    i32 4, label %c4
    i32 5, label %c5
    i32 6, label %c6
    i32 7, label %c7
  ]
c0:
  %p0 = getelementptr i64, ptr addrspace(200) %p, i64 0
  %v0 = load i64, ptr addrspace(200) %p0
  ret i64 %v0
c1:
  %p1 = getelementptr i64, ptr addrspace(200) %p, i64 1
  %v1 = load i64, ptr addrspace(200) %p1
  ret i64 %v1
c2:
  %p2 = getelementptr i64, ptr addrspace(200) %p, i64 2
  %v2 = load i64, ptr addrspace(200) %p2
  ret i64 %v2
c3:
  %p3 = getelementptr i64, ptr addrspace(200) %p, i64 3
  %v3 = load i64, ptr addrspace(200) %p3
  ret i64 %v3
c4:
  %p4 = getelementptr i64, ptr addrspace(200) %p, i64 4
  %v4 = load i64, ptr addrspace(200) %p4
  ret i64 %v4
c5:
  %p5 = getelementptr i64, ptr addrspace(200) %p, i64 5
  %v5 = load i64, ptr addrspace(200) %p5
  ret i64 %v5
c6:
  %p6 = getelementptr i64, ptr addrspace(200) %p, i64 6
  %v6 = load i64, ptr addrspace(200) %p6
  ret i64 %v6
c7:
  %p7 = getelementptr i64, ptr addrspace(200) %p, i64 7
  %v7 = load i64, ptr addrspace(200) %p7
  ret i64 %v7
invalid:
  unreachable
}

define i64 @caller(i32 %x, ptr addrspace(200) %p) #0 {
  %v = call i64 @dispatch(i32 %x, ptr addrspace(200) %p)
  ret i64 %v
}

attributes #0 = { "no-jump-tables"="false" }
