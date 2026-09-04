*** IR Dump After Loop Strength Reduction (loop-reduce) ***
; Preheader:
if.then:                                          ; preds = %entry
  %a2 = getelementptr inbounds nuw i8, ptr addrspace(200) %pWC, i64 48
  %4 = load ptr addrspace(200), ptr addrspace(200) %a2, align 16, !tbaa !35
  %5 = zext nneg i32 %3 to i64
  %6 = mul nuw nsw i64 %5, 80
  %7 = add nsw i64 %6, -80
  br label %while.cond

; Loop:
while.cond:                                       ; preds = %if.end22, %if.then
  %lsr.iv = phi i64 [ %lsr.iv.next, %if.end22 ], [ %7, %if.then ]
  %a.0 = phi ptr addrspace(200) [ %4, %if.then ], [ %incdec.ptr, %if.end22 ]
  %scevgep = getelementptr i8, ptr addrspace(200) null, i64 %lsr.iv
  %wtFlags = getelementptr inbounds nuw i8, ptr addrspace(200) %a.0, i64 34
  %8 = load i16, ptr addrspace(200) %wtFlags, align 2, !tbaa !36
  %9 = and i16 %8, 1
  %tobool.not = icmp eq i16 %9, 0
  br i1 %tobool.not, label %if.end, label %if.then5

if.then5:                                         ; preds = %while.cond
  %10 = load ptr addrspace(200), ptr addrspace(200) %a.0, align 16, !tbaa !40
  %tobool.not.i = icmp eq ptr addrspace(200) %10, null
  br i1 %tobool.not.i, label %if.end, label %if.then.i

if.then.i:                                        ; preds = %if.then5
  call fastcc addrspace(200) void @sqlite3ExprDeleteNN(ptr addrspace(200) noundef %2, ptr addrspace(200) noundef %10) #42
  br label %if.end

if.end:                                           ; preds = %if.then.i, %if.then5, %while.cond
  %11 = load i16, ptr addrspace(200) %wtFlags, align 2, !tbaa !36
  %12 = and i16 %11, 48
  %tobool9.not = icmp eq i16 %12, 0
  br i1 %tobool9.not, label %if.end18, label %if.then10

if.then10:                                        ; preds = %if.end
  %u16 = getelementptr inbounds nuw i8, ptr addrspace(200) %a.0, i64 48
  %13 = load ptr addrspace(200), ptr addrspace(200) %u16, align 16, !tbaa !41
  call fastcc addrspace(200) void @sqlite3WhereClauseClear(ptr addrspace(200) noundef %13) #42
  call addrspace(200) void @sqlite3DbFree(ptr addrspace(200) noundef %2, ptr addrspace(200) noundef %13) #42
  br label %if.end18

if.end18:                                         ; preds = %if.then10, %if.end
  %cmp19 = icmp eq ptr addrspace(200) %scevgep, null
  br i1 %cmp19, label %if.end23.loopexit, label %if.end22

if.end22:                                         ; preds = %if.end18
  %incdec.ptr = getelementptr inbounds nuw i8, ptr addrspace(200) %a.0, i64 80
  %lsr.iv.next = add i64 %lsr.iv, -80
  br label %while.cond, !llvm.loop !42

; Exit blocks
if.end23.loopexit:                                ; preds = %if.end18
  br label %if.end23
