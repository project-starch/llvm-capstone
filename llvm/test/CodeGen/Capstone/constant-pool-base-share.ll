; RUN: llc -mtriple=capstone64-unknown-elf -mattr=+m,+a,+f,+d -target-abi lp64 \
; RUN:   -capstone-gp-free -capstone-image-gp -verify-machineinstrs < %s \
; RUN:   | FileCheck %s --implicit-check-not='%pcrel_hi(.LCPI1_1)' \
; RUN:       --implicit-check-not='%pcrel_hi(.LCPI1_255)'

target datalayout = "e-m:e-p:64:128-p200:128:128:128:64-i64:64-i128:128-n32:64-S128-ni:200-A200-P200-G200"

; Constant-pool base sharing under the image-gp ABI (CapstoneTargetObjectFile
; puts a function's mergeable entries contiguously into .rodata;
; CapstoneFoldMemOffset::rebaseConstantPool addresses them from the lowest
; candidate entry; Early MachineLICM + MachineCSE then merge the chains).
;
; The rule every case respects: no surviving capability cursor changes value.
; The one cursor the rewritten chains form, &cp#0, is one the program already
; forms through cp#0's own chain; every access address is unchanged
; (&cp#0 + off_i = &cp#i), the displacements are simm12, and both the base and
; the accesses lie inside the pool, inside gp's bounds.

; Four doubles: one auipc/addi/scc, four loads at the entries' byte offsets.
; The addi stays because the loads are at different offsets (the %pcrel_lo
; fold in CapstoneMergeBaseOffset needs one common offset).
; CHECK-LABEL: poly4:
; CHECK:       auipc [[HI:a[0-9]+]], %pcrel_hi(.LCPI0_0)
; CHECK:       addi [[ADDR:a[0-9]+]], [[HI]], %pcrel_lo(
; CHECK:       scc [[BASE:a[0-9]+]], gp, [[ADDR]]
; CHECK-NOT:   scc {{.*}}, gp,
; CHECK-DAG:   fld {{f[a-z0-9]+}}, 0([[BASE]])
; CHECK-DAG:   fld {{f[a-z0-9]+}}, 8([[BASE]])
; CHECK-DAG:   fld {{f[a-z0-9]+}}, 16([[BASE]])
; CHECK-DAG:   fld {{f[a-z0-9]+}}, 24([[BASE]])
; CHECK:       ret
define double @poly4(double %x) addrspace(200) nounwind {
entry:
  %m0 = fmul double %x, 0x3FD5555555555555
  %a0 = fadd double %m0, 0x3FE5555555555555
  %m1 = fmul double %a0, 0x3FC999999999999A
  %a1 = fadd double %m1, 0x3FF3333333333333
  ret double %a1
}

; Negative control, the simm12 bound: 260 doubles span 2080 bytes. Entries 0
; to 255 (byte offsets 0 to 2040) share cp#0's base; entries 256 to 259
; (offsets 2048 and up, outside a load's simm12) keep their own chain, each
; with its own auipc. The --implicit-check-not lines above reject a chain
; for entry 1 or entry 255 anywhere in the output.
; CHECK-LABEL: far_pool:
; CHECK:       auipc {{a[0-9]+}}, %pcrel_hi(.LCPI1_0)
; CHECK-DAG:   fld {{f[a-z0-9]+}}, 2040(
; CHECK-DAG:   auipc {{a[0-9]+}}, %pcrel_hi(.LCPI1_256)
; CHECK-DAG:   auipc {{a[0-9]+}}, %pcrel_hi(.LCPI1_259)
; CHECK:       ret
define double @far_pool(double %x) addrspace(200) nounwind {
entry:
  %m0 = fmul double %x, 0x3FD5555555555555
  %m1 = fmul double %m0, 0x3FD5655555555556
  %m2 = fmul double %m1, 0x3FD5755555555557
  %m3 = fmul double %m2, 0x3FD5855555555558
  %m4 = fmul double %m3, 0x3FD5955555555559
  %m5 = fmul double %m4, 0x3FD5A5555555555A
  %m6 = fmul double %m5, 0x3FD5B5555555555B
  %m7 = fmul double %m6, 0x3FD5C5555555555C
  %m8 = fmul double %m7, 0x3FD5D5555555555D
  %m9 = fmul double %m8, 0x3FD5E5555555555E
  %m10 = fmul double %m9, 0x3FD5F5555555555F
  %m11 = fmul double %m10, 0x3FD6055555555560
  %m12 = fmul double %m11, 0x3FD6155555555561
  %m13 = fmul double %m12, 0x3FD6255555555562
  %m14 = fmul double %m13, 0x3FD6355555555563
  %m15 = fmul double %m14, 0x3FD6455555555564
  %m16 = fmul double %m15, 0x3FD6555555555565
  %m17 = fmul double %m16, 0x3FD6655555555566
  %m18 = fmul double %m17, 0x3FD6755555555567
  %m19 = fmul double %m18, 0x3FD6855555555568
  %m20 = fmul double %m19, 0x3FD6955555555569
  %m21 = fmul double %m20, 0x3FD6A5555555556A
  %m22 = fmul double %m21, 0x3FD6B5555555556B
  %m23 = fmul double %m22, 0x3FD6C5555555556C
  %m24 = fmul double %m23, 0x3FD6D5555555556D
  %m25 = fmul double %m24, 0x3FD6E5555555556E
  %m26 = fmul double %m25, 0x3FD6F5555555556F
  %m27 = fmul double %m26, 0x3FD7055555555570
  %m28 = fmul double %m27, 0x3FD7155555555571
  %m29 = fmul double %m28, 0x3FD7255555555572
  %m30 = fmul double %m29, 0x3FD7355555555573
  %m31 = fmul double %m30, 0x3FD7455555555574
  %m32 = fmul double %m31, 0x3FD7555555555575
  %m33 = fmul double %m32, 0x3FD7655555555576
  %m34 = fmul double %m33, 0x3FD7755555555577
  %m35 = fmul double %m34, 0x3FD7855555555578
  %m36 = fmul double %m35, 0x3FD7955555555579
  %m37 = fmul double %m36, 0x3FD7A5555555557A
  %m38 = fmul double %m37, 0x3FD7B5555555557B
  %m39 = fmul double %m38, 0x3FD7C5555555557C
  %m40 = fmul double %m39, 0x3FD7D5555555557D
  %m41 = fmul double %m40, 0x3FD7E5555555557E
  %m42 = fmul double %m41, 0x3FD7F5555555557F
  %m43 = fmul double %m42, 0x3FD8055555555580
  %m44 = fmul double %m43, 0x3FD8155555555581
  %m45 = fmul double %m44, 0x3FD8255555555582
  %m46 = fmul double %m45, 0x3FD8355555555583
  %m47 = fmul double %m46, 0x3FD8455555555584
  %m48 = fmul double %m47, 0x3FD8555555555585
  %m49 = fmul double %m48, 0x3FD8655555555586
  %m50 = fmul double %m49, 0x3FD8755555555587
  %m51 = fmul double %m50, 0x3FD8855555555588
  %m52 = fmul double %m51, 0x3FD8955555555589
  %m53 = fmul double %m52, 0x3FD8A5555555558A
  %m54 = fmul double %m53, 0x3FD8B5555555558B
  %m55 = fmul double %m54, 0x3FD8C5555555558C
  %m56 = fmul double %m55, 0x3FD8D5555555558D
  %m57 = fmul double %m56, 0x3FD8E5555555558E
  %m58 = fmul double %m57, 0x3FD8F5555555558F
  %m59 = fmul double %m58, 0x3FD9055555555590
  %m60 = fmul double %m59, 0x3FD9155555555591
  %m61 = fmul double %m60, 0x3FD9255555555592
  %m62 = fmul double %m61, 0x3FD9355555555593
  %m63 = fmul double %m62, 0x3FD9455555555594
  %m64 = fmul double %m63, 0x3FD9555555555595
  %m65 = fmul double %m64, 0x3FD9655555555596
  %m66 = fmul double %m65, 0x3FD9755555555597
  %m67 = fmul double %m66, 0x3FD9855555555598
  %m68 = fmul double %m67, 0x3FD9955555555599
  %m69 = fmul double %m68, 0x3FD9A5555555559A
  %m70 = fmul double %m69, 0x3FD9B5555555559B
  %m71 = fmul double %m70, 0x3FD9C5555555559C
  %m72 = fmul double %m71, 0x3FD9D5555555559D
  %m73 = fmul double %m72, 0x3FD9E5555555559E
  %m74 = fmul double %m73, 0x3FD9F5555555559F
  %m75 = fmul double %m74, 0x3FDA0555555555A0
  %m76 = fmul double %m75, 0x3FDA1555555555A1
  %m77 = fmul double %m76, 0x3FDA2555555555A2
  %m78 = fmul double %m77, 0x3FDA3555555555A3
  %m79 = fmul double %m78, 0x3FDA4555555555A4
  %m80 = fmul double %m79, 0x3FDA5555555555A5
  %m81 = fmul double %m80, 0x3FDA6555555555A6
  %m82 = fmul double %m81, 0x3FDA7555555555A7
  %m83 = fmul double %m82, 0x3FDA8555555555A8
  %m84 = fmul double %m83, 0x3FDA9555555555A9
  %m85 = fmul double %m84, 0x3FDAA555555555AA
  %m86 = fmul double %m85, 0x3FDAB555555555AB
  %m87 = fmul double %m86, 0x3FDAC555555555AC
  %m88 = fmul double %m87, 0x3FDAD555555555AD
  %m89 = fmul double %m88, 0x3FDAE555555555AE
  %m90 = fmul double %m89, 0x3FDAF555555555AF
  %m91 = fmul double %m90, 0x3FDB0555555555B0
  %m92 = fmul double %m91, 0x3FDB1555555555B1
  %m93 = fmul double %m92, 0x3FDB2555555555B2
  %m94 = fmul double %m93, 0x3FDB3555555555B3
  %m95 = fmul double %m94, 0x3FDB4555555555B4
  %m96 = fmul double %m95, 0x3FDB5555555555B5
  %m97 = fmul double %m96, 0x3FDB6555555555B6
  %m98 = fmul double %m97, 0x3FDB7555555555B7
  %m99 = fmul double %m98, 0x3FDB8555555555B8
  %m100 = fmul double %m99, 0x3FDB9555555555B9
  %m101 = fmul double %m100, 0x3FDBA555555555BA
  %m102 = fmul double %m101, 0x3FDBB555555555BB
  %m103 = fmul double %m102, 0x3FDBC555555555BC
  %m104 = fmul double %m103, 0x3FDBD555555555BD
  %m105 = fmul double %m104, 0x3FDBE555555555BE
  %m106 = fmul double %m105, 0x3FDBF555555555BF
  %m107 = fmul double %m106, 0x3FDC0555555555C0
  %m108 = fmul double %m107, 0x3FDC1555555555C1
  %m109 = fmul double %m108, 0x3FDC2555555555C2
  %m110 = fmul double %m109, 0x3FDC3555555555C3
  %m111 = fmul double %m110, 0x3FDC4555555555C4
  %m112 = fmul double %m111, 0x3FDC5555555555C5
  %m113 = fmul double %m112, 0x3FDC6555555555C6
  %m114 = fmul double %m113, 0x3FDC7555555555C7
  %m115 = fmul double %m114, 0x3FDC8555555555C8
  %m116 = fmul double %m115, 0x3FDC9555555555C9
  %m117 = fmul double %m116, 0x3FDCA555555555CA
  %m118 = fmul double %m117, 0x3FDCB555555555CB
  %m119 = fmul double %m118, 0x3FDCC555555555CC
  %m120 = fmul double %m119, 0x3FDCD555555555CD
  %m121 = fmul double %m120, 0x3FDCE555555555CE
  %m122 = fmul double %m121, 0x3FDCF555555555CF
  %m123 = fmul double %m122, 0x3FDD0555555555D0
  %m124 = fmul double %m123, 0x3FDD1555555555D1
  %m125 = fmul double %m124, 0x3FDD2555555555D2
  %m126 = fmul double %m125, 0x3FDD3555555555D3
  %m127 = fmul double %m126, 0x3FDD4555555555D4
  %m128 = fmul double %m127, 0x3FDD5555555555D5
  %m129 = fmul double %m128, 0x3FDD6555555555D6
  %m130 = fmul double %m129, 0x3FDD7555555555D7
  %m131 = fmul double %m130, 0x3FDD8555555555D8
  %m132 = fmul double %m131, 0x3FDD9555555555D9
  %m133 = fmul double %m132, 0x3FDDA555555555DA
  %m134 = fmul double %m133, 0x3FDDB555555555DB
  %m135 = fmul double %m134, 0x3FDDC555555555DC
  %m136 = fmul double %m135, 0x3FDDD555555555DD
  %m137 = fmul double %m136, 0x3FDDE555555555DE
  %m138 = fmul double %m137, 0x3FDDF555555555DF
  %m139 = fmul double %m138, 0x3FDE0555555555E0
  %m140 = fmul double %m139, 0x3FDE1555555555E1
  %m141 = fmul double %m140, 0x3FDE2555555555E2
  %m142 = fmul double %m141, 0x3FDE3555555555E3
  %m143 = fmul double %m142, 0x3FDE4555555555E4
  %m144 = fmul double %m143, 0x3FDE5555555555E5
  %m145 = fmul double %m144, 0x3FDE6555555555E6
  %m146 = fmul double %m145, 0x3FDE7555555555E7
  %m147 = fmul double %m146, 0x3FDE8555555555E8
  %m148 = fmul double %m147, 0x3FDE9555555555E9
  %m149 = fmul double %m148, 0x3FDEA555555555EA
  %m150 = fmul double %m149, 0x3FDEB555555555EB
  %m151 = fmul double %m150, 0x3FDEC555555555EC
  %m152 = fmul double %m151, 0x3FDED555555555ED
  %m153 = fmul double %m152, 0x3FDEE555555555EE
  %m154 = fmul double %m153, 0x3FDEF555555555EF
  %m155 = fmul double %m154, 0x3FDF0555555555F0
  %m156 = fmul double %m155, 0x3FDF1555555555F1
  %m157 = fmul double %m156, 0x3FDF2555555555F2
  %m158 = fmul double %m157, 0x3FDF3555555555F3
  %m159 = fmul double %m158, 0x3FDF4555555555F4
  %m160 = fmul double %m159, 0x3FDF5555555555F5
  %m161 = fmul double %m160, 0x3FDF6555555555F6
  %m162 = fmul double %m161, 0x3FDF7555555555F7
  %m163 = fmul double %m162, 0x3FDF8555555555F8
  %m164 = fmul double %m163, 0x3FDF9555555555F9
  %m165 = fmul double %m164, 0x3FDFA555555555FA
  %m166 = fmul double %m165, 0x3FDFB555555555FB
  %m167 = fmul double %m166, 0x3FDFC555555555FC
  %m168 = fmul double %m167, 0x3FDFD555555555FD
  %m169 = fmul double %m168, 0x3FDFE555555555FE
  %m170 = fmul double %m169, 0x3FDFF555555555FF
  %m171 = fmul double %m170, 0x3FE0055555555600
  %m172 = fmul double %m171, 0x3FE0155555555601
  %m173 = fmul double %m172, 0x3FE0255555555602
  %m174 = fmul double %m173, 0x3FE0355555555603
  %m175 = fmul double %m174, 0x3FE0455555555604
  %m176 = fmul double %m175, 0x3FE0555555555605
  %m177 = fmul double %m176, 0x3FE0655555555606
  %m178 = fmul double %m177, 0x3FE0755555555607
  %m179 = fmul double %m178, 0x3FE0855555555608
  %m180 = fmul double %m179, 0x3FE0955555555609
  %m181 = fmul double %m180, 0x3FE0A5555555560A
  %m182 = fmul double %m181, 0x3FE0B5555555560B
  %m183 = fmul double %m182, 0x3FE0C5555555560C
  %m184 = fmul double %m183, 0x3FE0D5555555560D
  %m185 = fmul double %m184, 0x3FE0E5555555560E
  %m186 = fmul double %m185, 0x3FE0F5555555560F
  %m187 = fmul double %m186, 0x3FE1055555555610
  %m188 = fmul double %m187, 0x3FE1155555555611
  %m189 = fmul double %m188, 0x3FE1255555555612
  %m190 = fmul double %m189, 0x3FE1355555555613
  %m191 = fmul double %m190, 0x3FE1455555555614
  %m192 = fmul double %m191, 0x3FE1555555555615
  %m193 = fmul double %m192, 0x3FE1655555555616
  %m194 = fmul double %m193, 0x3FE1755555555617
  %m195 = fmul double %m194, 0x3FE1855555555618
  %m196 = fmul double %m195, 0x3FE1955555555619
  %m197 = fmul double %m196, 0x3FE1A5555555561A
  %m198 = fmul double %m197, 0x3FE1B5555555561B
  %m199 = fmul double %m198, 0x3FE1C5555555561C
  %m200 = fmul double %m199, 0x3FE1D5555555561D
  %m201 = fmul double %m200, 0x3FE1E5555555561E
  %m202 = fmul double %m201, 0x3FE1F5555555561F
  %m203 = fmul double %m202, 0x3FE2055555555620
  %m204 = fmul double %m203, 0x3FE2155555555621
  %m205 = fmul double %m204, 0x3FE2255555555622
  %m206 = fmul double %m205, 0x3FE2355555555623
  %m207 = fmul double %m206, 0x3FE2455555555624
  %m208 = fmul double %m207, 0x3FE2555555555625
  %m209 = fmul double %m208, 0x3FE2655555555626
  %m210 = fmul double %m209, 0x3FE2755555555627
  %m211 = fmul double %m210, 0x3FE2855555555628
  %m212 = fmul double %m211, 0x3FE2955555555629
  %m213 = fmul double %m212, 0x3FE2A5555555562A
  %m214 = fmul double %m213, 0x3FE2B5555555562B
  %m215 = fmul double %m214, 0x3FE2C5555555562C
  %m216 = fmul double %m215, 0x3FE2D5555555562D
  %m217 = fmul double %m216, 0x3FE2E5555555562E
  %m218 = fmul double %m217, 0x3FE2F5555555562F
  %m219 = fmul double %m218, 0x3FE3055555555630
  %m220 = fmul double %m219, 0x3FE3155555555631
  %m221 = fmul double %m220, 0x3FE3255555555632
  %m222 = fmul double %m221, 0x3FE3355555555633
  %m223 = fmul double %m222, 0x3FE3455555555634
  %m224 = fmul double %m223, 0x3FE3555555555635
  %m225 = fmul double %m224, 0x3FE3655555555636
  %m226 = fmul double %m225, 0x3FE3755555555637
  %m227 = fmul double %m226, 0x3FE3855555555638
  %m228 = fmul double %m227, 0x3FE3955555555639
  %m229 = fmul double %m228, 0x3FE3A5555555563A
  %m230 = fmul double %m229, 0x3FE3B5555555563B
  %m231 = fmul double %m230, 0x3FE3C5555555563C
  %m232 = fmul double %m231, 0x3FE3D5555555563D
  %m233 = fmul double %m232, 0x3FE3E5555555563E
  %m234 = fmul double %m233, 0x3FE3F5555555563F
  %m235 = fmul double %m234, 0x3FE4055555555640
  %m236 = fmul double %m235, 0x3FE4155555555641
  %m237 = fmul double %m236, 0x3FE4255555555642
  %m238 = fmul double %m237, 0x3FE4355555555643
  %m239 = fmul double %m238, 0x3FE4455555555644
  %m240 = fmul double %m239, 0x3FE4555555555645
  %m241 = fmul double %m240, 0x3FE4655555555646
  %m242 = fmul double %m241, 0x3FE4755555555647
  %m243 = fmul double %m242, 0x3FE4855555555648
  %m244 = fmul double %m243, 0x3FE4955555555649
  %m245 = fmul double %m244, 0x3FE4A5555555564A
  %m246 = fmul double %m245, 0x3FE4B5555555564B
  %m247 = fmul double %m246, 0x3FE4C5555555564C
  %m248 = fmul double %m247, 0x3FE4D5555555564D
  %m249 = fmul double %m248, 0x3FE4E5555555564E
  %m250 = fmul double %m249, 0x3FE4F5555555564F
  %m251 = fmul double %m250, 0x3FE5055555555650
  %m252 = fmul double %m251, 0x3FE5155555555651
  %m253 = fmul double %m252, 0x3FE5255555555652
  %m254 = fmul double %m253, 0x3FE5355555555653
  %m255 = fmul double %m254, 0x3FE5455555555654
  %m256 = fmul double %m255, 0x3FE5555555555655
  %m257 = fmul double %m256, 0x3FE5655555555656
  %m258 = fmul double %m257, 0x3FE5755555555657
  %m259 = fmul double %m258, 0x3FE5855555555658
  ret double %m259
}

; Negative controls for the rewrites this lane does NOT carry (the
; constant-pool lane's first version reassociated and split cursor arithmetic
; on table pointers; every rewrite that forms a new cursor value was dropped):
; the interior table pointer keeps its cincoffsetimm 112 ahead of the dynamic
; cursor add, and the disjoint-or element offset stays an addiw.
%struct.exp_table = type { [14 x double], [128 x { double, i64 }] }
@__exp_data_test = external hidden addrspace(200) constant %struct.exp_table, align 8

; CHECK-LABEL: table_lookup:
; CHECK:       shrink [[TAB:a[0-9]+]],
; CHECK:       cincoffsetimm [[T:a[0-9]+]], [[TAB]], 112
; CHECK-DAG:   cincoffset [[P1:a[0-9]+]], [[T]], {{a[0-9]+}}
; CHECK-DAG:   ld {{a[0-9]+}}, 0([[P1]])
; CHECK-DAG:   fld {{f[a-z0-9]+}}, 0({{a[0-9]+}})
; CHECK-NOT:   112(
; CHECK-NOT:   120(
; CHECK:       ret
define double @table_lookup(i64 %ki) addrspace(200) nounwind {
entry:
  %shl = shl i64 %ki, 57
  %off0 = lshr i64 %shl, 53
  %tab_base = getelementptr inbounds nuw i8, ptr addrspace(200) @__exp_data_test, i64 112
  %p0 = getelementptr inbounds nuw i8, ptr addrspace(200) %tab_base, i64 %off0
  %v0 = load double, ptr addrspace(200) %p0, align 8
  %off1 = or disjoint i64 %off0, 8
  %p1 = getelementptr inbounds nuw i8, ptr addrspace(200) %tab_base, i64 %off1
  %v1_i = load i64, ptr addrspace(200) %p1, align 8
  %sbits = add i64 %v1_i, %ki
  %v1 = bitcast i64 %sbits to double
  %sum = fadd double %v0, %v1
  ret double %sum
}

%struct.log_table = type { [9 x double], [64 x { double, double }], [64 x { double, double }] }
@__log_data_test = external hidden addrspace(200) constant %struct.log_table, align 8

; The split large offset (2047 + 145) stays in integer arithmetic ahead of the
; cursor add; the loads keep 0 and 8.
; CHECK-LABEL: log_table_lookup:
; CHECK-DAG:   addi [[X:a[0-9]+]], {{a[0-9]+}}, 2047
; CHECK-DAG:   addi {{a[0-9]+}}, [[X]], 145
; CHECK-NOT:   145(
; CHECK-NOT:   144(
; CHECK:       ret
define double @log_table_lookup(i64 %x) addrspace(200) nounwind {
entry:
  %idx = and i64 %x, 1008
  %off_a0 = add nuw nsw i64 %idx, 144
  %pa0 = getelementptr inbounds nuw i8, ptr addrspace(200) @__log_data_test, i64 %off_a0
  %va0 = load double, ptr addrspace(200) %pa0, align 8
  %pa1 = getelementptr inbounds nuw i8, ptr addrspace(200) %pa0, i64 8
  %va1 = load double, ptr addrspace(200) %pa1, align 8
  %off_b0 = add nuw nsw i64 %idx, 2192
  %pb0 = getelementptr inbounds nuw i8, ptr addrspace(200) @__log_data_test, i64 %off_b0
  %vb0 = load double, ptr addrspace(200) %pb0, align 8
  %pb1 = getelementptr inbounds nuw i8, ptr addrspace(200) %pb0, i64 8
  %vb1 = load double, ptr addrspace(200) %pb1, align 8
  %s0 = fadd double %va0, %va1
  %s1 = fadd double %vb0, %vb1
  %s = fadd double %s0, %s1
  ret double %s
}

; The reassociation G(G(B,C),Off) -> G(G(B,Off),C) (the review's first shape:
; with bounds [2^38-4096, 2^38), B = 2^38-2048, C = -1, Off = 2048 the new
; intermediate cursor is 2^38, refused under sv39) is NOT something this lane
; adds. This pins what the base already does with the shape: SelectionDAG
; folds the two GEPs into cincoffset(B, add(Off, C)) and the pre-existing
; target combine cincoffset(X, add(Y, C)) -> cincoffset(cincoffset(X, Y), C)
; then forms B+Off. That combine predates this lane and is recorded as an
; open question of the offset-folding lane's third review; this lane neither
; adds to it nor removes it.
; CHECK-LABEL: reassoc_preexisting:
; CHECK:       cincoffset [[P:a[0-9]+]], a0, a1
; CHECK-NEXT:  lbu a0, -1([[P]])
; CHECK-NEXT:  ret
define i8 @reassoc_preexisting(ptr addrspace(200) %b, i64 %off) addrspace(200) nounwind {
entry:
  %p1 = getelementptr i8, ptr addrspace(200) %b, i64 -1
  %p2 = getelementptr i8, ptr addrspace(200) %p1, i64 %off
  %v = load i8, ptr addrspace(200) %p2
  ret i8 %v
}
