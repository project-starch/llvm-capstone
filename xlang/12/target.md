# target.md — mruby #4001 (Row 12)

* **CVE/Issue:** mruby #4001
* **Product:** mruby (with `mruby-io` gem)
* **Vulnerability Type:** Heap Use-After-Free (UAF) in `File#initialize_copy` via dangling `DATA_PTR`
* **Vulnerable Tag/Commit:** `13a318b0c70573af45f76a79f902f95845177107` (mruby) and `b84656eaf3496876b91b2528f011f899964f5f3a` (mruby-io)
* **Fix Commit:** `9b2d861` (mruby-io) or pull request #27
