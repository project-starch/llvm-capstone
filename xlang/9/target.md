# target.md — mruby #3829 (Row 9)

* **CVE/Issue:** mruby #3829
* **Product:** mruby
* **Vulnerability Type:** Heap Use-After-Free (UAF) in garbage collector (`mrb_gc_mark`) via shared string slice from irep pool string
* **Vulnerable Tag/Commit:** `13a318b0c70573af45f76a79f902f95845177107` (vulnerable parent of `e4662d77e`)
* **Fix Commit:** `e4662d77e75de4cc6d8e98e56bb0395cbbedbaf7`
