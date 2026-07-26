# target.md — CVE-2018-10191 (Row 11)

* **CVE:** CVE-2018-10191
* **Product:** mruby
* **Status:** SKIPPED
* **Technical Rationale:** Infeasible to reproduce deterministically under AddressSanitizer (ASan). In older mruby <= 1.4.0 releases, deeply nested scopes (128+ levels of instance_eval blocks) designed to trigger the scope-level offset byte wrap-around in `OP_GETUPVAR` are rejected by the parser's compilation engine (throwing "too complex expression" or stack overflow memory limits) before the runtime virtual machine's integer overflow/out-of-bounds pointer read can be executed.
