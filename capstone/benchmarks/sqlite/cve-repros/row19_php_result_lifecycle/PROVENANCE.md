# PROVENANCE — row19 (PHP php-src#5204 result-lifecycle)

- Upstream / fix : https://github.com/php/php-src/pull/5204 (also bugs #64531, #79293)
- Tier           : **STALE-STATE, NOT memory-safety** (confirms N/A-for-literal)

**Real memory-safety bug: NO.** `SQLite3::query()` / `SQLite3Stmt::execute()`
called `sqlite3_reset()` after stepping, so `fetchArray()`'s later `sqlite3_step()`
**re-executed the query with side effects**. The fix caches the step result
instead of resetting:

Verbatim from the fix:
```c
result->stmt_obj->has_stepped = 1;
result->stmt_obj->last_step_result = return_code;
...
if (result_obj->stmt_obj->has_stepped) {
    ret = result_obj->stmt_obj->last_step_result;
} else {
    ret = sqlite3_step(result_obj->stmt_obj->stmt);
}
```

**Consequence.** row19 is a logic/stale-state bug — memory-safe on the host
(ASan-clean, oracle `none`). It has **no free/revoke event** for a capability to
fault on. This confirms the fidelity-ceiling decision: row19 stays a BORROW-REVOKE
**probe by nature** (a borrowed-result lifetime we *could* impose, but the real bug
is not memory unsafety). Skipped for literal in task-010 for exactly this reason.
