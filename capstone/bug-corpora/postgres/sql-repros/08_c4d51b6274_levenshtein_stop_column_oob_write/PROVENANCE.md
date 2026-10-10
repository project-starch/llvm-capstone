# levenshtein_less_equal writes out of bounds on an overflowed cost product

Upstream commit `c4d51b6274`, "Avoid overflow in Levenshtein distance
calculations". It postdates the 17.5 pin, so the pin is affected.

Upstream's own account:

> `levenshtein()` and `levenshtein_less_equal()` let the caller specify the
> insertion, deletion, and substitution costs, and fuzzystrmatch's
> corresponding SQL functions accept any 32-bit integer for each. Since the
> distances are calculated with 32-bit arithmetic, large costs can cause
> overflows, thereby producing nonsensical results. **Certain inputs to
> `levenshtein_less_equal()` can even cause out-of-bounds writes.**

## Established on the pin by inspection

`src/backend/utils/adt/levenshtein.c` (the file is `#include`d twice by
`varlena.c`, once plain and once with `LEVENSHTEIN_LESS_EQUAL` defined; only
the second compilation has `stop_column` as a real variable):

- `:152` -- `min_theo_d = net_inserts < 0 ? -net_inserts * del_c :
  net_inserts * ins_c;` -- `int * int` into an `int`. The costs are whatever
  the SQL caller passed.
- `:175` -- `int slack_d = max_d - min_theo_d;` -- inherits the overflowed
  value.
- `:178` -- `stop_column = best_column + (slack_d / (ins_c + del_c)) + 1;`
- `:179-180` -- `if (stop_column > m) stop_column = m + 1;`
  **This is the defect.** The clamp is one-sided. A `stop_column` that came
  out below range is not corrected.
- `:212` -- `prev = (int *) palloc(2 * m * sizeof(int));`
- `:240` -- `prev[stop_column] = max_d + 1;` -- the write.

Upstream's fix widens `min_theo_d`, `slack_d` and the cost variables to
`int64` and replaces the one-sided clamp with `stop_column = Min(tmp, m + 1)`
computed in 64-bit.

## Reachability

`contrib/fuzzystrmatch/fuzzystrmatch.c:198`
(`levenshtein_less_equal_with_costs`) passes the caller's three costs and
`max_d` straight through to `varstr_levenshtein_less_equal` at `:218`. There is
no validation of the costs on the way, which is upstream's point: the SQL
function accepts any `int32`.

## Why it is nested

`prev` is a `palloc` chunk. How far out of the chunk the write lands is a
function of the costs the caller supplies, so unlike most cases in this corpus
this one can be aimed -- near-miss writes that stay inside the aset block, and
larger ones that leave it. That makes it the most useful single case here for
separating "the mechanism cannot see inside the allocator" from "the mechanism
did not fire", and the three arms should be run with more than one cost
triple.
