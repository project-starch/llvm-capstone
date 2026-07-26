#ifndef LADDER_BASE_NULL_KERNEL_H
#define LADDER_BASE_NULL_KERNEL_H
/* Null "kernel": the control that prices the MEASUREMENT ITSELF.
 *
 * Every other rung's cycle count is a compute plus whatever the counter bracket
 * costs. This rung is the bracket with nothing inside it, so it bounds that cost
 * and answers "is the instrument small compared to what it measures?" -- which no
 * rung can answer about itself.
 *
 * Note what this does NOT measure: the baseline's demand-paging / first-touch
 * cost. Those faults happen inside a real kernel's own arrays while it runs, and
 * this touches none, so the null rung cannot see them. Separating the two is the
 * job of the cold-vs-warm pass pair, not of this control. (Getting that backwards
 * is easy: both feel like "overhead".)
 *
 * `volatile` so -O0 still emits a real load and the call cannot fold to nothing.
 */
static volatile unsigned null_sink;
static unsigned null_compute(void) { return null_sink; }
#endif
