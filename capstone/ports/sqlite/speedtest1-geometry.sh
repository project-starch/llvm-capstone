# ONE SOURCE OF TRUTH FOR THE SPEEDTEST1 GEOMETRY. Sourced, not executed.
#
# These three numbers were carried in three places and disagreed in all three: the run script
# defaulted to a 1.75 MiB arena and a 2 MiB declared stack, the board handover specified 2 MiB and
# 1 MiB, and the baseline build defaulted to a third arena size of 2 MiB that no committed flow ever
# reached. Nothing gated any of it. Two arms with different arena sizes take different buddy-split
# paths and different allocation counts, so a ratio silently absorbs the difference -- and an
# operator running the committed script against `orm` got a 1.75 MiB arena under orm's measured
# 2 MiB minimum, i.e. the abort path, which under emulation looks like a fault rather than a
# misconfiguration.
#
# THE ARENA IS 2 MiB because orm --size 1 needs 2 and main --size 1 needs 1.5 (measured; see
# tools/speedtest1-heap-sweep.sh). THE DECLARED STACK IS 1 MiB because the carve plus a 2 MiB
# declaration demands order 11 and the domain is refused at creation, before an instruction runs.
# DO NOT RAISE THE ARENA on the strength of domdata-budget.py passing: 2.5 MiB passed that gate and
# then faulted at SQ: E/share1 before entry.
SPEEDTEST1_GEOM_HEAP=${SPEEDTEST1_HEAP:-$((2048 * 1024))}
SPEEDTEST1_GEOM_STACK=${SPEEDTEST1_STACK:-$((1024 * 1024))}
SPEEDTEST1_GEOM_REGION=${SPEEDTEST1_REGION_SIZE:-65536}
