/* SPDX-License-Identifier: MIT */
/* long rt_service(long nr, long a0, long a1): a direct service call of the
 * race-test module build (CV_RACE_TEST). Scalars only. */
.text
.global rt_service
.type rt_service, @function
rt_service:
    mv a7, a0
    mv a0, a1
    mv a1, a2
    ecall
    ret
.size rt_service, .-rt_service
