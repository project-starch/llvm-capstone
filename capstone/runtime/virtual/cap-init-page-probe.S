/* A first data access under either image-wide or narrowed PCC. The body and
 * alternate target deliberately occupy disjoint, exactly representable ranges.
 * The ordinary module resumes the load. A diagnostic redirect to the alternate
 * target must not be mistaken for permission to widen the interrupted PCC. */
.text
.balign 16
.global cap_init_page_probe
.type cap_init_page_probe,@function
cap_init_page_probe:
    cincoffsetimm sp, sp, -16
    stc ra, 0(sp)
    lla t0, cap_init_page_probe_body
    scc t0, gp, t0
    beqz a1, 1f
    mv t1, t0
    addi t2, t1, 16
    shrink t0, t1, t2
    /* The gp-free caller supplied an integer return address. CJALR must
     * create a capability return link before entering the narrowed PCC. */
1:  cjalr ra, 0(t0)
    ldc ra, 0(sp)
    cincoffsetimm sp, sp, 16
    ret
.size cap_init_page_probe, .-cap_init_page_probe

.balign 16
.global cap_init_page_probe_body
.type cap_init_page_probe_body,@function
cap_init_page_probe_body:
    lbu a0, 0(a0)
    cjalr zero, 0(ra)
    .space 8
.size cap_init_page_probe_body, .-cap_init_page_probe_body

.balign 16
.global cap_init_page_probe_redirect
.type cap_init_page_probe_redirect,@function
cap_init_page_probe_redirect:
    li a0, 0
    cjalr zero, 0(ra)
.size cap_init_page_probe_redirect, .-cap_init_page_probe_redirect
