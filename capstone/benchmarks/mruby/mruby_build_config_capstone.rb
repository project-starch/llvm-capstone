# Host build config used ONLY to generate mruby's sources for the Capstone domain.
#
# We do not cross-compile through rake. rake runs on the host to produce three
# things the domain needs and that cannot be obtained any other way:
#
#   * the presymbol tables (mruby/presym/id.h and presym.inc), generated from the
#     gem set -- 22 of 32 core files fail to compile without them, and every failure
#     is a missing MRB_SYM__* identifier rather than anything about the target;
#   * mrblib.c and gem_init.c, the Ruby standard library as bytecode;
#   * build/host/amalgam/mruby.c, ONE translation unit, which the gp-captable ABI
#     requires because getGpCaptableIndex numbers globals per module.
#
# The domain is then compiled from that amalgamation with our own flags. The
# defines here therefore only have to be consistent enough to generate the right
# sources; the flags that matter for the capability model are applied at our
# compile, in build-mruby-silicon.sh.
#
# THE GEMBOX IS THE LOAD-BEARING CHOICE. default-no-stdio pulls stdlib and math and
# nothing else, so no mruby-io, -socket, -dir, -env, -process or -signal. That is
# what keeps <time.h>, <sys/types.h> and the file APIs out of the amalgamation
# rather than stubbing them afterwards.
MRuby::Build.new do |conf|
  conf.toolchain :gcc

  # No mrbc, no mirb, no mruby binary in the target set. mrbc IS still built for
  # the host, because it is what turns a .rb specimen into the bytecode the domain
  # runs; it just does not go into the image.
  # stdlib, but NOT the math gembox as a whole: it pulls mruby-bigint, whose
  # bigint.c is built on `unsigned __int128`. On this target i128 IS the capability
  # width, and the backend's custom BITCAST/i128 legalisation returns a mismatched
  # type there -- clang asserts in LegalizeOp on mpz_gcd. That is a real codegen
  # defect and it is recorded separately; it is not this corpus's problem, because
  # what we measure is the GC and the object heap, and 64-bit integers are enough
  # for every specimen. Dropping the gem is the honest small change; fixing i128
  # arithmetic on a capability-width integer type is a different piece of work.
  conf.gembox "stdlib"
  # mruby-math is dropped for the same reason as mruby-bigint and by the same
  # standard: it needs fifteen libm functions beebs does not carry (acosh, asinh,
  # atanh, cbrt, cosh, expm1, log1p, log2, log10, sinh, tan, tanh, trunc, asin,
  # atan2), and no specimen in this corpus computes a hyperbolic. Writing fifteen
  # soft-float implementations to satisfy a linker would be work in service of
  # nothing being measured.

  conf.cc.defines << "MRB_NO_STDIO"

  # Not needed by the generator, but kept so a host run of mrbc and a domain run
  # agree on the object model rather than differing silently.
  conf.cc.defines << "MRB_NO_BOXING"
  conf.cc.defines << "MRB_USE_METHOD_T_STRUCT"
end
