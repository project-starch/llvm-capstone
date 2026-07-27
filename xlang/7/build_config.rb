# Build configuration for Row 7 (mruby-bigint mrb_bint_reduce GC hazard).
#
# Two things differ from the other mruby rows and both are required:
#
#  1. `mruby-bigint` and `mruby-rational` are NOT in the default gembox, so they
#     are added explicitly. mrb_bint_reduce() only exists when MRB_USE_RATIONAL
#     is defined (see the #ifdef around it in bigint.c), and MRB_USE_RATIONAL is
#     set by mruby-rational's mrbgem.rake -- so BOTH gems are needed to compile
#     the function under test at all.
#
#  2. The only caller of mrb_bint_reduce is mruby-rational's rational_new_b(),
#     reached from Ruby as Rational(bignum, bignum).

MRuby::Build.new('host') do |conf|
  conf.toolchain :clang

  conf.enable_debug

  conf.cc.flags << "-fsanitize=address" << "-g" << "-O1" << "-fno-omit-frame-pointer"
  conf.linker.flags << "-fsanitize=address"

  conf.gembox 'default'
  conf.gem core: 'mruby-bigint'
  conf.gem core: 'mruby-rational'
end

MRuby::CrossBuild.new('riscv64') do |conf|
  conf.toolchain :gcc

  conf.cc do |cc|
    cc.command = 'riscv64-linux-gnu-gcc'
    cc.flags << "-O3" << "-g"
  end

  conf.linker do |linker|
    linker.command = 'riscv64-linux-gnu-gcc'
    linker.flags << "-O3"
  end

  conf.archiver do |archiver|
    archiver.command = 'riscv64-linux-gnu-ar'
  end

  conf.gembox 'default'
  conf.gem core: 'mruby-bigint'
  conf.gem core: 'mruby-rational'
end
