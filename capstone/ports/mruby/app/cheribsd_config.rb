# mruby for CheriBSD purecap, flags from ports/common/cmake/toolchains/cheribsd.cmake.
#
# The host build makes mrbc; the cross build makes the guest binary. Two things here are
# not optional and both cost a run to find:
#
#   POOL_ALIGNMENT=16  mruby's parser pool hands out blocks for structures that hold
#                      pointers, and at 16-byte capabilities an 8-byte-aligned block
#                      makes every capability field in it unaligned -- the guest dies
#                      with a Bus error before it reaches any script.
#   MRB_NO_BOXING      word boxing packs a value into a pointer-sized word, which cannot
#                      hold a capability.
#
# Patches 0001 and 0002 are likewise required to BUILD at all, not just to run: at 16-byte
# pointers RSTRING_EMBED_LEN_MAX is 59 and needs six bits, which trips a static assert in
# src/string.c, and symbol.c's literal-flag arithmetic drops the tag through uintptr_t.
#
# Arms: no define is the CHERI baseline; -DMRB_POISONCAP_HASH (patch 0012) adds the
# per-slot poison adapter, selected at run time with MRB_POISON_MODE=0|1.
SDK = ENV.fetch('CHERI_SDK')
SYSROOT = ENV.fetch('CHERI_SYSROOT')
CHERI = %W(-target riscv64-unknown-freebsd13 --sysroot=#{SYSROOT}
           -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax -B#{SDK}/bin)

gems = lambda do |conf|
  conf.gembox 'stdlib'; conf.gembox 'stdlib-ext'; conf.gembox 'math'; conf.gembox 'metaprog'
  conf.gem :core => 'mruby-bin-mruby'
end

MRuby::Build.new do |conf|          # host: makes mrbc for the cross build
  conf.toolchain :gcc
  conf.cc.defines += %w(MRB_DEBUG)
  gems.call(conf)
  conf.gem :core => 'mruby-bin-mrbc'
end

MRuby::CrossBuild.new('cheribsd') do |conf|
  conf.toolchain :clang
  conf.cc.command = "#{SDK}/bin/clang"
  conf.cc.flags = ['-g', '-O1', '-std=gnu99', '-DPOOL_ALIGNMENT=16', *CHERI]
  conf.cc.defines += %w(MRB_NO_BOXING)
  conf.linker.command = "#{SDK}/bin/clang"
  conf.linker.flags = [*CHERI, '-fuse-ld=lld']
  conf.archiver.command = "#{SDK}/bin/llvm-ar"
  gems.call(conf)
  conf.test_runner.command = 'false'
end
