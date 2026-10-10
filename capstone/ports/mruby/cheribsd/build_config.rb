# mruby 4.0.0-rc2 for CheriBSD purecap (CHERI-RISC-V, l64pc128d).
SDK  = ENV.fetch('CHERI_SDK')
ROOTFS = ENV.fetch('CHERI_SYSROOT')
CHERI = %W(--target=riscv64-unknown-freebsd13 --sysroot=#{ROOTFS}
           -march=rv64imafdcxcheri -mabi=l64pc128d -mno-relax -B#{SDK}/bin)
# The same configuration the domain port needs on a 16-byte-pointer target
# (ports/mruby/app/build_config.rb): no boxing, the parser pool aligned for a
# capability, and the switch dispatch, because labels-as-values tables are
# emitted as integers and jumping through one is not a capability.
common = %w(-g -Wall -Wno-unused-function -std=gnu99 -DPOOL_ALIGNMENT=16
            -DMRB_NO_DIRECT_THREADING)
DEFINES = %w(MRB_NO_BOXING)

# The native build only supplies mrbc, the bytecode compiler the cross build runs.
MRuby::Build.new do |conf|
  conf.toolchain :gcc
  conf.build_dir = "#{MRUBY_ROOT}/build/native"
  conf.cc.flags += %w(-std=gnu99 -DPOOL_ALIGNMENT=16 -DMRB_NO_DIRECT_THREADING)
  conf.cc.defines += DEFINES
  conf.gem :core => 'mruby-bin-mrbc'
end

MRuby::CrossBuild.new('cheribsd') do |conf|
  conf.toolchain :clang
  conf.build_dir = "#{MRUBY_ROOT}/build/cheribsd"
  conf.cc.command      = "#{SDK}/bin/clang"
  conf.cxx.command     = "#{SDK}/bin/clang++"
  conf.linker.command  = "#{SDK}/bin/clang"
  conf.archiver.command = "#{SDK}/bin/llvm-ar"
  conf.cc.flags     = ['-O1'] + CHERI + common
  conf.cc.defines  += DEFINES
  conf.linker.flags = CHERI + ['-static']
  conf.gembox 'stdlib'
  conf.gembox 'stdlib-ext'
  conf.gembox 'math'
  conf.gembox 'metaprog'
  conf.gembox 'stdlib-io'
  %w(mruby-pack mruby-bin-mruby).each { |g| conf.gem :core => g }
  conf.gem :core => (File.directory?("#{MRUBY_ROOT}/mrbgems/hal-posix-task") ? 'hal-posix-task' : 'mruby-task')
  conf.host_target = 'riscv64-unknown-freebsd13'
end
