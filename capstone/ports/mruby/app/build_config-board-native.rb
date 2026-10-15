# mruby as a NATIVE Linux process on the FPGA board: B5's unprotected baseline (docs/plans/b0-silicon-delegated-runtime.md).
# The board's own toolchain (buildroot's riscv64-buildroot-linux-gnu-gcc 12.3: rv64g, lp64d hard-float, glibc, the
# same as the board's busybox and libc), with build_config-silicon.rb's gems and defines, at two levels: -O2 (the x86
# oracle's) and -O1 (the silicon build's). $RV64_CROSS is the toolchain prefix (.../bin/riscv64-buildroot-linux-gnu-).
#
# What differs from the silicon build besides the protection: hard-float against soft-float, glibc and gcc against
# musl and clang, and dynamic linking against one full-LTO image. The comparison is an upper bound on the cost of
# the silicon ABI, not a measurement of capabilities alone.

common = ['-std=gnu99', '-DPOOL_ALIGNMENT=16', '-DMRB_NO_DIRECT_THREADING']
defines = ['MRB_NO_BOXING']

gems = lambda do |conf|
  conf.gembox 'stdlib'
  conf.gembox 'stdlib-ext'
  conf.gembox 'math'
  conf.gembox 'metaprog'
  %w(mruby-io mruby-errno mruby-dir mruby-bin-mruby).each { |g| conf.gem :core => g }
end

# The host build is build_config-silicon.rb's, so the two configurations share build/native (mrbc and the oracle).
MRuby::Build.new do |conf|
  conf.toolchain :gcc
  conf.build_dir = "#{MRUBY_ROOT}/build/native"
  conf.cc.flags += ENV.fetch('MRBD_OPT', '-O2').split + common
  conf.cc.defines += defines
  gems.call(conf)
  conf.gem :core => 'mruby-bin-mrbc'
end

%w(O2 O1).each do |opt|
  MRuby::CrossBuild.new("rv64-#{opt}") do |conf|
    conf.toolchain :gcc
    cross = ENV.fetch('RV64_CROSS')
    conf.cc.command = "#{cross}gcc"
    conf.linker.command = "#{cross}gcc"
    conf.archiver.command = "#{cross}ar"
    conf.cc.flags = ["-#{opt}"] + common
    conf.cc.defines += defines
    gems.call(conf)
  end
end
