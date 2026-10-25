# mruby for the SILICON delegated ABI (B5, docs/plans/b0-silicon-delegated-runtime.md), and the same configuration
# natively as the reference its output is compared with. capstone/runtime/silicon/build-b5-mruby.sh passes this as
# MRUBY_CONFIG at the 4.0.0-rc2 pin (parse.y, no Prism).
#
# The silicon build differs from build_config.rb's virtual-SDK one:
#   - compiled by $SILICON_CC, a wrapper that adds build-b0-hello.sh's own flags (gp-captable, +m +a soft-float,
#     -O1 -flto, the four silicon -mllvm options): the objects are bitcode for that script's full-LTO link, and
#     $SILICON_LD_RECORDER only records what rake would have linked;
#   - no mruby-task (a timer HAL and SIGALRM) and no mruby-socket. The board's runtime serves files, and nothing
#     here needs a timer or a socket;
#   - level0 heap, never MRB_CAPSTONE_SUBLET (ISSUES R-55: sublet_carve is wrong under the full bounds scheme).
# Floating point stays ON (the oracle's smoke.rb checks floats): soft-float, which is exact IEEE for + - * / and
# runs musl's own libm on both sides.

common = ['-std=gnu99', '-DPOOL_ALIGNMENT=16', '-DMRB_NO_DIRECT_THREADING']
defines = ['MRB_NO_BOXING']

gems = lambda do |conf|
  conf.gembox 'stdlib'
  conf.gembox 'stdlib-ext'
  conf.gembox 'math'
  conf.gembox 'metaprog'
  %w(mruby-io mruby-errno mruby-dir mruby-bin-mruby).each { |g| conf.gem :core => g }
end

# The host build makes mrbc for both, and is the native reference.
MRuby::Build.new do |conf|
  conf.toolchain :gcc
  conf.build_dir = "#{MRUBY_ROOT}/build/native"
  conf.cc.flags += ENV.fetch('MRBD_OPT', '-O2').split + common
  conf.cc.defines += defines
  gems.call(conf)
  conf.gem :core => 'mruby-bin-mrbc'
end

MRuby::CrossBuild.new('silicon') do |conf|
  conf.toolchain :clang
  conf.cc.command = ENV.fetch('SILICON_CC')
  conf.cc.flags = common
  # MRB_CAPSTONE_DELEGATE: patch 0009's delegated posix_spawn for IO.popen. capstone64-unknown-elf defines neither
  # __unix__ nor __linux__, so pread/pwrite and the string length cap are set as build_config.rb sets them.
  # _GNU_SOURCE: build-b0-hello.sh's flags define _XOPEN_SOURCE=700, under which musl hides chroot and friends that
  # mruby-dir's HAL calls.
  conf.cc.defines += defines + %w(MRB_CAPSTONE_DELEGATE MRB_WITH_IO_PREAD_PWRITE MRB_STR_LENGTH_MAX=0 _GNU_SOURCE)
  conf.archiver.command = ENV.fetch('LLVM_AR')
  conf.linker.command = ENV.fetch('SILICON_LD_RECORDER')
  gems.call(conf)
end
