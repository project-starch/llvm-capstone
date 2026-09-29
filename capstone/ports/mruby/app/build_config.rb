# mruby for a Capstone musl domain, and the same configuration natively as the
# reference its output is compared with. build-mruby-domain.sh passes this as
# MRUBY_CONFIG and sets the knobs:
#
#   MRBD_BOXING    no (default) | word   word needs patches/0006 (the boxed word a __uintcap_t)
#   MRBD_DISPATCH  switch (default) | direct   direct threading needs the compiler's
#                  labels-as-values fix (compiler/cap-init-blockaddress)
#   MRBD_OPT       optimisation flags, default -O2
#   MRBD_TESTS     1 builds mrbtest for both builds
#   MRBD_DEFINES   extra -D flags (space separated), both builds
#
# Gems: stdlib, stdlib-ext, math and metaprog, and from stdlib-io only what a
# domain can serve: mruby-io, mruby-errno, mruby-dir, mruby-pack. Not mruby-socket,
# mruby-process or mruby-signal: a domain has no sockets, processes or signals.

boxing   = ENV.fetch('MRBD_BOXING', 'no')
dispatch = ENV.fetch('MRBD_DISPATCH', 'switch')
opt      = ENV.fetch('MRBD_OPT', '-O2').split
tests    = ENV['MRBD_TESTS'] == '1'
extra    = ENV.fetch('MRBD_DEFINES', '').split

# POOL_ALIGNMENT: mruby's parser pool picks 8, and its cells hold pointers.
common = ['-std=gnu99', '-DPOOL_ALIGNMENT=16'] + extra
common << '-DMRB_NO_DIRECT_THREADING' if dispatch == 'switch'
defines = boxing == 'word' ? ['MRB_WORD_BOXING'] : ['MRB_NO_BOXING']

gems = lambda do |conf|
  conf.gembox 'stdlib'
  conf.gembox 'stdlib-ext'
  conf.gembox 'math'
  conf.gembox 'metaprog'
  %w(mruby-io mruby-errno mruby-dir mruby-pack mruby-bin-mruby).each { |g| conf.gem :core => g }
end

# The host build makes mrbc for both, and is the native reference.
MRuby::Build.new do |conf|
  conf.toolchain :gcc
  conf.build_dir = "#{MRUBY_ROOT}/build/native"
  conf.cc.flags += opt + common
  conf.cc.defines += defines
  gems.call(conf)
  conf.gem :core => 'mruby-bin-mrbc'
  conf.enable_test if tests
end

MRuby::CrossBuild.new('capstone') do |conf|
  conf.toolchain :clang
  # A cross build picks no port; musl is POSIX. Versions before the port layer
  # (4.0.0-rc2) have no `ports` and need none.
  conf.ports 'posix' if conf.respond_to?(:ports)
  conf.cc.command = 'capstone-cc'
  conf.cc.flags = opt + common
  # A domain cannot spawn a process: mruby-io's own veto over IO.popen, backticks
  # and friends, under which their tests skip rather than fail.
  # capstone64-unknown-elf defines neither __unix__ nor __linux__, and mruby reads
  # the platform from them: without the two below, IO#pread/#pwrite would be left
  # out (the runtime serves pread64/pwrite64) and a String would be capped at
  # 1 MiB, where the native build has no cap.
  conf.cc.defines += defines + %w(MRB_NO_IO_POPEN MRB_WITH_IO_PREAD_PWRITE MRB_STR_LENGTH_MAX=0)
  conf.archiver.command = ENV.fetch('LLVM_AR')
  conf.linker.command = 'capstone-cc'
  # MRBD_HEAP=sublet-gc: every GC object slot under Sublet (patch 0008), which
  # needs sublet.h from the runtime tree; the build script exports its path.
  if (inc = ENV['MRBD_GC_SUBLET_INCLUDE'])
    conf.cc.defines << 'MRB_CAPSTONE_GC_SUBLET'
    conf.cc.include_paths << inc
  end
  gems.call(conf)
  conf.enable_test if tests
  conf.test_runner.command = 'false'      # the domain runs it, not rake
end
