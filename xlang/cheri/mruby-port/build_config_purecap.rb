SDK = ENV['CHERI_SDK']
SYSROOT = ENV['CHERI_SYSROOT']
CAPFLAGS = %W[--target=riscv64-unknown-freebsd -march=rv64gcxcheri -mabi=l64pc128d
              -mno-relax -ftls-model=initial-exec -cheri-tgot-tls --sysroot=#{SYSROOT} -O0 -g]

MRuby::Build.new do |conf|
  toolchain :clang          # host build: mrbc and friends
  conf.gembox 'default'
end

MRuby::CrossBuild.new('purecap') do |conf|
  toolchain :clang
  conf.cc.command      = "#{SDK}/bin/clang"
  conf.cc.flags        = [CAPFLAGS, "-DMRB_USE_METHOD_T_STRUCT", "-DPOOL_ALIGNMENT=16"]
  conf.linker.command  = "#{SDK}/bin/clang"
  conf.linker.flags    = [CAPFLAGS]
  conf.archiver.command = "#{SDK}/bin/llvm-ar"
  conf.gembox 'default'
  conf.test_runner.command = nil
end
