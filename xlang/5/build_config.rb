# Complete build configuration for CVE-2022-1934

# Target 1: Host native build with Clang + AddressSanitizer (ASan)
MRuby::Build.new('host-asan') do |conf|
  conf.toolchain :clang

  conf.enable_debug

  conf.cc.flags << "-fsanitize=address" << "-g" << "-O1" << "-fno-omit-frame-pointer"
  conf.linker.flags << "-fsanitize=address"

  conf.gembox 'default'
end

# Target 2: Cross-compilation build for riscv64 using GCC
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
end
