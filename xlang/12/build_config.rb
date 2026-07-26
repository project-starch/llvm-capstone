# Complete Build Configuration for mruby #4001 (Row 12)
#
# In mruby, build_config.rb is the central configuration file (replacing standard Makefiles).
# It directs mruby's build engine (Rake) on which compilers, toolchains, flags, and gems
# to build. This configuration defines two distinct build targets to support our dual-verification plan.

# ========================================================================================
# Target 1: Host Native Debug Build with AddressSanitizer (ASan)
# ========================================================================================
# This target compiles the native interpreter ("build/host/bin/mruby") used to natively
# verify, triage, and debug our Use-After-Free (UAF) reproduction locally.
#
# Note: Renamed from 'host-asan' to 'host' to support the older 2018 build system's expectations.
MRuby::Build.new('host') do |conf|
  # We use the GCC toolchain because its conservative stack-scanning GC layout
  # is highly stable and fully compatible with AddressSanitizer on x86_64 Linux.
  conf.toolchain :gcc

  # Enable debug symbols (-g) so that ASan can output exact source file line numbers.
  conf.enable_debug

  # Enable AddressSanitizer (ASan) compilation flags:
  # -fsanitize=address      : Instruments all malloc/free and memory reads/writes to catch UAFs.
  # -O1                     : Standard debug optimization (keeps frames readable and prevents bloating).
  # -fno-omit-frame-pointer : Keeps frame pointers on the stack for perfect ASan backtrace reports.
  conf.cc.flags << "-fsanitize=address" << "-g" << "-O1" << "-fno-omit-frame-pointer"
  conf.linker.flags << "-fsanitize=address"

  # Load the core standard library box (Strings, Arrays, Hashes, etc.).
  conf.gembox 'default'

  # CRUCIAL: Statically compile and link our local, pinned, vulnerable clone of the mruby-io gem
  # located inside our case directory, rather than pulling the latest patched version from GitHub.
  conf.gem File.expand_path('../mruby-io', __FILE__)
end

# ========================================================================================
# Target 2: Cross-Compilation Build for RISC-V (QEMU and FPGA Protection)
# ========================================================================================
# This target cross-compiles the vulnerable interpreter ("build/riscv64/bin/mruby")
# into a stock riscv64 ELF binary so it can be emulated under QEMU or run inside
# capability-protected hardware domains on our FPGA board.
MRuby::CrossBuild.new('riscv64') do |conf|
  # Directs the build system to use cross-compilers
  conf.toolchain :gcc

  # Configure the C Compiler to target RISC-V 64-bit architecture
  conf.cc do |cc|
    cc.command = 'riscv64-linux-gnu-gcc' # The GNU RISC-V cross-compiler
    cc.flags << "-O3" << "-g"            # High optimization (-O3) to match our production firmware layout
  end

  # Configure the Linker for RISC-V
  conf.linker do |linker|
    linker.command = 'riscv64-linux-gnu-gcc'
    linker.flags << "-O3"
  end

  # Configure the Archiver to package internal static libraries for RISC-V
  conf.archiver do |archiver|
    archiver.command = 'riscv64-linux-gnu-ar'
  end

  conf.gembox 'default'
  
  # Compiles and links the exact same local vulnerable mruby-io gem into our RISC-V binary.
  conf.gem File.expand_path('../mruby-io', __FILE__)
end
