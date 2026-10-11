# Same boxing, alignment, dispatch and gems as the Capstone interpreter port.
common = ['-std=gnu99', '-DPOOL_ALIGNMENT=16', '-DMRB_NO_DIRECT_THREADING']
gems = lambda do |conf|
  %w(stdlib stdlib-ext math metaprog).each { |box| conf.gembox box }
  %w(mruby-io mruby-errno mruby-dir mruby-pack mruby-bin-mruby).each do |gem|
    conf.gem :core => gem
  end
end
MRuby::Build.new do |conf|
  conf.toolchain :gcc
  conf.build_dir = "#{MRUBY_ROOT}/build/native"
  conf.cc.flags += ['-O1'] + common
  conf.cc.defines += ['MRB_NO_BOXING']
  gems.call(conf)
  conf.gem :core => 'mruby-bin-mrbc'
end
MRuby::CrossBuild.new('cheribsd') do |conf|
  conf.toolchain :clang
  conf.cc.command = ENV.fetch('EXP_CC')
  conf.cc.flags = ['-O1'] + common
  conf.cc.flags += ['-DMRB_GC_STUDY_POISONCAP'] if ENV['EXP_GC_STUDY'] == '1'
  conf.cc.defines += %w(MRB_NO_BOXING MRB_NO_IO_POPEN MRB_WITH_IO_PREAD_PWRITE MRB_STR_LENGTH_MAX=0)
  conf.archiver.command = ENV.fetch('EXP_AR')
  conf.linker.command = ENV.fetch('EXP_CC')
  conf.linker.flags += ['-Wl,--wrap=main,--wrap=write', ENV.fetch('EXP_MEMORY_SOURCE')]
  conf.linker.flags += ['-DEXP_MRB_GC_STUDY'] if ENV['EXP_GC_STUDY'] == '1'
  if ENV['EXP_ALLOCATIONS'] == '1'
    conf.linker.flags += ['-DEXP_ALLOCATIONS', ENV.fetch('EXP_ALLOC_SOURCE'),
      '-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free,--wrap=posix_memalign,--wrap=aligned_alloc']
  end
  gems.call(conf)
end
