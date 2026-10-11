# Four native builds for the corpus sweep. Same four arms probe_config.rb uses
# (assertions; assertions plus a collection at every allocation; ASan with
# assertions off so ASan answers first; ASan with one object per GC page, which
# turns a slot release into a page release ASan can see), with the gem set the
# Capstone port actually builds, so mruby-io, mruby-pack and mruby-task defects
# are reachable too.
gems = lambda do |conf|
  conf.gembox 'stdlib'
  conf.gembox 'stdlib-ext'
  conf.gembox 'math'
  conf.gembox 'metaprog'
  conf.gembox 'stdlib-io'
  conf.gem :core => 'mruby-pack'
  conf.gem :core => (File.directory?("#{MRUBY_ROOT}/mrbgems/hal-posix-task") ? 'hal-posix-task' : 'mruby-task')
  conf.gem :core => 'mruby-bin-mruby'
end

MRuby::Build.new do |conf|                # 'host'
  conf.toolchain :gcc
  conf.enable_debug
  conf.cc.defines += %w(MRB_DEBUG)
  gems.call(conf)
  conf.gem :core => 'mruby-bin-mrbc'
end

MRuby::Build.new('stress') do |conf|
  conf.toolchain :gcc
  conf.enable_debug
  conf.cc.defines += %w(MRB_DEBUG MRB_GC_STRESS)
  gems.call(conf)
end

MRuby::Build.new('asan') do |conf|
  conf.toolchain :gcc
  conf.cc.flags += %w(-fsanitize=address -fno-omit-frame-pointer -g -O1)
  conf.linker.flags += %w(-fsanitize=address)
  gems.call(conf)
end

MRuby::Build.new('asan-page1') do |conf|
  conf.toolchain :gcc
  conf.cc.flags += %w(-fsanitize=address -fno-omit-frame-pointer -g -O1)
  conf.cc.defines += %w(MRB_HEAP_PAGE_SIZE=1)
  conf.linker.flags += %w(-fsanitize=address)
  gems.call(conf)
end
