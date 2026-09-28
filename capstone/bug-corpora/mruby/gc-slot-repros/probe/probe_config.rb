# Four native builds of the pin, for survey.sh. 'host' also makes the mrbc the
# others need: assertions on; assertions plus a collection at every allocation;
# ASan with assertions off so ASan answers first; and ASan with one object per GC
# page, which turns a slot release into a page release ASan can see -- so a defect
# silent in 'asan' and loud in 'asan-page1' is reusing a GC slot, and one silent in
# both is nested below the collector.
gems = lambda do |conf|
  conf.gembox 'stdlib'
  conf.gembox 'stdlib-ext'
  conf.gembox 'math'
  conf.gembox 'metaprog'
  conf.gem :core => 'mruby-bin-mruby'
end

MRuby::Build.new do |conf|                # 'host'
  conf.toolchain :gcc
  conf.enable_debug
  conf.cc.defines += %w(MRB_DEBUG)
  gems.call(conf)
  conf.gem :core => 'mruby-bin-mrbc'
  conf.enable_test
end

MRuby::Build.new('stress') do |conf|
  conf.toolchain :gcc
  conf.enable_debug
  conf.cc.defines += %w(MRB_DEBUG MRB_GC_STRESS)
  gems.call(conf)
  conf.enable_test
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
