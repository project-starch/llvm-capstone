# Three native builds of the pin, for survey.sh. 'host' also makes the mrbc the
# others need: assertions on, assertions plus a collection at every allocation,
# and ASan with assertions off so ASan is what answers first.
gems = lambda do |conf|
  conf.gembox 'stdlib'
  conf.gembox 'stdlib-ext'
  conf.gembox 'math'
  conf.gembox 'metaprog'
  conf.gem :core => 'mruby-bin-mruby'
end

MRuby::Build.new do |conf|          # 'host': assertions on
  conf.toolchain :gcc
  conf.enable_debug
  conf.cc.defines += %w(MRB_DEBUG)
  gems.call(conf)
  conf.gem :core => 'mruby-bin-mrbc'
  conf.enable_test
end

MRuby::Build.new('stress') do |conf|   # assertions + collect at every allocation
  conf.toolchain :gcc
  conf.enable_debug
  conf.cc.defines += %w(MRB_DEBUG MRB_GC_STRESS)
  gems.call(conf)
  conf.enable_test
end

MRuby::Build.new('asan') do |conf|   # ASan, assertions OFF so ASan speaks first
  conf.toolchain :gcc
  conf.cc.flags += %w(-fsanitize=address -fno-omit-frame-pointer -g -O1)
  conf.linker.flags += %w(-fsanitize=address)
  conf.gembox 'stdlib'
  conf.gembox 'stdlib-ext'
  conf.gembox 'math'
  conf.gembox 'metaprog'
  conf.gem :core => 'mruby-bin-mruby'
end
