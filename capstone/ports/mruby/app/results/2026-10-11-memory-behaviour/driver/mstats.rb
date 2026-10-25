# mstats.rb: mruby's own heap statistics around one workload, 2026-10-11 memory-behaviour run.
#
#   mruby mstats.rb WORKLOAD.rb [WORKLOAD ARG...]
#
# Reads WORKLOAD.rb, evaluates it at the top level with ARGV set to its own arguments, and writes
# ObjectSpace.count_objects before it, right after it, and after a GC.start, to stderr. The
# workload's stdout is untouched: it is the oracle compared against stock and native. After the
# last reading it prints MM_DONE and sleeps 4 s, so the guest shell can read the process's VmHWM
# while the process still exists.
def mm_count(tag)
  h = ObjectSpace.count_objects
  $stderr.write("MM_COUNT #{tag} " + h.map { |k, v| [k.to_s, v] }.sort.map { |k, v| "#{k}=#{v}" }.join(" ") + "\n")
end

mm_workload = ARGV.shift
mm_source = File.read(mm_workload)
mm_count("before")
eval(mm_source)
$stdout.flush
mm_count("after")
GC.start
mm_count("after_gc")
$stderr.write("MM_GC interval_ratio=#{GC.interval_ratio} step_ratio=#{GC.step_ratio} generational=#{GC.generational_mode}\n")
$stderr.write("MM_DONE\n")
$stderr.flush
sleep 4
