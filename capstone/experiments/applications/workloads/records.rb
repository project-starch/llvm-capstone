# mruby: CSV-like request parsing, a retained graph and transient request batches.
n, batches, retained = ARGV.map { |v| v.to_i }
def phase(name)
  STDERR.syswrite("MEMPHASE #{name}\n")
end
def records(count)
  (0...count).map do |i|
    row = "#{i},alpha:beta," + "x" * 96
    id, tags, body = row.split(',')
    {id: id.to_i, tags: tags.split(':'), body: body}
  end
end
keep = records(retained)
phase('baseline')
checksum = 0
batches.times do |epoch|
  batch = records(n * (epoch == batches / 2 ? 4 : 1))
  batch.each { |r| checksum += r[:id] }
  phase("live-#{epoch}")
  batch = nil
  GC.start
  phase("released-#{epoch}")
end
raise 'retained oracle' unless keep.inject(0) { |s,r| s+r[:id] } == retained*(retained-1)/2
burst = 4*n
raise 'batch oracle' unless checksum == (batches-1)*n*(n-1)/2 + burst*(burst-1)/2
puts "EXP-OK mruby #{checksum}"
