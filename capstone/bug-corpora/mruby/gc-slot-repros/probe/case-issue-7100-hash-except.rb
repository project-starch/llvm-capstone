class MovingKey
  attr_accessor :owner, :armed
  def initialize(id); @id = id; @armed = false; end
  def hash; @id; end
  def ==(_)
    if @armed
      @armed = false
      2_000.times { |i| @owner[i + 10_000] = i }
    end
    false
  end
  alias eql? ==
end

class DeconstructingOwner
  def initialize(hash); @hash = hash; end
  def deconstruct_keys(_); @hash; end
end

keys = 30.times.map { |i| MovingKey.new(i + 1) }
hash = { a: 1 }
keys.each { |key| hash[key] = key; key.owner = hash }
keys[0].armed = true

case DeconstructingOwner.new(hash)
in { a: 1, **rest }
  p rest.length
end
