if $fails.empty?
  p ["PASS"]
else
  $fails.each { |f| p f }
end
