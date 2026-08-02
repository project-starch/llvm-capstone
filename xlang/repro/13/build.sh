#!/usr/bin/env bash
set -e

# Resolve paths
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MRUBY_DIR="$DIR/mruby"

echo "=== [1/2] Sourcing and prepping mruby repo ==="
if [ ! -d "$MRUBY_DIR" ]; then
    echo "Cloning mruby upstream..."
    git clone https://github.com/mruby/mruby.git "$MRUBY_DIR"
fi

cd "$MRUBY_DIR"
# Check out the exact vulnerable commit for mruby #4927 (Row 13)
git fetch origin 70e574689664c10ed2c47581999cc2ce3e3c5afb
git checkout fc8fb41451b07b3fda0726ba80e88e509ad02452

# Apply Rakefile compatibility patch for FileUtils (rm_f / cp / mkdir_p)
sed -i 's/FileUtils.mkdir_p bin_path, { :verbose => $verbose }/FileUtils.mkdir_p bin_path/g' Rakefile
sed -i 's/FileUtils.rm_f t.name, { :verbose => $verbose }/FileUtils.rm_f t.name/g' Rakefile
sed -i 's/FileUtils.cp t.prerequisites.first, t.name, { :verbose => $verbose }/FileUtils.cp t.prerequisites.first, t.name/g' Rakefile

echo "=== [2/2] Building mruby with unified config (host-asan + riscv64) ==="
rake clean || true
MRUBY_CONFIG="$DIR/build_config.rb" rake

echo "=== Build Complete ==="
