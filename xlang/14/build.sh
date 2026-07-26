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
# Check out the exact vulnerable commit for mruby #3596 (Row 14)
git fetch origin 5c114c91d4ff31859fcd84cf8bf349b737b90d99
git checkout 491d68bb3004eb8d7deec4a3a682b25de0d4afc2

# Apply Rakefile compatibility patch for FileUtils (rm_f / cp / mkdir_p)
sed -i 's/FileUtils.mkdir_p bin_path, { :verbose => $verbose }/FileUtils.mkdir_p bin_path/g' Rakefile
sed -i 's/FileUtils.rm_f t.name, { :verbose => $verbose }/FileUtils.rm_f t.name/g' Rakefile
sed -i 's/FileUtils.cp t.prerequisites.first, t.name, { :verbose => $verbose }/FileUtils.cp t.prerequisites.first, t.name/g' Rakefile

echo "=== [2/2] Building mruby with unified config (host-asan + riscv64) ==="
rake clean || true
MRUBY_CONFIG="$DIR/build_config.rb" rake

echo "=== Build Complete ==="
