#!/usr/bin/env bash
# Automated Build and Setup Script for Row 12 (mruby-io File#initialize_copy UAF)
#
# This script handles the retrieval, checkout, compatibility patching, and compilation
# of the exact vulnerable versions of both the mruby core interpreter and the mruby-io C gem.
set -e

# Resolve absolute paths for the build workspace
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MRUBY_DIR="$DIR/mruby"
IO_DIR="$DIR/mruby-io"

echo "=== [1/3] Sourcing and prepping mruby-io repo ==="
# Step 1: Obtain the vulnerable C gem source code.
# We clone the official mruby-io repository and checkout the exact unmerged
# Pull-Request commit (b84656e) where the vulnerable File#initialize_copy method was introduced.
# Since b84656e is in an unmerged Pull Request, we must explicitly fetch refs/pull/*
# so the local git database can locate and resolve the commit.
if [ ! -d "$IO_DIR" ]; then
    echo "Cloning mruby-io..."
    git clone https://github.com/iij/mruby-io.git "$IO_DIR"
fi
cd "$IO_DIR"
git fetch origin "+refs/pull/*:refs/remotes/origin/pull/*"
git checkout b84656eaf3496876b91b2528f011f899964f5f3a

echo "=== [2/3] Sourcing and prepping mruby repo ==="
# Step 2: Obtain the compatible core mruby interpreter.
# We clone the official mruby core and checkout an April 2018 commit (13a318b).
# This aligns the C-level internal APIs of the core interpreter with our checked-out
# version of mruby-io from Step 1, ensuring they compile and link together flawlessly.
if [ ! -d "$MRUBY_DIR" ]; then
    echo "Cloning mruby upstream..."
    git clone https://github.com/mruby/mruby.git "$MRUBY_DIR"
fi
cd "$MRUBY_DIR"
git fetch origin e4662d77e75de4cc6d8e98e56bb0395cbbedbaf7
git checkout 13a318b0c70573af45f76a79f902f95845177107

# Step 3: Apply the 2018-to-2026 Rakefile Compatibility Patch.
#
# WHY THIS IS LOAD-BEARING:
# In 2018, mruby's Rake build file passed an option hash `{ :verbose => $verbose }`
# as a second parameter to FileUtils file-operations (mkdir_p, rm_f, cp).
# Under modern Ruby environments (like the stable 24/26 versions we are running in),
# these file utilities no longer accept a hash parameter in this position and throw
# a strict ArgumentError, crashing the build. 
# We use sed to strip the obsolete second argument on-the-fly, allowing the 2018 
# codebase to compile cleanly and stably on modern machines.
sed -i 's/FileUtils.mkdir_p bin_path, { :verbose => $verbose }/FileUtils.mkdir_p bin_path/g' Rakefile
sed -i 's/FileUtils.rm_f t.name, { :verbose => $verbose }/FileUtils.rm_f t.name/g' Rakefile
sed -i 's/FileUtils.cp t.prerequisites.first, t.name, { :verbose => $verbose }/FileUtils.cp t.prerequisites.first, t.name/g' Rakefile

echo "=== [3/3] Building mruby with unified config (host-asan + riscv64) ==="
# Step 4: Clean old compilation artifacts and run the compiler.
# We direct Rake (mruby's build engine) to use our custom target-configuration profile
# located in build_config.rb, which compiles a native GCC + ASan target (for verification)
# and cross-compiles a stock RISC-V target (for QEMU/FPGA protection verification).
rake clean || true
MRUBY_CONFIG="$DIR/build_config.rb" rake

echo "=== Build Complete ==="
