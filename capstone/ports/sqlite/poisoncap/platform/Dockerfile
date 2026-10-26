# The build host for the published PoisonCap platform: Ubuntu 24.04 with clang-18, the host the
# recipe in ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/ was written for. Used by
# run-in-container.sh; the platform sources and outputs stay on the host, mounted in.
# The base is pinned by digest. Package versions are not pinned by apt; the wrapper records
# `dpkg -l` of every build (build.sh) into the platform's logs instead.
FROM ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      clang-18 lld-18 llvm-18 cmake ninja-build python3 git ca-certificates build-essential \
      bison flex pkg-config autoconf automake libtool texinfo bc time file xz-utils \
      libarchive-dev libarchive-tools libbz2-dev libglib2.0-dev libpixman-1-dev libexpat1-dev \
      libgmp-dev libattr1-dev libcap-ng-dev zlib1g-dev libssl-dev meson openssh-client \
    && rm -rf /var/lib/apt/lists/*
# cheribuild's sqlite target requires tclsh (its check_system_dependencies), used by
# speedtest1/build.sh. A separate layer, so the platform's earlier layers are unchanged.
RUN apt-get update && apt-get install -y --no-install-recommends tcl \
    && rm -rf /var/lib/apt/lists/*
