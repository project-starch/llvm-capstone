FROM ubuntu:22.04@sha256:3b06811b2afd352be909dd088a004166d665dc76d38b13eada33522a9d915c6f
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git cmake ninja-build pkg-config \
    python3 python3-venv python3-pip ruby bison flex autoconf automake libtool \
    xz-utils unzip zlib1g-dev libbz2-dev libffi-dev libssl-dev libreadline-dev \
    libsqlite3-dev liblzma-dev libncurses-dev libgdbm-dev libevent-dev \
    libglib2.0-dev libpcap-dev libpcre2-dev libgcrypt20-dev libc-ares-dev \
    libnghttp2-dev liblz4-dev libzstd-dev libxml2-dev && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /repo
