#!/bin/sh
set -e

BUILD_TYPE=${1:-Release}

cmake -S . -B build -DCMAKE_BUILD_TYPE=$BUILD_TYPE
cmake --build build -j$(nproc)
