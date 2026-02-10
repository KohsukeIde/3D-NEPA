#!/usr/bin/env bash
set -euo pipefail

mkdir -p data
cd data

if [ ! -f ModelNet40.zip ]; then
  wget http://modelnet.cs.princeton.edu/ModelNet40.zip
fi

if [ ! -d ModelNet40 ]; then
  unzip -q ModelNet40.zip
fi

echo "OK: data/ModelNet40"
