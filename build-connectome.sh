#!/usr/bin/env bash
# build-connectome.sh — the two build shapes of the connectome engine.
#
#   production  (default): native vector path OFF. DEFINE INDEX … HNSW|DISKANN and the
#               KNN operators (<|k|>, <|k,ef|>, <|k,dist|>) reject at execution. The vector
#               organ is the merged-in TotalRecall engine (Phase 4). This is what the
#               clyffy-connectome.service ships.
#   bench:      native vector path ON (upstream default features). The SurrealDB-HNSW
#               comparison arm for benches, and the shape that runs the upstream knn
#               language-tests. NOT for production.
#
# The vector path is gated by the `vector-index` cargo feature (default-ON in every
# crate so upstream tests + diff-based syncs stay shaped). Production opts out via
# --no-default-features + the `connectome-production` meta-feature (all defaults minus
# vector-index), so the feature list lives versioned in Cargo.toml, not in this script.
set -euo pipefail

build_production() {
  echo ">> connectome PRODUCTION build (vector-index OFF — TotalRecall owns vectors)"
  cargo build --release --locked \
    --no-default-features \
    --features connectome-production
}

build_bench() {
  echo ">> connectome BENCH build (vector-index ON — SurrealDB-HNSW comparison arm)"
  cargo build --release --locked   # default features include vector-index
}

case "${1:-production}" in
  production) build_production ;;
  bench)      build_bench ;;
  *) echo "usage: $0 {production|bench}" >&2; exit 2 ;;
esac
