# connectome

**connectome = SurrealDB, truly forked.** This is a clean private-lineage fork of **SurrealDB v3.1.5**
being converted into the **map/graph spine of a fused recall engine**: its native vector machinery
(HNSW / DiskANN / KNN) is excised and **replaced by [TotalRecall](https://github.com/EonsofStupid/totalrecall)**
— a Qdrant-lineage vector engine — running **in-process**. One binary. SurrealQL in; graph, document and
relational served by the SurrealDB core; ANN served by TotalRecall's `segment`/`collection` crates. No HTTP
hop between the map and the treasure.

> **SurrealDB is the map. TotalRecall is the treasure. Every recall is a treasure hunt.**
> connectome is where the map and the treasure become one engine.

## Design law (fork discipline)

Identity = **connectome** (repo, top-level package, the `connectome` binary). Internals = upstream:
the library crates (`surrealdb-core`, `surrealdb-server`, `surrealdb-types`, `surrealdb-parser`, …) keep
their upstream names so releases sync **diff-based**, never merge-based. Same law as TotalRecall's
(identity = TotalRecall, internals = Qdrant).

## Provenance

| | |
|---|---|
| Upstream | `surrealdb/surrealdb` **v3.1.5** (`f8d7c511`) |
| Baseline tags | `surrealdb-v3.1.5-baseline` · `v3.1.5-connectome.0` |
| History | single `main`, one clean orphan baseline commit |
| Sibling engine | `EonsofStupid/totalrecall` (Qdrant v1.18.2 fork — the vector arm) |
| Consumer | clyffy (`clyffy-connectome.service`; `clyffy-storage` ConnectomeAdapter over HTTP `/sql`) |

## Local deltas vs stock SurrealDB v3.1.5 (baseline commit)

- Root `[package] name = "surreal"` → **`connectome`** (binary rename; `CARGO_BIN_EXE_connectome` /
  `NEXTEST_BIN_EXE_connectome` updated in `tests/surrealism_integration.rs`).
- Upstream `README.md` preserved at `docs/UPSTREAM_README.md`; this file replaces it.
- Nothing else. The vector excision and the TotalRecall fusion land as subsequent tagged work
  (`v3.1.5-connectome.1`, …) — see `docs/` as those phases land.

## Build

```bash
cargo build            # full workspace (toolchain pinned by rust-toolchain.toml)
./target/debug/connectome start --bind 127.0.0.1:8000 rocksdb:/path/to/kv
```

`build-connectome.sh` (production vector-OFF vs bench vector-ON shapes) arrives with the excision phase.

## Upstream sync (when a newer SurrealDB 3.x releases)

History is a clean orphan baseline, so syncs are diff-based: fetch upstream, `git diff vOLD vNEW |
git apply --3way` on a `sync/surrealdb-vNEW` branch, re-apply the identity + fusion deltas, re-verify,
retag `surrealdb-vNEW-baseline`. The upstream remote is push-disabled by design.
