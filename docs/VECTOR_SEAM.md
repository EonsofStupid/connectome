# The vector seam — where SurrealDB's native vectors were excised, and where TotalRecall plugs in

**Status:** Phase 2 (2026-07-11). This documents the internal boundary created by excising
SurrealDB v3.1.5's native vector machinery, so Phase 4 (THE MERGE) can plug TotalRecall's
`segment`/`collection` engine into exactly these sites — one binary, coded as one, fused at recall.

## Why

connectome (the map: graph/relations/docs) and TotalRecall (the treasure: the deep Qdrant fork,
sole vector engine) become ONE embedded engine. The first step is to sever SurrealDB's *own* vector
path cleanly, exposing a defined socket at two points: **(1) index build** and **(2) KNN execution**.
Gated behind the `vector-index` cargo feature (default-ON → upstream tests + diff-based syncs stay
shaped; the `connectome-production` build omits it).

## The excision (feature `vector-index` OFF → reject; the sites are the socket)

v3.1.5 routes KNN through **two** planners (mid-migration): the streaming executor
(`exec/planner`) and a legacy `.compute()` fallback (`idx/planner`). Both must be sealed, plus the
physical-eval backstop. Four guard sites, all `#[cfg(not(feature = "vector-index"))]`:

| # | Socket point | File | What it guards |
|---|---|---|---|
| a | **Index build** | `core/src/expr/statements/define/index.rs` (`DefineIndexStatement::compute`, after `is_allowed`) | Rejects `DEFINE INDEX … HNSW\|DISKANN`. Because no such index can be created, every downstream index-backed KNN path is transitively neutralized. **This is the index-build socket.** |
| b | **KNN exec (streaming)** | `core/src/exec/planner/select/mod.rs` (`has_knn` funnel, ~L937) | Rejects `<\|k\|>`, `<\|k,ef\|>`, `<\|k,dist\|>` in SELECT before any physical plan. |
| c | **KNN exec (legacy)** | `core/src/idx/planner/tree.rs` (`Expr::Binary` NearestNeighbor arm, ~L232) | Rejects KNN in the legacy chokepoint (SELECT fallback + UPDATE/DELETE/CREATE/UPSERT). |
| d | **Physical backstop** | `core/src/exec/physical_expr/ops.rs` (`NearestNeighbor` arm) | Fail-LOUD if a KNN op ever reaches physical eval — replaces the stock `Value::Bool(true)` (which silently passes *all* rows) with a hard error. Unreachable given a–c; defense against silent-wrong. |

Rejection message (all sites): `vector indexing is disabled in connectome; recall is served by TotalRecall`.

**Kept compiled (deliberately):** the HNSW/DiskANN modules (`idx/trees/hnsw`, `idx/trees/diskann`)
and the pure-math `vector::` functions (`fnc/vector.rs`, `fnc/util/math/vector.rs` — similarity,
distance) stay compiled in both shapes. cfg-ing out the modules would cascade `#[cfg]` across ~20
match arms + the on-disk revision enum — a large, non-additive diff that fights upstream syncs.
Runtime rejection at the two chokepoints is a ~4-block additive diff. `vector::distance::knn()`
already returns `Value::None` outside a KNN context, so it needs no guard.

## Where TotalRecall plugs in (Phase 4)

The same two socket points become the bind sites for the merged vector organ:

- **Socket (a) — index build**: `DEFINE INDEX … ` (a connectome vector index) builds a **TotalRecall
  segment** (Qdrant-class HNSW via `SegmentBuilder`) instead of SurrealDB's tree. The reject guard
  is replaced by a build dispatch into `segment`/`collection`.
- **Socket (b/c) — KNN execution**: the KNN operator executes ANN **through TotalRecall in-process**
  against that segment, returning candidate record-ids that flow back into SurrealDB's row pipeline.
  The reject guards are replaced by an execution dispatch.

SurrealQL surface unchanged; graph traversal + ANN in the same query, same process, no HTTP hop.
The trecall-side groundwork (`geo` 0.32 / `parking_lot` sans `deadlock_detection` pins) already
exists so `segment`/`collection` co-compile with `surrealdb-core` — see `TRECALL_RENAME_HANDOFF.md`.

## Build shapes

`./build-connectome.sh production` (vector-OFF, ships) · `./build-connectome.sh bench` (vector-ON,
the SurrealDB-HNSW comparison arm + upstream knn language-tests).
