# A3 wiring spec — TotalRecall `segment` as connectome's vector index (implementation-depth)

**Status:** spec, 2026-07-11. Grounds the merge (A3) in the actual `surrealdb-core` code. Companion to
`VECTOR_SEAM.md` (the reject sockets) and the co-compile proof (branch `merge-a3-vector-organ`). All paths
under `surrealdb/core/src/`.

## The de-risking finding (read first)

**SurrealDB's native HNSW is ALREADY not synchronously-transactional at the graph level.** On a record
write, the txn-bound path writes only a durable **pending** row (`!hr`, `hnsw/index.rs:291`); the real graph
mutation runs later, out-of-band, in a compaction task (`kvs/ds.rs:2466 apply_hnsw_compaction` →
`hnsw/index.rs:436 apply_compaction`), in its own write txn, replay-safe and idempotent. **So the
"transactional write vs non-transactional segment" mismatch is solved by MIRRORING this existing two-phase
design — no 2PC, no new distributed invariant.** This is the load-bearing insight.

## The four seams (each: exact site → what to do)

### 1. INDEX BUILD (`DEFINE INDEX … `) — reuse the durable `IndexBuilder`
- Build is NOT synchronous in `compute`; it's delegated: `define/index.rs:240 run_indexing_with_builder` →
  `:308 index_builder.build(...)` → `kvs/index/builder.rs:227` (spawns a task) → the row-iteration loop
  `builder.rs:1040-1184` → per-batch `replay.rs:505 index_initial_batch` (decodes each row, computes
  indexed values, applies via `IndexOperation`).
- **Do:** keep the whole builder machine (free: durable build-state, generation ownership, cluster-safe
  takeover, `INFO FOR INDEX` progress, `CONCURRENTLY`). Only re-point the `Index::Hnsw` dispatch (seam 2) to
  the segment. The initial scan then populates the segment unchanged; `builder.rs:1304 compact_hnsw_pendings`
  becomes the segment flush.

### 2. WRITE PATH (index maintenance) — pending in txn, segment in compaction
- Dispatch: `idx/index.rs:118 IndexOperation::compute` → `Index::Hnsw(p) => self.index_hnsw(...)`
  (`idx/index.rs:128`). Caller: `doc/index.rs:127 IndexOperation::new` → `:145 compute` → `:148
  trigger_compaction` (all inside the record-write txn, `ctx.tx()`).
- **Do (phase A, txn-bound, `idx/index.rs:481 index_hnsw`):** KEEP writing the durable `!hr` pending
  transactionally (this is the write-ahead intent log; commits/rolls back atomically with the record). Do
  NOT upsert the segment here (non-transactional side effect in a rollback-able txn = forbidden).
- **Do (phase B, out-of-band, `idx/index.rs:330 apply_hnsw_compaction` / `kvs/ds.rs:2466`):** perform the
  actual `segment.upsert_point` / `segment.delete` here — already replay-safe + idempotent. old/new values:
  `Some/Some`=update, `Some/None`=delete, `None/Some`=insert.

### 3. KNN EXECUTION — single reroute point
- Both planners converge on `hnsw/index.rs:558 HnswIndex::knn_search(ctx, stk, pt, k, ef, cond_filter) ->
  VecDeque<KnnIteratorResult>` (result = `(Arc<RecordId>, dist f64, Option<record>)`). Streaming caller:
  `scan/knn.rs:288`; legacy caller: `executor.rs:981`.
- **Do:** implement ONE `knn_search`-signature method on the segment-backed index → satisfies both callers,
  zero pipeline changes. Convert `segment.search_batch` `ScoredPoint`s → `(Arc<RecordId>, f64, None)`
  (map segment score → SurrealDB distance, nearest-first). **`cond_filter` (residual WHERE):** preserve the
  permission-before-cond gate (`hnsw/filter.rs::is_record_truthy`, documented `scan/knn.rs:57-69`) — safest:
  over-fetch candidates from the segment and post-filter on the SurrealDB side (don't push perms into segment).
- **Read-your-writes:** `knn_search` today folds in un-compacted pendings (reads `!hg` generation). The
  segment path must either compact pendings before search, or search pendings-then-segment, so a committed
  record that hasn't been applied to the segment yet is still found.

### 4. PERSISTENCE — segment on FS, catalog/pending in KV
- Native HNSW persists as KV keys inside the same RocksDB (no separate file). The segment CANNOT be KV keys
  (it's a self-managing dir: vectors/, payload_storage/, graph). Give it a filesystem dir under the
  datastore path (`kvs/rocksdb/mod.rs:349 Datastore::new(path,…)`): **`<datastore_path>/trecall/<ns>_<db>_
  <table>_<index_id>/`**, keyed by the same `(ns,db,table,index_id)` tuple as the `!h*` keys. In-memory KV
  backends → a temp/RAM dir. `REMOVE INDEX`/overwrite (`define/index.rs:195-197 retire_durable_index`) must
  also delete the segment dir; process eviction mirrors `store/mod.rs:116 remove_hnsw_index`.
- **Crash consistency (FS↔KV split):** KV `!hr` pending queue + `!hg` generation are the source of truth. On
  open, reconcile: replay un-applied pendings into the segment; make segment upserts idempotent (point-id =
  deterministic fn of `RecordIdKey`, mirroring the `!hd/!hi` doc-id maps) so a redo after a crashed
  compaction is harmless.

## Implementation sequence (increments, each independently checkable)
1. **Segment lifecycle module** (`idx/vector_organ.rs` grows): open/create a segment at the FS dir for a
   given `(ns,db,table,index_id)` + `HnswParams`→SegmentConfig; the process-local cache mirroring
   `IndexStores::get_index_hnsw`. Unit-test: create → upsert → search → reopen.
2. **KNN reroute** (seam 3): the `knn_search`-signature method over the segment. Gate: a KNN query returns
   correct record-ids (over-fetch + post-filter). Both planners satisfied.
3. **Write path** (seam 2): segment upsert/delete in `apply_hnsw_compaction`; keep `!hr` pending in
   `index_hnsw`. Gate: create records → compaction → KNN finds them; rollback leaves segment untouched.
4. **Build** (seam 1): `Index::Hnsw` dispatch → segment; reuse `IndexBuilder`. Gate: DEFINE INDEX on a
   populated table builds a searchable segment; `INFO FOR INDEX` reports progress.
5. **Persistence/crash** (seam 4): reconcile-on-open; REMOVE INDEX cleanup. Gate: kill mid-compaction →
   reopen → recall intact.

Only once these pass end-to-end is `DEFINE INDEX … ` in the connectome engine building a TotalRecall segment
and KNN executing through it in-process — the unified engine.
