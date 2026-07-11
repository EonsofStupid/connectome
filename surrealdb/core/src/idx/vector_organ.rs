//! connectome A3 (THE MERGE) — TotalRecall's vector engine compiled INTO surrealdb-core.
//!
//! The in-process vector ORGAN: the `segment` crate (Qdrant-lineage, from `EonsofStupid/totalrecall`)
//! linked directly into the connectome engine, so `DEFINE INDEX … ` builds a TotalRecall segment and the
//! KNN operator searches it WITHOUT an HTTP hop. Gated behind the `vector-organ` feature.
//!
//! This module is **increment 1** of the wiring (see `docs/MERGE_WIRING_SPEC.md`): the segment LIFECYCLE —
//! open-or-load a persistent segment for one index, upsert/delete points keyed by SurrealDB record id,
//! search, and survive reopen. The DEFINE-INDEX build hook, the write-path (pending→compaction) hook, and
//! the `knn_search` reroute plug into THIS in the next increments.
//!
//! Modeled on the proven embeddable pattern (clyffy `TotalRecallAdapter`): a single appendable Plain/exact
//! cosine `Segment`, single-writer behind a `Mutex`, flushed to disk on drop. Plain is exact (recall = 1 by
//! construction) — the HNSW optimizer path (sub-linear ANN at scale) is a later increment; correctness of
//! the wiring is proven first on the exact segment.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Context as _, Result};
use segment::data_types::query_context::QueryContext;
use segment::data_types::vectors::{only_default_vector, QueryVector, DEFAULT_VECTOR_NAME};
use segment::entry::entry_point::{
    NonAppendableSegmentEntry, ReadSegmentEntry, SegmentEntry, StorageSegmentEntry,
};
use segment::segment::Segment;
use segment::segment_constructor::{build_segment, load_segment, normalize_segment_dir};
use segment::types::{
    Distance, ExtendedPointId, Indexes, Payload, PayloadStorageType, SegmentConfig, VectorDataConfig,
    VectorStorageType, WithPayload, WithVector,
};
use segment_common::counter::hardware_accumulator::{HwMeasurementAcc, HwSharedDrain};
use uuid::Uuid;

/// The payload key under which each point stores its originating SurrealDB record id (read back on search).
const RID_KEY: &str = "rid";

/// Captured hardware/usage signals for one organ (CLEAR SIGNALS — the segment's own CPU/IO measurement,
/// which the consuming clyffy layer drains into `metrics.recall_evals` alongside the funnel's StageSignals).
#[derive(Debug, Clone, Copy)]
pub struct VectorOrganStats {
    /// Live point count in the segment.
    pub points: usize,
    /// Cumulative CPU units measured across all ops on this organ.
    pub cpu: usize,
    /// Cumulative vector-storage IO read units.
    pub vector_io_read: usize,
    /// Cumulative vector-storage IO write units.
    pub vector_io_write: usize,
}

/// A TotalRecall `segment` serving one connectome vector index. Appendable Plain/exact cosine; persistent
/// (load-or-build under `dir`, flush on drop).
pub struct VectorOrgan {
    /// The single in-process segment (writes take `&mut`; callers hold `&self`, so a `Mutex`).
    seg: Mutex<Segment>,
    /// Monotonic op counter — every write must exceed the point's current version to apply.
    seq: AtomicU64,
    /// Vector dimension (fixed at open; every upsert must match).
    dim: usize,
    /// Organ-lifetime hardware-measurement accumulator. Every upsert/search hands the segment a counter cell
    /// tied to this (drains on drop) — so the engine's CPU/IO cost is CAPTURED, not discarded. Exposed via
    /// `stats()` for the consumer to persist. HwMeasurementAcc is a cheap Arc-shared drain.
    hw: HwMeasurementAcc,
}

impl VectorOrgan {
    /// Open-or-build the segment for an index under `dir` (a directory unique to this index; the spec derives
    /// it as `<datastore>/trecall/<ns>_<db>_<table>_<index_id>/`). Reloads an existing segment or builds a
    /// fresh Plain/cosine one of dimension `dim`.
    ///
    /// # Errors
    /// If the directory can't be created, or the segment can't be loaded/built.
    pub fn open(dir: &Path, dim: usize) -> Result<Self> {
        std::fs::create_dir_all(dir).with_context(|| format!("vector-organ dir {dir:?}"))?;
        let seg = match Self::load_existing(dir)? {
            Some(seg) => seg,
            None => {
                let config = SegmentConfig {
                    vector_data: HashMap::from([(
                        DEFAULT_VECTOR_NAME.to_owned(),
                        VectorDataConfig {
                            size: dim,
                            distance: Distance::Cosine,
                            storage_type: VectorStorageType::InRamChunkedMmap,
                            index: Indexes::Plain {},
                            quantization_config: None,
                            multivector_config: None,
                            datatype: None,
                        },
                    )]),
                    sparse_vector_data: Default::default(),
                    payload_storage_type: PayloadStorageType::Mmap,
                };
                build_segment(dir, &config, None, true).context("vector-organ segment build")?
            }
        };
        // Seed the op counter PAST the loaded version, else reloaded points decline new writes.
        let seq = AtomicU64::new(seg.version() + 1);
        // Un-gated capturing accumulator (`new()`/`Default` are `testing`-gated). Cells `accumulate` into
        // `request_drain`, which `get_cpu()`/`get_vector_io_*()` read — so op cost is captured.
        let hw = HwMeasurementAcc::new_with_metrics_drain(Arc::new(HwSharedDrain::default()));
        Ok(Self { seg: Mutex::new(seg), seq, dim, hw })
    }

    /// Captured usage signals (CLEAR SIGNALS): live point count + cumulative CPU/IO measured by the segment
    /// engine across this organ's ops. The consumer persists these to the warehouse.
    ///
    /// # Errors
    /// On lock poison.
    pub fn stats(&self) -> Result<VectorOrganStats> {
        let seg = self.seg.lock().map_err(|e| anyhow::anyhow!("vector-organ lock: {e}"))?;
        Ok(VectorOrganStats {
            points: seg.available_point_count(),
            cpu: self.hw.get_cpu(),
            vector_io_read: self.hw.get_vector_io_read(),
            vector_io_write: self.hw.get_vector_io_write(),
        })
    }

    /// Reload the first valid persisted segment under `dir` (if any).
    fn load_existing(dir: &Path) -> Result<Option<Segment>> {
        for entry in std::fs::read_dir(dir).with_context(|| format!("vector-organ read {dir:?}"))? {
            let path = entry?.path();
            if !path.is_dir() {
                continue;
            }
            if let Some((seg_path, uuid)) = normalize_segment_dir(&path).context("vector-organ normalize")? {
                let seg = load_segment(&seg_path, uuid, None, &AtomicBool::new(false))
                    .context("vector-organ load")?;
                return Ok(Some(seg));
            }
        }
        Ok(None)
    }

    /// Deterministic point id for a SurrealDB record id — stable across runs ⇒ idempotent upsert (re-indexing
    /// the same record updates the point, never duplicates). Mirrors the doc-id map role of the `!hd/!hi` keys.
    fn point_id(rid: &str) -> ExtendedPointId {
        ExtendedPointId::Uuid(Uuid::new_v5(&Uuid::NAMESPACE_OID, rid.as_bytes()))
    }

    /// Vector dimension this organ was opened at.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Insert-or-update the point for record `rid` with `vector`. The record id is stored in the payload so
    /// search returns it. Idempotent (same `rid` ⇒ same point).
    ///
    /// # Errors
    /// On dimension mismatch, lock poison, or a segment write failure.
    #[tracing::instrument(level = "debug", name = "vector_organ.upsert", skip_all, fields(rid = rid, dim = self.dim), err)]
    pub fn upsert(&self, rid: &str, vector: &[f32]) -> Result<()> {
        anyhow::ensure!(
            vector.len() == self.dim,
            "vector-organ dim mismatch: got {}, expected {}",
            vector.len(),
            self.dim
        );
        let point = Self::point_id(rid);
        let vectors = only_default_vector(vector);
        let mut map = serde_json::Map::new();
        map.insert(RID_KEY.to_string(), serde_json::Value::String(rid.to_string()));
        let pl = Payload(map);

        // Counter cell tied to the organ's accumulator: the segment's CPU/IO cost drains in on drop (CAPTURED).
        let hw = self.hw.get_counter_cell();
        let op_vec = self.seq.fetch_add(1, Ordering::SeqCst);
        let op_pl = self.seq.fetch_add(1, Ordering::SeqCst);
        let mut seg = self.seg.lock().map_err(|e| anyhow::anyhow!("vector-organ lock: {e}"))?;
        seg.upsert_point(op_vec, point, vectors, &hw).context("vector-organ upsert_point")?;
        seg.set_payload(op_pl, point, &pl, &None, &hw).context("vector-organ set_payload")?;
        Ok(())
    }

    /// Delete the point for record `rid` (no-op if absent).
    ///
    /// # Errors
    /// On lock poison or a segment delete failure.
    #[tracing::instrument(level = "debug", name = "vector_organ.delete", skip_all, fields(rid = rid), err)]
    pub fn delete(&self, rid: &str) -> Result<()> {
        let point = Self::point_id(rid);
        let hw = self.hw.get_counter_cell();
        let op = self.seq.fetch_add(1, Ordering::SeqCst);
        let mut seg = self.seg.lock().map_err(|e| anyhow::anyhow!("vector-organ lock: {e}"))?;
        seg.delete_point(op, point, &hw).context("vector-organ delete_point")?;
        Ok(())
    }

    /// Exact top-`k` cosine search. Returns `(record_id, score)` nearest-first (higher score = nearer).
    ///
    /// # Errors
    /// On lock poison or a segment search failure.
    #[tracing::instrument(level = "debug", name = "vector_organ.search", skip_all, fields(k = k, dim = self.dim, hits = tracing::field::Empty), err)]
    pub fn search(&self, vector: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        let qv: QueryVector = vector.to_vec().into();
        // Real accumulator (not disposable): search CPU/IO cost drains into the organ's stats.
        let ctx = QueryContext::new(usize::MAX, self.hw.clone());
        let sqc = ctx.get_segment_query_context();
        let seg = self.seg.lock().map_err(|e| anyhow::anyhow!("vector-organ lock: {e}"))?;
        let mut batches = seg
            .search_batch(
                DEFAULT_VECTOR_NAME,
                &[&qv],
                &WithPayload::from(true),
                &WithVector::from(false),
                None,
                k,
                None,
                &sqc,
            )
            .context("vector-organ search_batch")?;
        let scored = batches.drain(..).next().unwrap_or_default();
        let hits: Vec<(String, f32)> = scored
            .into_iter()
            .map(|sp| {
                let rid = sp
                    .payload
                    .as_ref()
                    .and_then(|p| p.0.get(RID_KEY))
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                (rid, sp.score)
            })
            .collect();
        tracing::Span::current().record("hits", hits.len());
        Ok(hits)
    }

    /// Flush to disk so `open` reloads the latest state (called on drop; exposed for the compaction flush).
    ///
    /// # Errors
    /// On lock poison or a segment flush failure.
    pub fn flush(&self) -> Result<()> {
        let seg = self.seg.lock().map_err(|e| anyhow::anyhow!("vector-organ lock: {e}"))?;
        seg.flush(true).context("vector-organ flush")?;
        Ok(())
    }
}

impl Drop for VectorOrgan {
    fn drop(&mut self) {
        if let Ok(seg) = self.seg.lock() {
            let _ = seg.flush(true);
        }
    }
}

/// Process-local cache of open `VectorOrgan`s, keyed by index identity — mirrors SurrealDB's
/// `IndexStores::get_index_hnsw` (`idx/trees/store/hnsw.rs`). The KNN reroute (increment 2) and the
/// DEFINE-INDEX build (increment 4) both fetch the organ for an index through here, so a segment is opened
/// once and shared. The on-disk dir is derived deterministically from the datastore path + index identity,
/// per `docs/MERGE_WIRING_SPEC.md` §4: `<base>/trecall/<ns>_<db>_<table>_<index_id>/`.
#[derive(Default)]
pub struct VectorOrganStore {
    organs: RwLock<HashMap<String, Arc<VectorOrgan>>>,
}

impl VectorOrganStore {
    /// The deterministic on-disk dir for one index's segment under the datastore `base`.
    pub fn organ_dir(base: &Path, ns: &str, db: &str, table: &str, index_id: u64) -> PathBuf {
        base.join("trecall").join(format!("{ns}_{db}_{table}_{index_id}"))
    }

    fn key(ns: &str, db: &str, table: &str, index_id: u64) -> String {
        format!("{ns}/{db}/{table}/{index_id}")
    }

    /// Get the cached organ for an index, or open-or-build it at its derived dir (dimension `dim`). Idempotent
    /// — the same index returns the same shared `VectorOrgan`.
    ///
    /// # Errors
    /// On lock poison or a segment open failure.
    pub fn get_or_open(
        &self,
        base: &Path,
        ns: &str,
        db: &str,
        table: &str,
        index_id: u64,
        dim: usize,
    ) -> Result<Arc<VectorOrgan>> {
        let key = Self::key(ns, db, table, index_id);
        if let Some(organ) = self
            .organs
            .read()
            .map_err(|e| anyhow::anyhow!("vector-organ store lock: {e}"))?
            .get(&key)
            .cloned()
        {
            return Ok(organ);
        }
        let mut w = self.organs.write().map_err(|e| anyhow::anyhow!("vector-organ store lock: {e}"))?;
        // Re-check under the write lock (another thread may have opened it).
        if let Some(organ) = w.get(&key).cloned() {
            return Ok(organ);
        }
        let dir = Self::organ_dir(base, ns, db, table, index_id);
        let organ = Arc::new(VectorOrgan::open(&dir, dim)?);
        w.insert(key, Arc::clone(&organ));
        Ok(organ)
    }

    /// Evict an index's organ from the cache (process-local; mirrors `remove_hnsw_index`). Flushes on drop of
    /// the last `Arc`. Returns whether an entry was present.
    ///
    /// # Errors
    /// On lock poison.
    pub fn evict(&self, ns: &str, db: &str, table: &str, index_id: u64) -> Result<bool> {
        let key = Self::key(ns, db, table, index_id);
        Ok(self
            .organs
            .write()
            .map_err(|e| anyhow::anyhow!("vector-organ store lock: {e}"))?
            .remove(&key)
            .is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Increment-1 gate: create → upsert → search → reopen. Two records in, exact search returns the nearer
    /// by its record id; after drop+reopen at the same dir the data survives.
    #[test]
    fn lifecycle_upsert_search_reopen() {
        let dir = std::env::temp_dir().join(format!("vector-organ-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);

        {
            let organ = VectorOrgan::open(&dir, 4).expect("open");
            organ.upsert("rec:a", &[1.0, 0.0, 0.0, 0.0]).expect("upsert a");
            organ.upsert("rec:b", &[0.0, 1.0, 0.0, 0.0]).expect("upsert b");

            let hits = organ.search(&[0.9, 0.1, 0.0, 0.0], 2).expect("search");
            assert!(!hits.is_empty(), "search returned hits");
            assert_eq!(hits[0].0, "rec:a", "nearest is rec:a (record id preserved through the segment)");
            assert!(hits[0].1 >= hits.get(1).map_or(0.0, |h| h.1), "scores descend");

            // idempotent upsert: re-indexing rec:a updates, doesn't duplicate
            organ.upsert("rec:a", &[1.0, 0.0, 0.0, 0.0]).expect("re-upsert a");
            let again = organ.search(&[1.0, 0.0, 0.0, 0.0], 5).expect("search");
            assert_eq!(again.iter().filter(|(r, _)| r == "rec:a").count(), 1, "no duplicate point");

            // CLEAR SIGNALS: the segment's CPU/IO cost is captured, not discarded.
            let stats = organ.stats().expect("stats");
            assert_eq!(stats.points, 2, "stats report live point count");
            assert!(stats.cpu > 0, "hardware CPU cost captured across ops (was discarded before)");
        } // drop → flush

        // reopen the SAME dir → data survives
        let reopened = VectorOrgan::open(&dir, 4).expect("reopen");
        let hits = reopened.search(&[1.0, 0.0, 0.0, 0.0], 1).expect("search after reopen");
        assert_eq!(hits.first().map(|h| h.0.as_str()), Some("rec:a"), "survives reopen");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Store gate: same index → same shared organ; distinct index → distinct organ; dir derivation; evict.
    #[test]
    fn store_caches_by_index_identity() {
        let base = std::env::temp_dir().join(format!("vorg-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let store = VectorOrganStore::default();

        // deterministic dir derivation
        let d = VectorOrganStore::organ_dir(&base, "clyffy", "connectome", "memory", 7);
        assert!(d.ends_with("trecall/clyffy_connectome_memory_7"), "dir derived from index identity: {d:?}");

        // same identity → same shared Arc (opened once)
        let a1 = store.get_or_open(&base, "clyffy", "connectome", "memory", 7, 4).expect("open a");
        let a2 = store.get_or_open(&base, "clyffy", "connectome", "memory", 7, 4).expect("get a");
        assert!(Arc::ptr_eq(&a1, &a2), "same index returns the same cached organ");

        // distinct index → distinct organ
        let b = store.get_or_open(&base, "clyffy", "connectome", "memory", 8, 4).expect("open b");
        assert!(!Arc::ptr_eq(&a1, &b), "different index → different organ");

        // the shared organ actually works
        a1.upsert("rec:x", &[1.0, 0.0, 0.0, 0.0]).expect("upsert via store organ");
        assert_eq!(a2.search(&[1.0, 0.0, 0.0, 0.0], 1).expect("search")[0].0, "rec:x", "shared instance");

        // evict
        assert!(store.evict("clyffy", "connectome", "memory", 7).expect("evict"), "evicted present entry");
        assert!(!store.evict("clyffy", "connectome", "memory", 7).expect("evict"), "already gone");

        let _ = std::fs::remove_dir_all(&base);
    }
}
