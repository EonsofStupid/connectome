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
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

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
use segment_common::counter::hardware_accumulator::HwMeasurementAcc;
use segment_common::counter::hardware_counter::HardwareCounterCell;
use uuid::Uuid;

/// The payload key under which each point stores its originating SurrealDB record id (read back on search).
const RID_KEY: &str = "rid";

/// A TotalRecall `segment` serving one connectome vector index. Appendable Plain/exact cosine; persistent
/// (load-or-build under `dir`, flush on drop).
pub struct VectorOrgan {
    /// The single in-process segment (writes take `&mut`; callers hold `&self`, so a `Mutex`).
    seg: Mutex<Segment>,
    /// Monotonic op counter — every write must exceed the point's current version to apply.
    seq: AtomicU64,
    /// Vector dimension (fixed at open; every upsert must match).
    dim: usize,
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
        Ok(Self { seg: Mutex::new(seg), seq, dim })
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

        let hw = HardwareCounterCell::disposable();
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
    pub fn delete(&self, rid: &str) -> Result<()> {
        let point = Self::point_id(rid);
        let hw = HardwareCounterCell::disposable();
        let op = self.seq.fetch_add(1, Ordering::SeqCst);
        let mut seg = self.seg.lock().map_err(|e| anyhow::anyhow!("vector-organ lock: {e}"))?;
        seg.delete_point(op, point, &hw).context("vector-organ delete_point")?;
        Ok(())
    }

    /// Exact top-`k` cosine search. Returns `(record_id, score)` nearest-first (higher score = nearer).
    ///
    /// # Errors
    /// On lock poison or a segment search failure.
    pub fn search(&self, vector: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        let qv: QueryVector = vector.to_vec().into();
        let ctx = QueryContext::new(usize::MAX, HwMeasurementAcc::disposable());
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
        Ok(scored
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
            .collect())
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
        } // drop → flush

        // reopen the SAME dir → data survives
        let reopened = VectorOrgan::open(&dir, 4).expect("reopen");
        let hits = reopened.search(&[1.0, 0.0, 0.0, 0.0], 1).expect("search after reopen");
        assert_eq!(hits.first().map(|h| h.0.as_str()), Some("rec:a"), "survives reopen");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
