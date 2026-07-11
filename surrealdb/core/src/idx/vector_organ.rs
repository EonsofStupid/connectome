//! connectome A3 (THE MERGE) — TotalRecall's vector engine compiled INTO surrealdb-core.
//!
//! This is the in-process vector ORGAN: the `segment` crate (Qdrant-lineage, from the sibling
//! `EonsofStupid/totalrecall` fork) linked directly into the connectome engine binary, so a
//! `DEFINE INDEX … ` can build a TotalRecall segment and the KNN operator can execute through it
//! WITHOUT an HTTP hop. Gated behind the `vector-organ` feature (additive; keeps upstream syncs clean).
//!
//! This first module is the co-compile PROOF-OF-LINK — it references a real `segment` type so the
//! build must resolve and link the crate in the same binary as surrealdb-core. The DEFINE INDEX →
//! segment-build and KNN → segment-search wiring plugs in on top of this, at the sockets the
//! `vector-index` excision exposes (see `docs/VECTOR_SEAM.md`).

use segment::types::Distance;

/// Map connectome's index distance intent onto TotalRecall's `segment::types::Distance`.
/// (First real cross-engine binding: proves the two forks link in one process.)
pub(crate) fn organ_distance_cosine() -> Distance {
    Distance::Cosine
}

/// Proof-of-link probe — returns the linked engine's distance-metric name in-process.
pub fn organ_probe() -> &'static str {
    match organ_distance_cosine() {
        Distance::Cosine => "vector-organ: TotalRecall segment linked in-process (cosine)",
        _ => "vector-organ: TotalRecall segment linked in-process",
    }
}
