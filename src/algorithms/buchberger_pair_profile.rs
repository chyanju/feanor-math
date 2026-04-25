//! Pair-count profiler observer (Sprint 2.6b instrumentation).
//!
//! Captures one row per sugar batch processed by [`buchberger_observed`]:
//!
//! ```text
//! sugar, n_pairs_processed, n_new_polys, n_zero_reductions,
//!     basis_size_before, basis_size_after, wall_time_ms
//! ```
//!
//! The CSV-friendly serialisation lets us answer the question:
//! *"is high-sugar pair processing dominating runtime on the 11 hard
//! benchmarks?"* before investing in Hilbert-driven pair pruning.
//!
//! Usage:
//!
//! ```ignore
//! use feanor_math::algorithms::buchberger::buchberger_observed;
//! use feanor_math::algorithms::buchberger_pair_profile::PairCountProfiler;
//!
//! let mut profiler = PairCountProfiler::new();
//! let gb = buchberger_observed(ring, gens, order, sort_fn, never_abort,
//!                              controller, &mut profiler).unwrap();
//! profiler.write_csv("pair_profile.csv").unwrap();
//! ```

use std::time::Instant;

use crate::ring::*;
use crate::rings::multivariate::MultivariatePolyRing;

use super::buchberger::BuchbergerObserver;

#[derive(Debug, Clone, Copy)]
pub struct BatchRow {
    pub sugar: usize,
    pub n_pairs_processed: usize,
    pub n_new_polys: usize,
    pub n_zero_reductions: usize,
    pub basis_size_before: usize,
    pub basis_size_after: usize,
    pub wall_time_ms: u128,
}

pub struct PairCountProfiler {
    rows: Vec<BatchRow>,
    pending_start: Option<(usize /*sugar*/, usize /*basis_size_before*/, Instant)>,
}

impl PairCountProfiler {
    pub fn new() -> Self {
        Self { rows: Vec::new(), pending_start: None }
    }

    pub fn rows(&self) -> &[BatchRow] {
        &self.rows
    }

    pub fn into_rows(self) -> Vec<BatchRow> {
        self.rows
    }

    /// Emit a CSV with header.  Path is opened with truncate semantics.
    pub fn write_csv<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        writeln!(
            f,
            "sugar,n_pairs_processed,n_new_polys,n_zero_reductions,basis_size_before,basis_size_after,wall_time_ms"
        )?;
        for r in &self.rows {
            writeln!(
                f,
                "{},{},{},{},{},{},{}",
                r.sugar,
                r.n_pairs_processed,
                r.n_new_polys,
                r.n_zero_reductions,
                r.basis_size_before,
                r.basis_size_after,
                r.wall_time_ms
            )?;
        }
        Ok(())
    }

    /// Convenience aggregator: total wall-time in milliseconds, partitioned
    /// by whether the sugar batch is "low" (sugar ≤ threshold) or "high"
    /// (sugar > threshold).  Returns `(low_ms, high_ms, n_pairs_high)`.
    pub fn split_by_sugar_threshold(&self, threshold: usize) -> (u128, u128, usize) {
        let mut low = 0u128;
        let mut high = 0u128;
        let mut n_high = 0usize;
        for r in &self.rows {
            if r.sugar <= threshold {
                low += r.wall_time_ms;
            } else {
                high += r.wall_time_ms;
                n_high += r.n_pairs_processed;
            }
        }
        (low, high, n_high)
    }
}

impl<P: RingStore> BuchbergerObserver<P> for PairCountProfiler
where
    P::Type: MultivariatePolyRing,
{
    fn on_sugar_batch_start(&mut self, sugar: usize, _n_pairs_to_process: usize, basis_size: usize) {
        self.pending_start = Some((sugar, basis_size, Instant::now()));
    }

    fn on_sugar_batch_end(
        &mut self,
        sugar: usize,
        n_pairs_processed: usize,
        n_new_polys: usize,
        n_zero_reductions: usize,
        basis_size_after: usize,
    ) {
        let (start_sugar, basis_size_before, start_at) = self
            .pending_start
            .take()
            .unwrap_or((sugar, basis_size_after, Instant::now()));
        debug_assert_eq!(start_sugar, sugar,
            "pair profiler: sugar drift between batch start ({}) and end ({})",
            start_sugar, sugar);
        let wall_time_ms = start_at.elapsed().as_millis();
        self.rows.push(BatchRow {
            sugar,
            n_pairs_processed,
            n_new_polys,
            n_zero_reductions,
            basis_size_before,
            basis_size_after,
            wall_time_ms,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_profiler_round_trip() {
        let p = PairCountProfiler::new();
        assert_eq!(p.rows().len(), 0);
        let (lo, hi, n) = p.split_by_sugar_threshold(10);
        assert_eq!((lo, hi, n), (0, 0, 0));
    }

    #[test]
    fn split_by_threshold_partitions_correctly() {
        let mut p = PairCountProfiler::new();
        p.rows.push(BatchRow {
            sugar: 3, n_pairs_processed: 5, n_new_polys: 2, n_zero_reductions: 3,
            basis_size_before: 10, basis_size_after: 12, wall_time_ms: 100,
        });
        p.rows.push(BatchRow {
            sugar: 8, n_pairs_processed: 7, n_new_polys: 1, n_zero_reductions: 6,
            basis_size_before: 12, basis_size_after: 13, wall_time_ms: 800,
        });
        let (lo, hi, n_high) = p.split_by_sugar_threshold(5);
        assert_eq!(lo, 100);
        assert_eq!(hi, 800);
        assert_eq!(n_high, 7);
    }

    /// Integration test: run buchberger on cyclic-6 with the profiler
    /// attached and verify the captured rows are sane.  This is the
    /// canonical canary for Sprint 2.6b instrumentation.
    #[test]
    fn integration_profiles_cyclic6() {
        use crate::computation::TEST_LOG_PROGRESS;
        use crate::algorithms::buchberger::{buchberger_observed, default_sort_fn};
        use crate::rings::multivariate::*;
        use crate::rings::multivariate::multivariate_impl::MultivariatePolyRingImpl;
        use crate::rings::zn::zn_static;

        let base = zn_static::Fp::<65537>::RING;
        let ring = MultivariatePolyRingImpl::new(base, 6);

        let cyclic6 = ring.with_wrapped_indeterminates_dyn(|[x, y, z, t, u, v]| {
            [
                x + y + z + t + u + v,
                x * y + y * z + z * t + t * u + x * v + u * v,
                x * y * z + y * z * t + z * t * u + x * y * v + x * u * v + t * u * v,
                x * y * z * t + y * z * t * u + x * y * z * v + x * y * u * v + x * t * u * v + z * t * u * v,
                x * y * z * t * u + x * y * z * t * v + x * y * z * u * v
                    + x * y * t * u * v + x * z * t * u * v + y * z * t * u * v,
                x * y * z * t * u * v - 1,
            ]
        });

        let mut profiler = PairCountProfiler::new();
        let gb = buchberger_observed(
            &ring,
            cyclic6,
            DegRevLex,
            default_sort_fn(&ring, DegRevLex),
            |_| false,
            TEST_LOG_PROGRESS,
            &mut profiler,
        )
        .expect("cyclic-6 should not abort");
        assert_eq!(45, gb.len(), "cyclic-6 GB cardinality mismatch");

        let rows = profiler.rows();
        assert!(!rows.is_empty(), "profiler captured no rows");
        // Every row must satisfy the count invariant.  Empty batches
        // (n_pairs_processed == 0) are legitimate: the driver dispatches
        // one whenever it advances `current_sugar`, even if no S-pair
        // qualifies (e.g. immediately after a basis reset to sugar 0).
        for r in rows {
            assert!(r.n_new_polys + r.n_zero_reductions == r.n_pairs_processed,
                "split_by_sugar invariant: {} new + {} zero != {} processed",
                r.n_new_polys, r.n_zero_reductions, r.n_pairs_processed);
        }
        let total_pairs: usize = rows.iter().map(|r| r.n_pairs_processed).sum();
        let total_new: usize = rows.iter().map(|r| r.n_new_polys).sum();
        // Cyclic-6 GB has 45 elements; we start with 6 input polys, so the
        // recursion must add ≥ 39 new polys.  (More if any are removed by
        // inter-reduction, which happens in this driver.)
        assert!(total_new >= 39,
            "expected ≥ 39 new polys produced; got {}", total_new);
        assert!(total_pairs > total_new,
            "expected zero reductions to occur (total {}, new {})",
            total_pairs, total_new);

        // Sanity dump if RUST_LOG=info or similar — but only on test
        // failure would this be noisy; print for manual inspection on
        // request via `--nocapture`.
        eprintln!("cyclic-6 pair profile: {} batches, {} total pairs ({} → polys, {} → 0)",
            rows.len(), total_pairs, total_new, total_pairs - total_new);
        for r in rows {
            eprintln!(
                "  sugar={:>3} pairs={:>4} new={:>3} zero={:>4} basis={}→{} t={}ms",
                r.sugar, r.n_pairs_processed, r.n_new_polys, r.n_zero_reductions,
                r.basis_size_before, r.basis_size_after, r.wall_time_ms
            );
        }
    }
}
