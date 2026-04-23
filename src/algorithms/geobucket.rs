//! Yan-style geobucket for polynomial reduction accumulation.
//!
//! During Buchberger's algorithm, `reduce_poly` repeatedly subtracts scaled
//! reducers from a target polynomial.  With a flat sorted Vec, each step
//! requires an O(|target| + |reducer|) merge.  Over N steps with a target
//! that grows to size A, total work is O(N * A).
//!
//! A geobucket stores the polynomial across logarithmically-sized buckets.
//! Adding a polynomial of length R goes into the bucket of matching size.
//! When a bucket overflows, it cascades (merges) into the next bucket.
//! Amortized cost per addition: O(R * log(A)).
//!
//! This matches CoCoA's `geobucket.C` implementation:
//!   - Bucket capacities grow by factor 4: 128, 512, 2048, 8192, ...
//!   - Cascade threshold: when bucket length > 2x capacity
//!   - 20 buckets pre-reserved (supports polynomials up to ~10^12 terms)
//!
//! Unlike CoCoA's linked-list-based geobucket, this uses sorted Vecs
//! (matching feanor-math's polynomial representation).  The merge operation
//! is O(n+m) but avoids pointer chasing and has better cache locality.

use std::cmp::Ordering;

/// Geobucket configuration (matching CoCoA's `geobucket.C:36-38`).
const MIN_BUCKET_LEN: usize = 128;
const BUCKET_FACTOR: usize = 4;
const NUM_BUCKETS: usize = 20;

/// A single bucket in the geobucket.
struct Bucket<C> {
    /// Sorted polynomial terms in **ascending** (deg, order) -- matches feanor-math convention.
    terms: Vec<(C, u16, u64)>,
    /// Maximum capacity before cascading.
    max_len: usize,
}

/// Yan-style geobucket for accumulating polynomial reduction results.
///
/// Terms are stored as `(coefficient, deg, order)` triples sorted in
/// ascending DegRevLex order within each bucket.
pub(crate) struct Geobucket<C> {
    buckets: Vec<Bucket<C>>,
}

#[allow(dead_code)] // is_zero / leading_term / into_ascending are part of the
                    // public-ish API of this internal module but currently
                    // unused by `reduce_poly_geobucket`; keep for completeness.
impl<C> Geobucket<C> {
    /// Create a new empty geobucket.
    pub(crate) fn new() -> Self {
        let mut buckets = Vec::with_capacity(NUM_BUCKETS);
        let mut cap = MIN_BUCKET_LEN;
        for _ in 0..NUM_BUCKETS {
            buckets.push(Bucket {
                terms: Vec::new(),
                max_len: cap,
            });
            cap = cap.saturating_mul(BUCKET_FACTOR);
        }
        Geobucket { buckets }
    }

    /// Find the bucket index for a polynomial of the given length.
    fn bucket_index(&self, len: usize) -> usize {
        let mut idx = 0;
        let mut cap = MIN_BUCKET_LEN;
        while idx + 1 < self.buckets.len() && len > cap {
            idx += 1;
            cap = cap.saturating_mul(BUCKET_FACTOR);
        }
        idx
    }

    /// Load an initial polynomial (ascending order) into the geobucket.
    pub(crate) fn load_ascending(&mut self, terms: Vec<(C, u16, u64)>) {
        let idx = self.bucket_index(terms.len());
        self.buckets[idx].terms = terms;
    }

    /// Check if the geobucket is empty (all buckets empty).
    pub(crate) fn is_zero(&self) -> bool {
        self.buckets.iter().all(|b| b.terms.is_empty())
    }

    /// Get the leading term (largest in DegRevLex = last in ascending order).
    /// Returns `(bucket_idx, &coeff, deg, order)` or None if empty.
    pub(crate) fn leading_term(&self) -> Option<(usize, &C, u16, u64)> {
        let mut best: Option<(usize, &C, u16, u64)> = None;
        for (i, bucket) in self.buckets.iter().enumerate() {
            if let Some((c, deg, order)) = bucket.terms.last() {
                match &best {
                    None => best = Some((i, c, *deg, *order)),
                    Some((_, _, best_deg, best_order)) => {
                        let cmp = deg.cmp(best_deg).then_with(|| order.cmp(best_order));
                        if cmp == Ordering::Greater {
                            best = Some((i, c, *deg, *order));
                        }
                    }
                }
            }
        }
        best
    }

    /// Add a scaled reducer polynomial (in ascending order) into the geobucket.
    ///
    /// `scaled_terms` should be the result of `-(quo_coeff * quo_mono * reducer)`,
    /// already computed and sorted in ascending order.
    ///
    /// After adding, cascades if needed.
    pub(crate) fn add_poly<AddZeroCheck>(
        &mut self,
        scaled_terms: Vec<(C, u16, u64)>,
        merge_fn: &AddZeroCheck,
    )
    where
        AddZeroCheck: Fn(&mut C, C) -> bool,
    {
        if scaled_terms.is_empty() {
            return;
        }
        let idx = self.bucket_index(scaled_terms.len());
        if self.buckets[idx].terms.is_empty() {
            self.buckets[idx].terms = scaled_terms;
        } else {
            let existing = std::mem::take(&mut self.buckets[idx].terms);
            self.buckets[idx].terms = merge_sorted(existing, scaled_terms, merge_fn);
        }
        self.cascade_from(idx, merge_fn);
    }

    /// Cascade from bucket `idx` upward if it exceeds capacity.
    fn cascade_from<AddZeroCheck>(&mut self, start: usize, merge_fn: &AddZeroCheck)
    where
        AddZeroCheck: Fn(&mut C, C) -> bool,
    {
        for i in start..self.buckets.len() - 1 {
            if self.buckets[i].terms.len() <= 2 * self.buckets[i].max_len {
                break;
            }
            // Cascade: merge bucket[i] into bucket[i+1]
            let from = std::mem::take(&mut self.buckets[i].terms);
            if self.buckets[i + 1].terms.is_empty() {
                self.buckets[i + 1].terms = from;
            } else {
                let to = std::mem::take(&mut self.buckets[i + 1].terms);
                self.buckets[i + 1].terms = merge_sorted(to, from, merge_fn);
            }
        }
    }

    /// Remove the leading term from the geobucket (the largest term across
    /// all buckets).  Also combines equal leading terms from different buckets.
    ///
    /// Returns `Some((coeff, deg, order))` or `None` if empty.
    pub(crate) fn pop_leading<AddZeroCheck>(
        &mut self,
        merge_fn: &AddZeroCheck,
    ) -> Option<(C, u16, u64)>
    where
        AddZeroCheck: Fn(&mut C, C) -> bool,
    {
        loop {
            // Find the bucket with the largest trailing (= leading) term
            let mut best_idx: Option<usize> = None;
            let mut best_deg: u16 = 0;
            let mut best_order: u64 = 0;

            for (i, bucket) in self.buckets.iter().enumerate() {
                if let Some((_, deg, order)) = bucket.terms.last() {
                    let is_better = match best_idx {
                        None => true,
                        Some(_) => {
                            let cmp = deg.cmp(&best_deg).then_with(|| order.cmp(&best_order));
                            cmp == Ordering::Greater
                        }
                    };
                    if is_better {
                        best_idx = Some(i);
                        best_deg = *deg;
                        best_order = *order;
                    }
                }
            }

            let primary = match best_idx {
                Some(i) => i,
                None => return None,
            };

            let (mut coeff, deg, order) = self.buckets[primary].terms.pop().unwrap();

            // Combine with equal terms from other buckets
            let mut is_zero = false;
            for i in 0..self.buckets.len() {
                if i == primary { continue; }
                if let Some((_, d, o)) = self.buckets[i].terms.last() {
                    if *d == deg && *o == order {
                        let (c2, _, _) = self.buckets[i].terms.pop().unwrap();
                        is_zero = merge_fn(&mut coeff, c2);
                    }
                }
            }

            if is_zero {
                // Coefficients cancelled -- try next term
                continue;
            }

            return Some((coeff, deg, order));
        }
    }

    /// Drain all terms into a Vec in ascending order.
    pub(crate) fn into_ascending<AddZeroCheck>(mut self, merge_fn: &AddZeroCheck) -> Vec<(C, u16, u64)>
    where
        AddZeroCheck: Fn(&mut C, C) -> bool,
    {
        // Merge all buckets into one
        let mut result: Vec<(C, u16, u64)> = Vec::new();
        for bucket in &mut self.buckets {
            if bucket.terms.is_empty() { continue; }
            let terms = std::mem::take(&mut bucket.terms);
            if result.is_empty() {
                result = terms;
            } else {
                result = merge_sorted(result, terms, merge_fn);
            }
        }
        result
    }
}

/// Merge two sorted (ascending) polynomial Vecs, combining equal monomials.
/// Removes zero coefficients.
fn merge_sorted<C, F>(
    a: Vec<(C, u16, u64)>,
    b: Vec<(C, u16, u64)>,
    add_zero_check: &F,
) -> Vec<(C, u16, u64)>
where
    F: Fn(&mut C, C) -> bool,
{
    let mut result = Vec::with_capacity(a.len() + b.len());

    let mut a_iter = a.into_iter().peekable();
    let mut b_iter = b.into_iter().peekable();

    loop {
        match (a_iter.peek(), b_iter.peek()) {
            (None, None) => break,
            (Some(_), None) => {
                result.extend(a_iter);
                break;
            }
            (None, Some(_)) => {
                result.extend(b_iter);
                break;
            }
            (Some(a_term), Some(b_term)) => {
                let cmp = a_term.1.cmp(&b_term.1).then_with(|| a_term.2.cmp(&b_term.2));
                match cmp {
                    Ordering::Less => {
                        result.push(a_iter.next().unwrap());
                    }
                    Ordering::Greater => {
                        result.push(b_iter.next().unwrap());
                    }
                    Ordering::Equal => {
                        let mut term_a = a_iter.next().unwrap();
                        let term_b = b_iter.next().unwrap();
                        let is_zero = add_zero_check(&mut term_a.0, term_b.0);
                        if !is_zero {
                            result.push(term_a);
                        }
                    }
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_check(a: &mut i64, b: i64) -> bool {
        *a += b;
        *a == 0
    }

    #[test]
    fn test_geobucket_basic() {
        let mut gb = Geobucket::<i64>::new();
        // Load polynomial: 3 + 5x + 7x^2 (ascending: deg 0, 1, 2)
        gb.load_ascending(vec![(3, 0, 0), (5, 1, 0), (7, 2, 0)]);

        assert!(!gb.is_zero());

        // Leading term should be (7, 2, 0)
        let (_, c, d, o) = gb.leading_term().unwrap();
        assert_eq!(*c, 7);
        assert_eq!(d, 2);
        assert_eq!(o, 0);

        // Drain to ascending
        let result = gb.into_ascending(&add_check);
        assert_eq!(result, vec![(3, 0, 0), (5, 1, 0), (7, 2, 0)]);
    }

    #[test]
    fn test_geobucket_add_and_cancel() {
        let mut gb = Geobucket::<i64>::new();
        gb.load_ascending(vec![(3, 0, 0), (5, 1, 0)]);

        // Add poly that cancels the degree-1 term: (-5)*x
        gb.add_poly(vec![(-5, 1, 0)], &add_check);

        let result = gb.into_ascending(&add_check);
        assert_eq!(result, vec![(3, 0, 0)]);
    }

    #[test]
    fn test_geobucket_full_cancel() {
        let mut gb = Geobucket::<i64>::new();
        gb.load_ascending(vec![(3, 0, 0), (5, 1, 0)]);
        gb.add_poly(vec![(-3, 0, 0), (-5, 1, 0)], &add_check);

        let result = gb.into_ascending(&add_check);
        assert_eq!(result, vec![]);
    }

    #[test]
    fn test_geobucket_pop_leading() {
        let mut gb = Geobucket::<i64>::new();
        gb.load_ascending(vec![(1, 0, 0), (2, 1, 0), (3, 2, 0)]);

        let lt = gb.pop_leading(&add_check).unwrap();
        assert_eq!(lt, (3, 2, 0));

        let lt = gb.pop_leading(&add_check).unwrap();
        assert_eq!(lt, (2, 1, 0));

        let lt = gb.pop_leading(&add_check).unwrap();
        assert_eq!(lt, (1, 0, 0));

        assert!(gb.pop_leading(&add_check).is_none());
    }

    #[test]
    fn test_merge_sorted() {
        let a = vec![(1i64, 0u16, 0u64), (3, 2, 0)];
        let b = vec![(2i64, 1u16, 0u64), (4, 3, 0)];
        let result = merge_sorted(a, b, &add_check);
        assert_eq!(result, vec![(1, 0, 0), (2, 1, 0), (3, 2, 0), (4, 3, 0)]);
    }

    #[test]
    fn test_merge_sorted_cancel() {
        let a = vec![(5i64, 1u16, 0u64)];
        let b = vec![(-5i64, 1u16, 0u64)];
        let result = merge_sorted(a, b, &add_check);
        assert_eq!(result, vec![]);
    }
}
