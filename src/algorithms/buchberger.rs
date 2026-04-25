use std::cmp::min;
use std::fmt::Debug;

use append_only_vec::AppendOnlyVec;

use crate::computation::*;
use crate::delegate::{UnwrapHom, WrapHom};
use crate::divisibility::{DivisibilityRing, DivisibilityRingStore};
use crate::field::Field;
use crate::homomorphism::Homomorphism;
use crate::local::{PrincipalLocalRing, PrincipalLocalRingStore};
use crate::pid::PrincipalIdealRingStore;
use crate::ring::*;
use crate::rings::multivariate::*;
use crate::seq::*;

/// Observer trait for Buchberger algorithm steps.
///
/// Implement this trait to trace the derivation history of polynomials
/// during a Groebner basis computation.  This enables UNSAT core
/// extraction by tracking which input polynomials contribute to each
/// derived polynomial.
///
/// All methods have default empty implementations, so you only need to
/// override the ones you care about.
#[stability::unstable(feature = "enable")]
pub trait BuchbergerObserver<P: RingStore>
where
    P::Type: MultivariatePolyRing,
{
    /// Called once after the initial inter-reduction of input polynomials,
    /// reporting how many basis elements survived (indices 0..count-1).
    fn on_initial_basis(&mut self, _count: usize) {}

    /// Called when an S-polynomial `spoly(basis[i], basis[j])` has been
    /// reduced to a non-zero polynomial `result`, which will be added to
    /// the basis.
    ///
    /// `parent_indices` contains the basis indices of the parents:
    ///   - For `SPoly::Standard(i, j)`: `[i, j]`
    ///   - For `SPoly::Nilpotent(i, _)`: `[i]`
    ///
    /// `result` is the reduced S-polynomial (before it gets its own basis
    /// index assigned).
    fn on_new_poly(&mut self, _parent_indices: &[usize], _result: &El<P>) {}

    /// Called when inter-reduction replaces `basis[index]` with a new
    /// reduced form.  The new form depends on all other basis elements
    /// that were used as reducers.
    fn on_inter_reduce(&mut self, _index: usize, _new_form: &El<P>) {}

    /// Sprint 2.3.5 (T-Sug-5): Called once per S-pair reduction *after*
    /// the running Giovini–Mora–Niesi–Robbiano sugar update has run.
    ///
    /// * `initial_sugar` — the S-pair's a-priori sugar (the old behavior
    ///   would have stored exactly this for the new basis element).
    /// * `final_sugar` — the running sugar after all reductions
    ///   (raise-only); equal to `initial_sugar` when no reduction
    ///   produced a higher candidate.
    /// * `n_raises` — number of `Sugar::my_update` calls that strictly
    ///   raised the value during this reduction.
    ///
    /// Profiling counters can use `final_sugar > initial_sugar` to
    /// measure how often the running update *actually* tightens the
    /// sugar estimate.  When `n_raises == 0` the running update was
    /// a no-op for this S-pair.
    fn on_running_sugar(&mut self, _initial_sugar: usize, _final_sugar: usize, _n_raises: usize) {}

    /// Sprint 2.6b — pair-count profiling hook.  Called once just before
    /// the loop dispatches the next sugar batch (i.e. `current_sugar` has
    /// advanced past the just-finished batch, or this is the first batch).
    ///
    /// Use [`on_sugar_batch_end`](Self::on_sugar_batch_end) to capture the
    /// counts produced by the batch we are about to dispatch.
    fn on_sugar_batch_start(&mut self, _sugar: usize, _n_pairs_to_process: usize, _basis_size: usize) {}

    /// Sprint 2.6b — pair-count profiling hook.  Called once after a sugar
    /// batch has been fully processed (all S-pairs of `sugar` reduced and
    /// any new basis elements appended; inter-reduction may still happen
    /// later).  Reports:
    /// * `n_pairs_processed` — number of S-pairs in this batch (before
    ///   any in-batch pruning by criteria already applied).
    /// * `n_new_polys` — number of S-pair reductions that produced a
    ///   non-zero remainder (i.e. would NOT have been pruned by an
    ///   omniscient Hilbert oracle).
    /// * `n_zero_reductions` — number that reduced to zero (the prime
    ///   target for Hilbert pair-count pruning).
    /// * `basis_size_after` — basis cardinality after appending new polys.
    fn on_sugar_batch_end(
        &mut self,
        _sugar: usize,
        _n_pairs_processed: usize,
        _n_new_polys: usize,
        _n_zero_reductions: usize,
        _basis_size_after: usize,
    ) {}
}

/// A no-op observer that does nothing.
#[stability::unstable(feature = "enable")]
pub struct NoObserver;

impl<P: RingStore> BuchbergerObserver<P> for NoObserver
where
    P::Type: MultivariatePolyRing,
{}

/// MISSING (Sprint 2.3.1, R4 §1.2 / `SugarDegree.C:214-218`):
/// `StdDegBase`-equivalent wrapper around the standard-degree sugar value.
///
/// This newtype mirrors CoCoA's `StdDegBase`, exposing the two canonical
/// operations:
///
/// * `my_mul(pp_deg)` — sugar after multiplying the underlying polynomial
///   by a power-product of total degree `pp_deg`:  `s += pp_deg`.
/// * `my_update(cofactor_deg, other)` — sugar after the in-place addition
///   `f += cofactor * g`, where `cofactor` is a power-product of total
///   degree `cofactor_deg` and `other = sugar(g)`:
///     `s := max(s, cofactor_deg + other)`.
///
/// This is the canonical Giovini–Mora–Niesi–Robbiano sugar update and is
/// intentionally a *raise-only* operation — `my_update` can never lower
/// the sugar value.
///
/// We keep `Sugar` as a `#[repr(transparent)]` newtype around `usize` so
/// it remains `Copy` and zero-cost vs. the previous bare `usize`.  The
/// trait-light surface (no allocation, no virtual dispatch) lets us thread
/// it through the per-S-pair reduction loop without overhead.
#[stability::unstable(feature = "enable")]
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Sugar(usize);

impl Sugar {
    #[inline] pub fn new(value: usize) -> Self { Self(value) }
    #[inline] pub fn value(self) -> usize { self.0 }

    /// Sugar after `f *= pp` where `pp` has total degree `pp_deg`.
    /// Mirrors `StdDegBase::myMul` (`SugarDegree.C:214`).
    #[inline]
    pub fn my_mul(&mut self, pp_deg: usize) { self.0 += pp_deg; }

    /// Sugar after `f += cofactor * g` where `cofactor` has total degree
    /// `cofactor_deg` and `sugar(g) = other`.
    /// Mirrors `StdDegBase::myUpdate` (`SugarDegree.C:215-218`).
    #[inline]
    pub fn my_update(&mut self, cofactor_deg: usize, other: Sugar) {
        let candidate = cofactor_deg + other.0;
        if candidate > self.0 { self.0 = candidate; }
    }

    /// Build the S-pair sugar from the two parent polys' sugars and the
    /// total degree of `lcm(LT(f), LT(g))`.  Mirrors `NewSugar(GPair)`
    /// (R4 §1.4 / `TmpGPair.C:36-45`).
    ///
    /// Standard-degree formula:
    ///   `sugar(S(f,g)) = max(sugar(f) + lcm_deg - deg(LT(f)),
    ///                        sugar(g) + lcm_deg - deg(LT(g)))`.
    ///
    /// Equivalent unrolled form of `s := sugar(f); s.my_mul(c1_deg);
    /// s.my_update(c2_deg, sugar(g))`, which is what CoCoA does literally.
    /// We expose the closed form for callers that already cache the
    /// degrees (the original site does).
    #[inline]
    pub fn for_spair(
        sugar_f: Sugar,
        deg_lt_f: usize,
        sugar_g: Sugar,
        deg_lt_g: usize,
        lcm_deg: usize,
    ) -> Sugar {
        debug_assert!(deg_lt_f <= lcm_deg, "lcm degree must dominate LT(f) degree");
        debug_assert!(deg_lt_g <= lcm_deg, "lcm degree must dominate LT(g) degree");
        let c1_deg = lcm_deg - deg_lt_f;
        let c2_deg = lcm_deg - deg_lt_g;
        Sugar(std::cmp::max(sugar_f.0 + c1_deg, sugar_g.0 + c2_deg))
    }
}

impl From<usize> for Sugar { #[inline] fn from(v: usize) -> Self { Sugar(v) } }
impl From<Sugar> for usize { #[inline] fn from(s: Sugar) -> Self { s.0 } }

#[stability::unstable(feature = "enable")]
#[derive(PartialEq, Clone, Eq, Hash)]
pub enum SPoly {
    /// S-polynomial pair with cached sugar degree and lcm degree.
    Standard {
        i: usize,
        j: usize,
        /// Sugar degree of this S-pair (Giovini-Mora-Niesi-Robbiano heuristic).
        sugar: usize,
        /// Cached degree of lcm(LT(g_i), LT(g_j)).
        lcm_deg: usize,
        /// MISSING-2 (R1 §7 #2): Monotonic creation timestamp; FIFO
        /// tiebreaker on (sugar, lcm_deg) ties.  Mirrors `GPair::myAge`
        /// from `CoCoA-EP/src/AlgebraicCore/TmpGPair.[CH]`.
        age: u64,
    },
    Nilpotent {
        /// poly index
        idx: usize,
        /// power-of-p multiplier
        k: usize,
        /// Sugar degree.
        sugar: usize,
        /// Cached lcm degree.
        lcm_deg: usize,
        /// MISSING-2: see Standard.age.
        age: u64,
    },
}

/// MISSING-2: Source of monotonic age timestamps for S-polynomials.
///
/// Used as the third sort key in `default_sort_fn` (after sugar, then
/// lcm_deg).  Process-wide counter is fine: ages only need to be
/// monotonic *within* a single buchberger run, and `u64` overflow at
/// ~2e19 is unreachable.
fn next_spoly_age() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

impl SPoly {
    /// Convenience accessors for sugar degree.
    fn sugar(&self) -> usize {
        match self {
            SPoly::Standard { sugar, .. } => *sugar,
            SPoly::Nilpotent { sugar, .. } => *sugar,
        }
    }
    /// Convenience accessor for cached lcm degree.
    fn cached_lcm_deg(&self) -> usize {
        match self {
            SPoly::Standard { lcm_deg, .. } => *lcm_deg,
            SPoly::Nilpotent { lcm_deg, .. } => *lcm_deg,
        }
    }
    /// MISSING-2: Convenience accessor for FIFO age timestamp.
    fn age(&self) -> u64 {
        match self {
            SPoly::Standard { age, .. } => *age,
            SPoly::Nilpotent { age, .. } => *age,
        }
    }
}

impl Debug for SPoly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SPoly::Standard { i, j, sugar, .. } => write!(f, "S({}, {})s{}", i, j, sugar),
            SPoly::Nilpotent { idx, k, sugar, .. } => write!(f, "p^{} F({})s{}", k, idx, sugar),
        }
    }
}

fn term_xlcm<P>(
    ring: P,
    (l_c, l_m): (&PolyCoeff<P>, &PolyMonomial<P>),
    (r_c, r_m): (&PolyCoeff<P>, &PolyMonomial<P>),
) -> (
    (PolyCoeff<P>, PolyMonomial<P>),
    (PolyCoeff<P>, PolyMonomial<P>),
    (PolyCoeff<P>, PolyMonomial<P>),
)
where
    P: RingStore,
    P::Type: MultivariatePolyRing,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
{
    let d_c = ring.base_ring().ideal_gen(l_c, r_c);
    let m_m = ring.monomial_lcm(ring.clone_monomial(l_m), r_m);
    let l_factor = ring.base_ring().checked_div(r_c, &d_c).unwrap();
    let r_factor = ring.base_ring().checked_div(l_c, &d_c).unwrap();
    let m_c = ring
        .base_ring()
        .mul_ref_snd(ring.base_ring().mul_ref_snd(d_c, &r_factor), &l_factor);
    return (
        (
            l_factor,
            ring.monomial_div(ring.clone_monomial(&m_m), l_m).ok().unwrap(),
        ),
        (
            r_factor,
            ring.monomial_div(ring.clone_monomial(&m_m), r_m).ok().unwrap(),
        ),
        (m_c, m_m),
    );
}

fn term_lcm<P>(
    ring: P,
    (l_c, l_m): (&PolyCoeff<P>, &PolyMonomial<P>),
    (r_c, r_m): (&PolyCoeff<P>, &PolyMonomial<P>),
) -> (PolyCoeff<P>, PolyMonomial<P>)
where
    P: RingStore,
    P::Type: MultivariatePolyRing,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
{
    let d_c = ring.base_ring().ideal_gen(l_c, r_c);
    let m_m = ring.monomial_lcm(ring.clone_monomial(l_m), r_m);
    let l_factor = ring.base_ring().checked_div(r_c, &d_c).unwrap();
    let r_factor = ring.base_ring().checked_div(l_c, &d_c).unwrap();
    let m_c = ring
        .base_ring()
        .mul_ref_snd(ring.base_ring().mul_ref_snd(d_c, &r_factor), &l_factor);
    return (m_c, m_m);
}

impl SPoly {
    #[stability::unstable(feature = "enable")]
    pub fn lcm_term<P, O>(&self, ring: P, basis: &[El<P>], order: O) -> (PolyCoeff<P>, PolyMonomial<P>)
    where
        P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
        O: MonomialOrder + Copy,
    {
        match self {
            SPoly::Standard { i, j, .. } => term_lcm(
                &ring,
                ring.LT(&basis[*i], order).unwrap(),
                ring.LT(&basis[*j], order).unwrap(),
            ),
            SPoly::Nilpotent { idx, k, .. } => {
                let (c, m) = ring.LT(&basis[*idx], order).unwrap();
                (
                    ring.base_ring().mul_ref_fst(
                        c,
                        ring.base_ring()
                            .pow(ring.base_ring().clone_el(ring.base_ring().max_ideal_gen()), *k),
                    ),
                    ring.clone_monomial(m),
                )
            }
        }
    }

    #[stability::unstable(feature = "enable")]
    pub fn poly<P, O>(&self, ring: P, basis: &[El<P>], order: O) -> El<P>
    where
        P: RingStore + Copy,
        P::Type: MultivariatePolyRing,
        <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
        O: MonomialOrder + Copy,
    {
        match self {
            SPoly::Standard { i, j, .. } => {
                let (f1_factor, f2_factor, _) = term_xlcm(
                    &ring,
                    ring.LT(&basis[*i], order).unwrap(),
                    ring.LT(&basis[*j], order).unwrap(),
                );
                let mut f1_scaled = ring.clone_el(&basis[*i]);
                ring.mul_assign_monomial(&mut f1_scaled, f1_factor.1);
                ring.inclusion().mul_assign_map(&mut f1_scaled, f1_factor.0);
                let mut f2_scaled = ring.clone_el(&basis[*j]);
                ring.mul_assign_monomial(&mut f2_scaled, f2_factor.1);
                ring.inclusion().mul_assign_map(&mut f2_scaled, f2_factor.0);
                return ring.sub(f1_scaled, f2_scaled);
            }
            SPoly::Nilpotent { idx, k, .. } => {
                let mut result = ring.clone_el(&basis[*idx]);
                ring.inclusion().mul_assign_map(
                    &mut result,
                    ring.base_ring()
                        .pow(ring.base_ring().clone_el(ring.base_ring().max_ideal_gen()), *k),
                );
                return result;
            }
        }
    }
}

#[inline(never)]
fn find_reducer<'a, 'b, P, O, I>(
    ring: P,
    f: &El<P>,
    reducers: I,
    order: O,
) -> Option<(usize, &'a El<P>, PolyCoeff<P>, PolyMonomial<P>)>
where
    P: RingStore + Copy,
    P::Type: MultivariatePolyRing,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
    O: MonomialOrder + Copy,
    I: Iterator<Item = (&'a El<P>, &'b AugLm)>,
{
    if ring.is_zero(f) {
        return None;
    }
    let (f_lc, f_lm) = ring.LT(f, order).unwrap();
    let f_lm_expanded = ring.expand_monomial(f_lm);
    let f_mask = divmask(&f_lm_expanded);
    // MISSING-3: Select shortest reducer among all valid ones
    reducers
        .enumerate()
        .filter_map(|(i, (reducer, reducer_aug))| {
            // MISSING-4: Use cached DivMask from AugLm
            if (reducer_aug.mask & !f_mask) != 0 {
                return None;
            }
            if (0..ring.indeterminate_count()).all(|j| reducer_aug.exponents[j] <= f_lm_expanded[j]) {
                let (r_lc, r_lm) = ring.LT(reducer, order).unwrap();
                let quo_m = ring.monomial_div(ring.clone_monomial(f_lm), r_lm).ok().unwrap();
                if let Some(quo_c) = ring.base_ring().checked_div(f_lc, r_lc) {
                    return Some((i, reducer, quo_c, quo_m));
                }
            }
            return None;
        })
        .min_by_key(|(_, reducer, _, _)| ring.terms(*reducer).count())
}

#[inline(never)]
fn filter_spoly<P, O>(ring: P, new_spoly: &SPoly, basis: &[El<P>], order: O) -> Option<usize>
where
    P: RingStore + Copy,
    P::Type: MultivariatePolyRing,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
    O: MonomialOrder + Copy,
{
    match new_spoly {
        SPoly::Standard { i, j: k, .. } => {
            let (i, k) = (*i, *k);
            assert!(i < k);
            let (bi_c, bi_m) = ring.LT(&basis[i], order).unwrap();
            let (bk_c, bk_m) = ring.LT(&basis[k], order).unwrap();
            let (S_c, S_m) = term_lcm(ring, (bi_c, bi_m), (bk_c, bk_m));
            let S_c_val = ring.base_ring().valuation(&S_c).unwrap();

            if S_c_val == 0
                && order.eq_mon(
                    ring,
                    &ring.monomial_div(ring.clone_monomial(&S_m), bi_m).ok().unwrap(),
                    bk_m,
                )
            {
                return Some(usize::MAX);
            }

            (0..k)
                .filter_map(|j| {
                    if j == i {
                        return None;
                    }
                    // more experiments needed - for some weird reason, replacing "properly divides"
                    // with "divides" (assuming I didn't make a mistake) leads
                    // to terrible performance
                    let (bj_c, bj_m) = ring.LT(&basis[j], order).unwrap();
                    let (f_c, f_m) = term_lcm(ring, (bj_c, bj_m), (bk_c, bk_m));
                    let f_c_val = ring.base_ring().valuation(&f_c).unwrap();

                    if j < i && order.eq_mon(ring, &f_m, &S_m) && f_c_val <= S_c_val {
                        return Some(j);
                    }
                    if let Ok(quo) = ring.monomial_div(ring.clone_monomial(&S_m), &f_m)
                        && f_c_val <= S_c_val
                        && (f_c_val < S_c_val || ring.monomial_deg(&quo) > 0)
                    {
                        return Some(j);
                    }
                    return None;
                })
                .next()
        }
        SPoly::Nilpotent { idx: i, k, .. } => {
            let (i, k) = (*i, *k);
            let nilpotent_power = ring.base_ring().nilpotent_power().unwrap();
            let f = &basis[i];

            let mut smallest_elim_coeff_valuation = usize::MAX;
            let mut current = ring.LT(f, order).unwrap();
            while ring.base_ring().valuation(current.0).unwrap() + k >= nilpotent_power {
                smallest_elim_coeff_valuation = min(
                    smallest_elim_coeff_valuation,
                    ring.base_ring().valuation(current.0).unwrap(),
                );
                let next = ring.largest_term_lt(f, order, current.1);
                if next.is_none() {
                    return Some(usize::MAX);
                }
                current = next.unwrap();
            }
            assert!(
                smallest_elim_coeff_valuation == usize::MAX || smallest_elim_coeff_valuation + k >= nilpotent_power
            );
            if smallest_elim_coeff_valuation == usize::MAX || smallest_elim_coeff_valuation + k > nilpotent_power {
                return Some(usize::MAX);
            } else {
                return None;
            }
        }
    }
}

#[stability::unstable(feature = "enable")]
pub fn default_sort_fn<P, O>(_ring: P, _order: O) -> impl FnMut(&mut [SPoly], &[El<P>])
where
    P: RingStore + Copy,
    P::Type: MultivariatePolyRing,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
    O: MonomialOrder + Copy,
{
    move |open, _basis| {
        // Sort by (sugar, lcm_deg, age) ascending.  Age = FIFO order of
        // S-pair construction (MISSING-2, mirrors `GPair::operator<` in
        // CoCoA-EP `TmpGPair.C`).  Age as third key ensures a
        // deterministic, reproducible selection on (sugar, lcm_deg)
        // ties — earlier pairs get to react first, matching CoCoA's
        // `myAge` priority and avoiding permutation sensitivity.
        //
        // Since the main loop pops from the END of the list, sort
        // descending so smallest (sugar, lcm_deg, age) sits at the tail
        // and is selected first.
        open.sort_by(|a, b| {
            b.sugar().cmp(&a.sugar())
                .then_with(|| b.cached_lcm_deg().cmp(&a.cached_lcm_deg()))
                .then_with(|| b.age().cmp(&a.age()))
        })
    }
}

#[stability::unstable(feature = "enable")]
pub type ExpandedMonomial = Vec<usize>;

/// Compute a DivMask for an expanded monomial.
///
/// OPTIMIZATION (T2.2, CoCoA `DivMask.H` "EvenPowers" variant):
/// When `n_vars ≤ 64` distribute the 64 mask bits across variables,
/// allocating `b = ⌊64 / n_vars⌋` bits per variable.  Bit `k` (within
/// the variable's slot) is set iff the variable's exponent `e ≥ 2^k`.
/// Divisibility on exponents (`a_i ≤ b_i`) implies subset on per-
/// variable bit patterns (`(a_i ≥ 2^k) ⇒ (b_i ≥ 2^k)`), preserving
/// the invariant `(mask_a & !mask_b) == 0  ⇐  a | b`.
///
/// OPTIMIZATION (T2.3, CoCoA `DivMask.H` "Hashing" variant):
/// When `n_vars > 64` we cannot give each variable its own bit; the
/// legacy fallback gave the first 64 vars one bit each and ignored the
/// tail.  That was sound (tail-var exponents simply weren't tested,
/// keeping the rejection conservative) but very weak.  The hashing
/// variant maps each variable to one of `B = 8` buckets via a fixed
/// deterministic mixing function, then within each bucket runs the
/// same EvenPowers encoding on `max(exponents in bucket)`.  Subset
/// property is preserved because `max(a_i) ≤ max(b_i)` whenever
/// `a_i ≤ b_i` for all i in the bucket — so divisibility implies the
/// mask invariant, just as for EvenPowers.
fn divmask(exponents: &[usize]) -> u64 {
    let n_vars = exponents.len();
    if n_vars == 0 {
        return 0;
    }
    if n_vars <= 64 {
        // EvenPowers: b bits per variable, threshold 2^k (k = 0..b-1).
        let b = 64 / n_vars;
        let mut mask: u64 = 0;
        for (i, &e) in exponents.iter().enumerate() {
            if e == 0 {
                continue;
            }
            let base_bit = i * b;
            let mut threshold: usize = 1;
            for k in 0..b {
                if e >= threshold {
                    mask |= 1u64 << (base_bit + k);
                } else {
                    break;
                }
                threshold = match threshold.checked_shl(1) {
                    Some(v) => v,
                    None => break,
                };
            }
        }
        return mask;
    }
    // Hashing variant: 8 buckets × 8 bits each.  Bucket index for var i
    // is determined by a cheap deterministic mixing function on i.
    const NUM_BUCKETS: usize = 8;
    const BITS_PER_BUCKET: usize = 8;
    let mut bucket_max: [usize; NUM_BUCKETS] = [0; NUM_BUCKETS];
    for (i, &e) in exponents.iter().enumerate() {
        if e == 0 {
            continue;
        }
        // Mix the index with a multiplicative hash; reduce mod NUM_BUCKETS.
        // Wrapping arithmetic keeps it cheap and deterministic.
        let h = (i as u64).wrapping_mul(0x9E3779B97F4A7C15);
        let bucket = (h as usize) % NUM_BUCKETS;
        if e > bucket_max[bucket] {
            bucket_max[bucket] = e;
        }
    }
    let mut mask: u64 = 0;
    for bucket in 0..NUM_BUCKETS {
        let e = bucket_max[bucket];
        if e == 0 {
            continue;
        }
        let base_bit = bucket * BITS_PER_BUCKET;
        let mut threshold: usize = 1;
        for k in 0..BITS_PER_BUCKET {
            if e >= threshold {
                mask |= 1u64 << (base_bit + k);
            } else {
                break;
            }
            threshold = match threshold.checked_shl(1) {
                Some(v) => v,
                None => break,
            };
        }
    }
    mask
}

/// Augmented leading monomial info for a basis element.
/// Stores the expanded exponent vector and cached DivMask for fast divisibility rejection.
#[stability::unstable(feature = "enable")]
pub struct AugLm {
    /// Expanded exponent vector of LT.
    pub exponents: ExpandedMonomial,
    /// DivMask for fast divisibility rejection.
    pub mask: u64,
}

/// Augment a polynomial with its expanded LT exponents and cached DivMask.
fn augment_lm<P, O>(ring: P, f: El<P>, order: O) -> (El<P>, AugLm)
where
    P: RingStore + Copy,
    P::Type: MultivariatePolyRing,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
    O: MonomialOrder,
{
    let exponents = ring.expand_monomial(ring.LT(&f, order).unwrap().1);
    let mask = divmask(&exponents);
    return (f, AugLm { exponents, mask });
}

/// MISSING-3 (R1 §7 #3): Detect ideal = R, i.e. the GB contains a poly
/// reducing to a *unit* (a non-zero constant whose leading coefficient is
/// invertible).  Mirrors `TmpGReductor::IamComputingGroebnerForOne` from
/// `CoCoA-EP/src/AlgebraicCore/TmpGReductor.C:683-694`: as soon as a unit
/// shows up in the basis, the ideal is the whole ring and we can short
/// circuit, returning `[1]` as a canonical reduced GB.
///
/// A polynomial is a "unit" iff its leading monomial has total degree 0
/// AND the leading coefficient has valuation 0 (i.e., is invertible).
/// Over a field every non-zero element is a unit; over a local ring like
/// `Z/2^k`, only odd constants qualify.
fn ideal_one_short_circuit<P, O>(ring: P, reducers: &[(El<P>, AugLm)], order: O) -> bool
where
    P: RingStore + Copy,
    P::Type: MultivariatePolyRing,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
    O: MonomialOrder + Copy,
{
    for (f, aug) in reducers {
        if !aug.exponents.iter().all(|e| *e == 0) {
            continue;
        }
        if ring.is_zero(f) {
            continue;
        }
        // Leading monomial is degree 0 — check its coefficient is a unit.
        if let Some((lc, _lm)) = ring.LT(f, order) {
            if let Some(v) = ring.base_ring().valuation(lc) {
                if v == 0 {
                    return true;
                }
            }
        }
    }
    false
}


/// Computes a Groebner basis of the ideal generated by the input basis w.r.t. the given term
/// ordering.
///
/// For a variant of this function that uses sensible defaults for most parameters, see
/// [`buchberger_simple()`].
///
/// The algorithm proceeds F4-style, i.e. reduces multiple S-polynomials before adding them to the
/// basis. When using a fast polynomial ring implementation (e.g.
/// [`crate::rings::multivariate::multivariate_impl::MultivariatePolyRingImpl`]), this makes the
/// algorithm as efficient as standard F4. Furthermore, the behavior can be modified by passing
/// custom functions for `sort_spolys` and `abort_early_if`.
///
/// - `sort_spolys` should permute the given list of S-polynomials w.r.t. the given basis; this can
///   be used to customize in which order S-polynomials are reduced, which can have huge impact on
///   performance. Note that S-polynomials that are supposed to be reduced first should be put at
///   the end of the list.
/// - `abort_early_if` takes the current basis (unfortunately, currently with some additional
///   information that can be ignored), and can return `true` to abort the GB computation, yielding
///   the current basis. In this case, the basis will in general not be a GB, but can still be
///   useful (e.g. `abort_early_if` might decide that a GB up to a fixed degree is sufficient).
///
/// # Explanation of logging output
///
/// If the passed computation controller accepts the logging, it will receive the following symbols:
///  - `-` means an S-polynomial was reduced to zero
///  - `s` means an S-polynomial reduced to a nonzero value and will be added to the basis at the
///    next opportunity
///  - `b(n)` means that the list of all generated basis polynomials has length `n`
///  - `r(n)` means that the current basis of the ideal has length `n`
///  - `S(n)` means that the algorithm still has to reduce `n` more S-polynomials
///  - `f(n)` means that `n` S-polynomials have, in total, been discarded by using the Buchberger
///    criteria
///  - `{n}` means that the algorithm is currently reducing S-polynomials of degree `n`
///  - `!` means that the algorithm decided to discard all current S-polynomial, and restart the
///    computation with the current basis
#[stability::unstable(feature = "enable")]
pub fn buchberger<P, O, Controller, SortFn, AbortFn>(
    ring: P,
    input_basis: Vec<El<P>>,
    order: O,
    sort_spolys: SortFn,
    abort_early_if: AbortFn,
    controller: Controller,
) -> Result<Vec<El<P>>, Controller::Abort>
where
    P: RingStore + Copy + Send + Sync,
    El<P>: Send + Sync,
    P::Type: MultivariatePolyRing,
    <P::Type as RingExtension>::BaseRing: Sync,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
    O: MonomialOrder + Copy + Send + Sync,
    PolyCoeff<P>: Send + Sync,
    Controller: ComputationController,
    SortFn: FnMut(&mut [SPoly], &[El<P>]),
    AbortFn: FnMut(&[(El<P>, AugLm)]) -> bool,
{
    buchberger_observed(ring, input_basis, order, sort_spolys, abort_early_if, controller, &mut NoObserver)
}

/// Sprint 2.7 — incremental Buchberger (no-observer wrapper).
///
/// Convenience wrapper that delegates to
/// [`buchberger_incremental_observed`] with [`NoObserver`].  See that
/// function's docs for the caller invariants on `known_gb` (must be a
/// reduced GB w.r.t. `order`).
pub fn buchberger_incremental<P, O, Controller, SortFn, AbortFn>(
    ring: P,
    known_gb: Vec<El<P>>,
    new_polys: Vec<El<P>>,
    order: O,
    sort_spolys: SortFn,
    abort_early_if: AbortFn,
    controller: Controller,
) -> Result<Vec<El<P>>, Controller::Abort>
where
    P: RingStore + Copy + Send + Sync,
    El<P>: Send + Sync,
    P::Type: MultivariatePolyRing,
    <P::Type as RingExtension>::BaseRing: Sync,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
    O: MonomialOrder + Copy + Send + Sync,
    PolyCoeff<P>: Send + Sync,
    Controller: ComputationController,
    SortFn: FnMut(&mut [SPoly], &[El<P>]),
    AbortFn: FnMut(&[(El<P>, AugLm)]) -> bool,
{
    buchberger_incremental_observed(
        ring, known_gb, new_polys, order, sort_spolys, abort_early_if, controller, &mut NoObserver,
    )
}

/// Like [`buchberger`], but with an observer that receives callbacks for
/// each new polynomial derived during the computation.  This enables
/// dependency tracking for UNSAT core extraction.
#[stability::unstable(feature = "enable")]
pub fn buchberger_observed<P, O, Controller, SortFn, AbortFn, Obs>(
    ring: P,
    input_basis: Vec<El<P>>,
    order: O,
    mut sort_spolys: SortFn,
    mut abort_early_if: AbortFn,
    controller: Controller,
    observer: &mut Obs,
) -> Result<Vec<El<P>>, Controller::Abort>
where
    P: RingStore + Copy + Send + Sync,
    El<P>: Send + Sync,
    P::Type: MultivariatePolyRing,
    <P::Type as RingExtension>::BaseRing: Sync,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
    O: MonomialOrder + Copy + Send + Sync,
    PolyCoeff<P>: Send + Sync,
    Controller: ComputationController,
    SortFn: FnMut(&mut [SPoly], &[El<P>]),
    AbortFn: FnMut(&[(El<P>, AugLm)]) -> bool,
    Obs: BuchbergerObserver<P>,
{
    // Sprint 2.7: setup phase only — pair generation + main loop are in
    // `buchberger_loop_observed` (shared with `buchberger_incremental_observed`).
    let augmented: Vec<_> = input_basis.into_iter().map(|f| augment_lm(ring, f, order)).collect();

    // Quick check: skip inter_reduce if no leading term is divisible
    // by another's (the basis is already inter-reduced).  This saves
    // ~1.5ms on every call where the input is already a GB.
    let needs_inter_reduce = augmented.iter().enumerate().any(|(i, (_, aug_i))| {
        augmented.iter().enumerate().any(|(j, (_, aug_j))| {
            if i == j { return false; }
            // MISSING-4: Use cached DivMask
            if (aug_j.mask & !aug_i.mask) != 0 { return false; }
            (0..ring.indeterminate_count()).all(|v| aug_j.exponents[v] <= aug_i.exponents[v])
        })
    });

    let input_basis: Vec<El<P>> = if needs_inter_reduce {
        inter_reduce(&ring, augmented, order)
            .into_iter()
            .map(|(f, _)| f)
            .collect::<Vec<_>>()
    } else {
        augmented.into_iter().map(|(f, _)| f).collect::<Vec<_>>()
    };
    debug_assert!(input_basis.iter().all(|f| !ring.is_zero(f)));
    observer.on_initial_basis(input_basis.len());

    let nilpotent_power = ring
        .base_ring()
        .nilpotent_power()
        .and_then(|e| if e != 0 { Some(e) } else { None });
    assert!(
        nilpotent_power.is_none()
            || ring.base_ring().is_zero(&ring.base_ring().pow(
                ring.base_ring().clone_el(ring.base_ring().max_ideal_gen()),
                nilpotent_power.unwrap()
            ))
    );

    // OPTIMIZATION (T1.1): Principal ideal short-circuit.
    // CoCoA `SparsePolyOps-ideal.C:760-765`: if the (already inter-reduced)
    // input contains 0 or 1 nonzero generators *over a field* (no
    // nilpotents in the base ring), the basis IS a Groebner basis —
    // there are no S-pairs to compute and no Nilpotent S-polys to
    // reduce.  This avoids substantial setup cost (sort_reducers,
    // update_basis, pair generation, etc.) for trivial ideals.
    //
    // Local rings with nilpotents are excluded because a single
    // generator like `1 + 4*X0` over Z/16 still requires Nilpotent
    // S-poly reductions to reach the canonical form `1`.
    if input_basis.len() <= 1 && nilpotent_power.is_none() {
        return Ok(input_basis);
    }

    // invariant: `(reducers) = (basis)` and there exists a reduction to zero for every `f`
    // in `basis` modulo `reducers`; reducers are always stored with an expanded
    // version of their leading monomial, in order to simplify divisibility checks
    let mut reducers: Vec<(El<P>, AugLm)> = input_basis
        .iter()
        .map(|f| augment_lm(ring, ring.clone_el(f), order))
        .collect::<Vec<_>>();
    sort_reducers_by(ring, order, &mut reducers);

    // MISSING-1: Track sugar degree for each basis element.
    // For input polynomials, sugar = total degree.
    let mut basis_sugar: Vec<usize> = Vec::new();

    // MISSING-2: Track active status for each basis element.
    let mut basis_active: Vec<bool> = Vec::new();

    let mut open = Vec::new();
    let mut basis = Vec::new();
    update_basis(
        ring,
        input_basis.into_iter().map(|f| {
            let sugar = ring.terms(&f).map(|(_, m)| ring.monomial_deg(m)).max().unwrap_or(0);
            (f, sugar)
        }),
        &mut basis,
        &mut basis_sugar,
        &mut basis_active,
        &mut open,
        order,
        nilpotent_power,
        &mut 0,
        &mut sort_spolys,
    );

    buchberger_loop_observed(
        ring, order, sort_spolys, abort_early_if, controller, observer,
        nilpotent_power, basis, basis_sugar, basis_active, open, reducers,
    )
}

/// Sprint 2.7 — incremental Buchberger entry point.
///
/// Computes a Gröbner basis of `<known_gb> + <new_polys>` by *extending*
/// an existing reduced GB rather than recomputing it from scratch.  This
/// avoids the per-call setup cost (inter-reduce, sort_reducers, the
/// initial round of `update_basis` over all generators) and — more
/// importantly — avoids generating the `O(|known_gb|^2)` S-pairs among
/// known_gb elements (which by hypothesis would all reduce to zero).
///
/// Only the cross-pairs `(known_gb_i, new_polys_k)` and the new-new
/// pairs `(new_polys_i, new_polys_k)` are generated.
///
/// # CRITICAL caller invariants
///
/// The caller MUST guarantee:
/// 1. `known_gb` is a Gröbner basis with respect to `order`.  If not,
///    pairs among known_gb elements that would NOT reduce to zero are
///    silently skipped, producing a wrong result.
/// 2. `known_gb` is inter-reduced (no LT divides another LT).
/// 3. Every element of `known_gb` is non-zero in `ring`.
///
/// In picus, the `Ideal::basis` field is the output of a prior
/// `compute_gb_with_order` call, which establishes (1)–(3) by
/// construction.  In tests, callers that violate these invariants will
/// get wrong answers — invariants are checked under `debug_assertions`
/// only (cheap LT-divisibility check).
///
/// # Behaviour on edge cases
///
/// - If `new_polys.is_empty()`, returns `known_gb` unchanged (after
///   stripping any zero polys defensively, none expected).
/// - If `known_gb.is_empty()`, behaves as a full Buchberger run on
///   `new_polys` (i.e., delegates conceptually to [`buchberger_observed`],
///   but takes the same code path through [`buchberger_loop_observed`]).
/// - The internal restart at high pair-density still goes through
///   [`buchberger_observed`] (full recompute on `reducers`), which is
///   correct: by that point the incremental advantage is already gone.
pub fn buchberger_incremental_observed<P, O, Controller, SortFn, AbortFn, Obs>(
    ring: P,
    known_gb: Vec<El<P>>,
    new_polys: Vec<El<P>>,
    order: O,
    mut sort_spolys: SortFn,
    abort_early_if: AbortFn,
    controller: Controller,
    observer: &mut Obs,
) -> Result<Vec<El<P>>, Controller::Abort>
where
    P: RingStore + Copy + Send + Sync,
    El<P>: Send + Sync,
    P::Type: MultivariatePolyRing,
    <P::Type as RingExtension>::BaseRing: Sync,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
    O: MonomialOrder + Copy + Send + Sync,
    PolyCoeff<P>: Send + Sync,
    Controller: ComputationController,
    SortFn: FnMut(&mut [SPoly], &[El<P>]),
    AbortFn: FnMut(&[(El<P>, AugLm)]) -> bool,
    Obs: BuchbergerObserver<P>,
{
    // Defensive: drop zero polys from inputs.  `update_basis` and the
    // loop assume non-zero generators.
    let known_gb: Vec<El<P>> = known_gb.into_iter().filter(|f| !ring.is_zero(f)).collect();
    let new_polys: Vec<El<P>> = new_polys.into_iter().filter(|f| !ring.is_zero(f)).collect();

    // Edge case: nothing new to add — known_gb is already a GB.
    if new_polys.is_empty() {
        return Ok(known_gb);
    }

    // Edge case: no prior GB — degenerates to a full run.  Routing
    // through buchberger_observed gives us the same setup (inter-reduce,
    // principal-ideal short-circuit) and avoids subtle invariant skew.
    if known_gb.is_empty() {
        return buchberger_observed(
            ring, new_polys, order, sort_spolys, abort_early_if, controller, observer,
        );
    }

    // Debug invariants: known_gb is inter-reduced (no LT properly
    // divides another LT).  Cheap O(n^2) check; only under
    // debug_assertions.
    #[cfg(debug_assertions)]
    {
        let augs: Vec<AugLm> = known_gb.iter()
            .map(|f| augment_lm(ring, ring.clone_el(f), order).1)
            .collect();
        let n_vars = ring.indeterminate_count();
        for i in 0..augs.len() {
            for j in 0..augs.len() {
                if i == j { continue; }
                if (augs[j].mask & !augs[i].mask) != 0 { continue; }
                let i_divides_j = (0..n_vars).all(|v| augs[i].exponents[v] <= augs[j].exponents[v]);
                debug_assert!(
                    !i_divides_j,
                    "buchberger_incremental_observed: known_gb is not inter-reduced \
                     (LT of element {} divides LT of element {})", i, j
                );
            }
        }
    }

    let nilpotent_power = ring
        .base_ring()
        .nilpotent_power()
        .and_then(|e| if e != 0 { Some(e) } else { None });

    observer.on_initial_basis(known_gb.len() + new_polys.len());

    // Pre-seed `reducers` with known_gb augments + new_polys augments.
    // The loop relies on `<reducers> = <basis>` as ideals, but does NOT
    // require reducers to be inter-reduced on entry — the first
    // iteration's post-pair `inter_reduce` will catch any redundancy
    // introduced by new_polys.
    let mut reducers: Vec<(El<P>, AugLm)> = known_gb.iter()
        .chain(new_polys.iter())
        .map(|f| augment_lm(ring, ring.clone_el(f), order))
        .collect();
    sort_reducers_by(ring, order, &mut reducers);

    // Pre-seed `basis` with known_gb only — `basis_active[..known_gb.len()]`
    // are all true, and pair generation skips intra-known pairs because
    // we only call `update_basis` for the new polys (it generates pairs
    // (i, k) for i < k).
    let mut basis: Vec<El<P>> = known_gb.iter().map(|f| ring.clone_el(f)).collect();
    let mut basis_sugar: Vec<usize> = known_gb.iter()
        .map(|f| ring.terms(f).map(|(_, m)| ring.monomial_deg(m)).max().unwrap_or(0))
        .collect();
    let mut basis_active: Vec<bool> = vec![true; known_gb.len()];
    let mut open: Vec<SPoly> = Vec::new();

    // Generate the cross-pairs and new-new pairs by appending each
    // new_poly through update_basis.  This is the core incremental
    // optimisation: we skip the O(|known_gb|^2) intra-known pair
    // generation that a full buchberger_observed call would perform.
    update_basis(
        ring,
        new_polys.into_iter().map(|f| {
            let sugar = ring.terms(&f).map(|(_, m)| ring.monomial_deg(m)).max().unwrap_or(0);
            (f, sugar)
        }),
        &mut basis,
        &mut basis_sugar,
        &mut basis_active,
        &mut open,
        order,
        nilpotent_power,
        &mut 0,
        &mut sort_spolys,
    );

    buchberger_loop_observed(
        ring, order, sort_spolys, abort_early_if, controller, observer,
        nilpotent_power, basis, basis_sugar, basis_active, open, reducers,
    )
}

/// Sprint 2.7: shared sort comparator for `reducers`, factored out of the
/// inline closure so both `buchberger_observed` and
/// `buchberger_incremental_observed` can reuse it.
#[inline]
fn sort_reducers_by<P, O>(ring: P, order: O, reducers: &mut [(El<P>, AugLm)])
where
    P: RingStore + Copy,
    P::Type: MultivariatePolyRing,
    O: MonomialOrder + Copy,
{
    // I have no idea why, but this order seems to give the best results
    reducers.sort_by(|(lf, _), (rf, _)| {
        order
            .compare(ring, ring.LT(lf, order).unwrap().1, ring.LT(rf, order).unwrap().1)
            .then_with(|| ring.terms(lf).count().cmp(&ring.terms(rf).count()))
    })
}

/// Sprint 2.7: shared main loop of Buchberger's algorithm, parameterized
/// over the *initial seed state* prepared by the caller.  Both
/// [`buchberger_observed`] (full setup) and
/// [`buchberger_incremental_observed`] (caller-provided GB seed +
/// new_polys) prepare the seed differently and then drive the same loop.
///
/// Caller responsibilities (invariants enforced by debug_assert in the
/// individual entry points, NOT here):
/// - `basis`, `basis_sugar`, `basis_active` have equal length.
/// - For every `i`, `basis_sugar[i] >= total_degree(basis[i])`.
/// - `reducers` covers `basis` (i.e. `<reducers> = <basis>` as ideals,
///   and every `f` in `basis` reduces to zero modulo `reducers`).
/// - `open` contains exactly the S-pairs that the caller's seed
///   `update_basis` calls would have produced — i.e., every (i, j) pair
///   among basis elements except those filtered out by Gebauer–Möller.
fn buchberger_loop_observed<P, O, Controller, SortFn, AbortFn, Obs>(
    ring: P,
    order: O,
    mut sort_spolys: SortFn,
    mut abort_early_if: AbortFn,
    controller: Controller,
    observer: &mut Obs,
    nilpotent_power: Option<usize>,
    mut basis: Vec<El<P>>,
    mut basis_sugar: Vec<usize>,
    mut basis_active: Vec<bool>,
    mut open: Vec<SPoly>,
    mut reducers: Vec<(El<P>, AugLm)>,
) -> Result<Vec<El<P>>, Controller::Abort>
where
    P: RingStore + Copy + Send + Sync,
    El<P>: Send + Sync,
    P::Type: MultivariatePolyRing,
    <P::Type as RingExtension>::BaseRing: Sync,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
    O: MonomialOrder + Copy + Send + Sync,
    PolyCoeff<P>: Send + Sync,
    Controller: ComputationController,
    SortFn: FnMut(&mut [SPoly], &[El<P>]),
    AbortFn: FnMut(&[(El<P>, AugLm)]) -> bool,
    Obs: BuchbergerObserver<P>,
{
    controller.run_computation(
        format_args!("buchberger_loop(len={}, vars={})", basis.len(), ring.indeterminate_count()),
        |controller| {
            let mut current_sugar: usize = 0;
            let mut filtered_spolys = 0;
            let mut changed = false;
            loop {
                // MISSING-3: short-circuit if a unit appeared in `reducers`
                // (e.g. via the initial basis or after a prior inter_reduce).
                    if ideal_one_short_circuit(ring, &reducers, order) {
                    log_progress!(controller, "(I=R)");
                    return Ok(vec![ring.one()]);
                }
                // Plan v3 Task 06 — one-at-a-time S-pair processing.
                // CoCoA processes a single S-pair per main-loop iteration
                // (`TmpGReductor.C:670-707`). The previous code collected
                // all same-sugar S-pairs into a parallel batch. With
                // restart removed (Task 01) and inter-reduce moved to
                // termination (Task 02), batching no longer offers
                // benefit; matching CoCoA's single-pair semantics
                // ensures the same processing order.
                let spolys_to_reduce_index = if open.is_empty() {
                    0
                } else {
                    open.len() - 1
                };
                let spolys_to_reduce = &open[spolys_to_reduce_index..];

                // Sprint 2.6b: pair-count profiling — start of a sugar batch.
                observer.on_sugar_batch_start(
                    current_sugar,
                    spolys_to_reduce.len(),
                    basis.len(),
                );

                let computation = ShortCircuitingComputation::new();
                let new_polys = AppendOnlyVec::new();
                let new_poly_parents = AppendOnlyVec::new();
                let new_poly_sugars = AppendOnlyVec::new();
                // Sprint 2.3.5: per-S-pair running-sugar profile rows.
                // Stored as (initial_sugar, final_sugar, n_raises) and
                // emitted via `on_running_sugar` after the parallel
                // join completes.  Pushed for *every* spoly we process
                // (zero-result included) so callers see the full trace.
                let new_poly_sugar_profile = AppendOnlyVec::new();
                let new_polys_ref = &new_polys;
                let new_poly_parents_ref = &new_poly_parents;
                let new_poly_sugars_ref = &new_poly_sugars;
                let new_poly_sugar_profile_ref = &new_poly_sugar_profile;
                let basis_ref = &basis;
                let reducers_ref = &reducers;
                // Sprint 2.3.2: snapshot reducer sugars so the running-
                // sugar update can map reducer indices to their sugar
                // values.  The first `reducers.len()` entries correspond
                // to elements of `reducers` (whose sugars live in
                // `basis_sugar`); the tail entries grow per-S-pair as
                // `new_polys` is appended.
                let basis_sugar_ref = &basis_sugar;

                computation
                    .handle(controller.clone())
                    .join_many(spolys_to_reduce.as_fn().map_fn(move |spoly| {
                        let spoly_sugar_init = spoly.sugar();
                        move |handle: ShortCircuitingComputationHandle<(), _>| {
                            let parent_info: Vec<usize> = match spoly {
                                SPoly::Standard { i, j, .. } => vec![*i, *j],
                                SPoly::Nilpotent { idx, .. } => vec![*idx],
                            };
                            let mut f = spoly.poly(ring, basis_ref, order);

                            // Sprint 2.3.2: running sugar starts at the
                            // S-pair sugar (matches `myAssignSPoly` in
                            // `TmpGPoly.C:301-317`) and is raised
                            // monotonically by each reducer application
                            // via `Sugar::my_update`.
                            let mut running_sugar = Sugar::new(spoly_sugar_init);
                            // Snapshot of new_polys at *call* time.  The
                            // AppendOnlyVec may grow during this S-pair
                            // (parallel join), but only via *push*; existing
                            // entries are immutable, so reading the length
                            // here gives a valid prefix.
                            let new_polys_snapshot_len = new_polys_ref.len();
                            let n_reducers = reducers_ref.len();
                            let lookup_sugar = |i: usize| -> Sugar {
                                if i < n_reducers {
                                    Sugar::new(basis_sugar_ref[i])
                                } else {
                                    let j = i - n_reducers;
                                    if j < new_poly_sugars_ref.len() {
                                        Sugar::new(new_poly_sugars_ref[j])
                                    } else {
                                        // Past the snapshot: a sibling
                                        // S-pair appended after we built
                                        // our reducer list; treat as
                                        // sugar 0 (conservative — at
                                        // worst no update happens).
                                        Sugar::new(0)
                                    }
                                }
                            };
                            let mut reducers_iter_fn = || {
                                reducers_ref.iter()
                                    .chain(new_polys_ref.iter().take(new_polys_snapshot_len))
                                    .map(|(f, aug)| (f, aug))
                            };
                            let n_raises = reduce_poly_with_sugar(
                                ring,
                                &mut f,
                                &mut reducers_iter_fn,
                                order,
                                Some(&mut running_sugar),
                                lookup_sugar,
                            );
                            // Sprint 2.3.5: record the running-sugar
                            // outcome for this S-pair (whether or not
                            // it survives reduction).
                            _ = new_poly_sugar_profile_ref.push(
                                (spoly_sugar_init, running_sugar.value(), n_raises)
                            );

                            if !ring.is_zero(&f) {
                                log_progress!(handle, "s");
                                _ = new_poly_parents_ref.push(parent_info);
                                // Sprint 2.3.2: store the *running* sugar
                                // (raised during reduction), not the raw
                                // S-pair sugar.  This is the value future
                                // S-pair sugar computations will see for
                                // this new basis element.
                                _ = new_poly_sugars_ref.push(running_sugar.value());
                                _ = new_polys_ref.push(augment_lm(ring, f, order));
                            } else {
                                log_progress!(handle, "-");
                            }

                            checkpoint!(handle);
                            return Ok(None);
                        }
                    }));

                drop(open.drain(spolys_to_reduce_index..));
                let new_polys = new_polys.into_vec();
                let new_poly_parents = new_poly_parents.into_vec();
                let new_poly_sugars = new_poly_sugars.into_vec();
                let new_poly_sugar_profile = new_poly_sugar_profile.into_vec();
                _ = computation.finish()?;

                // Notify observer of newly derived polynomials
                for (parents, (poly, _)) in new_poly_parents.iter().zip(new_polys.iter()) {
                    observer.on_new_poly(parents, poly);
                }
                // Sprint 2.3.5: emit one running-sugar profile event
                // per S-pair processed (zero-result included).
                for &(init_s, final_s, n_raises) in new_poly_sugar_profile.iter() {
                    observer.on_running_sugar(init_s, final_s, n_raises);
                }

                // Sprint 2.6b: pair-count profiling — end of a sugar batch.
                {
                    let n_pairs_processed = new_poly_sugar_profile.len();
                    let n_new = new_polys.len();
                    observer.on_sugar_batch_end(
                        current_sugar,
                        n_pairs_processed,
                        n_new,
                        n_pairs_processed.saturating_sub(n_new),
                        basis.len(),
                    );
                }

                // process the generated new polynomials
                if new_polys.is_empty() && open.is_empty() {
                    if changed {
                        log_progress!(controller, "!");
                        // this seems necessary, as the invariants for `reducers` don't imply that
                        // it already is a GB; more concretely, reducers
                        // contains polys of basis that are reduced with eath other, but the
                        // S-polys between two of them might not have been considered
                        return buchberger_observed::<P, O, _, _, _, Obs>(
                            ring,
                            reducers.into_iter().map(|(f, _)| f).collect(),
                            order,
                            sort_spolys,
                            abort_early_if,
                            controller,
                            observer,
                        );
                    } else {
                        // Plan v3 Task 02 — final inter-reduce.
                        // CoCoA inter-reduces once at the end of the GB
                        // computation rather than after every sugar batch.
                        // This matches that pattern.
                        let mut final_reducers = inter_reduce(ring, reducers, order);
                        sort_reducers_by(ring, order, &mut final_reducers);
                        return Ok(final_reducers.into_iter().map(|(f, _)| f).collect());
                    }
                } else if new_polys.is_empty() {
                    current_sugar = open.last().unwrap().sugar();
                    log_progress!(controller, "{{{}}}", current_sugar);
                } else {
                    changed = true;
                    current_sugar = 0;
                    update_basis(
                        ring,
                        new_polys.iter().zip(new_poly_sugars.iter()).map(|((f, _), &sugar)| {
                            (ring.clone_el(f), sugar)
                        }),
                        &mut basis,
                        &mut basis_sugar,
                        &mut basis_active,
                        &mut open,
                        order,
                        nilpotent_power,
                        &mut filtered_spolys,
                        &mut sort_spolys,
                    );
                    log_progress!(
                        controller,
                        "(b={})(S={})(f={})",
                        basis.len(),
                        open.len(),
                        filtered_spolys
                    );

                    reducers.extend(new_polys);
                    // Plan v3 Task 02 — per-batch inter_reduce removed.
                    // CoCoA inter-reduces only once at the end. Per-batch
                    // inter-reduction was O(|reducers|²) work paid every
                    // sugar batch (~thousands per circuit). The final
                    // inter-reduce at loop termination handles cleanup.
                    sort_reducers_by(ring, order, &mut reducers);
                    log_progress!(controller, "(r={})", reducers.len());
                    // MISSING-3 retained: short-circuit if a unit appeared.
                    // This check is cheap and useful even without inter_reduce.
                if ideal_one_short_circuit(ring, &reducers, order) {
                        log_progress!(controller, "(I=R)");
                        return Ok(vec![ring.one()]);
                    }
                    if abort_early_if(&reducers) {
                        log_progress!(controller, "(early_abort)");
                        return Ok(reducers.into_iter().map(|(f, _)| f).collect());
                    }
                }

                // Plan v3 Task 01 — restart heuristic removed.
                // CoCoA never restarts; it processes the pair queue
                // straight through to completion. The previous restart
                // (Sprint 2.8b) fired on essentially every top-level
                // call (29,216/29,217 on MontgomeryAdd) and was a
                // major source of redundant work. Removed.
            }
        },
    )
}

fn update_basis<I, P, O, SortFn>(
    ring: P,
    new_polys: I,
    basis: &mut Vec<El<P>>,
    basis_sugar: &mut Vec<usize>,
    basis_active: &mut Vec<bool>,
    open: &mut Vec<SPoly>,
    order: O,
    nilpotent_power: Option<usize>,
    filtered_spolys: &mut usize,
    sort_spolys: &mut SortFn,
) where
    P: RingStore + Copy,
    P::Type: MultivariatePolyRing,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
    O: MonomialOrder + Copy,
    SortFn: FnMut(&mut [SPoly], &[El<P>]),
    I: Iterator<Item = (El<P>, usize)>,
{
    let n_vars = ring.indeterminate_count();

    // MISSING-8: Preallocate exponent scratch vectors
    let mut tmp_gi_exp = vec![0usize; n_vars];
    let mut tmp_gj_exp = vec![0usize; n_vars];
    let mut tmp_lcm_ij = vec![0usize; n_vars];
    let mut tmp_lcm_ik = vec![0usize; n_vars];
    let mut tmp_lcm_jk = vec![0usize; n_vars];

    for (new_poly, new_sugar) in new_polys {
        let k = basis.len();
        basis.push(new_poly);
        basis_sugar.push(new_sugar);
        basis_active.push(true);

        let (_gk_c, gk_m) = ring.LT(&basis[k], order).unwrap();
        let gk_exp = ring.expand_monomial(gk_m);
        let gk_mask = divmask(&gk_exp);
        let gk_deg: usize = gk_exp.iter().copied().sum();

        // Plan v3 Task 04 — match CoCoA non-strict deactivation.
        // CoCoA (`TmpGReductor.C:500`) uses `IsDivisibleFast(LT_old, LT_new)`
        // — non-strict divisibility. When the new LT equals an old LT,
        // the old element is deactivated. The previous strict-divisibility
        // check kept both equal-LT generators, paying for redundant
        // S-pair processing instead.
        for i in 0..k {
            if !basis_active[i] { continue; }
            let gi_exp_tmp = ring.expand_monomial(ring.LT(&basis[i], order).unwrap().1);
            let gi_mask = divmask(&gi_exp_tmp);
            // Check if LT(g_k) divides LT(g_i) (non-strict, includes equality)
            if (gk_mask & !gi_mask) != 0 { continue; }
            let k_divides_i = (0..n_vars).all(|v| gk_exp[v] <= gi_exp_tmp[v]);
            if k_divides_i {
                basis_active[i] = false;
            }
        }

        // Plan v3 Task 04 — keep pairs of deactivated elements.
        // CoCoA does NOT remove pairs whose basis members were deactivated;
        // those pairs may still reduce to non-zero polynomials and matter
        // for correctness. Leaving them in `open` matches CoCoA's behavior.
        // (Previous code: `open.retain(...)` removed them — REMOVED.)

        // --- Gebauer-Möller B_k criterion ---
        // Remove old pairs S(i,j) from `open` where LT(g_k) divides
        // lcm(LT(g_i), LT(g_j)) and the triangular condition holds.
        // MISSING-8: Use preallocated exponent vectors
        open.retain(|spoly| {
            match spoly {
                SPoly::Standard { i, j, .. } => {
                    ring.expand_monomial_to(ring.LT(&basis[*i], order).unwrap().1, &mut tmp_gi_exp);
                    ring.expand_monomial_to(ring.LT(&basis[*j], order).unwrap().1, &mut tmp_gj_exp);
                    // Compute lcm exponents of (i, j)
                    for v in 0..n_vars {
                        tmp_lcm_ij[v] = tmp_gi_exp[v].max(tmp_gj_exp[v]);
                    }
                    // Check if LT(g_k) divides lcm(LT(g_i), LT(g_j))
                    let k_divides_lcm = (0..n_vars).all(|v| gk_exp[v] <= tmp_lcm_ij[v]);
                    if !k_divides_lcm {
                        return true; // keep this pair
                    }
                    // Check triangular condition: lcm(i,k) != lcm(i,j) AND lcm(j,k) != lcm(i,j)
                    for v in 0..n_vars {
                        tmp_lcm_ik[v] = tmp_gi_exp[v].max(gk_exp[v]);
                        tmp_lcm_jk[v] = tmp_gj_exp[v].max(gk_exp[v]);
                    }
                    let ik_eq_ij = tmp_lcm_ik[..n_vars] == tmp_lcm_ij[..n_vars];
                    let jk_eq_ij = tmp_lcm_jk[..n_vars] == tmp_lcm_ij[..n_vars];
                    if ik_eq_ij || jk_eq_ij {
                        return true; // keep — no shorter path through k
                    }
                    *filtered_spolys += 1;
                    false // remove — g_k provides a shorter path
                }
                SPoly::Nilpotent { .. } => true, // keep nilpotent pairs
            }
        });

        // --- Generate new pairs S(i, k) for all active i < k ---
        //
        // Sprint 2.4.1 + 2.4.2 (R5.B): mirror CoCoA's `myBuildNewPairs`
        // (`TmpGReductor.C:485-552`) ordering of criteria:
        //
        //   1. Build every pair (i, k) for active i < k, recording the
        //      lcm exponent vector AND a `coprime: bool` tag (cheap:
        //      `lcm_deg == gi_deg + gk_deg`).
        //   2. Apply M-criterion (bidirectional dominance) within
        //      `new_pairs`, with the **coprime-LCM equal-swap**
        //      (CoCoA `:463-466`): on equal lcms prefer the coprime
        //      variant because it can be discarded by the F-criterion
        //      later for free.
        //   3. ONLY THEN apply the F-criterion (drop coprime pairs).
        //      CoCoA defers F until after M because the M-criterion
        //      may rely on a coprime pair to dominate a non-coprime
        //      sibling — dropping the coprime first would needlessly
        //      keep the non-coprime one.  See R5 §2.2 lines 273-281.
        //
        // We keep `filter_spoly` (the existing post-construction
        // filter) intact — it kills pairs whose lcm equals the LT of
        // some active basis element, which is a strictly stronger
        // criterion than F and unrelated to GM dominance.
        let mut new_pairs: Vec<(SPoly, Vec<usize>, bool /*coprime*/)> = Vec::new();
        for i in 0..k {
            // MISSING-2: Skip deactivated basis elements
            if !basis_active[i] { continue; }

            // Compute lcm exponents for M criterion + sugar + cached lcm_deg
            let gi_exp = ring.expand_monomial(ring.LT(&basis[i], order).unwrap().1);
            let mut lcm_exp = vec![0usize; n_vars];
            for v in 0..n_vars {
                lcm_exp[v] = gi_exp[v].max(gk_exp[v]);
            }
            // MISSING-5: Cached lcm degree
            let lcm_deg_val: usize = lcm_exp.iter().copied().sum();
            let gi_deg: usize = gi_exp.iter().copied().sum();

            // Sprint 2.4.2: tag coprime status (cheap monomial-level
            // test: gcd(LT_i, LT_k) = 1 ⇔ lcm_deg = gi_deg + gk_deg).
            // The actual F-criterion drop is deferred to step 3 below
            // so the coprime-LCM swap can fire first.  Restricted to
            // fields (no nilpotents) — over local rings the F-criterion
            // is unsound (see line ~1193 OPTIMIZATION T1.2 comment).
            let is_coprime = nilpotent_power.is_none() && lcm_deg_val == gi_deg + gk_deg;

            // MISSING-1: Compute sugar degree for the pair via the
            // canonical `Sugar::for_spair` formula (R4 §1.4 /
            // `TmpGPair.C:36-45`).  Mathematically identical to the
            // previous inline `max(s_i + (lcm-gi_deg), s_k + (lcm-gk_deg))`
            // but routes through `Sugar`'s testable surface so 2.3.2's
            // running-sugar update can interoperate cleanly.
            let sugar_val = Sugar::for_spair(
                Sugar::new(basis_sugar[i]), gi_deg,
                Sugar::new(basis_sugar[k]), gk_deg,
                lcm_deg_val,
            ).value();

            let spoly = SPoly::Standard { i, j: k, sugar: sugar_val, lcm_deg: lcm_deg_val, age: next_spoly_age() };
            if filter_spoly(ring, &spoly, basis, order).is_some() {
                *filtered_spolys += 1;
                continue;
            }
            new_pairs.push((spoly, lcm_exp, is_coprime));
        }

        // --- Step 2: M criterion with coprime-LCM equal-swap ---
        // Bidirectional dominance: if lcm_a properly divides lcm_b,
        // drop b.  When lcm_a == lcm_b (equal), keep the coprime one
        // (CoCoA `:463-466` — non-coprime → coprime swap), so the
        // subsequent F-criterion can drop it for free.
        if new_pairs.len() > 1 {
            let mut keep = vec![true; new_pairs.len()];
            for a in 0..new_pairs.len() {
                if !keep[a] { continue; }
                for b in 0..new_pairs.len() {
                    if a == b || !keep[b] { continue; }
                    // Sprint 2.4.2: handle equal-lcm case explicitly
                    // BEFORE the proper-divisibility check.  When
                    // lcms are equal and exactly one of (a, b) is
                    // coprime, drop the non-coprime — equivalently,
                    // "swap in the coprime variant" in CoCoA's
                    // in-place mutation form.  The F-criterion at
                    // step 3 will then drop the surviving coprime
                    // pair, achieving the same final state CoCoA
                    // reaches via `:463-466` + later `:530`.
                    if new_pairs[a].1 == new_pairs[b].1 {
                        // Equal lcm.  Prefer coprime; if both or
                        // neither coprime, keep `a` arbitrarily
                        // (matches CoCoA's "first wins" iteration
                        // order; subsequent F-drop normalizes).
                        if new_pairs[a].2 && !new_pairs[b].2 {
                            keep[b] = false;
                            *filtered_spolys += 1;
                        } else if !new_pairs[a].2 && new_pairs[b].2 {
                            keep[a] = false;
                            *filtered_spolys += 1;
                            break; // a is dead; advance outer loop
                        } else {
                            // Both coprime or both non-coprime —
                            // keep one arbitrarily (a < b in iteration).
                            keep[b] = false;
                            *filtered_spolys += 1;
                        }
                        continue;
                    }
                    // Proper-divisibility (M-criterion direction (a)):
                    // lcm_a strictly divides lcm_b → b dominated.
                    let divides = (0..n_vars).all(|v| new_pairs[a].1[v] <= new_pairs[b].1[v]);
                    if divides {
                        keep[b] = false;
                        *filtered_spolys += 1;
                    }
                }
            }
            // Step 3: deferred F-criterion (drop surviving coprime
            // pairs).  CoCoA `:530-538` — must run AFTER M so that the
            // coprime-swap step can promote a non-coprime to coprime
            // before the drop.
            let mut kept: Vec<(SPoly, Vec<usize>, bool)> = Vec::new();
            for (idx, tup) in new_pairs.into_iter().enumerate() {
                if !keep[idx] { continue; }
                if tup.2 {
                    // F-criterion: coprime ⇒ S-poly reduces to zero.
                    *filtered_spolys += 1;
                    continue;
                }
                kept.push(tup);
            }
            for (spoly, _, _) in kept {
                open.push(spoly);
            }
        } else {
            // Single (or zero) new pair — M-criterion vacuous.  Apply
            // F-criterion directly.
            for (spoly, _, is_coprime) in new_pairs {
                if is_coprime {
                    *filtered_spolys += 1;
                    continue;
                }
                open.push(spoly);
            }
        }

        // Nilpotent S-polys (for local rings)
        if let Some(e) = nilpotent_power {
            for nk in 1..e {
                let lcm_deg_val = gk_deg; // LT doesn't change for nilpotent
                let sugar_val = basis_sugar[k]; // sugar inherited
                let spoly = SPoly::Nilpotent { idx: k, k: nk, sugar: sugar_val, lcm_deg: lcm_deg_val, age: next_spoly_age() };
                if filter_spoly(ring, &spoly, basis, order).is_none() {
                    open.push(spoly);
                } else {
                    *filtered_spolys += 1;
                }
            }
        }
    }
    sort_spolys(&mut *open, &*basis);
}

fn reduce_poly<'a, 'b, F, I, P, O>(ring: P, to_reduce: &mut El<P>, mut reducers: F, order: O)
where
    P: 'a + RingStore + Copy,
    P::Type: MultivariatePolyRing,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
    O: MonomialOrder + Copy,
    F: FnMut() -> I,
    I: Iterator<Item = (&'a El<P>, &'b AugLm)>,
{
    // Sprint 2.3.2: delegate to the sugar-aware variant with a no-op
    // sugar updater.  Used by `multivariate_division` and other callers
    // that don't track sugar.
    let _ = reduce_poly_with_sugar(
        ring,
        to_reduce,
        &mut reducers,
        order,
        /* sugar_self    */ None,
        /* reducer_sugar */ |_| Sugar::new(0),
    );
}

/// Sprint 2.3.2 (R4 §6 / `SugarDegree.C:62-63`): variant of
/// `reduce_poly` that maintains a running Giovini–Mora–Niesi–Robbiano
/// sugar value during reduction.
///
/// When a reducer at index `i` is applied with cofactor monomial `c`,
/// the in-flight sugar `s` is raised to
///   `max(s, deg(c) + reducer_sugar(i))`
/// matching `StdDegBase::myUpdate(c, g)` in CoCoA.
///
/// * `sugar_self` — the in-flight S-poly's sugar, mutated in place.
///   When `None`, no sugar tracking is performed (free fall-through to
///   the previous behavior; used by `multivariate_division`).
/// * `reducer_sugar(i)` — returns the sugar of the `i`-th reducer
///   yielded by the `reducers` iterator.  Iteration must be stable.
///
/// Returns the number of *strictly raising* `Sugar::my_update` calls
/// during reduction (Sprint 2.3.5 profile counter).  Always 0 when
/// `sugar_self` is `None`.
///
/// Reducer identification across the trait boundary uses pointer
/// identity: the iterator's first pass populates an `(*const El, Sugar)`
/// map; the trait's observer callback looks up the cofactor's reducer
/// pointer in that map.  This is O(1) amortized per reduction step
/// (small `Vec` linear scan, since reducer counts are bounded by basis
/// size, typically < 100).
#[inline]
fn reduce_poly_with_sugar<'a, 'b, F, I, P, O, S>(
    ring: P,
    to_reduce: &mut El<P>,
    reducers: &mut F,
    order: O,
    mut sugar_self: Option<&mut Sugar>,
    mut reducer_sugar: S,
) -> usize
where
    P: 'a + RingStore + Copy,
    P::Type: MultivariatePolyRing,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
    O: MonomialOrder + Copy,
    F: FnMut() -> I,
    I: Iterator<Item = (&'a El<P>, &'b AugLm)>,
    S: FnMut(usize) -> Sugar,
{
    let mut n_raises: usize = 0;

    // Build a pointer→sugar map once.  Reducer iterator yields stable
    // references so the pointers remain valid for the whole reduction
    // (the underlying buffers — `reducers` Vec and `new_polys`
    // AppendOnlyVec — never reallocate while we hold these pointers).
    //
    // We only build this map when sugar tracking is requested; the
    // no-sugar path (`sugar_self == None`) skips this O(R) one-shot
    // setup entirely.
    let ptr_sugar: Vec<(*const El<P>, Sugar)> = if sugar_self.is_some() {
        reducers().enumerate()
            .map(|(i, (r, _))| (r as *const El<P>, reducer_sugar(i)))
            .collect()
    } else {
        Vec::new()
    };

    // The accumulator fast path is only valid for DegRevLex, since the
    // internal linked-list is sorted by (deg, order) which matches DegRevLex.
    if order.is_same(&DegRevLex) {
        // Sugar observer: for each reducer applied with cofactor of
        // degree `cof_deg`, raise `*sugar_self` per
        // `StdDegBase::myUpdate` (`SugarDegree.C:215-218`) and bump
        // `n_raises` if the value strictly increased.
        let sugar_self_cell = std::cell::RefCell::new(sugar_self.as_deref_mut());
        let n_raises_cell = std::cell::Cell::new(0usize);
        let ptr_sugar_ref = &ptr_sugar;
        let mut on_reducer_applied = |reducer_ptr: *const El<P>, cof_deg: usize| {
            let mut borrow = sugar_self_cell.borrow_mut();
            if let Some(s) = borrow.as_deref_mut() {
                if let Some(&(_, rs)) = ptr_sugar_ref.iter().find(|(p, _)| *p == reducer_ptr) {
                    let before = s.value();
                    s.my_update(cof_deg, rs);
                    if s.value() > before {
                        n_raises_cell.set(n_raises_cell.get() + 1);
                    }
                }
            }
        };
        let used_fast_path = ring.get_ring().reduce_poly_loop_with_observer(
            to_reduce,
            &mut |lt_c: &PolyCoeff<P>, lt_m: &PolyMonomial<P>| {
                let f_lm_expanded = ring.expand_monomial(lt_m);
                let f_mask = divmask(&f_lm_expanded);
                // MISSING-3: Select shortest reducer via min_by_key
                reducers()
                    .filter_map(|(reducer, reducer_aug)| {
                        // MISSING-4: Use cached DivMask from AugLm
                        if (reducer_aug.mask & !f_mask) != 0 {
                            return None;
                        }
                        if (0..ring.indeterminate_count()).all(|j| reducer_aug.exponents[j] <= f_lm_expanded[j]) {
                            let (r_lc, r_lm) = ring.LT(reducer, order).unwrap();
                            let quo_m = ring.monomial_div(ring.clone_monomial(lt_m), r_lm).ok().unwrap();
                            if let Some(quo_c) = ring.base_ring().checked_div(lt_c, r_lc) {
                                return Some((
                                    reducer as *const El<P>,
                                    quo_c,
                                    quo_m,
                                    ring.terms(reducer).count(),
                                ));
                            }
                        }
                        None
                    })
                    .min_by_key(|(_, _, _, count)| *count)
                    .map(|(ptr, c, m, _)| (ptr, c, m))
            },
            &mut on_reducer_applied,
        );
        if used_fast_path {
            return n_raises_cell.get();
        }
        // Fast path declined: harvest any raises that may have already
        // happened (none, since the closure isn't invoked) and continue
        // to the fallback below.
        n_raises = n_raises_cell.get();
    }

    // Fallback: generic loop.  Sugar update happens here directly since
    // we have the cofactor monomial in hand (`quo_m`).
    while let Some((_, reducer, quo_c, quo_m)) = find_reducer(ring, to_reduce, reducers(), order) {
        if let Some(ref mut s) = sugar_self {
            let cof_deg = ring.monomial_deg(&quo_m);
            let reducer_ptr = reducer as *const El<P>;
            if let Some(&(_, rs)) = ptr_sugar.iter().find(|(p, _)| *p == reducer_ptr) {
                let before = s.value();
                s.my_update(cof_deg, rs);
                if s.value() > before { n_raises += 1; }
            }
        }
        ring.sub_assign_mul_monomial(to_reduce, reducer, &quo_c, &quo_m);
    }
    n_raises
}

#[stability::unstable(feature = "enable")]
pub fn multivariate_division<'a, P, O, I>(ring: P, mut f: El<P>, reducers: I, order: O) -> El<P>
where
    P: 'a + RingStore + Copy,
    P::Type: MultivariatePolyRing,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
    O: MonomialOrder + Copy,
    I: Clone + Iterator<Item = &'a El<P>>,
{
    let augs = reducers
        .clone()
        .map(|f| {
            let exponents = ring.expand_monomial(ring.LT(f, order).unwrap().1);
            let mask = divmask(&exponents);
            AugLm { exponents, mask }
        })
        .collect::<Vec<_>>();
    reduce_poly(ring, &mut f, || reducers.clone().zip(augs.iter()), order);
    return f;
}

fn inter_reduce<P, O>(ring: P, mut polys: Vec<(El<P>, AugLm)>, order: O) -> Vec<(El<P>, AugLm)>
where
    P: RingStore + Copy,
    P::Type: MultivariatePolyRing,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: DivisibilityRing,
    O: MonomialOrder + Copy,
{
    let mut changed = true;
    while changed {
        changed = false;
        let mut i = 0;
        while i < polys.len() {
            let last_i = polys.len() - 1;
            polys.swap(i, last_i);
            let (reducers, to_reduce) = polys.split_at_mut(last_i);
            let to_reduce = &mut to_reduce[0];

            // MISSING-6: Remember old LT to detect if reduction changed it
            let old_deg = to_reduce.1.exponents.iter().copied().sum::<usize>();
            let old_mask = to_reduce.1.mask;

            reduce_poly(
                ring,
                &mut to_reduce.0,
                || reducers.iter().map(|(f, aug)| (f, aug)),
                order,
            );

            // undo swap so that the outer loop still iterates over every poly
            if !ring.is_zero(&to_reduce.0) {
                let new_exponents = ring.expand_monomial(ring.LT(&to_reduce.0, order).unwrap().1);
                let new_mask = divmask(&new_exponents);
                let new_deg: usize = new_exponents.iter().copied().sum();
                // MISSING-6: If LT changed, mark changed to trigger another pass
                if new_deg != old_deg || new_mask != old_mask {
                    changed = true;
                }
                to_reduce.1 = AugLm { exponents: new_exponents, mask: new_mask };
                polys.swap(i, last_i);
                i += 1;
            } else {
                _ = polys.pop().unwrap();
                // A polynomial was reduced to zero — this changes the set, trigger another pass
                changed = true;
            }
        }
    }
    return polys;
}

use crate::rings::local::AsLocalPIR;
use crate::rings::multivariate::multivariate_impl::MultivariatePolyRingImpl;

/// Computes a Groebner basis of the ideal generated by the input basis w.r.t. the given term
/// ordering.
///
/// For a variant of this function that allows for more configuration, see [`buchberger()`].
pub fn buchberger_simple<P, O>(ring: P, input_basis: Vec<El<P>>, order: O) -> Vec<El<P>>
where
    P: RingStore + Copy + Send + Sync,
    El<P>: Send + Sync,
    P::Type: MultivariatePolyRing,
    <P::Type as RingExtension>::BaseRing: Sync,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: Field,
    O: MonomialOrder + Copy + Send + Sync,
    PolyCoeff<P>: Send + Sync,
{
    let as_local_pir = AsLocalPIR::from_field(ring.base_ring());
    let new_poly_ring = MultivariatePolyRingImpl::new(&as_local_pir, ring.indeterminate_count());
    let from_ring = new_poly_ring.lifted_hom(ring, WrapHom::to_delegate_ring(as_local_pir.get_ring()));
    let result = buchberger::<_, _, _, _, _>(
        &new_poly_ring,
        input_basis.into_iter().map(|f| from_ring.map(f)).collect(),
        order,
        default_sort_fn(&new_poly_ring, order),
        |_| false,
        DontObserve,
    )
    .unwrap_or_else(no_error);
    let to_ring = ring.lifted_hom(&new_poly_ring, UnwrapHom::from_delegate_ring(as_local_pir.get_ring()));
    return result.into_iter().map(|f| to_ring.map(f)).collect();
}

#[cfg(test)]
use crate::integer::BigIntRing;
#[cfg(test)]
use crate::rings::poly::{PolyRingStore, dense_poly};
#[cfg(test)]
use crate::rings::rational::RationalField;
#[cfg(test)]
use crate::rings::zn::zn_static;

/// Sprint 2.3.4 (T-Sug-4): unit tests for the `Sugar` wrapper.
///
/// These exercise the canonical Giovini–Mora–Niesi–Robbiano formulas
/// in isolation — no polynomial ring needed — so that any regression in
/// the wrapper's arithmetic is caught immediately, independent of the
/// Buchberger plumbing in 2.3.2.
#[test]
fn test_sugar_my_mul_adds_pp_degree() {
    // sugar(f * pp) = sugar(f) + deg(pp) — `StdDegBase::myMul`
    let mut s = Sugar::new(7);
    s.my_mul(3);
    assert_eq!(s.value(), 10);
    s.my_mul(0); // identity
    assert_eq!(s.value(), 10);
}

#[test]
fn test_sugar_my_update_is_raise_only() {
    // sugar(f + c*g) = max(sugar(f), deg(c) + sugar(g)) —
    // `StdDegBase::myUpdate`.  Critical property: never lowers.
    let mut s = Sugar::new(10);

    // Case A: cofactor_deg + other > current → raises.
    s.my_update(/* cofactor_deg */ 4, /* other */ Sugar::new(8));
    assert_eq!(s.value(), 12, "12 = max(10, 4+8) should raise");

    // Case B: cofactor_deg + other < current → no change (raise-only).
    s.my_update(2, Sugar::new(3));
    assert_eq!(s.value(), 12, "max(12, 2+3)=12 unchanged");

    // Case C: equality → no change.
    s.my_update(0, Sugar::new(12));
    assert_eq!(s.value(), 12);
}

#[test]
fn test_sugar_for_spair_matches_classical_formula() {
    // sugar(S(f,g)) = max(sugar(f) + (lcm-deg_LT(f)),
    //                     sugar(g) + (lcm-deg_LT(g))).
    // Mirrors `NewSugar(GPair)` in `TmpGPair.C:36-45`.

    // Case A: balanced — both branches agree.  Sugars 5, 4; LT degs 2, 1;
    // lcm 3 → max(5+1, 4+2) = max(6, 6) = 6.
    let s = Sugar::for_spair(Sugar::new(5), 2, Sugar::new(4), 1, /* lcm */ 3);
    assert_eq!(s.value(), 6);

    // Case B: f-branch dominates.  Sugars 10, 4; LT degs 2, 1;
    // lcm 3 → max(10+1, 4+2) = 11.
    let s = Sugar::for_spair(Sugar::new(10), 2, Sugar::new(4), 1, 3);
    assert_eq!(s.value(), 11);

    // Case C: g-branch dominates.  Sugars 4, 10; LT degs 2, 1;
    // lcm 3 → max(4+1, 10+2) = 12.
    let s = Sugar::for_spair(Sugar::new(4), 2, Sugar::new(10), 1, 3);
    assert_eq!(s.value(), 12);

    // Case D: coprime LTs (lcm = LT(f) + LT(g) componentwise; total
    // degree adds).  LT(f)=x^2 deg 2, LT(g)=y^3 deg 3, lcm=x^2 y^3 deg 5;
    // sugars 0,0 → S-pair sugar = max(0+3, 0+2) = 3.
    let s = Sugar::for_spair(Sugar::new(0), 2, Sugar::new(0), 3, 5);
    assert_eq!(s.value(), 3);
}

#[test]
fn test_sugar_for_spair_equals_unrolled_my_mul_my_update() {
    // The closed-form `for_spair` must match the unrolled CoCoA sequence
    //   s := sugar(f); s.my_mul(c1_deg); s.my_update(c2_deg, sugar(g))
    // for all reasonable inputs.  This is the contract that makes the
    // wrapper a drop-in for `NewSugar(GPair)`.
    for sf in 0..=8 {
        for sg in 0..=8 {
            for df in 0..=4 {
                for dg in 0..=4 {
                    let lcm = std::cmp::max(df, dg) + 1; // any lcm ≥ both
                    let closed = Sugar::for_spair(
                        Sugar::new(sf), df, Sugar::new(sg), dg, lcm
                    );
                    let mut unrolled = Sugar::new(sf);
                    unrolled.my_mul(lcm - df);          // c1_deg
                    unrolled.my_update(lcm - dg, Sugar::new(sg));
                    assert_eq!(closed, unrolled,
                        "mismatch for sf={},sg={},df={},dg={},lcm={}",
                        sf, sg, df, dg, lcm);
                }
            }
        }
    }
}

#[test]
fn test_sugar_running_update_demonstrates_raise() {
    // Synthetic running-sugar trace.  Initial S-pair sugar = 5;
    // reductions apply cofactors of degrees 1, 0, 4 against reducers
    // with sugars 6, 3, 2.  Final sugar should be max(5, 1+6, 0+3, 4+2)
    // = max(5, 7, 3, 6) = 7.  Demonstrates that the *third* reduction
    // (cof 0, sugar 3) does NOT lower the value below the prior 7 —
    // running sugar is monotonically non-decreasing.
    let mut s = Sugar::new(5);
    s.my_update(1, Sugar::new(6)); assert_eq!(s.value(), 7);
    s.my_update(0, Sugar::new(3)); assert_eq!(s.value(), 7);
    s.my_update(4, Sugar::new(2)); assert_eq!(s.value(), 7);
    // A strictly larger contribution finally raises further.
    s.my_update(5, Sugar::new(5)); assert_eq!(s.value(), 10);
}

#[test]
fn test_buchberger_small() {
    let base = zn_static::F17;
    let ring = MultivariatePolyRingImpl::new(base, 2);

    let f1 = ring.from_terms(
        [
            (1, ring.create_monomial([2, 0])),
            (1, ring.create_monomial([0, 2])),
            (16, ring.create_monomial([0, 0])),
        ]
        .into_iter(),
    );
    let f2 = ring.from_terms([(1, ring.create_monomial([1, 1])), (15, ring.create_monomial([0, 0]))].into_iter());

    let actual = buchberger(
        &ring,
        vec![ring.clone_el(&f1), ring.clone_el(&f2)],
        DegRevLex,
        default_sort_fn(&ring, DegRevLex),
        |_| false,
        TEST_LOG_PROGRESS,
    )
    .unwrap_or_else(no_error);

    let expected = ring.from_terms(
        [
            (16, ring.create_monomial([0, 3])),
            (15, ring.create_monomial([1, 0])),
            (1, ring.create_monomial([0, 1])),
        ]
        .into_iter(),
    );

    assert_eq!(3, actual.len());
    assert_el_eq!(
        ring,
        ring.zero(),
        multivariate_division(&ring, f1, actual.iter(), DegRevLex)
    );
    assert_el_eq!(
        ring,
        ring.zero(),
        multivariate_division(&ring, f2, actual.iter(), DegRevLex)
    );
    assert_el_eq!(
        ring,
        ring.zero(),
        multivariate_division(&ring, expected, actual.iter(), DegRevLex)
    );
}

#[test]
fn test_buchberger_larger() {
    let base = zn_static::F17;
    let ring = MultivariatePolyRingImpl::new(base, 3);

    let f1 = ring.from_terms(
        [
            (1, ring.create_monomial([2, 1, 1])),
            (1, ring.create_monomial([0, 2, 0])),
            (1, ring.create_monomial([1, 0, 1])),
            (2, ring.create_monomial([1, 0, 0])),
            (1, ring.create_monomial([0, 0, 0])),
        ]
        .into_iter(),
    );
    let f2 = ring.from_terms(
        [
            (1, ring.create_monomial([0, 3, 1])),
            (1, ring.create_monomial([0, 0, 3])),
            (1, ring.create_monomial([1, 1, 0])),
        ]
        .into_iter(),
    );
    let f3 = ring.from_terms(
        [
            (1, ring.create_monomial([1, 0, 2])),
            (1, ring.create_monomial([1, 0, 1])),
            (2, ring.create_monomial([0, 1, 1])),
            (7, ring.create_monomial([0, 0, 0])),
        ]
        .into_iter(),
    );

    let actual = buchberger(
        &ring,
        vec![ring.clone_el(&f1), ring.clone_el(&f2), ring.clone_el(&f3)],
        DegRevLex,
        default_sort_fn(&ring, DegRevLex),
        |_| false,
        TEST_LOG_PROGRESS,
    )
    .unwrap_or_else(no_error);

    let g1 = ring.from_terms(
        [
            (1, ring.create_monomial([0, 4, 0])),
            (8, ring.create_monomial([0, 3, 1])),
            (12, ring.create_monomial([0, 1, 3])),
            (6, ring.create_monomial([0, 0, 4])),
            (1, ring.create_monomial([0, 3, 0])),
            (13, ring.create_monomial([0, 2, 1])),
            (11, ring.create_monomial([0, 1, 2])),
            (10, ring.create_monomial([0, 0, 3])),
            (11, ring.create_monomial([0, 2, 0])),
            (12, ring.create_monomial([0, 1, 1])),
            (6, ring.create_monomial([0, 0, 2])),
            (6, ring.create_monomial([0, 1, 0])),
            (13, ring.create_monomial([0, 0, 1])),
            (9, ring.create_monomial([0, 0, 0])),
        ]
        .into_iter(),
    );

    assert_el_eq!(
        ring,
        ring.zero(),
        multivariate_division(&ring, g1, actual.iter(), DegRevLex)
    );
}

#[test]
fn test_generic_computation() {
    let base = zn_static::F17;
    let ring = MultivariatePolyRingImpl::new(base, 6);
    let poly_ring = dense_poly::DensePolyRing::new(&ring, "X");

    let var_i = |i: usize| {
        ring.create_term(
            base.one(),
            ring.create_monomial((0..ring.indeterminate_count()).map(|j| if i == j { 1 } else { 0 })),
        )
    };
    let X1 = poly_ring.mul(
        poly_ring.from_terms([(var_i(0), 0), (ring.one(), 1)].into_iter()),
        poly_ring.from_terms([(var_i(1), 0), (ring.one(), 1)].into_iter()),
    );
    let X2 = poly_ring.mul(
        poly_ring.add(
            poly_ring.clone_el(&X1),
            poly_ring.from_terms([(var_i(2), 0), (var_i(3), 1)].into_iter()),
        ),
        poly_ring.add(
            poly_ring.clone_el(&X1),
            poly_ring.from_terms([(var_i(4), 0), (var_i(5), 1)].into_iter()),
        ),
    );
    let basis = vec![
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 0)),
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 1)),
        ring.sub_ref_snd(ring.int_hom().map(1), poly_ring.coefficient_at(&X2, 2)),
    ];

    let start = std::time::Instant::now();
    let gb1 = buchberger(
        &ring,
        basis.iter().map(|f| ring.clone_el(f)).collect(),
        DegRevLex,
        default_sort_fn(&ring, DegRevLex),
        |_| false,
        TEST_LOG_PROGRESS,
    )
    .unwrap_or_else(no_error);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    assert_eq!(11, gb1.len());
}

#[test]
fn test_gb_local_ring() {
    let base = AsLocalPIR::from_zn(zn_static::Zn::<16>::RING).unwrap();
    let ring: MultivariatePolyRingImpl<_> = MultivariatePolyRingImpl::new(base, 1);

    let f = ring.from_terms(
        [
            (base.int_hom().map(4), ring.create_monomial([1])),
            (base.one(), ring.create_monomial([0])),
        ]
        .into_iter(),
    );
    let gb = buchberger(
        &ring,
        vec![f],
        DegRevLex,
        default_sort_fn(&ring, DegRevLex),
        |_| false,
        TEST_LOG_PROGRESS,
    )
    .unwrap_or_else(no_error);

    assert_eq!(1, gb.len());
    assert_el_eq!(ring, ring.one(), gb[0]);
}

#[test]
fn test_gb_lex() {
    let ZZ = BigIntRing::RING;
    let QQ = AsLocalPIR::from_field(RationalField::new(ZZ));
    let QQYX = MultivariatePolyRingImpl::new(&QQ, 2);
    let [f, g] = QQYX.with_wrapped_indeterminates(|[Y, X]| {
        [
            1 + X.pow_ref(2) + 2 * Y + (1 + X) * Y.pow_ref(2),
            3 + X + (2 + X) * Y + (1 + X + X.pow_ref(2)) * Y.pow_ref(2),
        ]
    });
    let expected = QQYX.with_wrapped_indeterminates(|[Y, X]| {
        [
            X.pow_ref(8) + 2 * X.pow_ref(7) + 3 * X.pow_ref(6)
                - 5 * X.pow_ref(5)
                - 10 * X.pow_ref(4)
                - 7 * X.pow_ref(3)
                + 8 * X.pow_ref(2)
                + 8 * X
                + 4,
            2 * Y + X.pow_ref(6) + 3 * X.pow_ref(5) + 6 * X.pow_ref(4) + X.pow_ref(3) - 7 * X.pow_ref(2) - 12 * X - 2,
        ]
    });

    let mut gb = buchberger_simple(&QQYX, vec![f, g], Lex);

    assert_eq!(2, gb.len());
    gb.sort_unstable_by_key(|f| QQYX.appearing_indeterminates(f).len());
    for (mut f, mut e) in gb.into_iter().zip(expected.into_iter()) {
        let f_lc_inv = QQ.invert(QQYX.LT(&f, Lex).unwrap().0).unwrap();
        QQYX.inclusion().mul_assign_map(&mut f, f_lc_inv);
        let e_lc_inv = QQ.invert(QQYX.LT(&e, Lex).unwrap().0).unwrap();
        QQYX.inclusion().mul_assign_map(&mut e, e_lc_inv);
        assert_el_eq!(QQYX, e, f);
    }
}

#[cfg(test)]
#[cfg(feature = "parallel")]
static TEST_COMPUTATION_CONTROLLER: ExecuteMultithreaded<LogProgress> = RunMultithreadedLogProgress;
#[cfg(test)]
#[cfg(not(feature = "parallel"))]
static TEST_COMPUTATION_CONTROLLER: LogProgress = TEST_LOG_PROGRESS;

#[ignore]
#[test]
fn test_expensive_gb_1() {
    let base = AsLocalPIR::from_zn(zn_static::Zn::<16>::RING).unwrap();
    let ring: MultivariatePolyRingImpl<_> = MultivariatePolyRingImpl::new(base, 12);

    let system = ring.with_wrapped_indeterminates(|[Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11]| {
        [
            Y0 * Y1 * Y2 * Y3.pow_ref(2) * Y4.pow_ref(2)
                + 4 * Y0 * Y1 * Y2 * Y3 * Y4 * Y4 * Y8
                + Y0 * Y1 * Y2 * Y5.pow_ref(2) * Y8.pow_ref(2)
                + Y0 * Y2 * Y3 * Y4 * Y6
                + Y0 * Y1 * Y3 * Y4 * Y7
                + Y0 * Y2 * Y5 * Y6 * Y8
                + Y0 * Y1 * Y5 * Y7 * Y8
                + Y0 * Y2 * Y3 * Y5 * Y10
                + Y0 * Y1 * Y3 * Y5 * Y11
                + Y0 * Y6 * Y7
                + Y3 * Y5 * Y9
                - 4,
            2 * Y0 * Y1 * Y2 * Y3.pow_ref(2) * Y4 * Y5
                + 2 * Y0 * Y1 * Y2 * Y3 * Y5.pow_ref(2) * Y8
                + Y0 * Y2 * Y3 * Y5 * Y6
                + Y0 * Y1 * Y3 * Y5 * Y7
                + 8,
            Y0 * Y1 * Y2 * Y3.pow_ref(2) * Y5.pow_ref(2) - 5,
        ]
    });

    let part_of_result =
        ring.with_wrapped_indeterminates(|[_Y0, Y1, Y2, _Y3, _Y4, _Y5, Y6, Y7, _Y8, _Y9, _Y10, _Y11]| {
            [
                4 * Y2.pow_ref(2) * Y6.pow_ref(2) - 4 * Y1.pow_ref(2) * Y7.pow_ref(2),
                8 * Y2 * Y6 + 8 * Y1 * Y7.clone(),
            ]
        });

    let start = std::time::Instant::now();
    let gb = buchberger(
        &ring,
        system.iter().map(|f| ring.clone_el(f)).collect(),
        DegRevLex,
        default_sort_fn(&ring, DegRevLex),
        |_| false,
        TEST_COMPUTATION_CONTROLLER,
    )
    .unwrap_or_else(no_error);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    for f in &part_of_result {
        assert!(ring.is_zero(&multivariate_division(&ring, ring.clone_el(f), gb.iter(), DegRevLex)));
    }

    assert_eq!(108, gb.len());
}

#[test]
#[ignore]
fn test_expensive_gb_2() {
    let base = zn_static::Fp::<7>::RING;
    let ring = MultivariatePolyRingImpl::new(base, 7);

    let basis = ring.with_wrapped_indeterminates_dyn(|[X0, X1, X2, X3, X4, X5, X6]| {
        [
            6 + 2 * X5
                + 2 * X4
                + X6
                + 4 * X0
                + 5 * X6 * X5
                + X6 * X4
                + 3 * X0 * X4
                + 6 * X0 * X6
                + 2 * X0 * X3
                + X0 * X2
                + 4 * X0 * X1
                + 2 * X3 * X4 * X5
                + 4 * X0 * X6 * X5
                + 6 * X0 * X2 * X5
                + 5 * X0 * X6 * X4
                + 2 * X0 * X3 * X4
                + 4 * X0 * X1 * X4
                + X0 * X6.pow_ref(2)
                + 3 * X0 * X3 * X6
                + 5 * X0 * X2 * X6
                + 2 * X0 * X1 * X6
                + X0 * X3.pow_ref(2)
                + 2 * X0 * X2 * X3
                + 3 * X0 * X3 * X4 * X5
                + 4 * X0 * X3 * X6 * X5
                + 3 * X0 * X1 * X6 * X5
                + 3 * X0 * X2 * X3 * X5
                + 3 * X0 * X3 * X6 * X4
                + 2 * X0 * X1 * X6 * X4
                + 2 * X0 * X3.pow_ref(2) * X4
                + 2 * X0 * X2 * X3 * X4
                + 3 * X0 * X3.pow_ref(2) * X4 * X5
                + 4 * X0 * X1 * X3 * X4 * X5
                + X0 * X3.pow_ref(2) * X4.pow_ref(2),
            5 + 4 * X0
                + 6 * X4 * X5
                + 3 * X6 * X5
                + 4 * X0 * X4
                + 3 * X0 * X6
                + 6 * X0 * X3
                + 6 * X0 * X2
                + 6 * X6 * X4 * X5
                + 2 * X0 * X4 * X5
                + 4 * X0 * X6 * X5
                + 3 * X0 * X2 * X5
                + 3 * X0 * X6 * X4
                + 5 * X0 * X3 * X4
                + 6 * X0 * X2 * X4
                + 4 * X0 * X6.pow_ref(2)
                + 3 * X0 * X3 * X6
                + 3 * X0 * X2 * X6
                + 2 * X0 * X6 * X4 * X5
                + 6 * X0 * X3 * X4 * X5
                + 5 * X0 * X1 * X4 * X5
                + 6 * X0 * X6.pow_ref(2) * X5
                + 2 * X0 * X3 * X6 * X5
                + 2 * X0 * X2 * X6 * X5
                + 6 * X0 * X1 * X6 * X5
                + 6 * X0 * X2 * X3 * X5
                + 6 * X0 * X3 * X4.pow_ref(2)
                + 4 * X0 * X6.pow_ref(2) * X4
                + 6 * X0 * X3 * X6 * X4
                + 3 * X0 * X2 * X6 * X4
                + 4 * X0 * X3 * X6 * X4 * X5
                + 5 * X0 * X1 * X6 * X4 * X5
                + 6 * X0 * X3.pow_ref(2) * X4 * X5
                + 5 * X0 * X2 * X3 * X4 * X5
                + 3 * X0 * X3 * X6 * X4.pow_ref(2)
                + 6 * X0 * X3.pow_ref(2) * X4.pow_ref(2) * X5.clone(),
            2 + 2 * X0
                + 4 * X0 * X4
                + 2 * X0 * X6
                + 5 * X0 * X4 * X5
                + 2 * X0 * X6 * X5
                + 4 * X0 * X2 * X5
                + 2 * X0 * X4.pow_ref(2)
                + 4 * X0 * X6 * X4
                + 4 * X0 * X6.pow_ref(2)
                + 2 * X6 * X4 * X5.pow_ref(2)
                + 4 * X0 * X6 * X4 * X5
                + X0 * X3 * X4 * X5
                + X0 * X2 * X4 * X5
                + 3 * X0 * X6.pow_ref(2) * X5
                + 2 * X0 * X3 * X6 * X5
                + 4 * X0 * X2 * X6 * X5
                + 2 * X0 * X6 * X4.pow_ref(2)
                + X0 * X6.pow_ref(2) * X4
                + 3 * X0 * X6 * X4 * X5.pow_ref(2)
                + 2 * X0 * X6.pow_ref(2) * X5.pow_ref(2)
                + 3 * X0 * X2 * X6 * X5.pow_ref(2)
                + X0 * X3 * X4.pow_ref(2) * X5
                + X0 * X6.pow_ref(2) * X4 * X5
                + X0 * X3 * X6 * X4 * X5
                + 6 * X0 * X2 * X6 * X4 * X5
                + 4 * X0 * X6.pow_ref(2) * X4.pow_ref(2)
                + 6 * X0 * X3 * X6 * X4 * X5.pow_ref(2)
                + 4 * X0 * X1 * X6 * X4 * X5.pow_ref(2)
                + 4 * X0 * X2 * X3 * X4 * X5.pow_ref(2)
                + 6 * X0 * X3 * X6 * X4.pow_ref(2) * X5
                + 2 * X0 * X3.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(2),
            4 + 5 * X0 * X4 * X5
                + 6 * X0 * X6 * X5
                + 5 * X0 * X4.pow_ref(2) * X5
                + 3 * X0 * X6 * X4 * X5
                + 3 * X0 * X6.pow_ref(2) * X5
                + 6 * X0 * X6 * X4 * X5.pow_ref(2)
                + 5 * X0 * X2 * X4 * X5.pow_ref(2)
                + X0 * X6.pow_ref(2) * X5.pow_ref(2)
                + 6 * X0 * X2 * X6 * X5.pow_ref(2)
                + 4 * X0 * X6 * X4.pow_ref(2) * X5
                + 2 * X0 * X6.pow_ref(2) * X4 * X5
                + 5 * X0 * X3 * X4.pow_ref(2) * X5.pow_ref(2)
                + 3 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(2)
                + 5 * X0 * X3 * X6 * X4 * X5.pow_ref(2)
                + 4 * X0 * X2 * X6 * X4 * X5.pow_ref(2)
                + 6 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5
                + 4 * X0 * X3 * X6 * X4.pow_ref(2) * X5.pow_ref(2),
            4 + 4 * X0 * X4.pow_ref(2) * X5.pow_ref(2)
                + X0 * X6 * X4 * X5.pow_ref(2)
                + X0 * X6.pow_ref(2) * X5.pow_ref(2)
                + 5 * X0 * X6 * X4.pow_ref(2) * X5.pow_ref(2)
                + 6 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(2)
                + 3 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(3)
                + 4 * X0 * X2 * X6 * X4 * X5.pow_ref(3)
                + 6 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(2)
                + 4 * X0 * X3 * X6 * X4.pow_ref(2) * X5.pow_ref(3),
            5 * X0 * X6 * X4.pow_ref(2) * X5.pow_ref(3)
                + 6 * X0 * X6.pow_ref(2) * X4 * X5.pow_ref(3)
                + 5 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(3),
            2 * X0 * X6.pow_ref(2) * X4.pow_ref(2) * X5.pow_ref(4),
        ]
    });

    let start = std::time::Instant::now();
    let gb = buchberger(
        &ring,
        basis,
        DegRevLex,
        default_sort_fn(&ring, DegRevLex),
        |_| false,
        TEST_COMPUTATION_CONTROLLER,
    )
    .unwrap_or_else(no_error);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());

    assert_eq!(130, gb.len());
}

#[test]
#[ignore]
fn test_groebner_cyclic6() {
    let base = zn_static::Fp::<65537>::RING;
    let ring = MultivariatePolyRingImpl::new(base, 6);

    let cyclic6 = ring.with_wrapped_indeterminates_dyn(|[x, y, z, t, u, v]| {
        [
            x + y + z + t + u + v,
            x * y + y * z + z * t + t * u + x * v + u * v,
            x * y * z + y * z * t + z * t * u + x * y * v + x * u * v + t * u * v,
            x * y * z * t + y * z * t * u + x * y * z * v + x * y * u * v + x * t * u * v + z * t * u * v,
            x * y * z * t * u
                + x * y * z * t * v
                + x * y * z * u * v
                + x * y * t * u * v
                + x * z * t * u * v
                + y * z * t * u * v,
            x * y * z * t * u * v - 1,
        ]
    });

    let start = std::time::Instant::now();
    let gb = buchberger(
        &ring,
        cyclic6,
        DegRevLex,
        default_sort_fn(&ring, DegRevLex),
        |_| false,
        TEST_COMPUTATION_CONTROLLER,
    )
    .unwrap_or_else(no_error);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());
    assert_eq!(45, gb.len());
}

#[test]
#[ignore]
fn test_groebner_cyclic7() {
    let base = zn_static::Fp::<65537>::RING;
    let ring = MultivariatePolyRingImpl::new(base, 7);

    let cyclic7 = ring.with_wrapped_indeterminates_dyn(|[x, y, z, t, u, v, w]| {
        [
            x + y + z + t + u + v + w,
            x * y + y * z + z * t + t * u + u * v + x * w + v * w,
            x * y * z + y * z * t + z * t * u + t * u * v + x * y * w + x * v * w + u * v * w,
            x * y * z * t
                + y * z * t * u
                + z * t * u * v
                + x * y * z * w
                + x * y * v * w
                + x * u * v * w
                + t * u * v * w,
            x * y * z * t * u
                + y * z * t * u * v
                + x * y * z * t * w
                + x * y * z * v * w
                + x * y * u * v * w
                + x * t * u * v * w
                + z * t * u * v * w,
            x * y * z * t * u * v
                + x * y * z * t * u * w
                + x * y * z * t * v * w
                + x * y * z * u * v * w
                + x * y * t * u * v * w
                + x * z * t * u * v * w
                + y * z * t * u * v * w,
            x * y * z * t * u * v * w - 1,
        ]
    });

    let start = std::time::Instant::now();
    let gb = buchberger(
        &ring,
        cyclic7,
        DegRevLex,
        default_sort_fn(&ring, DegRevLex),
        |_| false,
        TEST_COMPUTATION_CONTROLLER,
    )
    .unwrap_or_else(no_error);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());
    assert_eq!(209, gb.len());
}

#[test]
#[ignore]
fn test_groebner_cyclic8() {
    let base = zn_static::Fp::<65537>::RING;
    let ring = MultivariatePolyRingImpl::new(base, 8);

    let cyclic7 = ring.with_wrapped_indeterminates_dyn(|[x, y, z, s, t, u, v, w]| {
        [
            x + y + z + s + t + u + v + w,
            x * y + y * z + z * s + s * t + t * u + u * v + x * w + v * w,
            x * y * z + y * z * s + z * s * t + s * t * u + t * u * v + x * y * w + x * v * w + u * v * w,
            x * y * z * s
                + y * z * s * t
                + z * s * t * u
                + s * t * u * v
                + x * y * z * w
                + x * y * v * w
                + x * u * v * w
                + t * u * v * w,
            x * y * z * s * t
                + y * z * s * t * u
                + z * s * t * u * v
                + x * y * z * s * w
                + x * y * z * v * w
                + x * y * u * v * w
                + x * t * u * v * w
                + s * t * u * v * w,
            x * y * z * s * t * u
                + y * z * s * t * u * v
                + x * y * z * s * t * w
                + x * y * z * s * v * w
                + x * y * z * u * v * w
                + x * y * t * u * v * w
                + x * s * t * u * v * w
                + z * s * t * u * v * w,
            x * y * z * s * t * u * v
                + x * y * z * s * t * u * w
                + x * y * z * s * t * v * w
                + x * y * z * s * u * v * w
                + x * y * z * t * u * v * w
                + x * y * s * t * u * v * w
                + x * z * s * t * u * v * w
                + y * z * s * t * u * v * w,
            x * y * z * s * t * u * v * w - 1,
        ]
    });

    let start = std::time::Instant::now();
    let gb = buchberger(
        &ring,
        cyclic7,
        DegRevLex,
        default_sort_fn(&ring, DegRevLex),
        |_| false,
        TEST_COMPUTATION_CONTROLLER,
    )
    .unwrap_or_else(no_error);
    let end = std::time::Instant::now();

    println!("Computed GB in {} ms", (end - start).as_millis());
    assert_eq!(372, gb.len());
}

// ===========================================================================
// Sprint 2.7 — incremental Buchberger tests
// ===========================================================================

/// Helper: assert that `gb_a` and `gb_b` generate the same ideal by
/// checking that every element of one reduces to zero modulo the other.
#[cfg(test)]
fn assert_same_ideal<P, O>(ring: P, gb_a: &[El<P>], gb_b: &[El<P>], order: O)
where
    P: RingStore + Copy,
    P::Type: MultivariatePolyRing,
    <<P::Type as RingExtension>::BaseRing as RingStore>::Type: PrincipalLocalRing,
    O: MonomialOrder + Copy,
{
    for f in gb_a {
        let r = multivariate_division(ring, ring.clone_el(f), gb_b.iter(), order);
        assert!(
            ring.is_zero(&r),
            "gb_a element does not reduce to 0 mod gb_b: {}",
            ring.format(f),
        );
    }
    for f in gb_b {
        let r = multivariate_division(ring, ring.clone_el(f), gb_a.iter(), order);
        assert!(
            ring.is_zero(&r),
            "gb_b element does not reduce to 0 mod gb_a: {}",
            ring.format(f),
        );
    }
}

#[test]
fn incremental_empty_new_polys_returns_known_gb() {
    let base = zn_static::F17;
    let ring = MultivariatePolyRingImpl::new(base, 2);
    let f1 = ring.from_terms([
        (1, ring.create_monomial([2, 0])),
        (1, ring.create_monomial([0, 2])),
        (16, ring.create_monomial([0, 0])),
    ].into_iter());
    let f2 = ring.from_terms([
        (1, ring.create_monomial([1, 1])),
        (15, ring.create_monomial([0, 0])),
    ].into_iter());

    // First compute a full GB to use as known_gb.
    let known_gb = buchberger(
        &ring, vec![ring.clone_el(&f1), ring.clone_el(&f2)],
        DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS,
    ).unwrap_or_else(no_error);

    let result = buchberger_incremental(
        &ring, known_gb.iter().map(|f| ring.clone_el(f)).collect(), vec![],
        DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS,
    ).unwrap_or_else(no_error);

    assert_eq!(known_gb.len(), result.len());
    assert_same_ideal(&ring, &known_gb, &result, DegRevLex);
}

#[test]
fn incremental_empty_known_gb_equals_full_buchberger() {
    let base = zn_static::F17;
    let ring = MultivariatePolyRingImpl::new(base, 2);
    let f1 = ring.from_terms([
        (1, ring.create_monomial([2, 0])),
        (1, ring.create_monomial([0, 2])),
        (16, ring.create_monomial([0, 0])),
    ].into_iter());
    let f2 = ring.from_terms([
        (1, ring.create_monomial([1, 1])),
        (15, ring.create_monomial([0, 0])),
    ].into_iter());

    let full_gb = buchberger(
        &ring, vec![ring.clone_el(&f1), ring.clone_el(&f2)],
        DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS,
    ).unwrap_or_else(no_error);

    let incr_gb = buchberger_incremental(
        &ring, vec![], vec![ring.clone_el(&f1), ring.clone_el(&f2)],
        DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS,
    ).unwrap_or_else(no_error);

    assert_same_ideal(&ring, &full_gb, &incr_gb, DegRevLex);
}

#[test]
fn incremental_one_new_poly_equals_full_recompute() {
    let base = zn_static::F17;
    let ring = MultivariatePolyRingImpl::new(base, 3);
    // Start with two generators
    let f1 = ring.from_terms([
        (1, ring.create_monomial([2, 1, 1])),
        (1, ring.create_monomial([0, 2, 0])),
        (1, ring.create_monomial([1, 0, 1])),
        (2, ring.create_monomial([1, 0, 0])),
        (1, ring.create_monomial([0, 0, 0])),
    ].into_iter());
    let f2 = ring.from_terms([
        (1, ring.create_monomial([0, 3, 1])),
        (1, ring.create_monomial([0, 0, 3])),
        (1, ring.create_monomial([1, 1, 0])),
    ].into_iter());

    // Compute GB of {f1, f2} as the seed
    let known_gb = buchberger(
        &ring, vec![ring.clone_el(&f1), ring.clone_el(&f2)],
        DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS,
    ).unwrap_or_else(no_error);

    // Add a third generator
    let f3 = ring.from_terms([
        (1, ring.create_monomial([1, 0, 2])),
        (1, ring.create_monomial([1, 0, 1])),
        (2, ring.create_monomial([0, 1, 1])),
        (7, ring.create_monomial([0, 0, 0])),
    ].into_iter());

    let incr_gb = buchberger_incremental(
        &ring, known_gb.iter().map(|f| ring.clone_el(f)).collect(),
        vec![ring.clone_el(&f3)],
        DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS,
    ).unwrap_or_else(no_error);

    let full_gb = buchberger(
        &ring, vec![ring.clone_el(&f1), ring.clone_el(&f2), ring.clone_el(&f3)],
        DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS,
    ).unwrap_or_else(no_error);

    assert_same_ideal(&ring, &full_gb, &incr_gb, DegRevLex);
}

#[test]
fn incremental_cyclic4_partial_seed() {
    // cyclic-4: x+y+z+t, xy+yz+zt+tx, xyz+yzt+ztx+txy, xyzt-1
    let base = zn_static::F17;
    let ring = MultivariatePolyRingImpl::new(base, 4);
    let g1 = ring.from_terms([
        (1, ring.create_monomial([1, 0, 0, 0])),
        (1, ring.create_monomial([0, 1, 0, 0])),
        (1, ring.create_monomial([0, 0, 1, 0])),
        (1, ring.create_monomial([0, 0, 0, 1])),
    ].into_iter());
    let g2 = ring.from_terms([
        (1, ring.create_monomial([1, 1, 0, 0])),
        (1, ring.create_monomial([0, 1, 1, 0])),
        (1, ring.create_monomial([0, 0, 1, 1])),
        (1, ring.create_monomial([1, 0, 0, 1])),
    ].into_iter());
    let g3 = ring.from_terms([
        (1, ring.create_monomial([1, 1, 1, 0])),
        (1, ring.create_monomial([0, 1, 1, 1])),
        (1, ring.create_monomial([1, 0, 1, 1])),
        (1, ring.create_monomial([1, 1, 0, 1])),
    ].into_iter());
    let g4 = ring.from_terms([
        (1, ring.create_monomial([1, 1, 1, 1])),
        (16, ring.create_monomial([0, 0, 0, 0])),
    ].into_iter());

    // Seed = GB of first 3 generators
    let known_gb = buchberger(
        &ring, vec![ring.clone_el(&g1), ring.clone_el(&g2), ring.clone_el(&g3)],
        DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS,
    ).unwrap_or_else(no_error);

    let incr_gb = buchberger_incremental(
        &ring, known_gb.iter().map(|f| ring.clone_el(f)).collect(),
        vec![ring.clone_el(&g4)],
        DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS,
    ).unwrap_or_else(no_error);

    let full_gb = buchberger(
        &ring, vec![ring.clone_el(&g1), ring.clone_el(&g2), ring.clone_el(&g3), ring.clone_el(&g4)],
        DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS,
    ).unwrap_or_else(no_error);

    assert_same_ideal(&ring, &full_gb, &incr_gb, DegRevLex);
}

#[test]
fn incremental_basis_to_unit() {
    // Adding 1 (a unit) should collapse the GB to {1}.
    let base = zn_static::F17;
    let ring = MultivariatePolyRingImpl::new(base, 2);
    let f1 = ring.from_terms([
        (1, ring.create_monomial([2, 0])),
        (1, ring.create_monomial([0, 2])),
        (16, ring.create_monomial([0, 0])),
    ].into_iter());

    let known_gb = buchberger(
        &ring, vec![ring.clone_el(&f1)],
        DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS,
    ).unwrap_or_else(no_error);

    let incr_gb = buchberger_incremental(
        &ring, known_gb.iter().map(|f| ring.clone_el(f)).collect(),
        vec![ring.one()],
        DegRevLex, default_sort_fn(&ring, DegRevLex), |_| false, TEST_LOG_PROGRESS,
    ).unwrap_or_else(no_error);

    assert_eq!(1, incr_gb.len(), "expected GB to collapse to {{1}}");
    assert!(ring.is_one(&incr_gb[0]), "expected the lone basis element to be 1");
}
