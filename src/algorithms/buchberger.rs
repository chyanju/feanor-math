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
}

/// A no-op observer that does nothing.
#[stability::unstable(feature = "enable")]
pub struct NoObserver;

impl<P: RingStore> BuchbergerObserver<P> for NoObserver
where
    P::Type: MultivariatePolyRing,
{}

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
    },
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
        // Sort by sugar degree (primary, ascending) then lcm degree (secondary, ascending).
        // Since the main loop processes from the END of the list, we sort in descending
        // order so that the smallest sugar/degree pair is at the end and is selected first.
        // The `_ring` and `_order` parameters are unused now that we sort by the cached
        // sugar/lcm_deg fields on `SPoly` itself, but are kept for API stability.
        open.sort_by(|a, b| {
            b.sugar().cmp(&a.sugar())
                .then_with(|| b.cached_lcm_deg().cmp(&a.cached_lcm_deg()))
        })
    }
}

#[stability::unstable(feature = "enable")]
pub type ExpandedMonomial = Vec<usize>;

/// Compute a DivMask for an expanded monomial.  Bit `i` is set iff
/// `exponents[i % n_vars] > 0`.  This allows O(1) rejection of most
/// non-divisibility checks: if `(mask_divisor & !mask_dividend) != 0`
/// then the divisor has a variable with positive exponent where the
/// dividend has zero — not divisible.
fn divmask(exponents: &[usize]) -> u64 {
    let mut mask: u64 = 0;
    for (i, &e) in exponents.iter().enumerate() {
        if e > 0 && i < 64 {
            mask |= 1u64 << i;
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
    controller.run_computation(
        format_args!(
            "buchberger(len={}, vars={})",
            input_basis.len(),
            ring.indeterminate_count()
        ),
        |controller| {
            // this are the basis polynomials we generated; we only append to this, such that the
            // S-polys remain valid
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

            let input_basis = if needs_inter_reduce {
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

            let sort_reducers = |reducers: &mut [(El<P>, AugLm)]| {
                // I have no idea why, but this order seems to give the best results
                reducers.sort_by(|(lf, _), (rf, _)| {
                    order
                        .compare(ring, ring.LT(lf, order).unwrap().1, ring.LT(rf, order).unwrap().1)
                        .then_with(|| ring.terms(lf).count().cmp(&ring.terms(rf).count()))
                })
            };

            // invariant: `(reducers) = (basis)` and there exists a reduction to zero for every `f`
            // in `basis` modulo `reducers`; reducers are always stored with an expanded
            // version of their leading monomial, in order to simplify divisibility checks
            let mut reducers: Vec<(El<P>, AugLm)> = input_basis
                .iter()
                .map(|f| augment_lm(ring, ring.clone_el(f), order))
                .collect::<Vec<_>>();
            sort_reducers(&mut reducers);

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

            let mut current_sugar: usize = 0;
            let mut filtered_spolys = 0;
            let mut changed = false;
            loop {
                // reduce all known S-polys of minimal sugar degree; in effect, this is the same as
                // the matrix reduction step during F4
                // MISSING-1: Use sugar degree for batching instead of raw lcm degree
                let spolys_to_reduce_index = open
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, spoly)| spoly.sugar() > current_sugar)
                    .map(|(i, _)| i + 1)
                    .unwrap_or(0);
                let spolys_to_reduce = &open[spolys_to_reduce_index..];

                let computation = ShortCircuitingComputation::new();
                let new_polys = AppendOnlyVec::new();
                let new_poly_parents = AppendOnlyVec::new();
                let new_poly_sugars = AppendOnlyVec::new();
                let new_polys_ref = &new_polys;
                let new_poly_parents_ref = &new_poly_parents;
                let new_poly_sugars_ref = &new_poly_sugars;
                let basis_ref = &basis;
                let reducers_ref = &reducers;

                computation
                    .handle(controller.clone())
                    .join_many(spolys_to_reduce.as_fn().map_fn(move |spoly| {
                        let spoly_sugar = spoly.sugar();
                        move |handle: ShortCircuitingComputationHandle<(), _>| {
                            let parent_info: Vec<usize> = match spoly {
                                SPoly::Standard { i, j, .. } => vec![*i, *j],
                                SPoly::Nilpotent { idx, .. } => vec![*idx],
                            };
                            let mut f = spoly.poly(ring, basis_ref, order);

                            reduce_poly(
                                ring,
                                &mut f,
                                || reducers_ref.iter().chain(new_polys_ref.iter()).map(|(f, aug)| (f, aug)),
                                order,
                            );

                            if !ring.is_zero(&f) {
                                log_progress!(handle, "s");
                                _ = new_poly_parents_ref.push(parent_info);
                                // MISSING-1: sugar is inherited from the S-pair
                                _ = new_poly_sugars_ref.push(spoly_sugar);
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
                _ = computation.finish()?;

                // Notify observer of newly derived polynomials
                for (parents, (poly, _)) in new_poly_parents.iter().zip(new_polys.iter()) {
                    observer.on_new_poly(parents, poly);
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
                        return Ok(reducers.into_iter().map(|(f, _)| f).collect());
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
                    reducers = inter_reduce(ring, reducers, order);
                    sort_reducers(&mut reducers);
                    log_progress!(controller, "(r={})", reducers.len());
                    if abort_early_if(&reducers) {
                        log_progress!(controller, "(early_abort)");
                        return Ok(reducers.into_iter().map(|(f, _)| f).collect());
                    }
                }

                // less S-polys if we restart from scratch with reducers
                if open.len() + filtered_spolys
                    > reducers.len() * reducers.len() / 2 + reducers.len() * nilpotent_power.unwrap_or(0) + 1
                {
                    log_progress!(controller, "!");
                    return buchberger_observed::<P, O, _, _, _, Obs>(
                        ring,
                        reducers.into_iter().map(|(f, _)| f).collect(),
                        order,
                        sort_spolys,
                        abort_early_if,
                        controller,
                        observer,
                    );
                }
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

        // MISSING-2: Deactivate old basis elements whose LT is properly divisible
        // by new LT(g_k). We require *strict* divisibility (not just equality) for
        // safety: when LTs are equal, the new generator's tail may differ from the
        // old one's, and reducing one against the other would produce a useful new
        // polynomial. CoCoA performs that interreduction explicitly; here we keep
        // both equal-LT generators so the standard S-pair processing handles it.
        for i in 0..k {
            if !basis_active[i] { continue; }
            let gi_exp_tmp = ring.expand_monomial(ring.LT(&basis[i], order).unwrap().1);
            let gi_mask = divmask(&gi_exp_tmp);
            // Check if LT(g_k) properly divides LT(g_i)
            if (gk_mask & !gi_mask) != 0 { continue; }
            let k_divides_i = (0..n_vars).all(|v| gk_exp[v] <= gi_exp_tmp[v]);
            let is_proper = k_divides_i && (0..n_vars).any(|v| gk_exp[v] < gi_exp_tmp[v]);
            if is_proper {
                basis_active[i] = false;
            }
        }

        // MISSING-2: Remove pairs involving deactivated basis elements
        open.retain(|spoly| {
            match spoly {
                SPoly::Standard { i, j, .. } => basis_active[*i] && basis_active[*j],
                SPoly::Nilpotent { idx, .. } => basis_active[*idx],
            }
        });

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
        let mut new_pairs: Vec<(SPoly, Vec<usize>)> = Vec::new();
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

            // MISSING-1: Compute sugar degree for the pair
            let sugar_val = std::cmp::max(
                basis_sugar[i] + lcm_deg_val - gi_deg,
                basis_sugar[k] + lcm_deg_val - gk_deg,
            );

            let spoly = SPoly::Standard { i, j: k, sugar: sugar_val, lcm_deg: lcm_deg_val };
            if filter_spoly(ring, &spoly, basis, order).is_some() {
                *filtered_spolys += 1;
                continue;
            }
            new_pairs.push((spoly, lcm_exp));
        }

        // --- M criterion: remove dominated pairs ---
        // Among new pairs, if lcm(i,k) divides lcm(j,k) for some i != j,
        // then S(j,k) is redundant. Keep only minimal lcm pairs.
        if new_pairs.len() > 1 {
            let mut keep = vec![true; new_pairs.len()];
            for a in 0..new_pairs.len() {
                if !keep[a] { continue; }
                for b in 0..new_pairs.len() {
                    if a == b || !keep[b] { continue; }
                    // Check if lcm_a divides lcm_b (a dominates b → remove b)
                    let divides = (0..n_vars).all(|v| new_pairs[a].1[v] <= new_pairs[b].1[v]);
                    if divides && new_pairs[a].1 != new_pairs[b].1 {
                        keep[b] = false;
                        *filtered_spolys += 1;
                    }
                }
            }
            for (idx, (spoly, _)) in new_pairs.into_iter().enumerate() {
                if keep[idx] {
                    open.push(spoly);
                }
            }
        } else {
            for (spoly, _) in new_pairs {
                open.push(spoly);
            }
        }

        // Nilpotent S-polys (for local rings)
        if let Some(e) = nilpotent_power {
            for nk in 1..e {
                let lcm_deg_val = gk_deg; // LT doesn't change for nilpotent
                let sugar_val = basis_sugar[k]; // sugar inherited
                let spoly = SPoly::Nilpotent { idx: k, k: nk, sugar: sugar_val, lcm_deg: lcm_deg_val };
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
    // The accumulator fast path is only valid for DegRevLex, since the
    // internal linked-list is sorted by (deg, order) which matches DegRevLex.
    if order.is_same(&DegRevLex) {
        let used_fast_path = ring.get_ring().reduce_poly_loop(
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
        );
        if used_fast_path {
            return;
        }
    }

    // Fallback: generic loop
    while let Some((_, reducer, quo_c, quo_m)) = find_reducer(ring, to_reduce, reducers(), order) {
        ring.sub_assign_mul_monomial(to_reduce, reducer, &quo_c, &quo_m);
    }
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
