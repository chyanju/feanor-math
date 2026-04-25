//! Univariate `Z[t]` helpers used by the Hilbert numerator engine.
//!
//! Two performance-sensitive primitives, expressed against the generic
//! [`PolyRing`] surface so they work for any dense (or sparse) univariate
//! polynomial ring over an arbitrary base ring:
//!
//! - [`mul_by_1_minus_xk`] computes `p := p * (1 - t^k)` in-place.
//! - [`add_mul_xk`] computes `p := p + c * t^k * q`.
//!
//! These mirror CoCoA's `myMulBy1MinusXExp` and `myAddMulLM`
//! (see `TmpHilbertDir/unipoly.{h,C}`).
//!
//! See `chat/plan-v2/R2_hilbert.md` §3c, §5 H1.

use crate::ring::*;
use crate::rings::poly::*;
use crate::homomorphism::Homomorphism;

/// Computes `p := p * (1 - t^k)` in place.
///
/// Equivalent (in math) to subtracting a `k`-shifted copy of `p` from `p`:
///
/// ```text
///   coeff[d]      ← coeff[d] - coeff[d - k]      for d >= k (descending)
///   coeff[d]      unchanged                       for d <  k
/// ```
///
/// Implementation is straightforward: `shifted = p << k`, then `p -= shifted`.
/// This is `O(deg(p))` ring ops (1 clone + 1 monomial-shift + 1 subtraction).
///
/// Special cases: if `k == 0` then `1 - t^0 = 0`, so the result is `0`.
pub fn mul_by_1_minus_xk<P>(ring: P, p: &mut El<P>, k: usize)
where
    P: RingStore + Copy,
    P::Type: PolyRing,
{
    if k == 0 {
        // (1 - 1) = 0
        *p = ring.zero();
        return;
    }
    let mut shifted = ring.clone_el(p);
    ring.get_ring().mul_assign_monomial(&mut shifted, k);
    // p -= shifted
    ring.sub_assign(p, shifted);
}

/// Computes `p := p + c * t^k * q`.
///
/// The base-ring scalar `c` is multiplied into every coefficient of `q`,
/// the result is shifted up by `k` degrees, and added into `p`.
///
/// Implementation walks `q.terms()` and feeds the shifted/scaled terms into
/// `add_assign_from_terms` on `p`.
pub fn add_mul_xk<P>(
    ring: P,
    p: &mut El<P>,
    c: &El<<P::Type as RingExtension>::BaseRing>,
    k: usize,
    q: &El<P>,
) where
    P: RingStore + Copy,
    P::Type: PolyRing,
{
    let base = ring.base_ring();
    let new_terms: Vec<(El<<P::Type as RingExtension>::BaseRing>, usize)> = ring
        .terms(q)
        .map(|(coef, deg)| (base.mul_ref(c, coef), deg + k))
        .collect();
    ring.get_ring().add_assign_from_terms(p, new_terms.into_iter());
}

/// Synthetic division by `(1 - t)`.
///
/// Given `p`, computes `q` and `rem ∈ R` such that `p = (1 - t) * q + rem`,
/// returns `Some(rem)` and overwrites `p` with `q` if the remainder is zero;
/// returns `None` (and leaves `p` unchanged) otherwise.
///
/// Note: `p(1) = (1 - 1) * q(1) + rem = rem`, so the remainder is always
/// the value `p(1)` regardless of whether the division is exact.
///
/// Algebra (in `R[t]`): write `p = Σ p_i t^i`, `q = Σ q_i t^i`. From
/// `(1 - t)·(q_0 + q_1 t + … + q_{d-1} t^{d-1})
///   = q_0 + (q_1 - q_0) t + … + (q_{d-1} - q_{d-2}) t^{d-1} - q_{d-1} t^d`
/// we read off
///   * `q_{d-1} = -p_d`
///   * `q_{i-1} = q_i - p_i`        for `0 < i < d`
///   * `rem    = p_0 - q_0`         (== `p(1)`).
///
/// Used by `HSSimplified` to cancel `(1 - t)` factors against the numerator
/// (revealing Krull dimension and multiplicity).  Each step is `O(1)` ring
/// ops, total `O(deg(p))`.
///
/// Returns `None` and leaves `p` unchanged if `(1 - t)` does **not** divide
/// `p` (i.e. `p(1) != 0`).
pub fn try_div_by_1_minus_t<P>(ring: P, p: &mut El<P>) -> Option<El<<P::Type as RingExtension>::BaseRing>>
where
    P: RingStore + Copy,
    P::Type: PolyRing,
{
    let base = ring.base_ring();
    let deg = match ring.degree(p) {
        Some(d) => d,
        None => {
            // p == 0 ⇒ q = 0, rem = 0.
            return Some(base.zero());
        }
    };
    // Snapshot coefficients of p (we may need to restore on failure, and the
    // ring API does not give us mutable slot access).
    let p_coefs: Vec<El<<P::Type as RingExtension>::BaseRing>> = (0..=deg)
        .map(|i| base.clone_el(ring.coefficient_at(p, i)))
        .collect();
    if deg == 0 {
        // p = p_0 ⇒ q = 0, rem = p_0.
        if !base.is_zero(&p_coefs[0]) {
            return None;
        }
        // Already zero; leave p as is and report exact division with rem=0.
        return Some(base.zero());
    }
    // Quotient buffer of length deg (coefficients of t^0..t^{deg-1}).
    // Start from the top: q_{d-1} = -p_d, then q_{i-1} = q_i - p_i descending.
    let mut quot: Vec<El<<P::Type as RingExtension>::BaseRing>> =
        (0..deg).map(|_| base.zero()).collect();
    quot[deg - 1] = base.negate(base.clone_el(&p_coefs[deg]));
    for i in (1..deg).rev() {
        // q_{i-1} = q_i - p_i
        let mut tmp = base.clone_el(&quot[i]);
        base.sub_assign_ref(&mut tmp, &p_coefs[i]);
        quot[i - 1] = tmp;
    }
    // Remainder = p_0 - q_0.
    let mut remainder = base.clone_el(&p_coefs[0]);
    base.sub_assign_ref(&mut remainder, &quot[0]);
    if !base.is_zero(&remainder) {
        return None;
    }
    // Build new polynomial from quot via from_terms (only nonzero coefs).
    let nonzero_terms: Vec<(El<<P::Type as RingExtension>::BaseRing>, usize)> = quot
        .into_iter()
        .enumerate()
        .filter_map(|(i, c)| {
            if base.is_zero(&c) {
                None
            } else {
                Some((c, i))
            }
        })
        .collect();
    *p = RingRef::new(ring.get_ring()).from_terms(nonzero_terms.into_iter());
    Some(base.zero())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integer::BigIntRing;
    use crate::rings::poly::dense_poly::DensePolyRing;

    #[test]
    fn test_mul_by_1_minus_xk_naive_agreement() {
        let zz = BigIntRing::RING;
        let pr = DensePolyRing::new(zz, "t");
        // p = 1 + 2t + 3t^2
        let p_orig = pr.from_terms([(zz.int_hom().map(1), 0), (zz.int_hom().map(2), 1), (zz.int_hom().map(3), 2)].into_iter());
        for k in 1usize..=4 {
            let mut p = pr.clone_el(&p_orig);
            mul_by_1_minus_xk(&pr, &mut p, k);
            // expected: p_orig - shifted_orig
            let mut shifted = pr.clone_el(&p_orig);
            pr.get_ring().mul_assign_monomial(&mut shifted, k);
            let mut expected = pr.clone_el(&p_orig);
            pr.sub_assign(&mut expected, shifted);
            assert!(pr.eq_el(&p, &expected), "k={}", k);
        }
    }

    #[test]
    fn test_mul_by_1_minus_xk_zero() {
        let zz = BigIntRing::RING;
        let pr = DensePolyRing::new(zz, "t");
        let mut p = pr.from_terms([(zz.int_hom().map(7), 0), (zz.int_hom().map(-3), 5)].into_iter());
        mul_by_1_minus_xk(&pr, &mut p, 0);
        assert!(pr.is_zero(&p));
    }

    #[test]
    fn test_mul_by_1_minus_xk_chain_regular_sequence() {
        // (1 - t^2)^3 should match `prod` of three `(1 - t^2)` factors.
        let zz = BigIntRing::RING;
        let pr = DensePolyRing::new(zz, "t");
        let mut acc = pr.from_terms([(zz.int_hom().map(1), 0)].into_iter()); // 1
        for _ in 0..3 {
            mul_by_1_minus_xk(&pr, &mut acc, 2);
        }
        // Expected: (1 - t^2)^3 = 1 - 3t^2 + 3t^4 - t^6
        let expected = pr.from_terms(
            [
                (zz.int_hom().map(1), 0),
                (zz.int_hom().map(-3), 2),
                (zz.int_hom().map(3), 4),
                (zz.int_hom().map(-1), 6),
            ]
            .into_iter(),
        );
        assert!(pr.eq_el(&acc, &expected));
    }

    #[test]
    fn test_add_mul_xk() {
        let zz = BigIntRing::RING;
        let pr = DensePolyRing::new(zz, "t");
        // p = 1, q = 1 + t, c = 5, k = 2 ⇒ p_new = 1 + 5t^2 + 5t^3
        let mut p = pr.from_terms([(zz.int_hom().map(1), 0)].into_iter());
        let q = pr.from_terms([(zz.int_hom().map(1), 0), (zz.int_hom().map(1), 1)].into_iter());
        let c = zz.int_hom().map(5);
        add_mul_xk(&pr, &mut p, &c, 2, &q);
        let expected = pr.from_terms(
            [
                (zz.int_hom().map(1), 0),
                (zz.int_hom().map(5), 2),
                (zz.int_hom().map(5), 3),
            ]
            .into_iter(),
        );
        assert!(pr.eq_el(&p, &expected));
    }

    #[test]
    fn test_try_div_by_1_minus_t_exact() {
        // (1-t) * (1 + t + t^2) = 1 - t^3.  Divide back.
        let zz = BigIntRing::RING;
        let pr = DensePolyRing::new(zz, "t");
        let mut p = pr.from_terms([(zz.int_hom().map(1), 0), (zz.int_hom().map(-1), 3)].into_iter());
        let r = try_div_by_1_minus_t(&pr, &mut p);
        assert!(r.is_some());
        let expected = pr.from_terms(
            [
                (zz.int_hom().map(1), 0),
                (zz.int_hom().map(1), 1),
                (zz.int_hom().map(1), 2),
            ]
            .into_iter(),
        );
        assert!(pr.eq_el(&p, &expected));
    }

    #[test]
    fn test_try_div_by_1_minus_t_inexact() {
        // 1 + t  is not divisible by (1 - t)  (remainder = 2).
        let zz = BigIntRing::RING;
        let pr = DensePolyRing::new(zz, "t");
        let mut p = pr.from_terms([(zz.int_hom().map(1), 0), (zz.int_hom().map(1), 1)].into_iter());
        let snapshot = pr.clone_el(&p);
        let r = try_div_by_1_minus_t(&pr, &mut p);
        assert!(r.is_none());
        // Per docstring, p must be unchanged on failure.
        assert!(pr.eq_el(&p, &snapshot));
    }
}
