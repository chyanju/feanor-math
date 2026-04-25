//! Leaf cases of the Hilbert numerator recursion.
//!
//! Three leaves, all returning `RingElem` in the dense univariate
//! polynomial ring `Z[t]` (`P`):
//!
//! * [`sp_poincare`] (CoCoA `SPPoincare`): base case when there are **no**
//!   mixed generators left.  The numerator is just `∏_{i ∈ occ(sp)} (1 - t^{sp[i]})`.
//!   Special case: if every occurring exponent is `1` (i.e. each generator
//!   is just `x_i`), the answer is `(1-t)^{|occ|}`.
//!
//! * [`len_one_poincare`] (CoCoA `LenOnePoincare`): one mixed generator `m`
//!   plus the SP list `sp`.  Two sub-cases:
//!     * `coprime(m, sp)`: numerator is `(1 - t^{deg m}) * ∏ (1 - t^{sp[i]})`.
//!     * Else: numerator is `p1 - t^{deg m} * p2` where
//!       `p1 = ∏ (1 - t^{sp[i]})` and
//!       `p2 = ∏ (1 - t^{sp[i] - m[i]})` (over occurring `i` of `sp`).
//!
//! * [`one_term_and_sp_poincare`] (CoCoA `OneTermAndSPPoincare`):
//!   identical formula to `len_one_poincare` but takes `(term, sp)`
//!   directly instead of unpacking from a `TermList` (used by the splitter
//!   to avoid round-tripping through `TermList`).
//!
//! See `chat/plan-v2/R2_hilbert.md` §5 H3.

use crate::homomorphism::Homomorphism;
use crate::ring::*;
use crate::rings::poly::*;
use crate::rings::poly::dense_poly::DensePolyRing;
use crate::integer::BigIntRing;

use super::term_list::{EMonom, TermList};
use super::uni_helpers::mul_by_1_minus_xk;

/// Concrete `Z[t]` ring used by the Hilbert engine.  Aliased here so the
/// rest of the engine doesn't have to spell out the long type.
pub type ZT = DensePolyRing<BigIntRing>;

/// Returns `(1-t)^k`, i.e. the numerator for an SP list whose `k` occurring
/// vars all have exponent 1.  Implemented by repeated `(1 - t^1)` mult.
pub fn pow_one_minus_t(zt: &ZT, k: usize) -> El<ZT> {
    let mut p = zt.one();
    for _ in 0..k {
        mul_by_1_minus_xk(zt, &mut p, 1);
    }
    p
}

/// CoCoA `SPPoincare`: base case with no mixed generators.
pub fn sp_poincare(zt: &ZT, sp: &EMonom) -> El<ZT> {
    let occ = sp.occ();
    let k = occ.len();
    // Optimisation: if every occurring exponent is 1, this is just (1-t)^k.
    if k as u32 == sp.degree() {
        return pow_one_minus_t(zt, k);
    }
    let mut res = zt.one();
    for &i in occ.iter().rev() {
        let e = sp.exp(i as usize) as usize;
        mul_by_1_minus_xk(zt, &mut res, e);
    }
    res
}

/// CoCoA `OneTermAndSPPoincare` / `LenOnePoincare` (same formula): one
/// mixed term `term` plus the SP list `sp`.
pub fn one_term_and_sp_poincare(zt: &ZT, term: &EMonom, sp: &EMonom) -> El<ZT> {
    let occ = sp.occ();
    let t_deg = term.degree() as usize;

    if term.coprime(sp) {
        // Numerator = ∏(1 - t^{sp[i]}) * (1 - t^{deg term}).
        let mut p1 = if (occ.len() as u32) == sp.degree() {
            pow_one_minus_t(zt, occ.len())
        } else {
            let mut p = zt.one();
            for &i in occ.iter().rev() {
                let e = sp.exp(i as usize) as usize;
                mul_by_1_minus_xk(zt, &mut p, e);
            }
            p
        };
        mul_by_1_minus_xk(zt, &mut p1, t_deg);
        return p1;
    }
    // Non-coprime branch: p1 = ∏(1 - t^{sp[i]}), p2 = ∏(1 - t^{sp[i] - term[i]}),
    // result = p1 - t^{deg term} * p2.
    let mut p1 = zt.one();
    let mut p2 = zt.one();
    for &i in occ.iter().rev() {
        let i_us = i as usize;
        let e_sp = sp.exp(i_us) as usize;
        let e_t = term.exp(i_us) as usize;
        debug_assert!(e_sp >= e_t,
            "len_one_poincare: SP exponent {e_sp} must dominate term exponent {e_t} at slot {i_us}");
        mul_by_1_minus_xk(zt, &mut p1, e_sp);
        mul_by_1_minus_xk(zt, &mut p2, e_sp - e_t);
    }
    // p1 += (-1) * t^{t_deg} * p2.
    let neg_one = zt.base_ring().negate(zt.base_ring().one());
    let p2_terms: Vec<_> = zt
        .terms(&p2)
        .map(|(c, d)| (zt.base_ring().mul_ref(&neg_one, c), d + t_deg))
        .collect();
    zt.get_ring().add_assign_from_terms(&mut p1, p2_terms);
    p1
}

/// CoCoA `LenOnePoincare`: convenience wrapper that takes a [`TermList`]
/// with exactly one mixed generator.
pub fn len_one_poincare(zt: &ZT, tl: &TermList) -> El<ZT> {
    debug_assert_eq!(tl.mixed.len(), 1);
    one_term_and_sp_poincare(zt, &tl.mixed[0], &tl.sp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn em(n: usize, sp: &[(usize, u32)]) -> EMonom {
        EMonom::from_sparse(n, sp)
    }

    fn zt() -> ZT {
        DensePolyRing::new(BigIntRing::RING, "t")
    }

    /// Convert `El<ZT>` to a `Vec<i64>` (low-degree first) for easy assertions.
    fn coeffs(z: &ZT, p: &El<ZT>) -> Vec<i64> {
        let d = z.degree(p).unwrap_or(0);
        let mut out = Vec::with_capacity(d + 1);
        for i in 0..=d {
            let c = z.coefficient_at(p, i);
            // Convert via integer base ring.
            let s = format!("{}", z.base_ring().format(&c));
            out.push(s.parse::<i64>().expect("coeff out of i64"));
        }
        out
    }

    #[test]
    fn sp_poincare_all_units_is_one_minus_t_to_k() {
        let z = zt();
        // SP = (x_0 x_2 x_3) → (1-t)^3 = 1 - 3t + 3t^2 - t^3.
        let sp = em(5, &[(0, 1), (2, 1), (3, 1)]);
        let p = sp_poincare(&z, &sp);
        assert_eq!(coeffs(&z, &p), vec![1, -3, 3, -1]);
    }

    #[test]
    fn sp_poincare_general_case() {
        let z = zt();
        // SP = (x_0^2, x_1^3) → (1-t^2)(1-t^3) = 1 - t^2 - t^3 + t^5.
        let sp = em(2, &[(0, 2), (1, 3)]);
        let p = sp_poincare(&z, &sp);
        assert_eq!(coeffs(&z, &p), vec![1, 0, -1, -1, 0, 1]);
    }

    #[test]
    fn sp_poincare_empty() {
        let z = zt();
        let sp = em(3, &[]);
        let p = sp_poincare(&z, &sp);
        assert_eq!(coeffs(&z, &p), vec![1]);
    }

    #[test]
    fn one_term_coprime_with_sp() {
        let z = zt();
        // SP = x_0^2; mixed = x_1 x_2 (coprime).  Numerator =
        //   (1 - t^2)(1 - t^2) = 1 - 2t^2 + t^4.
        let sp = em(3, &[(0, 2)]);
        let term = em(3, &[(1, 1), (2, 1)]);
        let p = one_term_and_sp_poincare(&z, &term, &sp);
        assert_eq!(coeffs(&z, &p), vec![1, 0, -2, 0, 1]);
    }

    #[test]
    fn one_term_overlapping_sp() {
        let z = zt();
        // SP = x_0^3; mixed = x_0 x_1.  Both share x_0.  By formula:
        //   p1 = (1 - t^3)
        //   p2 = (1 - t^{3-1}) = (1 - t^2)
        //   result = p1 - t^{deg(x_0 x_1) = 2} * p2
        //          = (1 - t^3) - t^2 (1 - t^2)
        //          = 1 - t^2 - t^3 + t^4.
        let sp = em(2, &[(0, 3)]);
        let term = em(2, &[(0, 1), (1, 1)]);
        let p = one_term_and_sp_poincare(&z, &term, &sp);
        assert_eq!(coeffs(&z, &p), vec![1, 0, -1, -1, 1]);
    }

    #[test]
    fn one_term_overlapping_multiple_sp_vars() {
        let z = zt();
        // SP = x_0^2 x_1^2; mixed = x_0 x_1.  Formula:
        //   p1 = (1 - t^2)(1 - t^2) = 1 - 2t^2 + t^4
        //   p2 = (1 - t)(1 - t)     = 1 - 2t   + t^2
        //   result = p1 - t^2 * p2 = (1 - 2t^2 + t^4) - t^2 (1 - 2t + t^2)
        //          = 1 - 2t^2 + t^4 - t^2 + 2t^3 - t^4
        //          = 1 - 3t^2 + 2t^3.
        let sp = em(2, &[(0, 2), (1, 2)]);
        let term = em(2, &[(0, 1), (1, 1)]);
        let p = one_term_and_sp_poincare(&z, &term, &sp);
        assert_eq!(coeffs(&z, &p), vec![1, 0, -3, 2]);
    }
}
