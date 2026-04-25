//! Exponent-vector monomials (`EMonom`) and term lists (`TermList`) used by
//! the Hilbert numerator recursion.
//!
//! Port of CoCoA-EP's `eterms.{h,C}` and `TermList.{h,C}`, simplified for
//! Rust:
//!
//! * **No flexarrays / RUM slab allocator.**  CoCoA used a custom recyclable
//!   memory pool keyed on `eterm_size(n_vars)` to amortise the cost of many
//!   short-lived monomials in the recursion.  We rely on `Vec` + the system
//!   allocator; clones are explicit.
//! * **No `sqfr_bits` short-circuit.**  CoCoA cached a `u64` bitset of which
//!   indeterminates occur, used as a fast pre-check in `coprime` / `divides`.
//!   The optimisation silently degrades to "always TRUE" past 64 vars; for
//!   our target problems (50-200 vars) it is rarely useful.  We always do
//!   the linear scan over the cached `occ` index list, which is `O(occ)`
//!   anyway.
//! * **Cached `occ` and `deg` are kept**, since they are the genuinely useful
//!   summary of a sparse exponent vector and are used by every operation.
//!
//! ## Data model
//!
//! An [`EMonom`] over `n_vars` indeterminates is a triple
//!
//! ```text
//!   exps : Vec<u32>     // length n_vars; exps[i] = exponent of x_{i+1}
//!   deg  : u32          // sum of exps
//!   occ  : Vec<u32>     // sorted list of indices i (0-based) with exps[i] > 0
//! ```
//!
//! A [`TermList`] is the recursion's working state:
//!
//! ```text
//!   sp    : EMonom         // "simple-power list" — one packed monomial
//!                          // whose exponent at slot i records the exponent
//!                          // of the simple-power generator x_i^{exps[i]}.
//!                          // Generators with exps[i] = 0 are absent.
//!   mixed : Vec<EMonom>    // mixed (≥ 2 vars occurring) generators
//! ```
//!
//! Indices throughout are **0-based** in the Rust port (CoCoA used 1-based).
//!
//! See `chat/plan-v2/R2_hilbert.md` §5 H2.

use std::cmp::min;

/// Exponent-vector monomial over a polynomial ring with `n_vars`
/// indeterminates.
///
/// Indices are 0-based: `self.exp(i)` is the exponent of `x_{i+1}` (in the
/// usual mathematical 1-based naming) for `0 <= i < n_vars`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EMonom {
    exps: Vec<u32>,
    deg: u32,
    /// Sorted list of indices `i` with `exps[i] > 0`.
    occ: Vec<u32>,
}

impl EMonom {
    /// Allocate the constant-1 monomial: every exponent zero.
    pub fn one(n_vars: usize) -> Self {
        Self { exps: vec![0; n_vars], deg: 0, occ: Vec::new() }
    }

    /// Build from a dense exponent vector.  Computes `deg` and `occ`.
    pub fn from_exps(exps: Vec<u32>) -> Self {
        let mut deg: u32 = 0;
        let mut occ = Vec::new();
        for (i, &e) in exps.iter().enumerate() {
            if e > 0 {
                deg = deg.checked_add(e).expect("EMonom degree overflow");
                occ.push(i as u32);
            }
        }
        Self { exps, deg, occ }
    }

    /// Build from a sparse `(index, exponent)` list.  Indices must be `< n_vars`.
    pub fn from_sparse(n_vars: usize, terms: &[(usize, u32)]) -> Self {
        let mut m = Self::one(n_vars);
        for &(i, e) in terms {
            assert!(i < n_vars, "EMonom::from_sparse: index out of range");
            assert!(e > 0, "EMonom::from_sparse: zero exponent");
            assert_eq!(m.exps[i], 0, "EMonom::from_sparse: duplicate index");
            m.exps[i] = e;
            m.deg = m.deg.checked_add(e).expect("EMonom degree overflow");
            m.occ.push(i as u32);
        }
        m.occ.sort_unstable();
        m
    }

    pub fn n_vars(&self) -> usize { self.exps.len() }
    pub fn degree(&self) -> u32 { self.deg }
    pub fn exp(&self, i: usize) -> u32 { self.exps[i] }
    pub fn exps(&self) -> &[u32] { &self.exps }
    pub fn occ(&self) -> &[u32] { &self.occ }
    pub fn occ_count(&self) -> usize { self.occ.len() }

    /// Set `self.exps[i] := e` (any `e >= 0`), updating cached `deg` and `occ`.
    ///
    /// `O(occ_count)` due to maintaining sorted `occ`.
    pub fn set(&mut self, i: usize, e: u32) {
        let old = self.exps[i];
        if old == e { return; }
        self.exps[i] = e;
        self.deg = self.deg + e - old;
        if old == 0 {
            // newly occurring; insert in sorted position
            let pos = self.occ.binary_search(&(i as u32)).unwrap_err();
            self.occ.insert(pos, i as u32);
        } else if e == 0 {
            // no longer occurs; remove
            let pos = self.occ.binary_search(&(i as u32))
                .expect("occ invariant: old > 0 implies index present");
            self.occ.remove(pos);
        }
    }

    /// Return `true` iff `self` and `other` share no occurring index.
    /// `O(occ(self) + occ(other))`.
    pub fn coprime(&self, other: &EMonom) -> bool {
        let (mut i, mut j) = (0usize, 0usize);
        while i < self.occ.len() && j < other.occ.len() {
            let a = self.occ[i];
            let b = other.occ[j];
            if a == b { return false; }
            if a < b { i += 1; } else { j += 1; }
        }
        true
    }

    /// Return `true` iff `self | other`, i.e. `self.exps[i] <= other.exps[i]`
    /// for every `i`.  `O(occ(self))`.
    pub fn divides(&self, other: &EMonom) -> bool {
        debug_assert_eq!(self.n_vars(), other.n_vars());
        for &i in &self.occ {
            if self.exps[i as usize] > other.exps[i as usize] { return false; }
        }
        true
    }

    /// In-place product: `self *= other`.  `O(n_vars + occ(other))` worst case.
    pub fn mul_assign(&mut self, other: &EMonom) {
        debug_assert_eq!(self.n_vars(), other.n_vars());
        for &i in &other.occ {
            let i = i as usize;
            let new = self.exps[i].checked_add(other.exps[i])
                .expect("EMonom mul overflow");
            self.set(i, new);
        }
    }

    /// In-place "monoid colon": `self := self / gcd(self, other)`.
    /// Equivalent to `self[i] := max(0, self[i] - other[i])`.
    /// Used by CoCoA `eterm_colon`.
    pub fn colon_assign(&mut self, other: &EMonom) {
        debug_assert_eq!(self.n_vars(), other.n_vars());
        // We must walk a copy of `self.occ` because `set` mutates it.
        let occ_snapshot: Vec<u32> = self.occ.clone();
        for &i in &occ_snapshot {
            let i = i as usize;
            let s = self.exps[i];
            let o = other.exps[i];
            let new = s.saturating_sub(o);
            if new != s {
                self.set(i, new);
            }
        }
    }

    /// Return `gcd(self, other)`.
    pub fn gcd(&self, other: &EMonom) -> EMonom {
        debug_assert_eq!(self.n_vars(), other.n_vars());
        let n = self.n_vars();
        let mut exps = vec![0u32; n];
        let mut deg = 0u32;
        let mut occ = Vec::new();
        // Walk the shorter occ list; gcd support is a subset of either.
        let (a, b) = if self.occ.len() <= other.occ.len() { (self, other) } else { (other, self) };
        for &i in &a.occ {
            let i = i as usize;
            let m = min(a.exps[i], b.exps[i]);
            if m > 0 {
                exps[i] = m;
                deg += m;
                occ.push(i as u32);
            }
        }
        EMonom { exps, deg, occ }
    }

    /// In-place union (lcm) on **simple-power lists**.  Required to support
    /// `SplitIndets`' merging of disjoint variable groups.  Equivalent to
    /// `self[i] := max(self[i], other[i])` for each `i`.
    pub fn union_assign(&mut self, other: &EMonom) {
        debug_assert_eq!(self.n_vars(), other.n_vars());
        for &i in other.occ.clone().iter() {
            let i = i as usize;
            let s = self.exps[i];
            let o = other.exps[i];
            if o > s { self.set(i, o); }
        }
    }
}

/// The recursion's working state: a "simple-power list" packed into a single
/// monomial, plus a vector of mixed (≥ 2 occurring vars) generators.
#[derive(Clone, Debug)]
pub struct TermList {
    pub sp: EMonom,
    pub mixed: Vec<EMonom>,
}

impl TermList {
    /// Empty term list (identity ideal): no SP generators, no mixed terms.
    pub fn empty(n_vars: usize) -> Self {
        Self { sp: EMonom::one(n_vars), mixed: Vec::new() }
    }

    /// Build a term list from an explicit list of monomial generators.
    /// Generators with exactly one occurring index are absorbed into `sp`
    /// (taking the **smaller** exponent if a simple power for that var
    /// already exists, since `x^a | x^b` means `(x^a)` is the stronger
    /// generator); all others go into `mixed`.
    ///
    /// **No interreduction** is performed — call [`crate::algorithms::hilbert::reduce::interreduce`]
    /// on the result if needed.
    pub fn from_generators<I>(n_vars: usize, gens: I) -> Self
    where
        I: IntoIterator<Item = EMonom>,
    {
        let mut tl = Self::empty(n_vars);
        for g in gens {
            assert_eq!(g.n_vars(), n_vars, "TermList::from_generators: arity mismatch");
            tl.insert(g);
        }
        tl
    }

    /// Insert a single generator.  See [`Self::from_generators`] for SP-vs-mixed
    /// dispatch rules.
    pub fn insert(&mut self, g: EMonom) {
        if g.occ_count() == 1 {
            let i = g.occ()[0] as usize;
            let new_e = g.exp(i);
            let cur = self.sp.exp(i);
            if cur == 0 || new_e < cur {
                self.sp.set(i, new_e);
            }
        } else if g.occ_count() >= 2 {
            self.mixed.push(g);
        }
        // occ_count == 0 ⇒ generator is the constant 1, ideal is the unit
        // ideal; we drop it silently here. Callers that care should detect
        // an `EMonom::one()` themselves before inserting.
    }

    pub fn n_vars(&self) -> usize { self.sp.n_vars() }
    pub fn mixed_len(&self) -> usize { self.mixed.len() }
}

// ----------------------------------------------------------------------------
// MoveNotCoprime / MoveNotCoprimeSP
// ----------------------------------------------------------------------------

/// Move from `from.sp` to `to.sp` every simple-power generator whose
/// underlying variable also appears in `pivot`.  After the call, `from.sp`
/// and `to.sp` remain disjoint on the moved indices.
///
/// CoCoA: `MoveNotCoprimeSP`.
pub fn move_not_coprime_sp(from_sp: &mut EMonom, to_sp: &mut EMonom, pivot: &EMonom) {
    debug_assert_eq!(from_sp.n_vars(), to_sp.n_vars());
    debug_assert_eq!(from_sp.n_vars(), pivot.n_vars());
    let occ_snapshot: Vec<u32> = from_sp.occ().to_vec();
    for &i in &occ_snapshot {
        let i = i as usize;
        if pivot.exp(i) != 0 {
            let e = from_sp.exp(i);
            to_sp.set(i, e);
            from_sp.set(i, 0);
        }
    }
}

/// Move from `from` to `to` every mixed generator that is **not coprime**
/// with `pivot`, then do the same for the simple-power list.
///
/// CoCoA: `MoveNotCoprime`.  Order is **not preserved** on `from.mixed`
/// (uses swap-remove, like CoCoA's `MTLMoveLastToNth`).
pub fn move_not_coprime(from: &mut TermList, to: &mut TermList, pivot: &EMonom) {
    debug_assert_eq!(from.n_vars(), to.n_vars());
    debug_assert_eq!(from.n_vars(), pivot.n_vars());
    let mut i = from.mixed.len();
    while i > 0 {
        i -= 1;
        if !from.mixed[i].coprime(pivot) {
            let m = from.mixed.swap_remove(i);
            to.mixed.push(m);
        }
    }
    move_not_coprime_sp(&mut from.sp, &mut to.sp, pivot);
}

// ----------------------------------------------------------------------------
// SplitIndets
// ----------------------------------------------------------------------------

/// If the **mixed** generators of `tl` partition into ≥ 2 groups by
/// shared occurring-index connectivity (treat each generator as a hyperedge
/// over the indet set), return one mixed-only [`TermList`] per connected
/// component.  Otherwise return `None`.
///
/// The returned list contains one `EMonom` per group, where that monom is
/// the **union** (lcm of supports, with maximum exponents) of all the
/// generators in that group.  It is used by the recursion as a "splitter"
/// to dispatch each group to its own sub-recursion via
/// [`move_not_coprime`].
///
/// CoCoA: `SplitIndets` (returns `NULL` if everything is in one component
/// or only one mixed generator exists).
pub fn split_indets(tl: &TermList) -> Option<Vec<EMonom>> {
    let n_vars = tl.n_vars();
    let mut groups: Vec<EMonom> = Vec::new();

    // For each mixed generator (in CoCoA: in reverse order, but irrelevant
    // for connectivity), attempt to merge with any existing group it shares
    // an indet with; merge all such groups into one.
    for tn in tl.mixed.iter() {
        // Collect indices of groups that intersect `tn`.
        let hits: Vec<usize> = (0..groups.len())
            .filter(|&j| !groups[j].coprime(tn))
            .collect();
        if hits.is_empty() {
            groups.push(tn.clone());
        } else {
            // Merge the first hit with `tn`, then absorb the rest into it.
            let primary = hits[0];
            groups[primary].union_assign(tn);
            // Pop the others (descending order so indices stay valid).
            for &k in hits[1..].iter().rev() {
                let g = groups.swap_remove(k);
                groups[primary].union_assign(&g);
            }
            // After swap_remove, `primary` may have been moved if it equalled
            // `groups.len() - 1`.  But hits[0] < hits[k] for k > 0, and we
            // never swap_remove `primary` itself, so primary's index stays
            // valid as long as the swap_remove didn't move primary's slot.
            // swap_remove(k) moves the LAST element to slot k; since primary < k
            // (hits is sorted ascending), primary is unaffected.  ✓
            let _ = hits;
        }
    }

    if groups.len() <= 1 {
        None
    } else {
        // Sanity: every emitted group monom must have n_vars indets.
        debug_assert!(groups.iter().all(|g| g.n_vars() == n_vars));
        Some(groups)
    }
}

// ----------------------------------------------------------------------------
// Pivots: BigPivotOf, GCD3PivotOf
// ----------------------------------------------------------------------------

/// Most-frequent indeterminate among mixed generators.  Returns `None` if
/// every variable occurs at most once (i.e. there are essentially no mixed
/// terms left to recurse on), else `Some(idx)` (0-based).
///
/// In case of ties, CoCoA returns the **median** of the tied indices.  We
/// preserve that behaviour to keep recursion shapes byte-identical with the
/// reference.
fn most_frequent_indet(tl: &TermList) -> Option<usize> {
    let n_vars = tl.n_vars();
    let mut count = vec![0u32; n_vars];
    for m in tl.mixed.iter() {
        for &i in m.occ() {
            count[i as usize] += 1;
        }
    }
    let max = *count.iter().max().unwrap_or(&0);
    if max <= 1 { return None; }
    // Collect tied indices in **CoCoA order**: CoCoA scans j = n_vars-1 down
    // to 0 (1-based j = IndNo .. 1) and pushes ties in that order, then
    // returns `MFI_Indets[MFIndNo / 2]` (1-based middle).  Reproduce that
    // exactly.
    let mut ties: Vec<usize> = Vec::new();
    for j in (0..n_vars).rev() {
        if count[j] == max { ties.push(j); }
    }
    if ties.len() == 1 {
        Some(ties[0])
    } else {
        // CoCoA: MFI_Indets[MFIndNo/2] with 1-based indexing means the
        // (MFIndNo/2)-th element, which in 0-based is index (MFIndNo/2 - 1).
        let mid = ties.len() / 2;
        Some(ties[mid - 1])
    }
}

/// BigPivot heuristic (CoCoA `BigPivotOf`): pick the most-frequent indet,
/// then take `min` of the **first** and **last** non-zero exponents seen
/// among mixed generators.  Returns `None` if no var has multiplicity > 1.
pub fn big_pivot_of(tl: &TermList) -> Option<(usize, u32)> {
    let idx = most_frequent_indet(tl)?;
    // First non-zero from front, last non-zero from back.
    let mut e_first: Option<u32> = None;
    for m in tl.mixed.iter() {
        let e = m.exp(idx);
        if e != 0 { e_first = Some(e); break; }
    }
    let mut e_last: Option<u32> = None;
    for m in tl.mixed.iter().rev() {
        let e = m.exp(idx);
        if e != 0 { e_last = Some(e); break; }
    }
    let e1 = e_first.expect("BigPivotOf: most_frequent_indet returned an absent var");
    let e2 = e_last.expect("BigPivotOf: most_frequent_indet returned an absent var");
    Some((idx, min(e1, e2)))
}

/// GCD3Pivot heuristic (CoCoA `GCD3PivotOf`): take the `gcd` of three
/// generators that contain the most-frequent indet.  Returns `None` if no
/// var has multiplicity > 1.
///
/// CoCoA randomises the pick via `Random(len)` which is `#define Random(len)
/// 1` (always picks first); we mirror that determinism.
pub fn gcd3_pivot_of(tl: &TermList) -> Option<EMonom> {
    let idx = most_frequent_indet(tl)?;
    // Collect generators containing `idx`.
    let mut with: Vec<&EMonom> = tl.mixed.iter().filter(|m| m.exp(idx) != 0).collect();
    if with.is_empty() { return None; }
    // CoCoA `Random(len)` ≡ 1 (1-based), i.e. take element at 1-based index 1
    // = element at 0-based index 0; then `MTLMoveLastToNth` swaps element 0
    // with the last and decrements length.  Reproduce that exact pattern.
    fn pop_first_via_swap(v: &mut Vec<&EMonom>) -> EMonom {
        let chosen = v[0].clone();
        let last = v.len() - 1;
        v.swap(0, last);
        v.pop();
        chosen
    }
    let t1 = pop_first_via_swap(&mut with);
    if with.is_empty() { return Some(t1); }
    let t2 = pop_first_via_swap(&mut with);
    let g12 = t1.gcd(&t2);
    if with.is_empty() {
        Some(g12)
    } else {
        // Third pick: again first-via-swap (CoCoA: `MTL[Random(MTLLen)]`).
        let t3 = with[0];
        Some(g12.gcd(t3))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn em(n: usize, sp: &[(usize, u32)]) -> EMonom {
        EMonom::from_sparse(n, sp)
    }

    #[test]
    fn emonom_basic_ops() {
        let m = em(5, &[(0, 2), (3, 1)]);
        assert_eq!(m.n_vars(), 5);
        assert_eq!(m.degree(), 3);
        assert_eq!(m.exps(), &[2, 0, 0, 1, 0]);
        assert_eq!(m.occ(), &[0u32, 3]);
    }

    #[test]
    fn emonom_set_updates_occ_and_deg() {
        let mut m = EMonom::one(4);
        m.set(2, 5);
        assert_eq!(m.degree(), 5);
        assert_eq!(m.occ(), &[2u32]);
        m.set(0, 1);
        assert_eq!(m.occ(), &[0u32, 2]);
        assert_eq!(m.degree(), 6);
        m.set(2, 0);
        assert_eq!(m.occ(), &[0u32]);
        assert_eq!(m.degree(), 1);
        m.set(0, 7); // overwrite, not new
        assert_eq!(m.degree(), 7);
        assert_eq!(m.occ(), &[0u32]);
    }

    #[test]
    fn emonom_coprime_and_divides() {
        let a = em(6, &[(0, 1), (2, 3)]);
        let b = em(6, &[(1, 1), (4, 1)]);
        let c = em(6, &[(2, 5), (5, 1)]);
        assert!(a.coprime(&b));
        assert!(!a.coprime(&c));
        assert!(b.coprime(&c) == false || true); // b∩c = {} so coprime
        assert!(b.coprime(&c));
        // x_0 * x_2^3 divides x_0 * x_2^5
        let d = em(6, &[(0, 1), (2, 5)]);
        assert!(a.divides(&d));
        // but x_0 * x_2^3 does NOT divide x_0 * x_2^2
        let e = em(6, &[(0, 1), (2, 2)]);
        assert!(!a.divides(&e));
    }

    #[test]
    fn emonom_mul_gcd_colon_union() {
        let a = em(5, &[(0, 2), (2, 1)]);
        let b = em(5, &[(0, 1), (3, 4)]);
        let mut p = a.clone();
        p.mul_assign(&b);
        assert_eq!(p, em(5, &[(0, 3), (2, 1), (3, 4)]));
        let g = a.gcd(&b);
        assert_eq!(g, em(5, &[(0, 1)]));
        let mut q = a.clone();
        q.colon_assign(&b);
        assert_eq!(q, em(5, &[(0, 1), (2, 1)]));
        // union = max, not sum (used for SP merging)
        let mut u = a.clone();
        u.union_assign(&b);
        assert_eq!(u, em(5, &[(0, 2), (2, 1), (3, 4)]));
    }

    #[test]
    fn termlist_dispatches_simple_powers_into_sp() {
        // I = (x_0^2, x_1^3, x_0 x_2)
        let n = 4;
        let tl = TermList::from_generators(
            n,
            vec![
                em(n, &[(0, 2)]),
                em(n, &[(1, 3)]),
                em(n, &[(0, 1), (2, 1)]),
            ],
        );
        assert_eq!(tl.sp.exps(), &[2, 3, 0, 0]);
        assert_eq!(tl.mixed_len(), 1);
        assert_eq!(tl.mixed[0], em(n, &[(0, 1), (2, 1)]));
    }

    #[test]
    fn termlist_sp_merge_keeps_min_exponent() {
        let n = 3;
        let tl = TermList::from_generators(
            n,
            vec![em(n, &[(0, 5)]), em(n, &[(0, 2)]), em(n, &[(0, 7)])],
        );
        // Strongest generator x_0^2 wins.
        assert_eq!(tl.sp.exp(0), 2);
        assert_eq!(tl.mixed_len(), 0);
    }

    #[test]
    fn move_not_coprime_basic() {
        // from has SP (x_0^3) (x_2^4), mixed { x_0 x_1, x_2 x_3, x_4 x_5 }.
        // pivot uses x_0 and x_5.  We expect:
        //   moved SP:    x_0^3
        //   moved mixed: x_0 x_1, x_4 x_5
        //   remaining SP:    x_2^4
        //   remaining mixed: x_2 x_3
        let n = 6;
        let mut from = TermList::from_generators(
            n,
            vec![
                em(n, &[(0, 3)]),
                em(n, &[(2, 4)]),
                em(n, &[(0, 1), (1, 1)]),
                em(n, &[(2, 1), (3, 1)]),
                em(n, &[(4, 1), (5, 1)]),
            ],
        );
        let mut to = TermList::empty(n);
        let pivot = em(n, &[(0, 1), (5, 1)]);
        move_not_coprime(&mut from, &mut to, &pivot);
        assert_eq!(to.sp.exp(0), 3);
        assert_eq!(to.sp.exp(2), 0);
        assert_eq!(from.sp.exp(0), 0);
        assert_eq!(from.sp.exp(2), 4);
        assert_eq!(to.mixed_len(), 2);
        assert_eq!(from.mixed_len(), 1);
        assert_eq!(from.mixed[0], em(n, &[(2, 1), (3, 1)]));
    }

    #[test]
    fn split_indets_two_components() {
        // I = (x_0 x_1, x_1 x_2, x_3 x_4)  →  components {0,1,2} and {3,4}.
        let n = 5;
        let tl = TermList::from_generators(
            n,
            vec![
                em(n, &[(0, 1), (1, 1)]),
                em(n, &[(1, 1), (2, 1)]),
                em(n, &[(3, 1), (4, 1)]),
            ],
        );
        let groups = split_indets(&tl).expect("expected two components");
        assert_eq!(groups.len(), 2);
        // One group covers indices {0,1,2}, the other {3,4}.
        let a: Vec<u32> = groups[0].occ().to_vec();
        let b: Vec<u32> = groups[1].occ().to_vec();
        let (small, large) = if a.len() < b.len() { (a, b) } else { (b, a) };
        assert_eq!(small, vec![3, 4]);
        assert_eq!(large, vec![0, 1, 2]);
    }

    #[test]
    fn split_indets_one_component_returns_none() {
        let n = 4;
        let tl = TermList::from_generators(
            n,
            vec![em(n, &[(0, 1), (1, 1)]), em(n, &[(1, 1), (2, 1)]), em(n, &[(2, 1), (3, 1)])],
        );
        assert!(split_indets(&tl).is_none());
    }

    #[test]
    fn most_frequent_indet_returns_none_when_all_unique() {
        let n = 4;
        let tl = TermList::from_generators(n, vec![em(n, &[(0, 1), (1, 1)])]);
        assert!(most_frequent_indet(&tl).is_none()); // each var appears exactly once
    }

    #[test]
    fn big_pivot_of_picks_most_frequent_min_exp() {
        // x_0 appears in two terms with exps 3 and 5; x_1 appears in one.
        // Pivot: (idx=0, deg=min(3, 5)=3).
        let n = 3;
        let tl = TermList::from_generators(
            n,
            vec![
                em(n, &[(0, 3), (1, 1)]),
                em(n, &[(0, 5), (2, 1)]),
            ],
        );
        let p = big_pivot_of(&tl).expect("expected a pivot");
        assert_eq!(p, (0, 3));
    }

    #[test]
    fn gcd3_pivot_of_takes_gcd_of_first_three() {
        // All three contain x_0; the gcd is x_0 (exponent 1, since min(2,1,3)=1)
        // and x_2^1 (since min on x_2 over those that have it = 0 → drops out).
        let n = 4;
        let tl = TermList::from_generators(
            n,
            vec![
                em(n, &[(0, 2), (1, 1)]),
                em(n, &[(0, 1), (2, 3)]),
                em(n, &[(0, 3), (3, 1)]),
            ],
        );
        let p = gcd3_pivot_of(&tl).expect("expected a gcd pivot");
        // gcd(x_0^2 x_1, x_0 x_2^3) = x_0; gcd(x_0, x_0^3 x_3) = x_0.
        assert_eq!(p, em(n, &[(0, 1)]));
    }
}
