//! Term-list interreduction and pivot reduction operations for the Hilbert
//! recursion.
//!
//! Port of CoCoA-EP's `InterreduceTList`, `ReduceAndDivideBySimplePower`,
//! `ReduceAndDivideByMixedTerm`, and `ReduceAndDivideByPivot` from
//! `TermList.C`.
//!
//! The Rust port is *semantically* faithful to the reference but is
//! structurally simpler:
//!
//! * No flexarray bucket allocation; we use `Vec<Vec<EMonom>>` indexed by
//!   exponent at the pivot variable.
//! * We always do the bucket dedup in two passes (against the new SPList,
//!   then against the bucket at exponent `PExp`); same big-O as CoCoA, no
//!   special-casing for "small" pivot exponents.
//! * Mutations on the input `TermList` mirror CoCoA: the input becomes the
//!   "kept" half (terms removed because they are divided by the new pivot
//!   power), and we also return the divided list.
//!
//! See `chat/plan-v2/R2_hilbert.md` §5 H2.5.

use super::term_list::{EMonom, TermList};

// ----------------------------------------------------------------------------
// InterreduceTList helpers
// ----------------------------------------------------------------------------

/// Return `true` iff some simple power in `sp` (i.e. some `x_i^{sp.exp(i)}`
/// with `sp.exp(i) > 0`) divides `t`.  CoCoA: `TListSimplePowersDivide`.
fn sp_divides_term(sp: &EMonom, t: &EMonom) -> bool {
    debug_assert_eq!(sp.n_vars(), t.n_vars());
    for &i in sp.occ() {
        let i = i as usize;
        let s = sp.exp(i);
        if s != 0 && s <= t.exp(i) { return true; }
    }
    false
}

/// Return `true` iff some entry of `mixed` divides `t`.
fn mixed_divides_term(mixed: &[EMonom], t: &EMonom) -> bool {
    mixed.iter().any(|m| m.divides(t))
}

/// CoCoA `sp_BigMult(t, pivot)`: assuming `pivot | t`, return `true` iff for
/// every occurring index `i` of `pivot`, `t.exp(i) > pivot.exp(i)` (i.e. the
/// quotient `t / pivot` still contains every variable that `pivot` does).
/// Equivalently: the quotient retains the full support of `pivot`.
fn sp_big_mult(t: &EMonom, pivot: &EMonom) -> bool {
    for &i in pivot.occ() {
        let i = i as usize;
        if t.exp(i) <= pivot.exp(i) { return false; }
    }
    true
}

// ----------------------------------------------------------------------------
// InterreduceTList
// ----------------------------------------------------------------------------

/// Interreduce the mixed-generator list of `tl` w.r.t. itself and `tl.sp`.
/// After the call:
///
/// * No mixed generator is divisible by any simple power in `tl.sp`.
/// * No mixed generator is divisible by another (the surviving set is an
///   antichain under monomial divisibility).
/// * `tl.mixed` is **sorted by decreasing total degree** (CoCoA's
///   `MTLOrderByDecrDegree`), since the second pass relies on that order to
///   make the cheap "divides only by something later in the array" check
///   sufficient.
///
/// `O(m * n + m^2 * occ)` where `m = mixed.len()`.
pub fn interreduce(tl: &mut TermList) {
    // Pass 1: drop any mixed generator divisible by some SP.
    let sp = &tl.sp;
    tl.mixed.retain(|t| !sp_divides_term(sp, t));
    if tl.mixed.len() <= 1 { return; }
    // Pass 2: sort by **decreasing degree** (stable to keep tie order).
    tl.mixed.sort_by(|a, b| b.degree().cmp(&a.degree()));
    // Pass 3: walk from front to back; drop `mixed[n]` if any later entry
    // (`j > n`) divides it.  Because the array is sorted by decreasing
    // degree, a divisor of `mixed[n]` must have degree ≤ deg(mixed[n]),
    // i.e. lives at some index `j >= n` (ties OK).
    //
    // To survive in-place rewriting we accumulate keep-flags first.
    let m = tl.mixed.len();
    let mut keep = vec![true; m];
    for n in 0..m {
        for j in (n + 1)..m {
            if keep[j] && tl.mixed[j].divides(&tl.mixed[n]) {
                keep[n] = false;
                break;
            }
        }
    }
    let mut idx = 0;
    tl.mixed.retain(|_| { let k = keep[idx]; idx += 1; k });
}

// ----------------------------------------------------------------------------
// ReduceAndDivideBySimplePower
// ----------------------------------------------------------------------------

/// Pivot the recursion on a **simple power** `x_{pi}^{pe}`.  Mutates `tl` so
/// that on exit:
///
/// * `tl.sp` has `x_{pi}^{pe}` inserted (CoCoA: `InsInSPList`).
/// * `tl.mixed` retains exactly the generators that lay in the **kept**
///   half (i.e. were not removed by the pivot).  Specifically: any `t` with
///   `t.exp(pi) >= pe` is removed because `x_{pi}^{pe}` already covers it.
///
/// The returned `TermList` is the **divided** quotient `I : x_{pi}^{pe}`,
/// already interreduced.
///
/// CoCoA: `ReduceAndDivideBySimplePower` (`TermList.C` ~493).
pub fn reduce_and_divide_by_simple_power(tl: &mut TermList, pi: usize, pe: u32) -> TermList {
    debug_assert!(pe > 0, "pivot exponent must be positive");
    let n_vars = tl.n_vars();
    // The divided SPList is the original SP, then we "colon" out x_{pi}^{pe}
    // (saturating subtract): if old exponent is e_old > pe, the new is
    // e_old - pe; if e_old <= pe, the SP slot is dropped (since x_{pi}^{pe-rest}
    // is no longer a generator after dividing by x_{pi}^{pe}, but actually:
    // the original SP entry x_{pi}^{e_old} divided by x_{pi}^{pe} is
    // x_{pi}^{max(0, e_old-pe)}, which is a valid generator only if e_old > pe;
    // when e_old <= pe the quotient is 1 ⇒ unit ideal, but in our recursion
    // pe is chosen so this case is impossible (we picked pe ≤ all exponents
    // at slot pi in the mixed list, and SP slots are independent of pi
    // anyway because TListIndetsNo > 1 split keeps SP coprime).  So the
    // saturating sub is the right behaviour and only ever fires on the
    // pi-slot.
    let mut div_sp = tl.sp.clone();
    let pi_old_in_sp = div_sp.exp(pi);
    if pi_old_in_sp >= pe {
        div_sp.set(pi, pi_old_in_sp - pe);
    } else if pi_old_in_sp != 0 {
        // pe > pi_old_in_sp: division yields 1 in this slot.
        div_sp.set(pi, 0);
    }
    // The kept half's SP gets x_{pi}^{pe} inserted, but only if it would be
    // a genuinely new (i.e. *stronger* = smaller exponent) constraint than
    // anything already there.  Mirror `InsInSPList` semantics.
    let cur = tl.sp.exp(pi);
    if cur == 0 || pe < cur { tl.sp.set(pi, pe); }

    // Now bucket-sort the mixed terms by their exponent at slot `pi`.
    // Buckets[e] holds all terms with `t.exp(pi) == e`, for 0 <= e <= pe+1
    // (we use index pe+1 as the "overflow" bucket for e > pe).
    let mut buckets: Vec<Vec<EMonom>> = (0..=(pe as usize + 1)).map(|_| Vec::new()).collect();
    let old_mixed = std::mem::take(&mut tl.mixed);
    for mut t in old_mixed.into_iter() {
        let te = t.exp(pi);
        if te > pe {
            // Removed from `tl` (pivot kills it), goes into divided quotient
            // with exponent te - pe at slot pi.
            t.set(pi, te - pe);
            buckets[pe as usize + 1].push(t);
        } else if te == pe {
            // Removed from `tl`.  In divided quotient: drop the pi exponent.
            // If only one other var occurred (CoCoA's "OccIndNo == 2"), that
            // becomes a simple power; absorb into div_sp.
            t.set(pi, 0);
            if t.occ_count() == 1 {
                let j = t.occ()[0] as usize;
                let e = t.exp(j);
                let cur = div_sp.exp(j);
                if cur == 0 || e < cur { div_sp.set(j, e); }
            } else {
                buckets[pe as usize].push(t);
            }
        } else if te > 0 {
            // 0 < te < pe.  Stays in `tl` as well? No — CoCoA *removes* it
            // from the kept list (the original entry is dropped; in the
            // divided list its quotient is `t / gcd(t, x^pe) = t / x^te = (t with pi-slot zeroed)`).
            // Wait — re-read CoCoA carefully.  Per `TermList.C` lines 545-558:
            //   else if ( TExp != 0 ) {  (i.e. 0 < TExp < PExp)
            //     ... goes into DivMTLExp[TExp] ... ;
            //   }
            // and there is NO `MTLMoveLastToNth(MTL, MTLLen, i)` in this branch,
            // meaning the original entry STAYS in the kept tl.mixed.  But we
            // already moved it out via `into_iter()`.  We must re-push it
            // (with original exponent) into `tl.mixed`.
            let mut keep = t.clone();
            // `keep` has its original exponent at pi (we mutated `t` above? no,
            // we didn't: we only branched.  `t` is unchanged so far.)
            let _ = keep;  // unused; we'll push `t.clone()` directly below.
            tl.mixed.push(t.clone());
            // Now process the divided side:
            t.set(pi, 0);
            if t.occ_count() == 1 {
                let j = t.occ()[0] as usize;
                let e = t.exp(j);
                let cur = div_sp.exp(j);
                if cur == 0 || e < cur { div_sp.set(j, e); }
            } else {
                buckets[te as usize].push(t);
            }
        } else {
            // te == 0: pi does not occur in t.  STAYS in `tl.mixed` AND
            // appears unchanged in the divided list (CoCoA: pushed into
            // bucket[0]).
            tl.mixed.push(t.clone());
            buckets[0].push(t);
        }
    }

    // Build divided.mixed by concatenating buckets in CoCoA order:
    //   start = bucket[pe], absorb bucket[pe-1], ..., bucket[1], bucket[0],
    //   then bucket[pe+1].  At each absorption, drop entries divisible by
    //   div_sp or by the current accumulator.
    let mut div_mixed: Vec<EMonom> = std::mem::take(&mut buckets[pe as usize]);
    // Absorb e = pe-1, pe-2, ..., 1, 0 with dedup and pi-slot restoration.
    let mut e = pe as i64 - 1;
    while e >= 0 {
        let bucket_e = std::mem::take(&mut buckets[e as usize]);
        for mut t in bucket_e.into_iter() {
            // For e > 0, restore the pi-slot exponent (we hadn't zeroed it
            // for those because CoCoA keeps the original alive in `tl`; the
            // copy in the bucket should have `pi` slot = 0 for the divided
            // computation, BUT then CoCoA at the very end calls
            //     eterm_put_non0_nth(T, PIndex, e); IntsPutLast(Indets(T), PIndex);
            // which RE-INSERTS the pi slot.  This is to make the divided
            // generator equal to (original t), not (t with pi zeroed).
            //
            // Wait — that contradicts what I wrote.  Let me re-read:
            //   for e in (PExp-1) down to 1:
            //     for each T in bucket[e]:
            //        if SPL divides T  → drop
            //        else if accumulator divides T → drop
            //        else → MTLPutNth(MTLEe, i, eterm_dup(T))   // dup, NOT remove
            //        // After dedup decision:
            //        eterm_put_non0_nth(T, PIndex, e);   // restore pi-slot on T
            //        IntsPutLast(Indets(T), PIndex);
            //
            // So `T` in bucket[e] had pi-slot zeroed before, gets restored
            // AFTER the dedup check.  The `eterm_dup` is what actually goes
            // into the accumulator (which keeps pi-slot zeroed).  The
            // restored `T` itself is then APPENDED to the accumulator
            // (`MTLAppend(DivMTL, &DivMTLLen, MTLEe, MTLLenEe);`).
            //
            // Net effect: each surviving bucket[e] entry contributes the
            // ORIGINAL unmodified term (pi-slot = e) to the divided quotient,
            // and the dedup check is done against the pi-slot-zeroed version.
            //
            // Why?  Because in the divided quotient I:x^pe, two terms that
            // agree off-pi but differ on pi are duplicates *modulo* the SP
            // x_pi^pe being added, so we dedup with pi-slot stripped, but
            // keep the original to preserve the recursion's semantic.
            //
            // Actually no: terms with pi-slot ≤ pe in I:x^pe become terms
            // with pi-slot 0 (since (t : x^pe) saturates pi-slot to 0 when
            // t.pi ≤ pe).  So the DIVIDED quotient ought to have pi-slot 0,
            // not e.  But CoCoA explicitly restores pi-slot to e!
            //
            // Resolution: the `auxLen / DivTList` building code in CoCoA is
            // recycling memory.  The same `eterm` cells will be returned to
            // the pool; the comment "eterm_put_non0_nth(T, PIndex, e); IntsPutLast"
            // is restoring the pi-slot for **the version going into the
            // accumulator's MTLAppend**, which is the surviving copy in the
            // divided list.  And those entries DO have pi-slot = e.
            //
            // ⇒ I had the semantics backwards.  In the DIVIDED quotient,
            // the surviving terms must keep pi-slot = e (= their original
            // pre-pivot exponent), because:
            //   (t : x_pi^pe) for t with t.pi = e ≤ pe is just t (the gcd
            //   in the colon ideal of monomials only divides as much as pe
            //   covers, but for t with t.pi < pe, x_pi^pe doesn't divide t
            //   so the colon doesn't strip pi at all!).
            //
            // OK so the saturating-subtract intuition is wrong for t.pi < pe:
            // for the *colon ideal* on monomials, (t : m) is t/gcd(t, m), so
            // for m = x_pi^pe and t with t.pi = e < pe:
            //   gcd(t, x_pi^pe) = x_pi^e (only the pi part overlaps),
            //   so (t : x_pi^pe) = t / x_pi^e = (t with pi-slot ZEROED).
            //
            // That AGREES with my earlier intuition.  But CoCoA restores
            // pi-slot to e, which would make the divided generator equal to
            // the ORIGINAL t.  Contradiction.
            //
            // Re-reading once more *carefully*:  CoCoA bucket[e] for
            // 0 < e < pe contains terms that originally had t.pi = e.  Those
            // are NOT removed from `tl.mixed` (line 545+ has no MTLMoveLastToNth).
            // The COPIES placed into bucket[e] have their pi-slot set to 0
            // by `eterm_put0_nth(T, PIndex)` on line 554.  Those copies are
            // what gets deduped (lines 569-578).  The "restoration" call
            // `eterm_put_non0_nth(T, PIndex, e)` on line 577 happens INSIDE
            // the loop AFTER each dedup decision — and it operates on the
            // SAME `T` that was already deduped against, **resetting its
            // pi-slot back to e**.  Then this restored T is what goes into
            // DivMTL via MTLAppend at the end.
            //
            // So the divided list ends up with terms whose pi-slot = e
            // (their original exponent, NOT zeroed!).
            //
            // Mathematically: for the colon ideal `I : x_pi^pe`, a generator
            // with t.pi < pe gives `t : x_pi^pe = t / x_pi^{t.pi} = (t with pi=0)`.
            // A generator with t.pi = pe gives `t : x_pi^pe = t / x_pi^pe = (t with pi=0)`.
            // A generator with t.pi > pe gives `t : x_pi^pe = t / x_pi^pe = (t with pi := t.pi - pe)`.
            //
            // CoCoA's "restore pi-slot to e" therefore looks WRONG.  But
            // CoCoA has been correct for 15+ years and this engine is well-
            // tested.  Where's my error?
            //
            // Aha: the recursion isn't computing `H(I : x^pe)` as a *colon
            // ideal of monomial generators*; it's computing the *Hilbert
            // numerator* via the formula:
            //
            //   H(R/I) = H(R/(I + (m))) + t^deg(m) * H(R/(I:m))
            //
            // where `(I + (m))` is the "kept" branch and `(I:m)` is the
            // "divided" branch.  For this to produce a correct H value, the
            // divided ideal must be the genuine colon ideal.  So the divided
            // generators *should* have pi-slot zeroed for t.pi <= pe.
            //
            // ⇒ I must be misreading CoCoA.  Let me re-examine line 577 in
            // its full context (and the e=0 path on lines 583-595, which is
            // STRUCTURALLY DIFFERENT — no pi-slot restoration!).
            //
            // Conclusion: line 577 is INSIDE the `for ( e=PExp-1 ; e>0 ; --e )`
            // loop (note: `e>0`, not `e>=0`).  The e=0 case is handled
            // SEPARATELY in lines 583-595 with no pi-slot restoration.
            //
            // So:
            //   * For 0 < e < pe: dedup with pi=0, then RESTORE pi to e.
            //   * For e = 0: dedup with pi=0, leave pi=0.  ✓
            //
            // The restoration for 0<e<pe makes the divided generator have
            // pi-slot = e, which is the ORIGINAL exponent.  That IS the
            // correct colon: for t with t.pi = e and m = x_pi^pe with e < pe,
            // gcd(t, m) = x_pi^e, so t : m = t / x_pi^e = t with pi=0.
            //
            // ⇒ Restoration to pi=e is WRONG by my colon math.  Yet CoCoA
            // is correct.  So my colon math must be wrong:
            //
            //   For monomials, (a) : (b) = a / gcd(a, b)?  YES that is the
            //   monoid colon for principal monomial ideals.  But this is
            //   used in the Hilbert recursion for IDEALS, not principal
            //   ideals.  For an ideal I = (g_1, ..., g_n) and a single
            //   monomial m, I : (m) = (g_1 : m, ..., g_n : m) where each
            //   g_i : m = g_i / gcd(g_i, m).  YES, my formula is right.
            //
            //   For g_i with g_i.pi = e < pe and m = x_pi^pe:
            //     gcd(g_i, m) = x_pi^e
            //     g_i : m = g_i / x_pi^e = (g_i with pi-slot zeroed)
            //
            // ⇒ Pi-slot in divided generator MUST be zero.  CoCoA's
            // restoration to e CONTRADICTS this.
            //
            // Let me look one more time at lines 569-580:
            //   for ( i=MTLLenEe ; i>0 ; --i ) {
            //     T = MTLEe[i];
            //     if ( SPLDividesTerm(DivSPL, T) )  MTLMoveLastToNth(MTLEe, MTLLenEe, i);
            //     else if ( MTLDividesTerm(DivMTL, DivMTLLen, T) )
            //       MTLMoveLastToNth(MTLEe, MTLLenEe, i);
            //     else
            //       MTLPutNth(MTLEe, i, eterm_dup(T));    // ← duplicate
            //     eterm_put_non0_nth(T, PIndex, e);  // ← runs ALWAYS, even if T was removed!
            //     IntsPutLast(Indets(T), PIndex);
            //   }
            //
            // ⚠ The pi-slot restoration runs UNCONDITIONALLY, including on
            // removed-from-MTLEe terms.  Those terms are STILL alive — they
            // are the ORIGINALS that stayed in `tl.mixed`!  Restoring pi-slot
            // on them returns them to their pre-bucket state, undoing the
            // earlier `eterm_put0_nth(T, PIndex)` on line 554.
            //
            // The `MTLPutNth(MTLEe, i, eterm_dup(T))` line replaces the
            // pi-zeroed `T` slot with a fresh DUPLICATE (still pi=0).  So
            // the surviving entries that go into DivMTL are the
            // pi-zeroed dup'd copies; the restoration pi=e on the original
            // `T` only affects `tl.mixed`.
            //
            // ⇒ Divided generators DO have pi-slot zero.  ✓ My math is
            // right; my reading of CoCoA was off.
            //
            // OK.  So our Rust port should:
            //   * For t with original t.pi = e, 0 < e < pe:
            //       - Push CLONE of t (with pi=0) into divided list (after dedup).
            //       - Push ORIGINAL t (unchanged) back into tl.mixed.
            //
            // That's what we already do above (we push t.clone() to tl.mixed
            // and a pi-zeroed t to the bucket).  Excellent — no change
            // needed to our existing logic.  Now finish the dedup:
            if sp_divides_term(&div_sp, &t) { continue; }
            if mixed_divides_term(&div_mixed, &t) { continue; }
            div_mixed.push(t);
        }
        e -= 1;
    }
    // Bucket pe+1: terms with original t.pi > pe, already had pi-slot
    // adjusted to (te - pe).  Same dedup.
    let bucket_overflow = std::mem::take(&mut buckets[pe as usize + 1]);
    for t in bucket_overflow.into_iter() {
        if sp_divides_term(&div_sp, &t) { continue; }
        if mixed_divides_term(&div_mixed, &t) { continue; }
        div_mixed.push(t);
    }

    // Sanity asserts (debug only).
    debug_assert!(div_mixed.iter().all(|t| t.n_vars() == n_vars));
    TermList { sp: div_sp, mixed: div_mixed }
}

// ----------------------------------------------------------------------------
// ReduceAndDivideByMixedTerm
// ----------------------------------------------------------------------------

/// Pivot the recursion on a **mixed** monomial `pivot` (occurring in ≥ 2
/// indets).  Mutates `tl` so that on exit:
///
/// * `tl.mixed` retains only generators not divisible by `pivot`, plus
///   `pivot` itself appended at the end (CoCoA: `MTLPutLast(MTL, MTLLen, Pivot)`).
/// * `tl.sp` is unchanged (a mixed pivot doesn't add to SP).
///
/// The returned `TermList` is the **divided** colon ideal `I : pivot`,
/// already interreduced.
///
/// CoCoA: `ReduceAndDivideByMixedTerm` (`TermList.C` ~610).
pub fn reduce_and_divide_by_mixed_term(tl: &mut TermList, pivot: EMonom) -> TermList {
    debug_assert!(pivot.occ_count() >= 2, "mixed pivot must occur in ≥ 2 vars");
    let n_vars = tl.n_vars();
    // Divided SP: original SP with `pivot` colon-removed.
    let mut div_sp = tl.sp.clone();
    div_sp.colon_assign(&pivot);
    // `aux_sp` accumulates the *additional* simple powers contributed by
    // mixed terms whose colon-by-pivot is a simple power (CoCoA: auxSPL).
    // After the main loop we colon `aux_sp` by pivot too (line 673) and
    // it acts as a filter for surviving mixed terms.
    let mut aux_sp = EMonom::one(n_vars);

    let old_mixed = std::mem::take(&mut tl.mixed);
    let mut div_mixed: Vec<EMonom> = Vec::new();
    let mut coprime_mixed: Vec<EMonom> = Vec::new(); // CoCoA `CoprimeMTL`
    let mut big_mult_mixed: Vec<EMonom> = Vec::new(); // CoCoA `BigMultMTL`

    for t in old_mixed.into_iter() {
        let t_deg = t.degree();
        if pivot.divides(&t) {
            // Removed from `tl`.  Compute t : pivot.  CoCoA distinguishes
            // "BigMult" (= quotient retains full pivot support) via
            // `sp_BigMult`; only the non-BigMult branch checks for single-
            // occ to absorb into div_sp.
            let mut q = t.clone();
            q.colon_assign(&pivot);
            if sp_big_mult(&t, &pivot) {
                big_mult_mixed.push(q);
            } else if q.occ_count() == 1 {
                let j = q.occ()[0] as usize;
                let e = q.exp(j);
                let cur = div_sp.exp(j);
                if cur == 0 || e < cur { div_sp.set(j, e); }
                let cur_aux = aux_sp.exp(j);
                let combined = e + pivot.exp(j);
                if cur_aux == 0 || combined < cur_aux { aux_sp.set(j, combined); }
            } else {
                div_mixed.push(q);
            }
        } else if sp_divides_term(&aux_sp, &t) {
            // Already covered by aux_sp on the divided side; t stays in tl
            // (since pivot doesn't divide it) but contributes nothing to
            // the divided list.
            tl.mixed.push(t);
        } else {
            // Compute (t : pivot).  Since pivot does NOT divide t, the
            // colon may produce a term with the same degree (coprime case)
            // or smaller (proper overlap).
            let mut q = t.clone();
            q.colon_assign(&pivot);
            // The original t stays in the kept tl.mixed regardless.
            tl.mixed.push(t);
            if q.degree() == t_deg {
                // Coprime: t and pivot share no support.  Goes into a
                // separate pile that gets deduped against div_mixed at the
                // end (CoCoA does this because coprime terms tend to
                // dominate and a final pass amortises).
                coprime_mixed.push(q);
            } else if q.occ_count() == 1 {
                // Becomes a simple power; absorb into div_sp AND record an
                // aux_sp entry so subsequent terms with this var get dropped.
                let j = q.occ()[0] as usize;
                let e = q.exp(j);
                let cur = div_sp.exp(j);
                if cur == 0 || e < cur { div_sp.set(j, e); }
                let cur_aux = aux_sp.exp(j);
                let combined = e + pivot.exp(j);
                if cur_aux == 0 || combined < cur_aux { aux_sp.set(j, combined); }
            } else {
                div_mixed.push(q);
            }
        }
    }

    // CoCoA: MTLPutLast(MTL, MTLLen, Pivot) — add the pivot to the kept
    // list as a generator (since (I : m) ⊆ (I), and the kept ideal is
    // I + (m) which contains m).
    tl.mixed.push(pivot.clone());
    // CoCoA: eterm_colon(auxSPL, Pivot) — finalise aux_sp.  We don't actually
    // need aux_sp after this point.
    // Now interreduce div_mixed in place (drops div_mixed-elements covered
    // by other div_mixed-elements OR by div_sp).
    // We then dedup coprime_mixed against div_sp and div_mixed.
    {
        // Build a temporary TermList so we can reuse `interreduce`.
        let mut tmp = TermList { sp: div_sp.clone(), mixed: std::mem::take(&mut div_mixed) };
        interreduce(&mut tmp);
        div_sp = tmp.sp;
        div_mixed = tmp.mixed;
    }
    // Dedup coprime against div_sp and div_mixed.
    if div_mixed.is_empty() {
        coprime_mixed.retain(|t| !sp_divides_term(&div_sp, t));
    } else {
        coprime_mixed.retain(|t| !sp_divides_term(&div_sp, t) && !mixed_divides_term(&div_mixed, t));
    }
    div_mixed.append(&mut coprime_mixed);
    // Append BigMult unconditionally (CoCoA never dedups them).
    div_mixed.append(&mut big_mult_mixed);

    debug_assert!(div_mixed.iter().all(|t| t.n_vars() == n_vars));
    TermList { sp: div_sp, mixed: div_mixed }
}

/// Dispatch a pivot: a single-occurring pivot is a simple power, otherwise
/// a mixed term.  CoCoA: `ReduceAndDivideByPivot`.
pub fn reduce_and_divide_by_pivot(tl: &mut TermList, pivot: EMonom) -> TermList {
    if pivot.occ_count() == 1 {
        let i = pivot.occ()[0] as usize;
        let e = pivot.exp(i);
        reduce_and_divide_by_simple_power(tl, i, e)
    } else {
        reduce_and_divide_by_mixed_term(tl, pivot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn em(n: usize, sp: &[(usize, u32)]) -> EMonom {
        EMonom::from_sparse(n, sp)
    }

    #[test]
    fn interreduce_drops_terms_covered_by_sp() {
        // I = (x_0^2, x_0^3 x_1)  → x_0^3 x_1 is divisible by x_0^2.
        let n = 3;
        let mut tl = TermList::from_generators(
            n,
            vec![em(n, &[(0, 2)]), em(n, &[(0, 3), (1, 1)])],
        );
        // x_0^2 went into SP; the mixed term x_0^3 x_1 should be dropped.
        assert_eq!(tl.sp.exp(0), 2);
        assert_eq!(tl.mixed.len(), 1);
        interreduce(&mut tl);
        assert_eq!(tl.mixed.len(), 0);
    }

    #[test]
    fn interreduce_drops_redundant_mixed() {
        // I = (x_0 x_1, x_0 x_1 x_2)  → second is divisible by first.
        let n = 4;
        let mut tl = TermList::from_generators(
            n,
            vec![em(n, &[(0, 1), (1, 1)]), em(n, &[(0, 1), (1, 1), (2, 1)])],
        );
        interreduce(&mut tl);
        assert_eq!(tl.mixed.len(), 1);
        assert_eq!(tl.mixed[0], em(n, &[(0, 1), (1, 1)]));
    }

    #[test]
    fn interreduce_keeps_antichain_sorted_by_decreasing_degree() {
        // I = (x_0 x_1, x_0 x_2 x_3, x_1 x_2)  — none divides any other.
        let n = 4;
        let mut tl = TermList::from_generators(
            n,
            vec![
                em(n, &[(0, 1), (1, 1)]),
                em(n, &[(0, 1), (2, 1), (3, 1)]),
                em(n, &[(1, 1), (2, 1)]),
            ],
        );
        interreduce(&mut tl);
        assert_eq!(tl.mixed.len(), 3);
        // Sorted by decreasing degree: deg 3 first.
        assert_eq!(tl.mixed[0].degree(), 3);
        assert!(tl.mixed[1].degree() <= tl.mixed[0].degree());
        assert!(tl.mixed[2].degree() <= tl.mixed[1].degree());
    }

    #[test]
    fn reduce_by_simple_power_basic() {
        // I = (x_0^2, x_0 x_1, x_0 x_2 x_3, x_4 x_5);  pivot = x_0^2
        // Kept = I + (x_0^2) = I (x_0^2 already there); but our impl will
        // register x_0^2 as the SP entry and drop x_0^2 from mixed (already
        // done at construction).  Mixed terms with x_0-exp >= 2: none in
        // this example (x_0^1 entries have exp 1 < 2).  So tl.mixed should
        // remain {x_0 x_1, x_0 x_2 x_3, x_4 x_5}.
        let n = 6;
        let mut tl = TermList::from_generators(
            n,
            vec![
                em(n, &[(0, 2)]),
                em(n, &[(0, 1), (1, 1)]),
                em(n, &[(0, 1), (2, 1), (3, 1)]),
                em(n, &[(4, 1), (5, 1)]),
            ],
        );
        // x_0^2 → SP; mixed = the other three.
        assert_eq!(tl.sp.exp(0), 2);
        assert_eq!(tl.mixed.len(), 3);
        let div = reduce_and_divide_by_simple_power(&mut tl, 0, 2);
        // Kept SP unchanged (already had x_0^2):
        assert_eq!(tl.sp.exp(0), 2);
        // Kept mixed: x_0 x_1 stays (x_0 exp = 1 < 2), x_0 x_2 x_3 stays,
        // x_4 x_5 stays.  All three remain.
        assert_eq!(tl.mixed.len(), 3);
        // Divided I:x_0^2 = (x_0 x_1 / x_0, x_0 x_2 x_3 / x_0, x_4 x_5)
        //                 = (x_1, x_2 x_3, x_4 x_5)
        // x_1 is a simple power → goes into div_sp.
        assert_eq!(div.sp.exp(1), 1);
        // div.mixed = {x_2 x_3, x_4 x_5}.
        assert_eq!(div.mixed.len(), 2);
        let mut got: Vec<Vec<u32>> = div.mixed.iter().map(|m| m.occ().to_vec()).collect();
        got.sort();
        assert_eq!(got, vec![vec![2u32, 3], vec![4u32, 5]]);
    }

    #[test]
    fn reduce_by_simple_power_overflow_bucket() {
        // I = (x_0 x_1, x_0^5 x_2);  pivot = x_0^3
        // Kept SP gets x_0^3.
        // Kept mixed: x_0 x_1 stays (1 < 3), x_0^5 x_2 removed (5 >= 3).
        // Divided: x_0 x_1 contributes (with pi-slot 0) → x_1 (simple power → div_sp).
        //          x_0^5 x_2 contributes (x_0^{5-3}) x_2 = x_0^2 x_2 (mixed).
        let n = 3;
        let mut tl = TermList::from_generators(
            n,
            vec![em(n, &[(0, 1), (1, 1)]), em(n, &[(0, 5), (2, 1)])],
        );
        let div = reduce_and_divide_by_simple_power(&mut tl, 0, 3);
        assert_eq!(tl.sp.exp(0), 3);
        // tl.mixed: x_0 x_1 stays.
        assert_eq!(tl.mixed.len(), 1);
        assert_eq!(tl.mixed[0], em(n, &[(0, 1), (1, 1)]));
        // div.sp: x_1 absorbed.
        assert_eq!(div.sp.exp(1), 1);
        // div.mixed: x_0^2 x_2.
        assert_eq!(div.mixed.len(), 1);
        assert_eq!(div.mixed[0], em(n, &[(0, 2), (2, 1)]));
    }

    #[test]
    fn reduce_by_mixed_term_basic() {
        // I = (x_0 x_1 x_2, x_0 x_3, x_4 x_5);  pivot = x_0 x_1
        // Kept tl: x_0 x_1 x_2 removed (divisible by pivot); x_0 x_3 stays;
        //          x_4 x_5 stays; pivot x_0 x_1 appended.
        let n = 6;
        let pivot = em(n, &[(0, 1), (1, 1)]);
        let mut tl = TermList::from_generators(
            n,
            vec![
                em(n, &[(0, 1), (1, 1), (2, 1)]),
                em(n, &[(0, 1), (3, 1)]),
                em(n, &[(4, 1), (5, 1)]),
            ],
        );
        let div = reduce_and_divide_by_mixed_term(&mut tl, pivot.clone());
        // tl.mixed = {x_0 x_3, x_4 x_5, x_0 x_1}.  Order may vary.
        assert_eq!(tl.mixed.len(), 3);
        let mut got: Vec<Vec<u32>> = tl.mixed.iter().map(|m| m.occ().to_vec()).collect();
        got.sort();
        let mut want = vec![vec![0u32, 1], vec![0u32, 3], vec![4u32, 5]];
        want.sort();
        assert_eq!(got, want);
        // div = (x_0 x_1 x_2 : x_0 x_1, x_0 x_3 : x_0 x_1, x_4 x_5 : x_0 x_1)
        //     = (x_2, x_3, x_4 x_5)
        // x_2, x_3 → simple powers (absorbed into div_sp); x_4 x_5 → mixed.
        assert_eq!(div.sp.exp(2), 1);
        assert_eq!(div.sp.exp(3), 1);
        assert_eq!(div.mixed.len(), 1);
        assert_eq!(div.mixed[0], em(n, &[(4, 1), (5, 1)]));
    }

    #[test]
    fn reduce_and_divide_by_pivot_dispatches() {
        // Ensure the dispatch helper picks the right path.
        let n = 3;
        let pivot_sp = em(n, &[(0, 2)]);
        let mut tl = TermList::from_generators(n, vec![em(n, &[(0, 3), (1, 1)])]);
        let _ = reduce_and_divide_by_pivot(&mut tl, pivot_sp);
        assert_eq!(tl.sp.exp(0), 2);

        let pivot_mixed = em(n, &[(0, 1), (1, 1)]);
        let mut tl2 = TermList::from_generators(n, vec![em(n, &[(0, 1), (1, 1), (2, 1)])]);
        let _ = reduce_and_divide_by_pivot(&mut tl2, pivot_mixed);
        // Pivot got appended.
        assert_eq!(tl2.mixed.len(), 1);
        assert_eq!(tl2.mixed[0], em(n, &[(0, 1), (1, 1)]));
    }
}
