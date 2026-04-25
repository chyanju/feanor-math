//! Hilbert series / Hilbert numerator engine.
//!
//! Port of CoCoA-EP-1.3.3's `TmpHilbertDir/` engine to feanor-math.
//! Computes the Hilbert numerator `N(t) ∈ Z[t]` of `R/I` where `I` is a
//! homogeneous ideal in a standard-graded polynomial ring `R = k[x_1,…,x_n]`.
//!
//! Implementation follows the Bigatti–Caboara–Robbiano monomial recursion
//! (CoCoA `TmpPoincareCPP.{H,C}`).
//!
//! Module layout (Plan v2 §2.6, R2 §5):
//! - [`uni_helpers`] — `(1 - t^k)` multiplication and synthetic `(1-t)`
//!   division on dense univariate `Z[t]`.
//! - more sub-modules forthcoming (term_list, leaves, recursion, ...).
//!
//! See `chat/plan-v2/R2_hilbert.md` for the full design rationale.

pub mod uni_helpers;
pub mod term_list;
pub mod reduce;
pub mod leaves;
pub mod recursion;

pub use recursion::{hilbert_numerator, hilbert_numerator_in};
