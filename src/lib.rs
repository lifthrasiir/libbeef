//! A Rust translation of Fabrice Bellard's
//! [libbf](https://bellard.org/libbf/), a tiny arbitrary precision
//! floating point library.
//!
//! The two main value types are [`BigFloat`] (arbitrary precision binary
//! floating point, corresponding to libbf's `bf_t`) and [`BigDecimal`]
//! (arbitrary precision decimal floating point, corresponding to `bfdec_t`).
//! Type-level format wrappers [`Float<F>`] and [`Decimal<F>`] pair a value
//! with a [`StaticFormat`] so that arithmetic operators automatically apply
//! the correct precision and rounding mode.
//!
//! Features include basic arithmetic, fused multiply-add, square root,
//! transcendental functions (exp, log, pow, trig), and parsing/formatting
//! in arbitrary radices (2--36).
//!
//! The crate is `no_std` compatible (requires `alloc`).
//!
//! # Aliasing-safe in-place operations
//!
//! libbf's C API allows aliased pointers — the same `bf_t *` can appear as
//! both the output and one (or both) inputs of an operation. Rust's borrow
//! rules forbid this: `self.mul_assign(&self, …)` would require `&mut self`
//! and `&self` simultaneously. Three dedicated method families fill the gap:
//!
//! | Method | Computes | libbf pattern |
//! |--------|----------|---------------|
//! | [`sqr`](BigFloat::sqr) / [`sqr_assign`](BigFloat::sqr_assign) | `self * self` | `bf_mul(r, a, a, …)` |
//! | [`rsub_assign`](BigFloat::rsub_assign) | `lhs - self` (reverse sub) | `bf_sub(r, a, r, …)` |
//! | [`rdiv_assign`](BigFloat::rdiv_assign) | `dividend / self` (reverse div) | `bf_div(r, a, r, …)` |
//!
//! The "reverse" variants are only needed for the non-commutative operations
//! (subtraction, division), where `r == b` produces a different result from
//! `r == a`. Addition and multiplication are commutative, so their existing
//! `_assign` methods already cover the `r == b` case.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

/// Arbitrary-precision decimal floating-point arithmetic.
pub mod decimal;
/// Arbitrary-precision binary floating-point arithmetic.
pub mod float;
/// Floating-point format descriptors (precision, rounding, exponent range).
pub mod format;
/// String-to-number parsing utilities.
pub mod parse;
/// Arithmetic status flags (inexact, overflow, etc.).
pub mod status;

pub use decimal::{BigDecimal, Decimal};
pub use float::{mp_recip, mp_sqrtrem, BigFloat, DivRemMode, Float, FpCategory, Integer, Sign};
pub use format::{
    formats, mul_log2_radix, BigFormat, DecimalFormat, ExpBits, Format, NoRadixPointPrec,
    NoSubnormal, Precision, RadixPointPrec, Rounding, StaticFormat, StaticRadixPointPrecision,
    StaticRounding, StaticSubnormal, Subnormal,
};
pub use status::Status;
