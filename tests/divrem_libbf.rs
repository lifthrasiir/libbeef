//! Regression tests for `div_rem_status` / `rem_status` / `rem_quo_status`
//! matching libbf's `bf_divrem` / `bf_rem` / `bf_remquo`.
use libbeef::format::{ExpBits, Precision};
use libbeef::{BigFloat, BigFormat, DivRemMode, Rounding};

fn i(v: i64) -> BigFloat {
    BigFloat::from_i64(v)
}

fn to_i(x: &BigFloat) -> i64 {
    x.get_int64().expect("integer in range")
}

#[test]
fn divrem_nearest_even_tie_uses_units_bit_of_quotient() {
    // 3 / 2 = 1.5 -> tie, q = 1 is odd, so rounds to 2 with r = -1.
    let (q, r, st) = i(3).div_rem_status(&i(2), BigFormat::BINARY64, DivRemMode::NearestEven);
    assert!(st.is_empty());
    assert_eq!(to_i(&q), 2);
    assert_eq!(to_i(&r), -1);
    let (q, r, _) = i(3).rem_quo_status(&i(2), BigFormat::BINARY64, DivRemMode::NearestEven);
    assert_eq!((q, to_i(&r)), (2, -1));
}

#[test]
fn divrem_temporaries_ignore_the_users_exponent_range() {
    // 2^1023 / (3 * 2^-100): the quotient (~2^1121) exceeds the binary64 range.
    let a = BigFloat::from_f64(2f64.powi(1023));
    let b = BigFloat::from_f64(3.0 * 2f64.powi(-100));
    let fmt = BigFormat {
        rounding: Rounding::TowardZero,
        ..BigFormat::BINARY64
    };
    let (q, r, st) = a.div_rem_status(&b, fmt, DivRemMode::TowardZero);
    assert!(st.is_empty());
    assert_eq!(r.to_f64(Rounding::NearestEven), 2f64.powi(-99));
    assert!(q.is_finite() && !q.is_zero());
    // q * b + r == a exactly.
    let exact = BigFormat {
        precision: Precision::Infinite,
        rounding: Rounding::TowardZero,
        exp_bits: ExpBits::Max,
        subnormal: false,
        radix_point_precision: false,
    };
    let sum = q.mul(&b, exact).add(&r, exact);
    assert_eq!(sum, a);
}

#[test]
fn rem_quo_status_returns_low_bits_of_quotient() {
    let f = BigFormat::BINARY64;
    let (q, _, _) = BigFloat::from_f64(1e30).rem_quo_status(&i(3), f, DivRemMode::TowardZero);
    assert_eq!(q, 1692321384976700757);
    let two64_plus_1 = BigFloat::from_u64(u64::MAX).add(
        &i(2),
        BigFormat {
            precision: Precision::Infinite,
            ..BigFormat::BINARY64
        },
    );
    let (q, _, _) = two64_plus_1.rem_quo_status(&i(1), f, DivRemMode::TowardZero);
    assert_eq!(q, 1);
}
