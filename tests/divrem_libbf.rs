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

#[test]
fn divrem_supports_all_libbf_quotient_modes() {
    // (a, b, mode, expected q, expected r)
    use DivRemMode::*;
    let cases = [
        (4, 3, Ceil, 2, -2),
        (-4, 3, Ceil, -1, -1),
        (4, 3, AwayFromZero, 2, -2),
        (-4, 3, AwayFromZero, -2, 2),
        (5, 2, NearestAway, 3, -1),
        (-5, 2, NearestAway, -3, 1),
        (5, 2, NearestEven, 2, 1),
        (-4, 3, Euclidean, -2, 2),
        (4, -3, Euclidean, -1, 1),
        (-4, 3, Floor, -2, 2),
        (-4, 3, TowardZero, -1, -1),
    ];
    for (a, b, mode, eq, er) in cases {
        let f = BigFormat::BINARY64;
        let (q, r, st) = i(a).div_rem_status(&i(b), f, mode);
        assert!(st.is_empty());
        assert_eq!((to_i(&q), to_i(&r)), (eq, er), "divrem({a}, {b}, {mode:?})");
        let (r2, st) = i(a).rem_status(&i(b), f, mode);
        assert!(st.is_empty());
        assert_eq!(to_i(&r2), er, "rem({a}, {b}, {mode:?})");
    }
}

#[test]
fn remquo_maps_rounding_modes_like_libbf() {
    let f = BigFormat::BINARY64;
    let cases = [
        (4, 3, Rounding::TowardPositive, 2, -2),
        (4, 3, Rounding::AwayFromZero, 2, -2),
        (5, 2, Rounding::NearestAway, 3, -1),
        (-4, 3, Rounding::Faithful, -2, 2),
        (5, 2, Rounding::NearestEven, 2, 1),
        (-4, 3, Rounding::TowardNegative, -2, 2),
        (-4, 3, Rounding::TowardZero, -1, -1),
    ];
    for (a, b, rnd, eq, er) in cases {
        let (q, r) = i(a).remquo(&i(b), f, rnd);
        assert_eq!((q, to_i(&r)), (eq, er), "remquo({a}, {b}, {rnd:?})");
    }
}
