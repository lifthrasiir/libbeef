//! Regression tests for `div_rem_status` / `rem_status` / `rem_quo_status`
//! matching libbf's `bf_divrem` / `bf_rem` / `bf_remquo`.
use libbeef::{BigFloat, BigFormat, DivRemMode};

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
