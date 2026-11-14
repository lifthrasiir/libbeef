// Regression tests for the exponent range. libbf derives it from the exponent
// width: `BF_EXP_BITS_MAX` (61 bits) gives e_max = 2^60, and `BF_FLAG_EXT_EXP`
// (62 bits) gives e_max = 2^61 (e_min = 3 - e_max in both cases). libbeef used
// to map `ExpBits::Max` and `ExpBits::Extended` to +-2^62. Expected values
// were computed with C libbf.
use libbeef::format::ExpBits;
use libbeef::{BigFloat, BigFormat, Precision, Rounding, Sign, Status};

fn fmt(exp_bits: ExpBits) -> BigFormat {
    BigFormat {
        precision: Precision::Bits(53),
        rounding: Rounding::NearestEven,
        exp_bits,
        subnormal: false,
        radix_point_precision: false,
    }
}

/// 1 * 2^e (so `raw_exp` is e + 1), built without any range check.
fn pow2(e: i64) -> BigFloat {
    BigFloat::from_raw(Sign::Positive, e + 1, vec![1 << 63])
}

#[test]
fn mul_overflows_at_max_exp_bits() {
    let a = pow2(1 << 59);
    for exp_bits in [ExpBits::Max, ExpBits::Bits(61)] {
        let (r, st) = a.mul_status(&a, fmt(exp_bits));
        assert!(r.is_infinite() && !r.sign().is_negative());
        assert_eq!(st, Status::OVERFLOW | Status::INEXACT);
    }
    // Still in range with the extended exponent.
    let (r, st) = a.mul_status(&a, fmt(ExpBits::Extended));
    assert_eq!(st, Status::empty());
    assert_eq!(r.raw_exp(), (1 << 60) + 1);
}

#[test]
fn range_boundaries() {
    // e_max = 2^60 for Max and 2^61 for Extended; e_min = 3 - e_max.
    for (exp_bits, e_max) in [(ExpBits::Max, 1_i64 << 60), (ExpBits::Extended, 1 << 61)] {
        let f = fmt(exp_bits);
        let (r, st) = BigFloat::from_raw(Sign::Positive, e_max, vec![1 << 63]).round_status(f);
        assert_eq!((r.raw_exp(), st), (e_max, Status::empty()));
        let (r, st) = BigFloat::from_raw(Sign::Positive, e_max + 1, vec![1 << 63]).round_status(f);
        assert!(r.is_infinite());
        assert_eq!(st, Status::OVERFLOW | Status::INEXACT);

        let e_min = 3 - e_max;
        let (r, st) = BigFloat::from_raw(Sign::Positive, e_min, vec![1 << 63]).round_status(f);
        assert_eq!((r.raw_exp(), st), (e_min, Status::empty()));
        let (r, st) = BigFloat::from_raw(Sign::Positive, e_min - 1, vec![1 << 63]).round_status(f);
        assert!(r.is_zero());
        assert_eq!(st, Status::UNDERFLOW | Status::INEXACT);
    }
}

#[test]
fn huge_exponent_difference_does_not_overflow() {
    // Operands outside the range of any format (only reachable via `from_raw`)
    // must not panic in the exponent arithmetic of add.
    let f = fmt(ExpBits::Max);
    let big = pow2((1 << 62) - 2);
    let tiny = pow2(-(1 << 62) - 1);
    let _ = big.sub_status(&tiny, f);
}

#[test]
fn cos_of_tiny_values_at_range_limit() {
    // 2 * exp - 1 stays within i64 for every value of the extended range.
    let f = fmt(ExpBits::Extended);
    let tiny = BigFloat::from_raw(Sign::Positive, 3 - (1 << 61), vec![1 << 63]);
    let (r, st) = tiny.cos_status(f);
    assert_eq!(st, Status::INEXACT);
    assert_eq!(r.raw_exp(), 1);
}

#[test]
fn bits_62_is_extended() {
    // libbf encodes 62 exponent bits as BF_FLAG_EXT_EXP.
    let a = pow2(1 << 59);
    let (r, st) = a.mul_status(&a, fmt(ExpBits::Bits(62)));
    assert_eq!(st, Status::empty());
    assert_eq!(r.raw_exp(), (1 << 60) + 1);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "valid range")]
fn bits_out_of_range_is_rejected() {
    let _ = BigFloat::from_u64(1).round_status(fmt(ExpBits::Bits(2)));
}
