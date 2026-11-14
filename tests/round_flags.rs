// Regression tests for the rounding status flags and range checks of
// `round`, `rint` and the arithmetic built on `__bf_round`. Expected values
// were computed with C libbf.
use libbeef::format::ExpBits;
use libbeef::{BigFloat, BigFormat, Precision, Rounding, Sign, Status};

const UF: Status = Status::from_bits_retain(0x18);

fn fmt(prec: Precision, rounding: Rounding, e: u8, subnormal: bool) -> BigFormat {
    BigFormat {
        precision: prec,
        rounding,
        exp_bits: ExpBits::Bits(e),
        subnormal,
        radix_point_precision: false,
    }
}

/// `limb / 2^64 * 2^exp`, i.e. libbf's `expn = exp` with the given top limb.
fn val(exp: i64, limb: u64) -> BigFloat {
    BigFloat::from_raw(Sign::Positive, exp, vec![limb])
}

#[track_caller]
fn check_zero(r: &(BigFloat, Status), status: Status) {
    assert!(r.0.is_zero() && !r.0.sign().is_negative(), "{:?}", r.0);
    assert_eq!(r.1, status);
}

#[track_caller]
fn check_fin(r: &(BigFloat, Status), exp: i64, limb: u64, status: Status) {
    assert_eq!((r.0.raw_exp(), r.0.limbs()), (exp, &[limb][..]));
    assert_eq!(r.1, status);
}

#[test]
fn a2_exact_value_below_e_min_without_subnormal_flushes() {
    // 2^-1100 with 11 exponent bits (e_min = -1021), 64-bit precision.
    let f = fmt(Precision::Bits(64), Rounding::NearestEven, 11, false);
    check_zero(&val(-1099, 1 << 63).round_status(f), UF);
    // With subnormals the exact value stays.
    let f = fmt(Precision::Bits(64), Rounding::NearestEven, 11, true);
    check_fin(&val(-1030, 1 << 63).round_status(f), -1030, 1 << 63, Status::empty());
}
