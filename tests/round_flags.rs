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

#[test]
fn a3_nonpositive_precision_reports_underflow_and_inexact() {
    let rint = |v: &BigFloat, rnd| v.rint_status(rnd);
    // rint(0.75) and rint(0.25).
    check_fin(&rint(&val(0, 0xC000_0000_0000_0000), Rounding::NearestEven), 1, 1 << 63, UF);
    check_zero(&rint(&val(-1, 1 << 63), Rounding::NearestEven), UF);
    check_fin(&rint(&val(-1, 1 << 63), Rounding::TowardPositive), 1, 1 << 63, UF);
    // No underflow once the precision is positive.
    check_fin(
        &rint(&val(2, 0xA000_0000_0000_0000), Rounding::NearestEven),
        2,
        1 << 63,
        Status::INEXACT,
    );
}

#[test]
fn a3_subnormal_rounding_to_zero_is_inexact() {
    // 2^-1074 * 3 * 2^-12 in binary64.
    let a = val(-1073, 1 << 63);
    let b = val(-11, 0xC000_0000_0000_0000);
    for rnd in [Rounding::NearestEven, Rounding::Faithful] {
        let f = fmt(Precision::Bits(53), rnd, 11, true);
        check_zero(&a.mul_status(&b, f), UF);
    }
}

#[test]
fn a4_faithful_rounding_always_reports_inexact() {
    let f = fmt(Precision::Bits(53), Rounding::Faithful, 11, true);
    // round(1) and round(1 + 2^-100).
    check_fin(&val(1, 1 << 63).round_status(f), 1, 1 << 63, Status::INEXACT);
    let x = BigFloat::from_raw(Sign::Positive, 1, vec![1 << 27, 1 << 63]);
    check_fin(&x.round_status(f), 1, 1 << 63, Status::INEXACT);
    // Subnormal range: 2^-1031 is exact, but reported as inexact underflow.
    check_fin(&val(-1030, 1 << 63).round_status(f), -1030, 1 << 63, UF);
    // Rounding to 2 bits: the add_one decision is the guard bit.
    let f2 = fmt(Precision::Bits(2), Rounding::Faithful, 11, false);
    for (v, exp, limb) in [
        (3_u64, 2_i64, 0xC000_0000_0000_0000_u64),
        (7, 4, 1 << 63),
        (5, 3, 0xC000_0000_0000_0000),
        (11, 4, 0xC000_0000_0000_0000),
    ] {
        let x = BigFloat::from_u64(v);
        check_fin(&x.round_status(f2), exp, limb, Status::INEXACT);
    }
    // rint(0.75) and rint(1.25).
    check_fin(&val(0, 0xC000_0000_0000_0000).rint_status(Rounding::Faithful), 1, 1 << 63, UF);
    check_fin(
        &val(2, 0xA000_0000_0000_0000).rint_status(Rounding::Faithful),
        2,
        0xC000_0000_0000_0000,
        Status::INEXACT,
    );
}
