// Regression tests for transcendental-function details that differ from C
// libbf. Expected values were computed with C libbf.
use libbeef::format::ExpBits;
use libbeef::{BigFloat, BigFormat, Precision, Rounding, Sign, Status};

fn fmt(bits: u64, rounding: Rounding, exp_bits: ExpBits, subnormal: bool) -> BigFormat {
    BigFormat {
        precision: Precision::Bits(bits),
        rounding,
        exp_bits,
        subnormal,
        radix_point_precision: false,
    }
}

fn f200() -> BigFormat {
    fmt(200, Rounding::NearestEven, ExpBits::Bits(61), false)
}

fn pow2(e: i64) -> BigFloat {
    BigFloat::from_raw(Sign::Positive, e + 1, vec![1 << 63])
}

#[track_caller]
fn check(r: &BigFloat, negative: bool, exp: i64, limbs_hi_to_lo: &[u64]) {
    assert_eq!(r.sign() == Sign::Negative, negative);
    assert_eq!(r.raw_exp(), exp);
    let mut l: Vec<u64> = limbs_hi_to_lo.to_vec();
    l.reverse();
    assert_eq!(r.limbs(), &l[..]);
}

#[test]
fn exp_reduction_beyond_2_pow_53() {
    // n = floor(a / ln 2) must be exact in 64 bits, not an f64.
    let a = pow2(59).neg();
    let (r, st) = a.exp_status(f200());
    assert_eq!(st, Status::INEXACT);
    check(
        &r,
        false,
        -831657068615270155,
        &[
            0x99ea1065dfa7268d,
            0xc4ffddcea639d83e,
            0x48280dd048c861b9,
            0xa700000000000000,
        ],
    );
}

#[test]
fn pow_reduction_beyond_2_pow_53() {
    let x = BigFloat::from_f64(1.5);
    let (r, st) = x.pow_status(&pow2(60), f200());
    assert_eq!(st, Status::INEXACT);
    check(
        &r,
        false,
        674415846470019194,
        &[
            0xfbe066390734b9d4,
            0xef97090121ad59fe,
            0x29c65486f279a414,
            0x9600000000000000,
        ],
    );
}
