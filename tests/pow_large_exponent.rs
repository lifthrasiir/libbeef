// Regression tests for `pow` with large integer exponents. libbf rejects
// overflowing or underflowing results up front from a crude bound on
// y*log|x|, and only evaluates x^y as an integer power at a bounded precision
// (or not at all when the result cannot be exact). Earlier versions of
// libbeef computed x^y exactly, e.g. ~1.6e10 bits for 3^1e10, which took
// minutes and gigabytes before rounding to infinity. Expected values were
// computed with C libbf.
use libbeef::format::ExpBits;
use libbeef::{BigFloat, BigFormat, Precision, Rounding, Sign, Status};
use std::sync::mpsc;
use std::time::Duration;

fn fmt(bits: u64, rounding: Rounding, exp_bits: ExpBits) -> BigFormat {
    BigFormat {
        precision: Precision::Bits(bits),
        rounding,
        exp_bits,
        ..BigFormat::default()
    }
}

/// Runs `pow` on a separate thread so that a regression fails the test
/// instead of hanging it.
fn pow(x: f64, y: f64, format: BigFormat) -> (BigFloat, Status) {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(BigFloat::from_f64(x).pow_status(&BigFloat::from_f64(y), format));
    });
    rx.recv_timeout(Duration::from_secs(10))
        .unwrap_or_else(|_| panic!("pow({x}, {y}) did not finish in time"))
}

#[track_caller]
fn check_finite(r: &(BigFloat, Status), status: Status, negative: bool, exp: i64, limbs: &[u64]) {
    assert_eq!(r.1, status);
    assert_eq!(r.0.sign() == Sign::Negative, negative);
    assert_eq!(r.0.raw_exp(), exp);
    assert_eq!(r.0.limbs(), limbs);
}

const INEXACT: Status = Status::INEXACT;
const OVERFLOW: Status = Status::from_bits_retain(Status::OVERFLOW.bits() | Status::INEXACT.bits());
const UNDERFLOW: Status =
    Status::from_bits_retain(Status::UNDERFLOW.bits() | Status::INEXACT.bits());

#[test]
fn pow_large_exponent_max_exp_range() {
    use Rounding::*;
    let max = ExpBits::Max;
    check_finite(
        &pow(3.0, 1e10, fmt(53, NearestEven, max)),
        INEXACT,
        false,
        15849625008,
        &[0x94376998857a1000],
    );
    check_finite(
        &pow(3.0, -1e10, fmt(53, NearestEven, max)),
        INEXACT,
        false,
        -15849625007,
        &[0xdd15023cdc210000],
    );
    check_finite(
        &pow(3.0, 2147483647.0, fmt(53, NearestEven, max)),
        INEXACT,
        false,
        3403681052,
        &[0xa9b52eda043ea000],
    );
    check_finite(
        &pow(3.0, 2147483647.0, fmt(128, TowardZero, max)),
        INEXACT,
        false,
        3403681052,
        &[0xb4e96b066264773b, 0xa9b52eda043ea14f],
    );
    check_finite(
        &pow(-3.0, 1e10 + 1.0, fmt(53, NearestEven, max)),
        INEXACT,
        true,
        15849625009,
        &[0xde531e64c8371800],
    );
    check_finite(
        &pow(1.0000001, 1e15, fmt(53, NearestEven, max)),
        INEXACT,
        false,
        144269497,
        &[0xf8f0b5c0a4090000],
    );
}

#[test]
fn pow_large_exponent_overflow_underflow() {
    use Rounding::*;
    let e11 = ExpBits::Bits(11);
    let max_finite: &[u64] = &[0xfffffffffffff800];

    let r = pow(3.0, 1e10, fmt(53, NearestEven, e11));
    assert_eq!(r.1, OVERFLOW);
    assert!(r.0.is_infinite() && r.0.sign() == Sign::Positive);
    check_finite(
        &pow(3.0, 1e10, fmt(53, TowardZero, e11)),
        OVERFLOW,
        false,
        1024,
        max_finite,
    );
    check_finite(
        &pow(-3.0, 1e10 + 1.0, fmt(53, TowardZero, e11)),
        OVERFLOW,
        true,
        1024,
        max_finite,
    );

    let r = pow(3.0, -1e10, fmt(53, NearestEven, e11));
    assert_eq!(r.1, UNDERFLOW);
    assert!(r.0.is_zero() && r.0.sign() == Sign::Positive);
    check_finite(
        &pow(3.0, -1e10, fmt(53, TowardPositive, e11)),
        UNDERFLOW,
        false,
        -1073, // smallest subnormal: BigFormat::default() enables subnormals
        &[1 << 63],
    );

    let r = pow(-3.0, 1e10 + 1.0, fmt(53, NearestEven, e11));
    assert_eq!(r.1, OVERFLOW);
    assert!(r.0.is_infinite() && r.0.sign() == Sign::Negative);

    let r = pow(3.0, 1000.0, fmt(53, NearestEven, e11));
    assert_eq!(r.1, OVERFLOW);
    assert!(r.0.is_infinite());
}

#[test]
fn pow_small_integer_exponent_stays_exact() {
    // 3^40 needs exactly 64 bits; (-1.5)^-3 = -8/27 is inexact.
    for rnd in [
        Rounding::NearestEven,
        Rounding::TowardZero,
        Rounding::TowardPositive,
    ] {
        let r = pow(3.0, 40.0, fmt(64, rnd, ExpBits::Bits(11)));
        assert_eq!(r.1, Status::empty());
        assert_eq!(r.0.get_uint64(), Some(12157665459056928801));
        let r = pow(-1.5, -3.0, fmt(53, rnd, ExpBits::Bits(11)));
        assert_eq!(r.1, INEXACT);
        let expected =
            BigFloat::from_i64(-8).div(&BigFloat::from_i64(27), fmt(53, rnd, ExpBits::Bits(11)));
        assert_eq!(r.0.cmp_total(&expected), core::cmp::Ordering::Equal);
    }
}
