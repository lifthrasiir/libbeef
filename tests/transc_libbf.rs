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

fn inf_fmt(rounding: Rounding, exp_bits: ExpBits) -> BigFormat {
    BigFormat {
        precision: Precision::Infinite,
        rounding,
        exp_bits,
        subnormal: false,
        radix_point_precision: false,
    }
}

fn run_with_timeout<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(10))
        .expect("did not finish in time")
}

#[test]
fn pow_infinite_precision_overflow() {
    // libbf checks overflow and `y >= 2^61` before any exact power.
    for rounding in [Rounding::NearestEven, Rounding::TowardZero] {
        for base in [3, 5] {
            let (r, st) = run_with_timeout(move || {
                BigFloat::from_i64(base).pow_status(&pow2(62), inf_fmt(rounding, ExpBits::Bits(61)))
            });
            assert!(r.is_infinite() && !r.sign().is_negative(), "{r:?}");
            assert_eq!(st, Status::OVERFLOW | Status::INEXACT);
        }
    }
}

#[test]
fn pow_infinite_precision_power_of_two() {
    // pow(2^b, y) = 2^(b*y), with the range check of mul_2exp.
    let (r, st) = run_with_timeout(|| {
        BigFloat::from_i64(2).pow_status(&pow2(62), inf_fmt(Rounding::NearestEven, ExpBits::Max))
    });
    assert!(r.is_infinite() && !r.sign().is_negative());
    assert_eq!(st, Status::OVERFLOW | Status::INEXACT);
    let (r, st) = BigFloat::from_i64(2).pow_status(
        &BigFloat::from_i64(100),
        inf_fmt(Rounding::NearestEven, ExpBits::Max),
    );
    assert_eq!(st, Status::empty());
    assert_eq!(r, pow2(100));
}

#[test]
fn small_argument_epsilon_status() {
    // bf_add_epsilon returns the status of bf_add, so a subnormal or
    // underflowing result reports UNDERFLOW.
    let x = BigFloat::from_raw(Sign::Positive, -1073, vec![1 << 63]); // 2^-1074
    let f = BigFormat::BINARY64;
    let under = Status::UNDERFLOW | Status::INEXACT;
    let (r, st) = x.sin_status(f);
    assert_eq!((r.raw_exp(), st), (-1073, under));
    assert_eq!(x.tan_status(f).1, under);
    assert_eq!(x.asin_status(f).1, under);
    assert_eq!(x.neg().atan_status(f).1, under);
    // Without subnormals the result flushes to zero.
    let nosub = BigFormat {
        subnormal: false,
        ..f
    };
    let (r, st) = x.sin_status(nosub);
    assert!(r.is_zero());
    assert_eq!(st, under);
}
