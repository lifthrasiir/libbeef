// Regression tests for issue #2: internal computations must not be limited by
// the exponent range of the *result* format (or of binary64). libbf performs
// intermediate steps with `BF_EXP_BITS_MAX`; earlier versions of libbeef used
// an 11-bit exponent instead, so intermediates beyond ~2^1024 overflowed.
use libbeef::{BigFloat, BigFormat, Precision, Rounding};

const PI_50: &str = "3.141592653589793238462643383279502884197169399375";
const LN2_40: &str = "0.6931471805599453094172321214581765680";

fn fmt(bits: u64) -> BigFormat {
    // Deliberately keeps the default (binary64) exponent width, as in the issue.
    BigFormat {
        precision: Precision::Bits(bits),
        ..BigFormat::default()
    }
}

fn inf() -> BigFormat {
    BigFormat {
        precision: Precision::Infinite,
        ..BigFormat::default()
    }
}

/// Constants are cached per thread, so evaluate each on a fresh thread to
/// avoid a cached high-precision value masking a low-precision bug.
fn fresh<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::spawn(f).join().unwrap()
}

fn digits(v: &BigFloat, n: u64) -> String {
    v.to_string_radix(10, n, Rounding::TowardZero, false)
        .unwrap()
}

fn pow_u64(base: u64, exp: u32) -> BigFloat {
    let mut r = BigFloat::from_u64(1);
    for _ in 0..exp {
        r.mul_u64_assign(base, inf());
    }
    r
}

#[test]
fn pi_at_high_precision() {
    for bits in [687_u64, 688, 1024, 4096, 20000] {
        let pi = fresh(move || BigFloat::pi(fmt(bits)));
        assert_eq!(digits(&pi, 49), &PI_50[..50], "pi at {bits} bits");
    }
}

#[test]
fn atan_at_high_precision() {
    let v = fresh(|| BigFloat::from_i64(1).atan(fmt(1024)));
    let pi = v.mul_u64(4, fmt(1024));
    assert_eq!(digits(&pi, 49), &PI_50[..50]);
}

#[test]
fn log2_at_high_precision() {
    for bits in [1024_u64, 3000, 20000] {
        let ln2 = fresh(move || BigFloat::log2(fmt(bits)));
        assert_eq!(digits(&ln2, 37), LN2_40, "ln2 at {bits} bits");
    }
}

#[test]
fn to_string_of_large_exponent() {
    // 3^2000 = 1.7478...e954
    let big = pow_u64(3, 2000);
    let s = big
        .to_string_radix(10, 10, Rounding::TowardZero, true)
        .unwrap();
    assert_eq!(s, "1.747871251e954");
    // 3^-2000, rounded to 200 bits, needs an exponent range beyond binary64
    // only for the input; the conversion must not overflow internally.
    let tiny = BigFloat::from_u64(1).div(
        &big,
        BigFormat {
            exp_bits: libbeef::format::ExpBits::Max,
            ..fmt(200)
        },
    );
    let s = tiny
        .to_string_radix(10, 10, Rounding::TowardZero, true)
        .unwrap();
    assert_eq!(s, "5.721245194e-955");
}

#[test]
fn parse_large_exponent() {
    let v: BigFloat = "1e400".parse().unwrap();
    assert_eq!(v, pow_u64(10, 400));
    let v: BigFloat = "1e-400".parse().unwrap();
    let back = v.mul(
        &pow_u64(10, 400),
        BigFormat {
            exp_bits: libbeef::format::ExpBits::Max,
            ..fmt(53)
        },
    );
    assert_eq!(back.to_f64(Rounding::NearestEven), 1.0);
}

#[test]
fn sqrtrem_of_large_integer() {
    // (3^1000)^2 + 5 has an exponent of ~3170 bits.
    let r = pow_u64(3, 1000);
    let n = r.mul(&r, inf()).add(&BigFloat::from_u64(5), inf());
    let (root, rem, _) = n.sqrtrem();
    assert_eq!(root, r);
    assert_eq!(rem, BigFloat::from_u64(5));
}
