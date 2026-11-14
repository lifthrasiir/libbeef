// Regression tests for `pow` cases whose result is exactly representable
// although the exponent is not an integer. libbf handles them through
// `check_exact_power2n`; without it the Ziv loop never converges under the
// directed rounding modes, because the exact result lies on a rounding
// boundary.
use libbeef::{BigFloat, BigFormat, Precision, Rounding, Status};

const ROUNDINGS: [Rounding; 6] = [
    Rounding::NearestEven,
    Rounding::TowardZero,
    Rounding::TowardNegative,
    Rounding::TowardPositive,
    Rounding::NearestAway,
    Rounding::AwayFromZero,
];

fn fmt(bits: u64, rounding: Rounding) -> BigFormat {
    BigFormat {
        precision: Precision::Bits(bits),
        rounding,
        ..BigFormat::default()
    }
}

fn f(v: f64) -> BigFloat {
    BigFloat::from_f64(v)
}

#[test]
fn pow_exact_non_integer_exponent() {
    let cases = [
        (9.0, 0.5, 3.0),
        (2.25, 0.5, 1.5),
        (9.0, 1.5, 27.0),
        (81.0, 0.25, 3.0),
        (81.0, 0.75, 27.0),
        (6561.0, 0.125, 3.0),
        (0.5625, 0.5, 0.75),
        (12.25, 2.5, 525.21875), // 3.5^5
    ];
    for bits in [24, 53, 128, 1000] {
        for rnd in ROUNDINGS {
            for (x, y, expected) in cases {
                let (r, status) = f(x).pow_status(&f(y), fmt(bits, rnd));
                assert_eq!(
                    r.to_f64(Rounding::NearestEven),
                    expected,
                    "pow({x}, {y}) {bits} {rnd:?}"
                );
                assert_eq!(status, Status::empty(), "pow({x}, {y}) {bits} {rnd:?}");
            }
        }
    }
}

#[test]
fn pow_inexact_after_exact_root() {
    // 9^-0.5 = 1/3: the root is exact but the integer power is not.
    for bits in [24, 53, 128] {
        for rnd in ROUNDINGS {
            let format = fmt(bits, rnd);
            let (r, status) = f(9.0).pow_status(&f(-0.5), format);
            let (expected, expected_status) = f(1.0).div_status(&f(3.0), format);
            assert_eq!(
                r.cmp_total(&expected),
                core::cmp::Ordering::Equal,
                "{bits} {rnd:?}"
            );
            assert_eq!(status, expected_status, "{bits} {rnd:?}");
        }
    }
}

#[test]
fn pow_not_exact_power() {
    // Neither is a perfect square; must still match sqrt.
    for bits in [24, 53, 128] {
        for rnd in ROUNDINGS {
            let format = fmt(bits, rnd);
            for x in [2.0, 3.0, 10.0, 0.3] {
                let r = f(x).pow(&f(0.5), format);
                let expected = f(x).sqrt(format);
                assert_eq!(
                    r.cmp_total(&expected),
                    core::cmp::Ordering::Equal,
                    "pow({x}, 0.5) {bits} {rnd:?}"
                );
            }
        }
    }
}
