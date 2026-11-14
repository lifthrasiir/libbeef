// Regression tests for issue #4: argument reduction for sin/cos/tan must keep
// the low bits of the quotient by pi/2 even when it is far beyond i64. Earlier
// versions read the quotient through f64, which saturated (or lost its low
// bits above 2^53), so the wrong quadrant was selected and the Ziv loop in
// `sin` never converged for arguments like 1e300.
use libbeef::{BigFloat, BigFormat, Precision, Rounding};

fn fmt() -> BigFormat {
    BigFormat {
        precision: Precision::Bits(128),
        ..BigFormat::default()
    }
}

/// The first 36 significant digits, enough to pin the quadrant and value
/// while ignoring formatting differences in the last digits.
fn digits(v: &BigFloat) -> String {
    sig36(
        &v.to_string_radix(10, 40, Rounding::TowardZero, false)
            .unwrap(),
    )
}

fn sig36(s: &str) -> String {
    let mut n = 0;
    let mut seen = false;
    s.chars()
        .take_while(|&c| {
            if c.is_ascii_digit() {
                seen |= c != '0';
                n += seen as usize;
            }
            n <= 36
        })
        .collect()
}

/// (x, sin x, cos x, tan x) at 128 bits, computed with C libbf.
const CASES: &[(f64, &str, &str, &str)] = &[
    (
        1e22,
        "-0.8522008497671888017727058937530293682616",
        "0.5232147853951389454975944733847094921399",
        "-1.628778225606898878549375936939548513543",
    ),
    (
        1e100,
        "-0.3806377310050286660718709233321073032216",
        "0.9247242387519337858793439735742483788226",
        "-0.4116229628832497988834983009940174394892",
    ),
    (
        1e300,
        "-0.8178819121159085970458852827554262120104",
        "-0.5753861119575490466882442759658061506354",
        "1.421448823874724412366840883947823882437",
    ),
    (
        -1e300,
        "0.8178819121159085970458852827554262120104",
        "-0.5753861119575490466882442759658061506354",
        "-1.421448823874724412366840883947823882437",
    ),
    (
        f64::MAX,
        "0.004961954789184061790502671197074705750765",
        "-0.9999876894265599374648700663069343951403",
        "-0.004962015874444894900500884336163230418559",
    ),
    (
        36028797018963968.0, // 2^55
        "-0.7920784407908279816464686391409708666909",
        "-0.6104193178745007668131122477967428487743",
        "1.297597270592401933274355472741960776769",
    ),
];

#[test]
fn sin_large() {
    for &(x, s, _, _) in CASES {
        assert_eq!(
            digits(&BigFloat::from_f64(x).sin(fmt())),
            sig36(s),
            "sin({x:e})"
        );
    }
}

#[test]
fn cos_large() {
    for &(x, _, c, _) in CASES {
        assert_eq!(
            digits(&BigFloat::from_f64(x).cos(fmt())),
            sig36(c),
            "cos({x:e})"
        );
    }
}

#[test]
fn tan_large() {
    for &(x, _, _, t) in CASES {
        assert_eq!(
            digits(&BigFloat::from_f64(x).tan(fmt())),
            sig36(t),
            "tan({x:e})"
        );
    }
}
