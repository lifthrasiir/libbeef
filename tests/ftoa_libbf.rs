// Regression tests for number-to-string conversion (`bf_ftoa`). Expected
// strings were produced by C libbf.
use libbeef::{BigFloat, Rounding, Sign};

fn rnd(r: u8) -> Rounding {
    match r {
        0 => Rounding::NearestEven,
        1 => Rounding::TowardZero,
        2 => Rounding::TowardNegative,
        3 => Rounding::TowardPositive,
        4 => Rounding::NearestAway,
        5 => Rounding::AwayFromZero,
        _ => Rounding::Faithful,
    }
}

fn make(kind: &str, sign: u8, expn: i64, hex: &str) -> BigFloat {
    let sign = Sign::from_negative(sign == 1);
    match kind {
        "N" => BigFloat::nan(),
        "I" => BigFloat::infinity(sign),
        "Z" => BigFloat::zero(sign),
        _ => {
            let nl = hex.len() / 16;
            let mut limbs = vec![0u64; nl];
            for i in 0..nl {
                limbs[nl - 1 - i] = u64::from_str_radix(&hex[i * 16..i * 16 + 16], 16).unwrap();
            }
            BigFloat::from_raw(sign, expn, limbs)
        }
    }
}

type Case = (
    u8,
    u64,
    u8,
    bool,
    &'static str,
    u8,
    i64,
    &'static str,
    &'static str,
);

/// (radix, digits, rounding, force_exp, kind, sign, raw exp, limbs, expected)
const FIXED_POW2: &[Case] = &[
    (16, 1, 0, false, "V", 0, 3, "8000000000000000", "4"),
    (16, 4, 0, false, "V", 0, 3, "8000000000000000", "4.000"),
    (16, 1, 0, false, "V", 0, 8, "ff00000000000000", "1p8"),
    (16, 3, 0, false, "V", 0, 12, "1230000000000000", "123"),
    (2, 4, 0, false, "V", 0, 3, "a000000000000000", "101.0"),
    (32, 2, 0, false, "V", 0, 12, "8000000000000000", "2.0@2"),
    (32, 3, 0, true, "Z", 0, 0, "0", "0.00@0"),
    (16, 3, 0, true, "V", 0, 12, "1230000000000000", "1.23p8"),
    (16, 2, 0, false, "V", 1, -3, "c000000000000000", "-0.18"),
    (2, 3, 1, false, "V", 0, 3, "ffff000000000000", "111"),
    (8, 3, 3, false, "V", 0, 5, "ffff000000000000", "40.0"),
    (
        4,
        5,
        0,
        false,
        "V",
        0,
        -9,
        "9000000000000000",
        "0.000010200",
    ),
    (16, 1, 0, false, "Z", 1, 0, "0", "-0"),
    (16, 3, 0, false, "Z", 0, 0, "0", "0.00"),
];

#[test]
fn to_string_radix_power_of_two_matches_libbf() {
    for &(radix, prec, r, force, kind, sign, expn, hex, expected) in FIXED_POW2 {
        let a = make(kind, sign, expn, hex);
        let s = a.to_string_radix(radix, prec, rnd(r), force).unwrap();
        assert_eq!(
            s, expected,
            "radix={radix} prec={prec} rnd={r} force={force} {kind} {sign} {expn} {hex}"
        );
    }
}

const FIXED_GENERAL: &[Case] = &[
    (10, 3, 0, false, "V", 0, 14, "c0e4000000000000", "1.23e4"),
    (
        12,
        25,
        0,
        false,
        "V",
        0,
        101,
        "8000000000000000",
        "9.289b0aa342b15786599765a9@27",
    ),
    (10, 3, 0, false, "V", 0, -3, "8000000000000000", "0.0625"),
    (10, 5, 1, false, "V", 0, 4, "b000000000000000", "11.000"),
    (
        7,
        6,
        0,
        false,
        "V",
        1,
        40,
        "abcdef0123456789",
        "-1.04212e14",
    ),
];

/// The fixed format switches to exponential notation as soon as the exponent
/// exceeds the number of digits.
#[test]
fn to_string_radix_general_matches_libbf() {
    for &(radix, prec, r, force, kind, sign, expn, hex, expected) in FIXED_GENERAL {
        let a = make(kind, sign, expn, hex);
        let s = a.to_string_radix(radix, prec, rnd(r), force).unwrap();
        assert_eq!(
            s, expected,
            "radix={radix} prec={prec} rnd={r} force={force} {kind} {sign} {expn} {hex}"
        );
    }
}
