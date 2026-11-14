// Regression tests for string parsing (`bf_atof`). Expected values were
// produced by C libbf; each line is `<value> st=<status> n=<consumed>` where
// the value is `<sign><raw exp>:<limbs, most significant first>`.
use libbeef::format::ExpBits;
use libbeef::parse::ParseOptions;
use libbeef::{BigFloat, BigFormat, Precision, Rounding};

fn pr(a: &BigFloat) -> String {
    if a.is_nan() {
        return "NaN".into();
    }
    let mut s = String::from(if a.sign().is_negative() { "-" } else { "+" });
    if a.is_infinite() {
        s += "Inf";
    } else if a.is_zero() {
        s += "0";
    } else {
        s += &format!("{}:", a.raw_exp());
        let l = a.limbs();
        let lo = l.iter().take_while(|&&x| x == 0).count();
        for x in l[lo..].iter().rev() {
            s += &format!("{x:016x}");
        }
    }
    s
}

fn opts(radix: u8, prec: Option<u64>, rounding: Rounding) -> ParseOptions {
    ParseOptions {
        radix,
        format: BigFormat {
            precision: prec.map_or(Precision::Infinite, Precision::Bits),
            rounding,
            exp_bits: ExpBits::Max,
            subnormal: false,
            radix_point_precision: false,
        },
        ..ParseOptions::default()
    }
}

/// Parses with the given options and formats like the C harness.
fn show_with(o: ParseOptions, s: &str) -> String {
    match BigFloat::parse_status(s, o) {
        Ok((v, st, n)) => format!("{} st={} n={}", pr(&v), st.bits(), n),
        Err(_) => "ERR".into(),
    }
}

fn show(radix: u8, prec: u64, s: &str) -> String {
    show_with(opts(radix, Some(prec), Rounding::NearestEven), s)
}

#[test]
fn huge_exponents() {
    // Exponent accumulation stops once it exceeds (i64::MAX - 11) / 10.
    assert_eq!(show(10, 53, "1e-9999999999999999999"), "+0 st=24 n=21");
    assert_eq!(show(10, 53, "1e9999999999999999999"), "+Inf st=20 n=20");
    assert_eq!(show(10, 53, "-1e99999999999999999999x"), "-Inf st=20 n=21");
    assert_eq!(show(16, 53, "1p99999999999999999999"), "+Inf st=20 n=20");
    assert_eq!(show(16, 53, "1@9999999999999999999"), "+Inf st=20 n=20");
    // C wraps `expn * radix_bits` here (undefined behaviour) and returns +Inf;
    // we saturate, giving the mathematically right underflow.
    assert_eq!(show(16, 53, "1@-3000000000000000000"), "+0 st=24 n=22");
    assert_eq!(show(2, 53, "1p-99999999999999999999"), "+0 st=24 n=21");
    assert_eq!(show(10, 53, "0e99999999999999999999"), "+Inf st=20 n=20");
    assert_eq!(show(16, 53, "1@300000000000000000"), "+Inf st=20 n=20");
}

#[test]
fn large_but_valid_exponents() {
    assert_eq!(show(10, 53, "1e1000000000000000000"), "+Inf st=16 n=21");
    assert_eq!(
        show(10, 53, "1e-300000000000000000"),
        "+-996578428466208704:c7508ef40be09800 st=16 n=21"
    );
    assert_eq!(show(2, 53, "1p-4611686018427387000"), "+0 st=24 n=22");
    assert_eq!(
        show(2, 53, "1p461168601842738700"),
        "+461168601842738701:8000000000000000 st=0 n=20"
    );
    assert_eq!(
        show(32, 53, "1@-46116860184273870"),
        "+-230584300921369349:8000000000000000 st=0 n=20"
    );
}

#[test]
fn exact_parses_are_not_inexact() {
    let ne = |s: &str| show(10, 53, s);
    assert_eq!(ne(".5"), "+0:8000000000000000 st=0 n=2");
    assert_eq!(ne("5."), "+3:a000000000000000 st=0 n=2");
    assert_eq!(ne("1e5"), "+17:c350000000000000 st=0 n=3");
    assert_eq!(ne("1e1"), "+4:a000000000000000 st=0 n=3");
    assert_eq!(ne("1.5e3"), "+11:bb80000000000000 st=0 n=5");
    assert_eq!(ne("0.5"), "+0:8000000000000000 st=0 n=3");
    // Inexact parses still report INEXACT.
    assert_eq!(ne("0.1"), "+-3:ccccccccccccd000 st=16 n=3");
    assert_eq!(ne("1e30"), "+100:c9f2c9cd04675000 st=16 n=4");
}

#[test]
fn exact_parses_with_directed_rounding() {
    let o = |r| opts(10, Some(53), r);
    assert_eq!(
        show_with(o(Rounding::TowardZero), "0.5"),
        "+0:8000000000000000 st=0 n=3"
    );
    assert_eq!(
        show_with(o(Rounding::TowardNegative), "1.5e3"),
        "+11:bb80000000000000 st=0 n=5"
    );
    assert_eq!(
        show_with(o(Rounding::TowardPositive), "1e22"),
        "+74:878678326eac9000 st=0 n=4"
    );
    assert_eq!(
        show_with(o(Rounding::TowardPositive), "1e30"),
        "+100:c9f2c9cd04675000 st=16 n=4"
    );
    assert_eq!(
        show_with(o(Rounding::TowardZero), "1e30"),
        "+100:c9f2c9cd04674800 st=16 n=4"
    );
}

/// Exact powers of ten under directed rounding. The intermediate power of the
/// radix needs more bits than any fixed cap on the Ziv loop provides.
#[test]
fn ftoa_exact_value_directed_rounding() {
    let x: BigFloat = "1e50000".parse().unwrap();
    for r in [
        Rounding::TowardZero,
        Rounding::TowardNegative,
        Rounding::TowardPositive,
    ] {
        assert_eq!(x.to_string_radix(10, 1, r, true).unwrap(), "1e50000");
    }
    let y: BigFloat = "1e-50000".parse().unwrap();
    // 10^-50000 is not exactly representable; just check it rounds sensibly.
    let s = y
        .to_string_radix(10, 1, Rounding::NearestEven, true)
        .unwrap();
    assert_eq!(s, "1e-50000");
}

#[test]
fn infinite_precision_reports_inexact_division() {
    let inf = |radix, s| show_with(opts(radix, None, Rounding::NearestEven), s);
    assert_eq!(inf(10, "0.1"), "+-3:cccccccccccccccd st=16 n=3");
    assert_eq!(inf(10, "1e-3"), "+-9:83126e978d4fdf3b st=16 n=4");
    assert_eq!(inf(3, "1@-1"), "+-1:aaaaaaaaaaaaaaab st=16 n=4");
    assert_eq!(inf(10, "1.5"), "+1:c000000000000000 st=0 n=3");
    assert_eq!(inf(10, "125e-2"), "+1:a000000000000000 st=0 n=6");
    assert_eq!(inf(10, "1e5"), "+17:c350000000000000 st=0 n=3");
}

#[test]
fn negative_zero() {
    for (radix, s, n) in [
        (10, "-0.0", 4),
        (10, "-00.000e5", 9),
        (10, "-.0", 3),
        (10, "-0", 2),
        (10, "-0e5", 4),
        (3, "-0.0@3", 6),
        (16, "-0.0p3", 6),
        (2, "-.0", 3),
    ] {
        assert_eq!(show(radix, 53, s), format!("-0 st=0 n={n}"), "{s}");
    }
    assert_eq!(
        show_with(opts(10, None, Rounding::NearestEven), "-0.0"),
        "-0 st=0 n=4"
    );
    assert_eq!(show(10, 53, "+0.0"), "+0 st=0 n=4");
    assert!("-0.0".parse::<BigFloat>().unwrap().sign().is_negative());
    assert!(BigFloat::from_str_radix("-0.0@3", 3)
        .unwrap()
        .sign()
        .is_negative());
}

#[test]
fn zero_with_exponent_in_power_of_two_radix() {
    assert_eq!(show(2, 53, "0p5"), "+0 st=0 n=3");
    assert_eq!(show(16, 53, "0@5"), "+0 st=0 n=3");
    assert_eq!(show(2, 53, "-0p5"), "-0 st=0 n=4");
    assert_eq!(show(16, 53, "0p5x"), "+0 st=0 n=3");
    assert_eq!(show(16, 53, "00@-5z"), "+0 st=0 n=5");
    assert_eq!(show(4, 53, "0@5"), "+0 st=0 n=3");
    assert_eq!(show(2, 53, "0p"), "+0 st=0 n=2");
    assert_eq!(show(2, 53, "0p+"), "+0 st=0 n=3");
    assert_eq!(show(2, 53, "0.p5"), "+0 st=0 n=4");
    assert_eq!(show(2, 53, "-0.p5"), "-0 st=0 n=5");
    // The exponent overflow check applies to zeros as well.
    assert_eq!(show(16, 53, "0@99999999999999999999"), "+Inf st=20 n=20");
    assert_eq!(show(2, 53, "0p-99999999999999999999"), "+0 st=24 n=21");
    assert_eq!(show(2, 53, ".p5"), "ERR");
    assert!(BigFloat::from_str_radix("0p5", 2).unwrap().is_zero());
}

/// Options like libbf's flags: `no_hex` is `BF_ATOF_NO_HEX`, `bin_oct` is
/// `BF_ATOF_BIN_OCT`.
fn flag_opts(radix: u8, no_hex: bool, bin_oct: bool) -> ParseOptions {
    ParseOptions {
        allow_hex_prefix: !no_hex,
        allow_binary_prefix: bin_oct,
        allow_octal_prefix: bin_oct,
        ..opts(radix, Some(53), Rounding::NearestEven)
    }
}

#[test]
fn prefix_needs_a_digit() {
    let o = flag_opts(0, false, true);
    // A prefix without a following digit is NaN (consuming the prefix).
    assert_eq!(show_with(o, "0b.1"), "NaN st=0 n=2");
    assert_eq!(show_with(o, "0b"), "NaN st=0 n=2");
    assert_eq!(show_with(o, "0o8"), "NaN st=0 n=2");
    assert_eq!(show_with(o, "0x"), "NaN st=0 n=2");
    assert_eq!(show_with(o, "0x.8"), "NaN st=0 n=2");
    assert_eq!(show_with(o, "0b1"), "+1:8000000000000000 st=0 n=3");
    assert_eq!(show_with(o, "0b102"), "+2:8000000000000000 st=0 n=4");
    assert_eq!(show_with(o, "0o17"), "+4:f000000000000000 st=0 n=4");
    assert_eq!(show_with(o, "-0x1p4"), "-5:8000000000000000 st=0 n=6");
    assert_eq!(
        show_with(flag_opts(16, false, false), "+0x.8"),
        "NaN st=0 n=3"
    );
}

#[test]
fn prefix_depends_on_radix() {
    // 0x needs radix 0 or 16, 0b and 0o need radix 0.
    for radix in [2, 8, 10] {
        assert_eq!(
            show_with(flag_opts(radix, false, true), "0x1f"),
            "+0 st=0 n=1"
        );
    }
    assert_eq!(show_with(flag_opts(10, false, true), "0b1"), "+0 st=0 n=1");
    assert_eq!(show_with(flag_opts(2, false, true), "0b1"), "+0 st=0 n=1");
    assert_eq!(show_with(flag_opts(2, false, true), "0o7"), "+0 st=0 n=1");
    assert_eq!(
        show_with(flag_opts(36, false, false), "0x1"),
        "+11:94a0000000000000 st=0 n=3"
    );
    // In radix 16, `0b1` is just the hex number 0xb1.
    assert_eq!(
        show_with(flag_opts(16, false, true), "0b1"),
        "+8:b100000000000000 st=0 n=3"
    );
    assert_eq!(
        show_with(flag_opts(16, false, false), "0x1f"),
        "+5:f800000000000000 st=0 n=4"
    );
    assert_eq!(show_with(flag_opts(16, true, false), "0x1f"), "+0 st=0 n=1");
}

#[test]
fn radix_zero_is_auto() {
    assert_eq!(
        show_with(flag_opts(0, false, false), "0x1f"),
        "+5:f800000000000000 st=0 n=4"
    );
    assert_eq!(show_with(flag_opts(0, true, false), "0x1f"), "+0 st=0 n=1");
    assert_eq!(show_with(flag_opts(0, false, false), "0b1"), "+0 st=0 n=1");
    assert_eq!(
        show_with(flag_opts(0, false, false), "1e2"),
        "+7:c800000000000000 st=0 n=3"
    );
    assert_eq!(
        show_with(flag_opts(0, false, false), "12.5"),
        "+4:c800000000000000 st=0 n=4"
    );
    assert_eq!(
        show_with(flag_opts(0, false, false), "inf"),
        "+Inf st=0 n=3"
    );
    assert_eq!(
        show_with(ParseOptions::auto_radix(), "0b101"),
        "+3:a000000000000000 st=0 n=5"
    );
}

#[test]
fn invalid_radix_is_an_error() {
    for radix in [1, 37, 255] {
        assert_eq!(show_with(flag_opts(radix, false, false), "12"), "ERR");
        assert_eq!(show_with(flag_opts(radix, false, false), "nan"), "ERR");
    }
}

/// Like `show_with`, with the radix exponent appended (libbf's `bf_atof2`).
fn show_exp(radix: u8, prec: Option<u64>, s: &str) -> String {
    let o = ParseOptions {
        return_radix_exponent: true,
        ..opts(radix, prec, Rounding::NearestEven)
    };
    match BigFloat::parse_with_exponent_status(s, o) {
        Ok((v, e, st, n)) => format!("{} st={} n={} e={}", pr(&v), st.bits(), n, e),
        Err(_) => "ERR".into(),
    }
}

#[test]
fn return_radix_exponent() {
    let m = "+61:890e99acb9c20000";
    assert_eq!(
        show_exp(10, Some(53), "1.2345"),
        format!("{m} st=0 n=6 e=-18")
    );
    assert_eq!(
        show_exp(10, Some(53), "-1.2345e10"),
        "-61:890e99acb9c20000 st=0 n=10 e=-8"
    );
    assert_eq!(
        show_exp(10, None, "0.1"),
        "+60:de0b6b3a76400000 st=0 n=3 e=-19"
    );
    assert_eq!(
        show_exp(10, Some(5), "123456789"),
        "+61:891087a3b46da000 st=0 n=9 e=-10"
    );
    assert_eq!(show_exp(10, Some(53), "0"), "+0 st=0 n=1 e=0");
    assert_eq!(show_exp(10, Some(53), "-0.0"), "-0 st=0 n=4 e=-19");
    assert_eq!(show_exp(10, Some(53), "0e5"), "+0 st=0 n=3 e=0");
    assert_eq!(
        show_exp(10, Some(53), "1e-400"),
        "+60:de0b6b3a76400000 st=0 n=6 e=-418"
    );
    assert_eq!(
        show_exp(3, Some(53), "12.1@5"),
        "+63:c7f75a7456aa8e60 st=0 n=6 e=-33"
    );
    assert_eq!(
        show_exp(10, Some(53), ".5"),
        "+63:8ac7230489e80000 st=0 n=2 e=-19"
    );
    assert_eq!(
        show_exp(10, Some(53), "12345678901234567890123.5e7"),
        "+124:949b0f6f0023313c449b79db5cac0000 st=0 n=27 e=-8"
    );
    // Power-of-two radixes are scaled normally; the exponent is 0.
    assert_eq!(
        show_exp(16, Some(53), "ff.8"),
        "+8:ff80000000000000 st=0 n=4 e=0"
    );
    assert_eq!(
        show_exp(16, Some(53), "ff.8p3"),
        "+11:ff80000000000000 st=0 n=6 e=0"
    );
    assert_eq!(
        show_exp(2, Some(53), "101.1"),
        "+3:b000000000000000 st=0 n=5 e=0"
    );
    // Special values and overflowing exponents.
    assert_eq!(
        show_exp(10, Some(53), "1e99999999999999999999"),
        "+Inf st=20 n=20 e=0"
    );
    assert_eq!(show_exp(10, Some(53), "inf"), "+Inf st=0 n=3 e=0");
}

#[test]
fn radix_exponent_option_without_exponent_output() {
    // `parse_status` with the option set returns the unscaled mantissa.
    let o = ParseOptions {
        return_radix_exponent: true,
        ..opts(10, Some(53), Rounding::NearestEven)
    };
    assert_eq!(show_with(o, "1.2345"), "+61:890e99acb9c20000 st=0 n=6");
    // Without the option the exponent is always 0 and the value is scaled.
    let o = opts(10, Some(53), Rounding::NearestEven);
    let (v, e, _, _) = BigFloat::parse_with_exponent_status("1.2345", o).unwrap();
    assert_eq!(e, 0);
    assert_eq!(v.to_f64(Rounding::NearestEven), 1.2345);
}
