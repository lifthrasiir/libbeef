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
