// `{:e}` / `{:E}` formatting of `BigFloat`, checked against Rust's own `f64`
// formatting (which is exact for values with at most 53 significant bits).
use libbeef::parse::ParseOptions;
use libbeef::{BigFloat, BigFormat, Precision};

const VALUES: &[f64] = &[
    1.5,
    0.1,
    3.0,
    48.0,
    -2.5,
    100.0,
    1e300,
    -1.2345e-300,
    123456789.0,
    0.000123,
    1.0 / 3.0,
    f64::MAX,
    f64::MIN_POSITIVE,
    0.0,
    -0.0,
];

#[test]
fn lower_exp_without_precision_is_shortest() {
    for &x in VALUES {
        assert_eq!(
            format!("{:e}", BigFloat::from_f64(x)),
            format!("{x:e}"),
            "{x:e}"
        );
        assert_eq!(
            format!("{:E}", BigFloat::from_f64(x)),
            format!("{x:E}"),
            "{x:E}"
        );
    }
}

#[test]
fn exp_with_precision_has_that_many_fractional_digits() {
    for &x in VALUES {
        for p in [0, 1, 3, 10, 20] {
            let b = BigFloat::from_f64(x);
            assert_eq!(format!("{b:.p$e}"), format!("{x:.p$e}"), "{x:e} .{p}");
            assert_eq!(format!("{b:.p$E}"), format!("{x:.p$E}"), "{x:e} .{p}");
        }
    }
}

#[test]
fn exp_honours_width_and_sign() {
    for &x in &[1.5, -1.5, 0.0, 1e100] {
        let b = BigFloat::from_f64(x);
        assert_eq!(format!("{b:12e}"), format!("{x:12e}"));
        assert_eq!(format!("{b:<12e}|"), format!("{x:<12e}|"));
        assert_eq!(format!("{b:^+12.2e}"), format!("{x:^+12.2e}"));
        assert_eq!(format!("{b:012.2e}"), format!("{x:012.2e}"));
    }
}

#[test]
fn exp_of_non_finite_values() {
    assert_eq!(format!("{:e}", BigFloat::nan()), "NaN");
    assert_eq!(format!("{:.3E}", BigFloat::nan()), "NaN");
    assert_eq!(format!("{:e}", BigFloat::from_f64(f64::INFINITY)), "Inf");
    assert_eq!(
        format!("{:e}", BigFloat::from_f64(f64::NEG_INFINITY)),
        "-Inf"
    );
}

#[test]
fn exp_of_wide_values_uses_their_precision() {
    let fmt200 = BigFormat {
        precision: Precision::Bits(200),
        exp_bits: libbeef::format::ExpBits::Max,
        ..BigFormat::BINARY128
    };
    let x = BigFloat::from_f64(1.25).div(&BigFloat::from_f64(3.0), fmt200);
    let s = format!("{x:e}");
    // Many more digits than an `f64` needs, and they read back as `x`.
    assert!(s.len() > 40, "{s}");
    let opts = ParseOptions {
        format: fmt200,
        ..ParseOptions::default()
    };
    let (back, _) = BigFloat::parse(&s, opts).unwrap();
    assert_eq!(back, x, "{s}");
    assert_eq!(format!("{x:.20e}"), "4.16666666666666666667e-1");
}
