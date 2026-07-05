use beef::format::{formats, StaticFormat};
use beef::{BigDecimal, BigFloat, Decimal, Float, FpCategory, Sign};

#[test]
fn big_float_special_values_are_classified() {
    assert_eq!(BigFloat::new().classify(), FpCategory::Zero);
    assert_eq!(BigFloat::nan().classify(), FpCategory::Nan);
    assert_eq!(
        BigFloat::infinity(Sign::Negative).classify(),
        FpCategory::Infinite
    );
    assert!(BigFloat::zero(Sign::Negative).is_sign_negative());
}

#[test]
fn u64_constructor_matches_libbf_normalization_shape() {
    let one = BigFloat::from_u64(1);
    assert_eq!(one.raw_exp(), 1);
    assert_eq!(one.limbs(), &[1_u64 << 63]);

    let three = BigFloat::from_u64(3);
    assert_eq!(three.raw_exp(), 2);
    assert_eq!(three.limbs(), &[3_u64 << 62]);
}

#[test]
fn f64_constructor_uses_normalized_significand_shape() {
    let half = BigFloat::from_f64(0.5);
    assert_eq!(half.raw_exp(), 0);
    assert_eq!(half.limbs(), &[1_u64 << 63]);

    let one_and_half = BigFloat::from_f64(1.5);
    assert_eq!(one_and_half.raw_exp(), 1);
    assert_eq!(one_and_half.limbs(), &[3_u64 << 62]);
}

#[test]
fn static_format_converts_to_big_format() {
    assert_eq!(formats::Binary64::FORMAT, beef::BigFormat::BINARY64);
    assert_eq!(formats::Binary128::FORMAT, beef::BigFormat::BINARY128);
    assert_eq!(formats::Decimal64::FORMAT, beef::BigFormat::DECIMAL64);
}

#[test]
fn format_wrapper_conversion_is_noop() {
    type F64 = Float<formats::Binary64>;
    type F128 = Float<formats::Binary128>;

    let x = F64::from_big(BigFloat::from_u64(42));
    let y: F128 = x.into_format();
    let raw = y.into_big();

    assert_eq!(raw.raw_exp(), 6);
    assert_eq!(raw.limbs(), &[42_u64 << 58]);
}

#[test]
fn static_float_remainder_uses_its_format() {
    type F64 = Float<formats::Binary64>;

    let a = F64::from_big(BigFloat::from_f64(5.5));
    let b = F64::from_big(BigFloat::from_f64(2.0));
    let (value, status) = (a % b)
        .into_big()
        .to_f64_status(beef::Rounding::NearestEven);

    assert!(status.is_empty());
    assert_eq!(value.to_bits(), (5.5_f64 % 2.0).to_bits());
}

#[test]
fn static_float_transcendental_methods_use_their_format() {
    type F64 = Float<formats::Binary64>;

    let x = F64::from_big(BigFloat::from_f64(0.25));
    let y = x.exp().log();
    let value = y.into_big().to_f64(beef::Rounding::NearestEven);

    assert_eq!(value.to_bits(), 0.25_f64.exp().ln().to_bits());
}

#[test]
fn scaled_decimal_values_are_normalized() {
    assert_eq!(
        BigDecimal::from_scaled_i64(1250, 2),
        BigDecimal::from_scaled_i64(125, 1)
    );
}

#[test]
fn static_decimal_unary_methods_use_their_format() {
    type D64 = Decimal<formats::Decimal64>;

    let x = D64::from_big(BigDecimal::from_scaled_i64(144, 2));
    assert_eq!(x.sqrt().into_big(), BigDecimal::from_scaled_i64(12, 1));

    let y = D64::from_big(BigDecimal::from_scaled_i64(15, 1));
    assert_eq!(
        y.rint(beef::Rounding::NearestEven).into_big(),
        BigDecimal::from_i64(2)
    );
}

#[test]
fn static_decimal_binary_methods_use_digit_precision() {
    type D64 = Decimal<formats::Decimal64>;

    let a = D64::from_big(BigDecimal::from_i64(12_345_678_901_234_567));
    let b = D64::from_big(BigDecimal::from_i64(1));
    let (sum, status) = a.add_status(&b);

    assert_eq!(
        sum.into_big().to_decimal_string().as_deref(),
        Some("12345678901234570")
    );
    assert!(status.contains(beef::Status::INEXACT));
}

#[test]
fn to_string_radix_no_double_dot() {
    let val = BigFloat::from_f64(255.0);
    // Non-power-of-2 radix with multiple digits should not produce double dots
    for n_digits in 2..=8_u64 {
        let s = val
            .to_string_radix(10, n_digits, beef::Rounding::NearestEven, true)
            .unwrap();
        assert!(
            !s.contains(".."),
            "to_string_radix(10, {n_digits}, force_exp=true) produced double dot: {s:?}"
        );

        let s2 = val
            .to_string_radix(10, n_digits, beef::Rounding::NearestEven, false)
            .unwrap();
        assert!(
            !s2.contains(".."),
            "to_string_radix(10, {n_digits}, force_exp=false) produced double dot: {s2:?}"
        );
    }

    // Also test other non-power-of-2 radices
    for radix in [3_u8, 5, 6, 7, 9, 10, 12] {
        for n_digits in 1..=6_u64 {
            if let Some(s) = val.to_string_radix(radix, n_digits, beef::Rounding::NearestEven, true)
            {
                assert!(
                    !s.contains(".."),
                    "to_string_radix({radix}, {n_digits}, force_exp=true) produced double dot: {s:?}"
                );
            }
        }
    }
}

#[test]
fn logic_and_negative_with_wider_positive() {
    // -1 & x == x for any positive integer x (regression: the result length
    // must follow the positive operand when the other operand is negative)
    let a = beef::BigFloat::from_i64(-1);
    let b = beef::BigFloat::from_f64(79228162514264337593543950336.0); // 2^96
    let r = a.logic_and(&b);
    assert_eq!(r.cmp_total(&b), core::cmp::Ordering::Equal);
    // xor consistency: (-1 ^ x) == -(x + 1)
    let inf = beef::BigFormat {
        precision: beef::Precision::Infinite,
        ..beef::BigFormat::BINARY64
    };
    let x = beef::BigFloat::from_i64(-1).logic_xor(&b);
    let expected = b.add(&beef::BigFloat::from_i64(1), inf).neg();
    assert_eq!(x.cmp_total(&expected), core::cmp::Ordering::Equal);
}
