use libbeef::format::{formats, StaticFormat};
use libbeef::{BigDecimal, BigFloat, Decimal, Float, FpCategory, Sign};

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
    assert_eq!(formats::Binary64::FORMAT, libbeef::BigFormat::BINARY64);
    assert_eq!(formats::Binary128::FORMAT, libbeef::BigFormat::BINARY128);
    assert_eq!(formats::Decimal64::FORMAT, libbeef::BigFormat::DECIMAL64);
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
        .to_f64_status(libbeef::Rounding::NearestEven);

    assert!(status.is_empty());
    assert_eq!(value.to_bits(), (5.5_f64 % 2.0).to_bits());
}

#[test]
fn static_float_transcendental_methods_use_their_format() {
    type F64 = Float<formats::Binary64>;

    let x = F64::from_big(BigFloat::from_f64(0.25));
    let y = x.exp().log();
    let value = y.into_big().to_f64(libbeef::Rounding::NearestEven);

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
        y.rint(libbeef::Rounding::NearestEven).into_big(),
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
    assert!(status.contains(libbeef::Status::INEXACT));
}

#[test]
fn to_string_radix_no_double_dot() {
    let val = BigFloat::from_f64(255.0);
    // Non-power-of-2 radix with multiple digits should not produce double dots
    for n_digits in 2..=8_u64 {
        let s = val
            .to_string_radix(10, n_digits, libbeef::Rounding::NearestEven, true)
            .unwrap();
        assert!(
            !s.contains(".."),
            "to_string_radix(10, {n_digits}, force_exp=true) produced double dot: {s:?}"
        );

        let s2 = val
            .to_string_radix(10, n_digits, libbeef::Rounding::NearestEven, false)
            .unwrap();
        assert!(
            !s2.contains(".."),
            "to_string_radix(10, {n_digits}, force_exp=false) produced double dot: {s2:?}"
        );
    }

    // Also test other non-power-of-2 radices
    for radix in [3_u8, 5, 6, 7, 9, 10, 12] {
        for n_digits in 1..=6_u64 {
            if let Some(s) =
                val.to_string_radix(radix, n_digits, libbeef::Rounding::NearestEven, true)
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
    let a = libbeef::BigFloat::from_i64(-1);
    let b = libbeef::BigFloat::from_f64(79228162514264337593543950336.0); // 2^96
    let r = a.logic_and(&b);
    assert_eq!(r.cmp_total(&b), core::cmp::Ordering::Equal);
    // xor consistency: (-1 ^ x) == -(x + 1)
    let inf = libbeef::BigFormat {
        precision: libbeef::Precision::Infinite,
        ..libbeef::BigFormat::BINARY64
    };
    let x = libbeef::BigFloat::from_i64(-1).logic_xor(&b);
    let expected = b.add(&libbeef::BigFloat::from_i64(1), inf).neg();
    assert_eq!(x.cmp_total(&expected), core::cmp::Ordering::Equal);
}

#[test]
fn get_int64_is_exact_beyond_f64_mantissa() {
    let exact = libbeef::BigFormat {
        precision: libbeef::Precision::Infinite,
        ..libbeef::BigFormat::BINARY128
    };
    let two = BigFloat::from_i64(2);
    let pow2 = |e: i64| two.pow(&BigFloat::from_i64(e), exact);

    for v in [i64::MAX, i64::MIN, (1 << 60) + 1, -(1 << 60) - 1, 0, -1, 1] {
        assert_eq!(BigFloat::from_i64(v).get_int64(), Some(v));
    }
    // Out of range.
    assert_eq!(pow2(63).get_int64(), None);
    assert_eq!(
        pow2(63)
            .add(&BigFloat::from_i64(1), exact)
            .neg()
            .get_int64(),
        None
    );
    assert_eq!(pow2(10_000).get_int64(), None);
    assert_eq!(pow2(10_000).get_uint64(), None);
    // Not an integer, even though the integer part does not fit in f64.
    let half = BigFloat::from_f64(0.5);
    assert_eq!(pow2(60).add(&half, exact).get_int64(), None);
    assert_eq!(half.get_int64(), None);
    assert_eq!(BigFloat::from_f64(-2.5).get_int64(), None);
}

#[test]
fn comparisons_order_multi_limb_mantissas_from_the_top() {
    use core::cmp::Ordering::{Greater, Less};
    let exact = libbeef::BigFormat {
        precision: libbeef::Precision::Infinite,
        ..libbeef::BigFormat::BINARY128
    };
    let pow2 = |e: i64| BigFloat::from_i64(1).mul_exp2(e);
    // Same exponent and two limbs each; `a` has the larger low limb but the
    // smaller high limb.
    let a = pow2(100).add(&BigFloat::from_i64(3), exact);
    let b = pow2(100)
        .add(&pow2(99), exact)
        .add(&BigFloat::from_i64(1), exact);
    assert_eq!(a.raw_exp(), b.raw_exp());
    assert_eq!(a.limbs().len(), 2);
    assert_eq!(b.limbs().len(), 2);

    assert_eq!(a.cmp_abs(&b), Less);
    assert_eq!(a.cmp_num(&b), Some(Less));
    assert!(a < b);
    assert_eq!(a.cmp_total(&b), Less);
    assert_eq!(a.neg().cmp_num(&b.neg()), Some(Greater));
    assert_eq!(a.neg().cmp_total(&b.neg()), Greater);

    // Total order puts every negative value below every positive one,
    // regardless of exponent.
    let neg_big = BigFloat::from_i64(-1000);
    let pos_small = BigFloat::from_f64(0.5);
    assert_eq!(neg_big.cmp_total(&pos_small), Less);
    assert_eq!(pos_small.cmp_total(&neg_big), Greater);
    assert_eq!(
        BigFloat::zero(Sign::Negative).cmp_total(&BigFloat::new()),
        Less
    );
    assert_eq!(
        BigFloat::nan().cmp_total(&BigFloat::infinity(Sign::Positive)),
        Greater
    );
}

#[test]
fn to_string_radix_large_digit_counts() {
    // Exercises the multi-limb radix conversion, where the remainder must be
    // reduced below the radix power using a correct magnitude comparison.
    let v = BigFloat::parse_decimal_integer("7920784407908279816464686391409708666910").unwrap();
    assert_eq!(
        v.to_string_radix(10, 40, libbeef::Rounding::TowardZero, false)
            .unwrap(),
        "7920784407908279816464686391409708666910"
    );
    // A zero radix limb in the middle.
    let w =
        BigFloat::parse_decimal_integer("10000000000000000000000000000000000000000000000000000")
            .unwrap();
    assert_eq!(
        w.to_string_radix(10, 53, libbeef::Rounding::TowardZero, false)
            .unwrap(),
        "10000000000000000000000000000000000000000000000000000"
    );
}
