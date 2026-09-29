use libbeef::{
    mp_recip, mul_log2_radix, BigDecimal, BigFloat, BigFormat, DivRemMode, Rounding, Sign, Status,
};

#[path = "bftest/ops.rs"]
mod ops;

#[cfg(feature = "vs_num_bigint")]
#[path = "bftest/num_bigint.rs"]
mod vs_num_bigint;

#[cfg(feature = "vs_rug")]
#[path = "bftest/rug.rs"]
mod vs_rug;

#[cfg(feature = "vs_malachite")]
#[path = "bftest/malachite.rs"]
mod vs_malachite;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BfTestOp {
    MpSqrtRem,
    MpRecip,
    Mul,
    Add,
    Sub,
    Rint,
    Round,
    CmpEq,
    CmpLt,
    CmpLe,
    Div,
    Fmod,
    Rem,
    Sqrt,
    Or,
    Xor,
    And,
    CanRound,
    MulL2Radix,
    DivL2Radix,
    Atof,
    Ftoa,
    Exp,
    Log,
    Cos,
    Sin,
    Tan,
    Atan,
    Atan2,
    Asin,
    Acos,
    Pow,
    AddDec,
    MulDec,
    DivDec,
    SqrtDec,
    FmodDec,
    DivRemDec,
    RintDec,
}

const BFTEST_OPS: &[(BfTestOp, &str)] = &[
    (BfTestOp::MpSqrtRem, "mp_sqrtrem"),
    (BfTestOp::MpRecip, "mp_recip"),
    (BfTestOp::Mul, "mul"),
    (BfTestOp::Add, "add"),
    (BfTestOp::Sub, "sub"),
    (BfTestOp::Rint, "rint"),
    (BfTestOp::Round, "round"),
    (BfTestOp::CmpEq, "cmp_eq"),
    (BfTestOp::CmpLt, "cmp_lt"),
    (BfTestOp::CmpLe, "cmp_le"),
    (BfTestOp::Div, "div"),
    (BfTestOp::Fmod, "fmod"),
    (BfTestOp::Rem, "rem"),
    (BfTestOp::Sqrt, "sqrt"),
    (BfTestOp::Or, "or"),
    (BfTestOp::Xor, "xor"),
    (BfTestOp::And, "and"),
    (BfTestOp::CanRound, "can_round"),
    (BfTestOp::MulL2Radix, "mul_l2radix"),
    (BfTestOp::DivL2Radix, "div_l2radix"),
    (BfTestOp::Atof, "atof"),
    (BfTestOp::Ftoa, "ftoa"),
    (BfTestOp::Exp, "exp"),
    (BfTestOp::Log, "log"),
    (BfTestOp::Cos, "cos"),
    (BfTestOp::Sin, "sin"),
    (BfTestOp::Tan, "tan"),
    (BfTestOp::Atan, "atan"),
    (BfTestOp::Atan2, "atan2"),
    (BfTestOp::Asin, "asin"),
    (BfTestOp::Acos, "acos"),
    (BfTestOp::Pow, "pow"),
    (BfTestOp::AddDec, "add_dec"),
    (BfTestOp::MulDec, "mul_dec"),
    (BfTestOp::DivDec, "div_dec"),
    (BfTestOp::SqrtDec, "sqrt_dec"),
    (BfTestOp::FmodDec, "fmod_dec"),
    (BfTestOp::DivRemDec, "divrem_dec"),
    (BfTestOp::RintDec, "rint_dec"),
];

/// Signature of a per-op benchmark runner.
type BenchFn = fn(&mut Mt19937_64, u64, u64) -> (usize, u128);

struct Mt19937_64 {
    mt: [u64; 312],
    mti: usize,
}

impl Mt19937_64 {
    const NN: usize = 312;
    const MM: usize = 156;
    const MATRIX_A: u64 = 0xB502_6F5A_A966_19E9;
    const UM: u64 = 0xFFFF_FFFF_8000_0000;
    const LM: u64 = 0x7FFF_FFFF;

    fn new(seed: u64) -> Self {
        let mut mt = [0; Self::NN];
        mt[0] = seed;
        for i in 1..Self::NN {
            mt[i] = 6_364_136_223_846_793_005_u64
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 62))
                .wrapping_add(i as u64);
        }
        Self { mt, mti: Self::NN }
    }

    fn next_u64(&mut self) -> u64 {
        if self.mti >= Self::NN {
            for i in 0..(Self::NN - Self::MM) {
                let x = (self.mt[i] & Self::UM) | (self.mt[i + 1] & Self::LM);
                self.mt[i] =
                    self.mt[i + Self::MM] ^ (x >> 1) ^ if x & 1 == 0 { 0 } else { Self::MATRIX_A };
            }
            for i in (Self::NN - Self::MM)..(Self::NN - 1) {
                let x = (self.mt[i] & Self::UM) | (self.mt[i + 1] & Self::LM);
                self.mt[i] = self.mt[i + Self::MM - Self::NN]
                    ^ (x >> 1)
                    ^ if x & 1 == 0 { 0 } else { Self::MATRIX_A };
            }
            let x = (self.mt[Self::NN - 1] & Self::UM) | (self.mt[0] & Self::LM);
            self.mt[Self::NN - 1] =
                self.mt[Self::MM - 1] ^ (x >> 1) ^ if x & 1 == 0 { 0 } else { Self::MATRIX_A };
            self.mti = 0;
        }

        let mut x = self.mt[self.mti];
        self.mti += 1;
        x ^= (x >> 29) & 0x5555_5555_5555_5555;
        x ^= (x << 17) & 0x71D6_7FFF_EDA6_0000;
        x ^= (x << 37) & 0xFFF7_EEE0_0000_0000;
        x ^= x >> 43;
        x
    }

    fn small_i64(&mut self) -> i64 {
        (self.next_u64() % 2001) as i64 - 1000
    }

    fn small_u64(&mut self) -> u64 {
        self.next_u64() & ((1_u64 << 53) - 1)
    }

    fn pow2_float(&mut self, min_exp: i32, span: i32) -> BigFloat {
        let exp = min_exp + (self.next_u64() % (span as u64)) as i32;
        let mut value = BigFloat::from_f64(2.0_f64.powi(exp));
        if self.next_u64() & 1 != 0 {
            value = value.neg();
        }
        value
    }

    fn finite_f64_bits(&mut self) -> u64 {
        let sign = self.next_u64() & (1_u64 << 63);
        let exp = self.next_u64() % 2046 + 1;
        let frac = self.next_u64() & ((1_u64 << 52) - 1);
        sign | (exp << 52) | frac
    }

    fn moderate_f64(&mut self) -> f64 {
        let sign = self.next_u64() & (1_u64 << 63);
        let exp = 923 + (self.next_u64() % 201);
        let frac = self.next_u64() & ((1_u64 << 52) - 1);
        f64::from_bits(sign | (exp << 52) | frac)
    }

    fn integer_string(&mut self, radix: u8) -> String {
        let len = 1 + (self.next_u64() % 90) as usize;
        let mut out = String::new();
        if self.next_u64() & 1 != 0 {
            out.push('-');
        }
        out.push(digit_char(
            1 + (self.next_u64() % u64::from(radix - 1)) as u8,
        ));
        for _ in 1..len {
            out.push(digit_char((self.next_u64() % u64::from(radix)) as u8));
        }
        out
    }
}

fn digit_char(value: u8) -> char {
    if value < 10 {
        char::from(b'0' + value)
    } else {
        char::from(b'a' + value - 10)
    }
}

fn expected_decimal_digit_round(
    literal: &str,
    precision: usize,
    rounding: Rounding,
) -> (String, bool) {
    let (is_negative, digits) = literal
        .strip_prefix('-')
        .map_or((false, literal), |rest| (true, rest));
    if precision == 0 || digits.len() <= precision {
        return (literal.to_string(), false);
    }

    let kept = &digits[..precision];
    let discarded = &digits[precision..];
    let inexact = discarded.as_bytes().iter().any(|&digit| digit != b'0');
    if !inexact {
        return (literal.to_string(), false);
    }

    let increment = match rounding {
        Rounding::TowardZero | Rounding::Faithful => false,
        Rounding::TowardPositive => !is_negative,
        Rounding::TowardNegative => is_negative,
        Rounding::AwayFromZero => true,
        Rounding::NearestAway | Rounding::NearestEven => {
            let bytes = discarded.as_bytes();
            if bytes[0] > b'5' {
                true
            } else if bytes[0] < b'5' {
                false
            } else if bytes[1..].iter().any(|&digit| digit != b'0')
                || rounding == Rounding::NearestAway
            {
                true
            } else {
                !(kept.as_bytes()[kept.len() - 1] - b'0').is_multiple_of(2)
            }
        }
    };

    let mut rounded = if increment {
        let mut bytes = kept.as_bytes().to_vec();
        let mut carried = true;
        for idx in (0..bytes.len()).rev() {
            if bytes[idx] == b'9' {
                bytes[idx] = b'0';
            } else {
                bytes[idx] += 1;
                carried = false;
                break;
            }
        }
        let mut out = String::new();
        if carried {
            out.push('1');
        }
        out.push_str(core::str::from_utf8(&bytes).expect("digits"));
        out
    } else {
        kept.to_string()
    };
    for _ in 0..(digits.len() - precision) {
        rounded.push('0');
    }
    if is_negative {
        rounded.insert(0, '-');
    }
    (rounded, true)
}

fn special(idx: usize) -> BigFloat {
    match idx {
        0 => BigFloat::zero(Sign::Positive),
        1 => BigFloat::zero(Sign::Negative),
        2 => BigFloat::infinity(Sign::Positive),
        3 => BigFloat::infinity(Sign::Negative),
        4 => BigFloat::from_i64(1),
        5 => BigFloat::from_i64(-1),
        6 => BigFloat::nan(),
        _ => unreachable!(),
    }
}

fn special_dec(idx: usize) -> BigDecimal {
    match idx {
        0 => BigDecimal::zero(Sign::Positive),
        1 => BigDecimal::zero(Sign::Negative),
        2 => BigDecimal::infinity(Sign::Positive),
        3 => BigDecimal::infinity(Sign::Negative),
        4 => BigDecimal::from_i64(1),
        5 => BigDecimal::from_i64(-1),
        6 => BigDecimal::nan(),
        _ => unreachable!(),
    }
}

#[test]
fn bftest_op_catalog_matches_upstream_order() {
    assert_eq!(BFTEST_OPS.len(), 39);
    assert_eq!(BFTEST_OPS[0], (BfTestOp::MpSqrtRem, "mp_sqrtrem"));
    assert_eq!(BFTEST_OPS[38], (BfTestOp::RintDec, "rint_dec"));
}

#[test]
fn bftest_rng_is_seeded_and_reproducible() {
    let mut a = Mt19937_64::new(1234);
    let mut b = Mt19937_64::new(1234);
    let mut c = Mt19937_64::new(1235);

    let seq_a = [a.next_u64(), a.next_u64(), a.next_u64(), a.next_u64()];
    let seq_b = [b.next_u64(), b.next_u64(), b.next_u64(), b.next_u64()];
    let seq_c = [c.next_u64(), c.next_u64(), c.next_u64(), c.next_u64()];

    assert_eq!(seq_a, seq_b);
    assert_ne!(seq_a, seq_c);
}

#[test]
fn bftest_seeded_mp_sqrtrem_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_mp_sqrtrem(&mut rng, 128, 0);
    }
}

#[test]
fn bftest_seeded_mp_recip_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_mp_recip(&mut rng, 128, 0);
    }
    let exact_half = [1_u64 << 63];
    assert_eq!(mp_recip(&exact_half), Some(vec![u64::MAX, 1]));
}

#[test]
fn bftest_special_value_matrix_for_initial_binary_ops() {
    let f = BigFormat::BINARY64;
    for a_idx in 0..7 {
        for b_idx in 0..7 {
            let a = special(a_idx);
            let b = special(b_idx);

            let (_sum, add_status) = a.add_status(&b, f);
            if (a_idx, b_idx) == (2, 3) || (a_idx, b_idx) == (3, 2) {
                assert!(add_status.contains(Status::INVALID_OP));
            }

            let (_product, mul_status) = a.mul_status(&b, f);
            if matches!((a_idx, b_idx), (0 | 1, 2 | 3) | (2 | 3, 0 | 1)) {
                assert!(mul_status.contains(Status::INVALID_OP));
            }

            let (_quotient, div_status) = a.div_status(&b, f);
            if matches!((a_idx, b_idx), (0 | 1, 0 | 1) | (2 | 3, 2 | 3)) {
                assert!(div_status.contains(Status::INVALID_OP));
            }
            if matches!(a_idx, 4 | 5) && matches!(b_idx, 0 | 1) {
                assert!(div_status.contains(Status::DIVIDE_ZERO));
            }
        }
    }
}

#[test]
fn bftest_special_value_matrix_for_initial_decimal_ops() {
    let f = BigFormat::DECIMAL64;
    for a_idx in 0..7 {
        for b_idx in 0..7 {
            let a = special_dec(a_idx);
            let b = special_dec(b_idx);

            let (_sum, add_status) = a.add_status(&b, f);
            if (a_idx, b_idx) == (2, 3) || (a_idx, b_idx) == (3, 2) {
                assert!(add_status.contains(Status::INVALID_OP));
            }

            let (_product, mul_status) = a.mul_status(&b, f);
            if matches!((a_idx, b_idx), (0 | 1, 2 | 3) | (2 | 3, 0 | 1)) {
                assert!(mul_status.contains(Status::INVALID_OP));
            }

            let (_quotient, div_status) = a.div_status(&b, f);
            if matches!((a_idx, b_idx), (0 | 1, 0 | 1) | (2 | 3, 2 | 3)) {
                assert!(div_status.contains(Status::INVALID_OP));
            }
            if matches!(a_idx, 4 | 5) && matches!(b_idx, 0 | 1) {
                assert!(div_status.contains(Status::DIVIDE_ZERO));
            }
        }
    }
}

#[test]
fn bftest_seeded_small_integer_ops_match_i128_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_small_integer_ops(&mut rng, 256, 0);
    }
}

#[test]
fn bftest_seeded_exact_dyadic_division_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_exact_dyadic_division(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_seeded_large_integer_limb_ops_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_large_integer_limb_ops(&mut rng, 128, 0);
    }
}

#[test]
fn bftest_seeded_float64_roundtrip_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_float64_roundtrip(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_to_f64_directed_rounding_modes() {
    let half_ulp = BigFloat::from_f64(2.0_f64.powi(-53));
    let just_above_one = BigFloat::from_f64(1.0).add(
        &half_ulp,
        BigFormat {
            precision: libbeef::Precision::Infinite,
            ..BigFormat::BINARY64
        },
    );

    assert_eq!(
        just_above_one.to_f64(Rounding::TowardZero).to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        just_above_one.to_f64(Rounding::TowardPositive).to_bits(),
        f64::from_bits(1.0_f64.to_bits() + 1).to_bits()
    );

    let negative = just_above_one.neg();
    assert_eq!(
        negative.to_f64(Rounding::TowardZero).to_bits(),
        (-1.0_f64).to_bits()
    );
    assert_eq!(
        negative.to_f64(Rounding::TowardNegative).to_bits(),
        f64::from_bits((-1.0_f64).to_bits() + 1).to_bits()
    );
}

#[test]
fn bftest_seeded_float64_add_mul_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_float64_add_mul(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_faithful_mul_truncates_inputs_like_libbf() {
    let format = BigFormat {
        precision: libbeef::Precision::Bits(1),
        rounding: Rounding::Faithful,
        ..BigFormat::BINARY64
    };
    let exact = BigFormat {
        precision: libbeef::Precision::Infinite,
        ..BigFormat::BINARY64
    };

    let a =
        BigFloat::parse_integer_radix("100000000000000000000000000000003", 16).expect("parse a");
    let b =
        BigFloat::parse_integer_radix("100000000000000000000000000000005", 16).expect("parse b");
    let a_trunc = BigFloat::parse_integer_radix("100000000000000000000000000000000", 16)
        .expect("parse truncated a");
    let b_trunc = BigFloat::parse_integer_radix("100000000000000000000000000000000", 16)
        .expect("parse truncated b");

    assert_eq!(
        a.mul(&b, format),
        a_trunc.mul(&b_trunc, exact).round(format)
    );
}

#[test]
fn bftest_seeded_float64_div_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_float64_div(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_binary_division_uses_libbf_limb_precision() {
    let exact = BigFormat {
        precision: libbeef::Precision::Infinite,
        ..BigFormat::BINARY128
    };
    let p130 = BigFormat {
        precision: libbeef::Precision::Bits(130),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY128
    };

    let quotient = BigFloat::from_i64(1).div(&BigFloat::from_i64(3), p130);
    let scale =
        BigFloat::parse_integer_radix(&format!("1{}", "0".repeat(131)), 2).expect("parse 2^131");
    let scaled = quotient.mul(&scale, exact);
    let expected = BigFloat::parse_integer_radix("2aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab", 16)
        .expect("parse rounded 1/3 mantissa");

    assert_eq!(scaled, expected);
}

#[test]
fn bftest_binary_division_honors_decimal_digit_precision() {
    let p3 = BigFormat {
        precision: libbeef::Precision::Digits(3),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let exact = BigFormat {
        precision: libbeef::Precision::Infinite,
        ..BigFormat::BINARY64
    };

    let (one_eighth, status) = BigFloat::from_i64(1).div_status(&BigFloat::from_i64(8), p3);
    assert!(status.is_empty());
    assert_eq!(
        one_eighth,
        BigFloat::parse_decimal("0.125", exact).expect("parse 1/8")
    );

    let (one_third, status) = BigFloat::from_i64(1).div_status(&BigFloat::from_i64(3), p3);
    assert!(status.contains(Status::INEXACT));
    let expected = BigFloat::parse_decimal(
        "0.333",
        BigFormat {
            precision: libbeef::Precision::Bits(20),
            ..BigFormat::BINARY64
        },
    )
    .expect("parse rounded 1/3");
    assert_eq!(one_third, expected);
}

#[test]
fn bftest_seeded_float64_fmod_for_several_seeds() {
    let f = BigFormat::BINARY64;

    for (a, b) in [(-1.0, 1.0), (1.0, -1.0), (-5.5, 2.0), (5.5, -2.0)] {
        let bf_a = BigFloat::from_f64(a);
        let bf_b = BigFloat::from_f64(b);
        let remainder = bf_a
            .rem(&bf_b, f, DivRemMode::TowardZero)
            .to_f64(Rounding::NearestEven);
        assert_eq!(remainder.to_bits(), (a % b).to_bits(), "{a:?} % {b:?}");
    }
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_float64_fmod(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_seeded_remainder_nearest_even_for_several_seeds() {
    let f = BigFormat::BINARY64;

    for (a, b, expected) in [
        (5.5, 2.0, -0.5),
        (-5.5, 2.0, 0.5),
        (5.0, 2.0, 1.0),
        (7.0, 2.0, -1.0),
        (-7.0, 2.0, 1.0),
    ] as [(f64, f64, f64); 5]
    {
        let bf_a = BigFloat::from_f64(a);
        let bf_b = BigFloat::from_f64(b);
        let remainder = bf_a
            .rem(&bf_b, f, DivRemMode::NearestEven)
            .to_f64(Rounding::NearestEven);
        assert_eq!(remainder.to_bits(), expected.to_bits(), "{a:?} rem {b:?}");
    }

    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_remainder_nearest_even(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_seeded_remainder_floor_and_euclidean_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_remainder_floor_euclidean(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_seeded_float64_sqrt_for_several_seeds() {
    let f = BigFormat::BINARY64;

    let (neg_root, neg_status) = BigFloat::from_f64(-1.0).sqrt_status(f);
    assert!(neg_root.is_nan());
    assert!(neg_status.contains(Status::INVALID_OP));
    assert!(BigFloat::infinity(Sign::Positive).sqrt(f).is_infinite());
    assert!(BigFloat::nan().sqrt(f).is_nan());
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_float64_sqrt(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_binary_sqrt_rounds_non_square_dyadics() {
    let p4 = BigFormat {
        precision: libbeef::Precision::Bits(4),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY128
    };

    assert_eq!(BigFloat::from_i64(2).sqrt(p4), BigFloat::from_f64(1.375));
    assert_eq!(BigFloat::from_i64(3).sqrt(p4), BigFloat::from_f64(1.75));

    let (root, status) = BigFloat::from_f64(0.5).sqrt_status(p4);
    let root_f64 = root.to_f64(Rounding::NearestEven);
    assert_eq!(root_f64.to_bits(), 0.6875_f64.to_bits());
    assert!(status.contains(Status::INEXACT));
}

#[test]
fn bftest_binary_sqrt_uses_libbf_limb_precision() {
    let exact = BigFormat {
        precision: libbeef::Precision::Infinite,
        ..BigFormat::BINARY128
    };
    let p130 = BigFormat {
        precision: libbeef::Precision::Bits(130),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY128
    };

    let root = BigFloat::from_i64(2).sqrt(p130);
    let scale =
        BigFloat::parse_integer_radix(&format!("1{}", "0".repeat(129)), 2).expect("parse 2^129");
    let mantissa = root.mul(&scale, exact);
    let expected = BigFloat::parse_integer_radix("2d413cccfe779921165f626cdd52afa7c", 16)
        .expect("parse rounded sqrt(2) mantissa");

    assert_eq!(mantissa, expected);
}

#[test]
fn bftest_binary_sqrt_honors_decimal_digit_precision() {
    let p3 = BigFormat {
        precision: libbeef::Precision::Digits(3),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let expected_format = BigFormat {
        precision: libbeef::Precision::Bits(20),
        ..BigFormat::BINARY64
    };

    let (root, status) = BigFloat::from_i64(2).sqrt_status(p3);
    assert!(status.contains(Status::INEXACT));
    let expected = BigFloat::parse_decimal("1.41", expected_format).expect("parse sqrt(2)");
    assert_eq!(root, expected);

    let exact = BigFormat {
        precision: libbeef::Precision::Infinite,
        ..BigFormat::BINARY64
    };
    let (root, status) = BigFloat::from_i64(2).sqrt_status(exact);
    assert!(root.is_nan());
    assert!(status.contains(Status::INVALID_OP));
}

#[test]
fn bftest_binary_rounding_and_sqrt_do_not_stop_at_u128_mantissa() {
    let exact = BigFormat {
        precision: libbeef::Precision::Infinite,
        ..BigFormat::BINARY128
    };
    let p113 = BigFormat {
        precision: libbeef::Precision::Bits(113),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY128
    };

    let root = BigFloat::parse_integer_radix("1000000000000000000000000000000000001", 16)
        .expect("parse root");
    let rounded_root = BigFloat::parse_integer_radix("1000000000000000000000000000000000000", 16)
        .expect("parse rounded root");
    let square = root.mul(&root, exact);

    assert_eq!(root.round(p113), rounded_root);
    assert_eq!(square.sqrt(exact), root);
    assert_eq!(square.sqrt(p113), rounded_root);
}

#[test]
fn bftest_seeded_float64_transcendentals_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_float64_transcendentals(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_float64_transcendental_invalid_status() {
    let f = BigFormat::BINARY64;

    let (log_value, log_status) = BigFloat::from_f64(-1.0).log_status(f);
    assert!(log_value.is_nan());
    assert!(log_status.contains(Status::INVALID_OP));

    let (asin_value, asin_status) = BigFloat::from_f64(2.0).asin_status(f);
    assert!(asin_value.is_nan());
    assert!(asin_status.contains(Status::INVALID_OP));

    let (acos_value, acos_status) = BigFloat::from_f64(-2.0).acos_status(f);
    assert!(acos_value.is_nan());
    assert!(acos_status.contains(Status::INVALID_OP));

    let (pow_value, pow_status) = BigFloat::from_f64(-2.0).pow_status(&BigFloat::from_f64(0.5), f);
    assert!(pow_value.is_nan());
    assert!(pow_status.contains(Status::INVALID_OP));

    let (exp_value, exp_status) = BigFloat::from_f64(1024.0).exp_status(f);
    assert!(exp_value.is_infinite());
    assert!(exp_status.contains(Status::OVERFLOW));
    assert!(exp_status.contains(Status::INEXACT));

    let (underflow_value, underflow_status) = BigFloat::from_f64(-1000.0).exp_status(f);
    let underflow_f64 = underflow_value.to_f64(Rounding::NearestEven);
    assert_eq!(underflow_f64.to_bits(), 0.0_f64.to_bits());
    assert!(underflow_status.contains(Status::UNDERFLOW));
    assert!(underflow_status.contains(Status::INEXACT));
}

#[test]
fn bftest_float64_transcendental_inexact_status() {
    let f = BigFormat::BINARY64;

    for (value, op) in [
        (BigFloat::from_f64(1.0).exp_status(f), "exp(1)"),
        (BigFloat::from_f64(2.0).log_status(f), "log(2)"),
        (BigFloat::from_f64(1.0).sin_status(f), "sin(1)"),
        (BigFloat::from_f64(1.0).cos_status(f), "cos(1)"),
        (BigFloat::from_f64(0.5).asin_status(f), "asin(0.5)"),
        (BigFloat::from_f64(0.5).acos_status(f), "acos(0.5)"),
    ] {
        assert!(
            value.1.contains(Status::INEXACT),
            "missing INEXACT for {op}: {:?}",
            value.1
        );
    }

    assert!(BigFloat::from_f64(0.0).exp_status(f).1.is_empty());
    assert!(BigFloat::from_f64(1.0).log_status(f).1.is_empty());
    assert!(BigFloat::from_f64(0.0).sin_status(f).1.is_empty());
    assert!(BigFloat::from_f64(0.0).tan_status(f).1.is_empty());
    assert!(BigFloat::from_f64(0.0).atan_status(f).1.is_empty());
    assert!(BigFloat::from_f64(0.0).asin_status(f).1.is_empty());
    assert!(BigFloat::from_f64(1.0).acos_status(f).1.is_empty());
    assert!(BigFloat::from_f64(2.0)
        .pow_status(&BigFloat::from_f64(0.0), f)
        .1
        .is_empty());
}

#[test]
fn bftest_transcendental_exact_identities_work_without_f64_fallback() {
    let f = BigFormat::BINARY128;
    let zero = BigFloat::zero(Sign::Positive);
    let neg_zero = BigFloat::zero(Sign::Negative);
    let one = BigFloat::from_i64(1);

    assert_eq!(zero.exp(f), one);
    assert_eq!(one.log(f), zero);
    assert_eq!(zero.cos(f), one);
    assert_eq!(zero.sin(f), zero);
    assert_eq!(neg_zero.sin(f), neg_zero);
    assert_eq!(zero.tan(f), zero);
    assert_eq!(zero.atan(f), zero);
    assert_eq!(zero.asin(f), zero);
    assert_eq!(one.acos(f), zero);
    assert_eq!(zero.atan2(&one, f), zero);
    assert_eq!(one.pow(&BigFloat::from_f64(0.5), f), one);
}

#[test]
fn bftest_pow_integer_exponent_uses_limb_arithmetic() {
    let exact = BigFormat {
        precision: libbeef::Precision::Infinite,
        ..BigFormat::BINARY128
    };
    let base = BigFloat::parse_integer_radix("10000000000000001", 16).expect("parse large base");
    let exponent = BigFloat::from_i64(5);

    let mut expected = BigFloat::from_i64(1);
    for _ in 0..5 {
        expected = expected.mul(&base, exact);
    }

    let (power, status) = base.pow_status(&exponent, exact);
    assert_eq!(power, expected);
    assert!(status.is_empty());

    let (reciprocal, status) = BigFloat::from_i64(2).pow_status(&BigFloat::from_i64(-3), exact);
    assert_eq!(
        reciprocal,
        BigFloat::parse_decimal("0.125", exact).expect("parse 1/8")
    );
    assert!(status.is_empty());
}

#[test]
fn bftest_binary64_inexact_status_for_rounded_add() {
    let one = BigFloat::from_f64(1.0);
    let tiny = BigFloat::from_f64(2.0_f64.powi(-60));
    let (sum, status) = one.add_status(&tiny, BigFormat::BINARY64);
    let sum_f64 = sum.to_f64(Rounding::NearestEven);

    assert_eq!(sum_f64.to_bits(), 1.0_f64.to_bits());
    assert!(status.contains(Status::INEXACT));
}

#[test]
fn bftest_binary64_inexact_status_for_division() {
    let one = BigFloat::from_f64(1.0);
    let three = BigFloat::from_f64(3.0);
    let (quotient, status) = one.div_status(&three, BigFormat::BINARY64);
    let quotient_f64 = quotient.to_f64(Rounding::NearestEven);

    assert_eq!(quotient_f64.to_bits(), (1.0_f64 / 3.0).to_bits());
    assert!(status.contains(Status::INEXACT));
}

#[test]
fn bftest_seeded_comparisons_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_comparisons(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_total_comparison_is_stable_for_special_values() {
    let values = [
        BigFloat::nan(),
        BigFloat::infinity(Sign::Negative),
        BigFloat::from_i64(-1),
        BigFloat::zero(Sign::Negative),
        BigFloat::zero(Sign::Positive),
        BigFloat::from_i64(1),
        BigFloat::infinity(Sign::Positive),
    ];

    for value in &values {
        assert_eq!(value.cmp_total(value), core::cmp::Ordering::Equal);
    }
    for (i, a) in values.iter().enumerate() {
        for (j, b) in values.iter().enumerate() {
            assert_eq!(a.cmp_total(b), b.cmp_total(a).reverse());
            if i == j {
                assert_eq!(a.cmp_full(b), core::cmp::Ordering::Equal);
            }
        }
    }
}

#[test]
fn bftest_seeded_nonnegative_logic_ops_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_nonnegative_logic_ops(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_seeded_signed_logic_ops_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_signed_logic_ops(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_seeded_fractional_logic_ops_truncate_toward_zero_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_fractional_logic_ops(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_seeded_decimal_integer_parse_format_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_decimal_integer_parse_format(&mut rng, 256, 0);
    }
}

#[test]
fn bftest_seeded_decimal_digit_rounding_for_integer_values() {
    let directed_cases = [
        ("12345", 3, Rounding::TowardZero, "12300"),
        ("12345", 3, Rounding::TowardPositive, "12400"),
        ("-12345", 3, Rounding::TowardNegative, "-12400"),
        ("12500", 2, Rounding::NearestEven, "12000"),
        ("13500", 2, Rounding::NearestEven, "14000"),
        ("999500", 3, Rounding::NearestEven, "1000000"),
    ];
    for (literal, precision, rounding, expected) in directed_cases {
        let value = BigFloat::parse_integer_radix(literal, 10).expect("parse integer");
        let (rounded, status) = value.round_status(BigFormat {
            precision: libbeef::Precision::Digits(precision as u64),
            rounding,
            ..BigFormat::BINARY64
        });
        assert_eq!(
            rounded.to_decimal_integer_string().as_deref(),
            Some(expected)
        );
        assert!(status.contains(Status::INEXACT));
    }
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_decimal_digit_rounding(&mut rng, 256, 0);
    }
}

#[test]
fn bftest_decimal_digit_rounding_for_binary_fractions() {
    let exact = BigFormat {
        precision: libbeef::Precision::Infinite,
        ..BigFormat::BINARY64
    };
    for (numerator, shift, precision, rounding, expected) in [
        (15, 3, 1, Rounding::NearestEven, "2"),
        (15, 4, 1, Rounding::NearestEven, "0.9"),
        (1, 6, 2, Rounding::NearestEven, "0.016"),
        (-3, 5, 1, Rounding::TowardNegative, "-0.1"),
        (-3, 5, 1, Rounding::TowardZero, "-0.09"),
    ] {
        let value = BigFloat::from_i64(numerator).div(&BigFloat::from_i64(1_i64 << shift), exact);
        let expected_format = BigFormat {
            precision: libbeef::Precision::Bits(precision * 4 + 8),
            ..BigFormat::BINARY64
        };
        let rounded = value.round(BigFormat {
            precision: libbeef::Precision::Digits(precision),
            rounding,
            ..BigFormat::BINARY64
        });
        let expected = BigFloat::parse_decimal(expected, expected_format).expect("parse expected");
        assert_eq!(
            rounded, expected,
            "{numerator}/2^{shift} precision={precision}"
        );
    }
}

#[test]
fn bftest_seeded_decimal_scaled_parse_format_for_several_seeds() {
    for literal in [
        "",
        ".",
        "+",
        "-",
        "1e",
        "1e+",
        "1.2.3",
        "abc",
        "1e999999999999",
    ] {
        assert!(
            BigDecimal::parse_decimal(literal).is_err(),
            "invalid decimal parsed: {literal:?}"
        );
    }

    for literal in [
        "0", "-0", "1.25", "-12.500", "1e3", "-2.5e-2", "inf", "-inf", "NaN",
    ] {
        let parsed = BigDecimal::parse_decimal(literal).expect("parse decimal");
        let formatted = parsed.to_decimal_string().expect("format decimal");
        let reparsed = BigDecimal::parse_decimal(&formatted).expect("reparse decimal");
        if parsed.is_nan() {
            assert!(reparsed.is_nan());
        } else {
            assert_eq!(reparsed, parsed, "{literal}");
        }
    }

    for literal in [
        "1000000000000000000000000000000000000000000000000000000000000000000000001",
        "-1234567890123456789012345678901234567890.0005",
        "1.23456789012345678901234567890123456789e42",
    ] {
        let parsed = BigDecimal::parse_decimal(literal).expect("parse large decimal");
        let formatted = parsed.to_decimal_string().expect("format large decimal");
        let reparsed = BigDecimal::parse_decimal(&formatted).expect("reparse large decimal");
        assert_eq!(reparsed, parsed, "{literal}");
    }

    let normalized = BigDecimal::parse_decimal("123456789012345678901234567890000000000000.000000")
        .expect("parse large normalized decimal");
    assert_eq!(
        normalized.to_decimal_string().as_deref(),
        Some("123456789012345678901234567890000000000000")
    );
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_decimal_scaled_parse_format(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_seeded_radix_integer_parse_format_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_radix_integer_parse_format(&mut rng, 256, 0);
    }
}

#[test]
fn bftest_seeded_binary64_atof_ftoa_roundtrip_for_several_seeds() {
    let f = BigFormat::BINARY64;
    for literal in ["", ".", "+", "-", "1e", "1e+", "nanx", "1.2.3"] {
        assert!(
            BigFloat::parse_decimal(literal, f).is_err(),
            "invalid binary decimal parsed: {literal:?}"
        );
    }

    for literal in ["0", "-0", "1.5", "-2.25e3", "inf", "-inf", "NaN"] {
        let parsed = BigFloat::parse_decimal(literal, f).expect("parse decimal");
        let formatted = parsed.to_decimal_string(f).expect("format decimal");
        let reparsed = BigFloat::parse_decimal(&formatted, f).expect("reparse decimal");
        if parsed.is_nan() {
            assert!(reparsed.is_nan());
        } else {
            let a = parsed.to_f64(Rounding::NearestEven);
            let b = reparsed.to_f64(Rounding::NearestEven);
            assert_eq!(a.to_bits(), b.to_bits(), "{literal}");
        }
    }
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_binary64_atof_ftoa_roundtrip(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_binary_atof_does_not_use_f64_parser_limit() {
    let format = BigFormat {
        precision: libbeef::Precision::Bits(160),
        ..BigFormat::BINARY128
    };
    let parsed = BigFloat::parse_decimal("1e400", format).expect("parse large decimal");
    assert!(!parsed.is_infinite());
    assert!(!parsed.is_zero());

    let expected = BigFloat::parse_integer_radix("1", 10)
        .expect("parse one")
        .mul(
            &BigFloat::parse_integer_radix(
                "100000000000000000000000000000000000000000000000000",
                10,
            )
            .expect("parse 1e50"),
            BigFormat {
                precision: libbeef::Precision::Infinite,
                ..BigFormat::BINARY128
            },
        );
    assert!(parsed.cmp_abs(&expected) == core::cmp::Ordering::Greater);
}

#[test]
fn bftest_seeded_decimal_integer_ops_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_decimal_integer_ops(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_seeded_decimal_scaled_ops_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_decimal_scaled_ops(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_decimal_rint_ties_and_faithful_rounding() {
    for (literal, rounding, expected) in [
        ("2.5", Rounding::NearestEven, "2"),
        ("3.5", Rounding::NearestEven, "4"),
        ("2.5", Rounding::NearestAway, "3"),
        ("-2.5", Rounding::NearestAway, "-3"),
        ("2.5", Rounding::TowardZero, "2"),
        ("-2.5", Rounding::TowardZero, "-2"),
        ("2.5", Rounding::TowardPositive, "3"),
        ("-2.5", Rounding::TowardPositive, "-2"),
        ("2.5", Rounding::TowardNegative, "2"),
        ("-2.5", Rounding::TowardNegative, "-3"),
        ("2.5", Rounding::AwayFromZero, "3"),
        ("-2.5", Rounding::AwayFromZero, "-3"),
        ("2.5", Rounding::Faithful, "2"),
        ("-2.5", Rounding::Faithful, "-2"),
    ] {
        let value = BigDecimal::parse_decimal(literal).expect("parse decimal");
        let (rounded, status) = value.rint_status(rounding);
        assert_eq!(
            rounded.to_decimal_string().as_deref(),
            Some(expected),
            "{literal} with {rounding:?}"
        );
        assert!(status.contains(Status::INEXACT));
    }
}

#[test]
fn bftest_decimal_ops_honor_digit_precision() {
    let f = BigFormat {
        precision: libbeef::Precision::Digits(4),
        rounding: Rounding::NearestEven,
        ..BigFormat::DECIMAL64
    };

    let (sum, status) = BigDecimal::from_i64(12_345).add_status(&BigDecimal::from_i64(1), f);
    assert_eq!(sum.to_decimal_string().as_deref(), Some("12350"));
    assert!(status.contains(Status::INEXACT));

    let lhs = BigDecimal::from_scaled_i64(12_345, 2);
    let rhs = BigDecimal::from_scaled_i64(9_876, 2);
    let (product, status) = lhs.mul_status(&rhs, f);
    assert_eq!(product.to_decimal_string().as_deref(), Some("12190"));
    assert!(status.contains(Status::INEXACT));

    let (quotient, status) = BigDecimal::from_i64(1).div_status(&BigDecimal::from_i64(3), f);
    assert_eq!(quotient.to_decimal_string().as_deref(), Some("0.3333"));
    assert!(status.contains(Status::INEXACT));

    let (tiny, status) = BigDecimal::from_i64(1).div_status(&BigDecimal::from_i64(300_000), f);
    assert_eq!(tiny.to_decimal_string().as_deref(), Some("0.000003333"));
    assert!(status.contains(Status::INEXACT));

    let upward = BigFormat {
        rounding: Rounding::TowardPositive,
        ..f
    };
    let (tiny_up, status) =
        BigDecimal::from_i64(1).div_status(&BigDecimal::from_i64(300_000), upward);
    assert_eq!(tiny_up.to_decimal_string().as_deref(), Some("0.000003334"));
    assert!(status.contains(Status::INEXACT));

    let downward = BigFormat {
        rounding: Rounding::TowardNegative,
        ..f
    };
    let (tiny_down, status) =
        BigDecimal::from_i64(-1).div_status(&BigDecimal::from_i64(300_000), downward);
    assert_eq!(
        tiny_down.to_decimal_string().as_deref(),
        Some("-0.000003334")
    );
    assert!(status.contains(Status::INEXACT));

    let (sticky_up, status) =
        BigDecimal::from_i64(1).div_status(&BigDecimal::from_i64(111), upward);
    assert_eq!(sticky_up.to_decimal_string().as_deref(), Some("0.00901"));
    assert!(status.contains(Status::INEXACT));

    let (sticky_down, status) =
        BigDecimal::from_i64(-1).div_status(&BigDecimal::from_i64(111), downward);
    assert_eq!(sticky_down.to_decimal_string().as_deref(), Some("-0.00901"));
    assert!(status.contains(Status::INEXACT));

    let directed = BigFormat {
        precision: libbeef::Precision::Digits(3),
        rounding: Rounding::TowardNegative,
        ..BigFormat::DECIMAL64
    };
    let (rounded, status) =
        BigDecimal::from_i64(-12_345).add_status(&BigDecimal::from_i64(0), directed);
    assert_eq!(rounded.to_decimal_string().as_deref(), Some("-12400"));
    assert!(status.contains(Status::INEXACT));
}

#[test]
fn bftest_decimal_rejects_binary_precision_without_panicking() {
    let binary = BigFormat {
        precision: libbeef::Precision::Bits(64),
        ..BigFormat::BINARY64
    };

    let (quotient, div_status) =
        BigDecimal::from_i64(1).div_status(&BigDecimal::from_i64(3), binary);
    assert!(quotient.is_nan());
    assert!(div_status.contains(Status::INVALID_OP));

    let (root, sqrt_status) = BigDecimal::from_i64(2).sqrt_status(binary);
    assert!(root.is_nan());
    assert!(sqrt_status.contains(Status::INVALID_OP));
}

#[test]
fn bftest_decimal_add_mul_do_not_stop_at_i128_coefficients() {
    let f = BigFormat {
        precision: libbeef::Precision::Infinite,
        ..BigFormat::DECIMAL64
    };

    let ten_pow_18 = BigDecimal::from_i64(1_000_000_000_000_000_000);
    let ten_pow_36 = ten_pow_18.mul(&ten_pow_18, f);
    let ten_pow_72 = ten_pow_36.mul(&ten_pow_36, f);
    let one = BigDecimal::from_i64(1);

    let incremented = ten_pow_72.add(&one, f);
    assert_eq!(incremented.sub(&one, f), ten_pow_72);
    assert_eq!(
        incremented.to_decimal_string().as_deref(),
        Some("1000000000000000000000000000000000000000000000000000000000000000000000001")
    );
}

#[test]
fn bftest_decimal_rounding_does_not_stop_at_i128_coefficients() {
    let exact = BigFormat {
        precision: libbeef::Precision::Infinite,
        ..BigFormat::DECIMAL64
    };
    let rounded = BigFormat {
        precision: libbeef::Precision::Digits(4),
        rounding: Rounding::NearestEven,
        ..BigFormat::DECIMAL64
    };

    let ten_pow_18 = BigDecimal::from_i64(1_000_000_000_000_000_000);
    let ten_pow_36 = ten_pow_18.mul(&ten_pow_18, exact);
    let ten_pow_72 = ten_pow_36.mul(&ten_pow_36, exact);
    let one = BigDecimal::from_i64(1);

    let (value, status) = ten_pow_72
        .add(&one, exact)
        .add_status(&BigDecimal::new(), rounded);
    assert_eq!(value, ten_pow_72);
    assert!(status.contains(Status::INEXACT));
}

#[test]
fn bftest_decimal_rem_divrem_do_not_stop_at_i128_coefficients() {
    let f = BigFormat {
        precision: libbeef::Precision::Infinite,
        ..BigFormat::DECIMAL64
    };
    let a = BigDecimal::parse_decimal(
        "1000000000000000000000000000000000000000000000000000000000000000000012345",
    )
    .expect("parse large dividend");
    let b = BigDecimal::parse_decimal("1000000000000000000000000000000000000")
        .expect("parse large divisor");

    let r = a.rem(&b, f, DivRemMode::TowardZero);
    assert_eq!(r.to_decimal_string().as_deref(), Some("12345"));

    let ((q, r), status) = a.divrem_status(&b, f, DivRemMode::TowardZero);
    assert!(status.is_empty());
    assert_eq!(r.to_decimal_string().as_deref(), Some("12345"));
    assert_eq!(q.mul(&b, f).add(&r, f), a);
}

#[test]
fn bftest_decimal_rint_does_not_stop_at_i128_coefficients() {
    let value = BigDecimal::parse_decimal(
        "1000000000000000000000000000000000000000000000000000000000000000000000000.5",
    )
    .expect("parse large decimal");
    let (rounded, status) = value.rint_status(Rounding::NearestEven);
    assert_eq!(
        rounded.to_decimal_string().as_deref(),
        Some("1000000000000000000000000000000000000000000000000000000000000000000000000")
    );
    assert!(status.contains(Status::INEXACT));

    let negative = BigDecimal::parse_decimal("-999999999999999999999999999999999999999.1")
        .expect("parse large negative decimal");
    let (rounded, status) = negative.rint_status(Rounding::TowardNegative);
    assert_eq!(
        rounded.to_decimal_string().as_deref(),
        Some("-1000000000000000000000000000000000000000")
    );
    assert!(status.contains(Status::INEXACT));
}

#[test]
fn bftest_decimal_div_does_not_stop_at_i128_coefficients() {
    let f = BigFormat {
        precision: libbeef::Precision::Digits(6),
        rounding: Rounding::NearestEven,
        ..BigFormat::DECIMAL64
    };
    let a = BigDecimal::parse_decimal(
        "1000000000000000000000000000000000000000000000000000000000000000000000001",
    )
    .expect("parse large dividend");
    let b = BigDecimal::parse_decimal("3").expect("parse divisor");

    let (q, status) = a.div_status(&b, f);
    assert_eq!(
        q.to_decimal_string().as_deref(),
        Some("333333000000000000000000000000000000000000000000000000000000000000000000")
    );
    assert!(status.contains(Status::INEXACT));

    let exact = BigFormat {
        precision: libbeef::Precision::Infinite,
        ..BigFormat::DECIMAL64
    };
    let (half, status) = a.div_status(&BigDecimal::from_i64(2), exact);
    assert_eq!(
        half.to_decimal_string().as_deref(),
        Some("500000000000000000000000000000000000000000000000000000000000000000000000.5")
    );
    assert!(status.is_empty());
}

#[test]
fn bftest_decimal_sqrt_does_not_stop_at_i128_coefficients() {
    let exact = BigFormat {
        precision: libbeef::Precision::Digits(80),
        rounding: Rounding::NearestEven,
        ..BigFormat::DECIMAL64
    };
    let root =
        BigDecimal::parse_decimal("100000000000000000000000000000000000000").expect("parse root");
    let square = root.mul(&root, exact);
    let (sqrt, status) = square.sqrt_status(exact);
    assert_eq!(sqrt, root);
    assert!(status.is_empty());

    let rounded = BigFormat {
        precision: libbeef::Precision::Digits(8),
        rounding: Rounding::NearestEven,
        ..BigFormat::DECIMAL64
    };
    let large_two = BigDecimal::parse_decimal(
        "2000000000000000000000000000000000000000000000000000000000000000000000000",
    )
    .expect("parse large two");
    let (sqrt, status) = large_two.sqrt_status(rounded);
    assert_eq!(
        sqrt.to_decimal_string().as_deref(),
        Some("1414213600000000000000000000000000000")
    );
    assert!(status.contains(Status::INEXACT));
}

#[test]
fn bftest_seeded_decimal_tiny_division_keeps_significant_digits() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_decimal_tiny_division(&mut rng, 256, 0);
    }
}

#[test]
fn bftest_seeded_decimal_integer_rem_divrem_rint_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_decimal_integer_rem_divrem_rint(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_seeded_decimal_integer_sqrt_for_several_seeds() {
    let f = BigFormat::DECIMAL64;
    let (neg_root, neg_status) = BigDecimal::from_i64(-1).sqrt_status(f);
    assert!(neg_root.is_nan());
    assert!(neg_status.contains(Status::INVALID_OP));
    let (inf_root, inf_status) = BigDecimal::from_i64(4).sqrt_status(BigFormat {
        precision: libbeef::Precision::Infinite,
        ..f
    });
    assert!(inf_root.is_nan());
    assert!(inf_status.contains(Status::INVALID_OP));
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_decimal_integer_sqrt(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_decimal_non_square_sqrt_rounds_to_digit_precision() {
    for (literal, precision, expected) in [
        ("2", 5, "1.4142"),
        ("3", 5, "1.7321"),
        ("0.02", 5, "0.14142"),
        ("200", 5, "14.142"),
    ] {
        let value = BigDecimal::parse_decimal(literal).expect("parse decimal");
        let (root, status) = value.sqrt_status(BigFormat {
            precision: libbeef::Precision::Digits(precision),
            rounding: Rounding::NearestEven,
            ..BigFormat::DECIMAL64
        });
        assert_eq!(root.to_decimal_string().as_deref(), Some(expected));
        assert!(status.contains(Status::INEXACT));
    }

    for (rounding, expected) in [
        (Rounding::TowardZero, "1.4142"),
        (Rounding::TowardNegative, "1.4142"),
        (Rounding::TowardPositive, "1.4143"),
        (Rounding::AwayFromZero, "1.4143"),
        (Rounding::NearestAway, "1.4142"),
    ] {
        let (root, status) = BigDecimal::from_i64(2).sqrt_status(BigFormat {
            precision: libbeef::Precision::Digits(5),
            rounding,
            ..BigFormat::DECIMAL64
        });
        assert_eq!(
            root.to_decimal_string().as_deref(),
            Some(expected),
            "sqrt(2) with {rounding:?}"
        );
        assert!(status.contains(Status::INEXACT));
    }
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_decimal_non_square_sqrt(&mut rng, 256, 0);
    }
}

#[test]
fn bftest_seeded_decimal_scaled_sqrt_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_decimal_scaled_sqrt(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_seeded_integer_rounding_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_integer_rounding(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_seeded_rint_binary_fractions_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_rint_binary_fractions(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_seeded_can_round_property_for_several_seeds() {
    assert!(!BigFloat::new().can_round(53, Rounding::NearestEven, 64));
    assert!(BigFloat::from_i64(1).can_round(53, Rounding::Faithful, 54));
    assert!(!BigFloat::from_i64(1).can_round(53, Rounding::Faithful, 53));
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_can_round_property(&mut rng, 512, 0);
    }
}

#[test]
fn bftest_seeded_mul_div_l2radix_for_several_seeds() {
    for value in [-257, -3, -2, -1, 0, 1, 2, 3, 257] {
        assert_eq!(mul_log2_radix(value, 2, false, false), Some(value));
        assert_eq!(mul_log2_radix(value, 2, true, false), Some(value));
        assert_eq!(mul_log2_radix(value, 4, false, false), Some(value * 2));
        assert_eq!(
            mul_log2_radix(value, 4, true, false),
            Some((value as f64 / 2.0).floor() as i64)
        );
        assert_eq!(
            mul_log2_radix(value, 4, true, true),
            Some((value as f64 / 2.0).ceil() as i64)
        );
    }
    assert_eq!(mul_log2_radix(1, 1, false, false), None);
    assert_eq!(mul_log2_radix(1, 37, false, false), None);
    assert_eq!(mul_log2_radix(10, 10, false, false), Some(33));
    assert_eq!(mul_log2_radix(10, 10, false, true), Some(34));
    assert_eq!(mul_log2_radix(1000, 10, true, false), Some(301));
    assert_eq!(mul_log2_radix(1000, 10, true, true), Some(302));
    assert_eq!(mul_log2_radix(-1000, 10, true, false), Some(-302));
    assert_eq!(mul_log2_radix(-1000, 10, true, true), Some(-301));
    assert_eq!(
        mul_log2_radix(123_456_789, 3, false, false),
        Some(195_674_381)
    );
    assert_eq!(mul_log2_radix(123_456_789, 3, true, true), Some(77_892_562));
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_mul_div_l2radix(&mut rng, 512, 0);
    }
}

fn round_i128_to_precision(value: i64, precision: u32, rounding: Rounding) -> i64 {
    if value == 0 {
        return 0;
    }
    let sign_negative = value < 0;
    let magnitude = value.unsigned_abs();
    let bit_len = 64 - magnitude.leading_zeros();
    if bit_len <= precision {
        return value;
    }
    let drop = bit_len - precision;
    let kept = magnitude >> drop;
    let discarded = magnitude & ((1_u64 << drop) - 1);
    let guard = ((discarded >> (drop - 1)) & 1) != 0;
    let sticky = drop > 1 && (discarded & ((1_u64 << (drop - 1)) - 1)) != 0;
    let add_one = match rounding {
        Rounding::NearestEven => guard && (sticky || (kept & 1) != 0),
        Rounding::TowardZero => false,
        Rounding::TowardPositive => !sign_negative && discarded != 0,
        Rounding::TowardNegative => sign_negative && discarded != 0,
        _ => false,
    };
    let rounded = (kept + u64::from(add_one)) << drop;
    if sign_negative {
        -(rounded as i64)
    } else {
        rounded as i64
    }
}

fn assert_correctly_rounded(low_prec: BigFloat, high_prec: BigFloat, op: &str) {
    let a = low_prec.to_f64(Rounding::NearestEven);
    let b = high_prec.to_f64(Rounding::NearestEven);
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "{op}: low_prec and high_prec disagree: {a:e} vs {b:e}"
    );
}

fn remainder_nearest_even_i64(a: i64, b: i64) -> i64 {
    let a_abs = a.unsigned_abs() as u128;
    let b_abs = b.unsigned_abs() as u128;
    let q = a_abs / b_abs;
    let r = a_abs % b_abs;
    let adjust = match (r * 2).cmp(&b_abs) {
        core::cmp::Ordering::Greater => true,
        core::cmp::Ordering::Equal => (q & 1) != 0,
        core::cmp::Ordering::Less => false,
    };
    let (magnitude, sign_negative) = if adjust {
        (b_abs - r, a >= 0)
    } else {
        (r, a < 0)
    };
    if magnitude == 0 {
        0
    } else if sign_negative {
        -(magnitude as i64)
    } else {
        magnitude as i64
    }
}

fn remainder_floor_i64(a: i64, b: i64) -> i64 {
    let trunc = a % b;
    if trunc != 0 && ((a < 0) != (b < 0)) {
        trunc + b
    } else {
        trunc
    }
}

fn remainder_euclidean_i64(a: i64, b: i64) -> i64 {
    let trunc = a % b;
    if trunc < 0 {
        trunc + b.abs()
    } else {
        trunc
    }
}

fn signed_zero_like_dividend(value: f64, dividend: f64) -> f64 {
    if value == 0.0 {
        value.copysign(dividend)
    } else {
        value
    }
}

fn add_scaled_decimal_i64(a: i64, a_scale: i32, b: i64, b_scale: i32) -> (i64, i32) {
    let scale = a_scale.max(b_scale);
    (
        a * pow10_i64(scale - a_scale) + b * pow10_i64(scale - b_scale),
        scale,
    )
}

fn align_scaled_decimal_i64(a: i64, a_scale: i32, b: i64, b_scale: i32) -> (i64, i64, i32) {
    let scale = a_scale.max(b_scale);
    (
        a * pow10_i64(scale - a_scale),
        b * pow10_i64(scale - b_scale),
        scale,
    )
}

fn pow10_i64(exp: i32) -> i64 {
    let mut value = 1_i64;
    for _ in 0..exp {
        value *= 10;
    }
    value
}

fn round_rational_i64(numerator: i64, denominator: i64, rounding: Rounding) -> i64 {
    let q = numerator / denominator;
    let r = numerator % denominator;
    if r == 0 {
        return q;
    }

    let sign_negative = numerator < 0;
    let abs_r = r.unsigned_abs();
    let abs_den = denominator.unsigned_abs();
    let add_one = match rounding {
        Rounding::NearestEven => {
            let twice = abs_r * 2;
            twice > abs_den || (twice == abs_den && (q & 1) != 0)
        }
        Rounding::NearestAway => abs_r * 2 >= abs_den,
        Rounding::TowardZero => false,
        Rounding::TowardPositive => !sign_negative,
        Rounding::TowardNegative => sign_negative,
        Rounding::AwayFromZero => true,
        Rounding::Faithful => false,
    };

    if add_one {
        q + if sign_negative { -1 } else { 1 }
    } else {
        q
    }
}

fn mul_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = vec![0_u64; a.len() + b.len()];
    for (i, &av) in a.iter().enumerate() {
        let mut carry = 0_u128;
        for (j, &bv) in b.iter().enumerate() {
            let k = i + j;
            let value = u128::from(out[k]) + u128::from(av) * u128::from(bv) + carry;
            out[k] = value as u64;
            carry = value >> 64;
        }
        let mut k = i + b.len();
        while carry != 0 {
            if k == out.len() {
                out.push(0);
            }
            let value = u128::from(out[k]) + carry;
            out[k] = value as u64;
            carry = value >> 64;
            k += 1;
        }
    }
    trim_test_limbs(&mut out);
    out
}

fn add_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n + 1);
    let mut carry = 0_u128;
    for i in 0..n {
        let value = u128::from(a.get(i).copied().unwrap_or(0))
            + u128::from(b.get(i).copied().unwrap_or(0))
            + carry;
        out.push(value as u64);
        carry = value >> 64;
    }
    if carry != 0 {
        out.push(carry as u64);
    }
    trim_test_limbs(&mut out);
    out
}

fn add_small_test_limbs(limbs: &mut Vec<u64>, value: u64) {
    let mut carry = u128::from(value);
    let mut i = 0;
    while carry != 0 {
        if i == limbs.len() {
            limbs.push(0);
        }
        let sum = u128::from(limbs[i]) + carry;
        limbs[i] = sum as u64;
        carry = sum >> 64;
        i += 1;
    }
}

fn shl_test_limbs(limbs: &mut Vec<u64>, shift: u32) {
    if limbs.is_empty() || shift == 0 {
        return;
    }
    let word_shift = (shift / 64) as usize;
    let bit_shift = shift % 64;
    if word_shift != 0 {
        let old_len = limbs.len();
        limbs.resize(old_len + word_shift, 0);
        for i in (0..old_len).rev() {
            limbs[i + word_shift] = limbs[i];
        }
        for limb in &mut limbs[..word_shift] {
            *limb = 0;
        }
    }
    if bit_shift != 0 {
        let mut carry = 0_u64;
        for limb in limbs.iter_mut() {
            let next = *limb >> (64 - bit_shift);
            *limb = (*limb << bit_shift) | carry;
            carry = next;
        }
        if carry != 0 {
            limbs.push(carry);
        }
    }
}

fn cmp_test_limbs(a: &[u64], b: &[u64]) -> core::cmp::Ordering {
    let mut a_len = a.len();
    let mut b_len = b.len();
    while a_len > 0 && a[a_len - 1] == 0 {
        a_len -= 1;
    }
    while b_len > 0 && b[b_len - 1] == 0 {
        b_len -= 1;
    }
    a_len.cmp(&b_len).then_with(|| {
        for i in (0..a_len).rev() {
            match a[i].cmp(&b[i]) {
                core::cmp::Ordering::Equal => {}
                order => return order,
            }
        }
        core::cmp::Ordering::Equal
    })
}

fn trim_test_limbs(limbs: &mut Vec<u64>) {
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
}

fn bf_rrandom(rng: &mut Mt19937_64, prec: u64) -> BigFloat {
    if prec == 0 {
        return BigFloat::new();
    }
    let n = prec.div_ceil(64) as usize;
    let m = rng.next_u64() % 4 + 1;
    let max_run_len = (prec / m).max(1);
    let mut cur_state = (rng.next_u64() & 1) != 0;
    let mut cur_len = rng.next_u64() % max_run_len + 1;
    let nb_bits = (n as u64) * 64;
    let mut tab = vec![0_u64; n];
    let mut bit_index = nb_bits - prec;
    while bit_index < nb_bits {
        let len = cur_len.min(nb_bits - bit_index);
        if cur_state {
            for j in 0..len {
                let idx = (bit_index + j) as usize;
                tab[idx / 64] |= 1_u64 << (idx % 64);
            }
        }
        bit_index += len;
        cur_len -= len;
        if cur_len == 0 {
            cur_len = rng.next_u64() % max_run_len + 1;
            cur_state = !cur_state;
        }
    }
    let bf = BigFloat::from_raw(Sign::Positive, 0, tab);
    let fmt = BigFormat {
        precision: libbeef::Precision::Bits(prec),
        rounding: Rounding::TowardZero,
        ..BigFormat::BINARY64
    };
    let (normalized, _) = bf.normalize_and_round(fmt);
    normalized
}

fn bf_rrandom_large(rng: &mut Mt19937_64, prec: u64) -> BigFloat {
    let prec1 = rng.next_u64() % (2 * prec) + 1;
    let mut a = bf_rrandom(rng, prec1);
    if !a.is_zero() {
        let sign = rng.next_u64() & 1 != 0;
        if sign {
            a = a.neg();
        }
    }
    a
}

fn bf_rrandom_int(rng: &mut Mt19937_64, prec: u64) -> BigFloat {
    let prec1 = rng.next_u64() % prec + 1;
    let mut a = bf_rrandom(rng, prec1);
    if !a.is_zero() {
        a.set_exp(a.raw_exp() + prec1 as i64);
    }
    if rng.next_u64() & 1 != 0 {
        a = a.neg();
    }
    a
}

#[test]
fn bftest_seeded_arbitrary_precision_add_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237, 1238] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_arbitrary_precision_add(&mut rng, 128, 53);
    }
}

#[test]
fn bftest_seeded_arbitrary_precision_mul_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237, 1238] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_arbitrary_precision_mul(&mut rng, 128, 53);
    }
}

#[test]
fn bftest_seeded_arbitrary_precision_div_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237, 1238] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_arbitrary_precision_div(&mut rng, 128, 53);
    }
}

#[test]
fn bftest_seeded_arbitrary_precision_sqrt_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237, 1238] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_arbitrary_precision_sqrt(&mut rng, 128, 53);
    }
}

#[test]
fn bftest_pi_matches_known_digits() {
    let f = BigFormat {
        precision: libbeef::Precision::Bits(256),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let pi = BigFloat::pi(f);
    let pi_f64 = pi.to_f64(Rounding::NearestEven);
    assert_eq!(pi_f64.to_bits(), core::f64::consts::PI.to_bits());
}

#[test]
fn bftest_log2_matches_known_digits() {
    let f = BigFormat {
        precision: libbeef::Precision::Bits(256),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let log2 = BigFloat::log2(f);
    let log2_f64 = log2.to_f64(Rounding::NearestEven);
    assert_eq!(log2_f64.to_bits(), core::f64::consts::LN_2.to_bits());
}

#[test]
fn bftest_decimal_pow_u64_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237, 1238] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_decimal_pow_u64(&mut rng, 256, 0);
    }
}

#[test]
fn bftest_seeded_higher_precision_transcendentals_for_several_seeds() {
    for seed in [1234, 1235, 1236, 1237, 1238] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_higher_precision_transcendentals(&mut rng, 16, 113);
    }
}

#[test]
fn bftest_rrandom_add_sub_matches_higher_precision() {
    for prec in [53_u64, 113, 256] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_add_sub(&mut rng, 128, prec);
        }
    }
}

#[test]
fn bftest_rrandom_mul_matches_higher_precision() {
    for prec in [53_u64, 113, 256] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_mul(&mut rng, 128, prec);
        }
    }
}

#[test]
fn bftest_rrandom_div_matches_higher_precision() {
    for prec in [53_u64, 113, 256] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_div(&mut rng, 128, prec);
        }
    }
}

#[test]
fn bftest_rrandom_sqrt_matches_higher_precision() {
    for prec in [53_u64, 113, 256] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_sqrt(&mut rng, 128, prec);
        }
    }
}

#[test]
fn bftest_rrandom_rint_matches_higher_precision() {
    for prec in [53_u64, 113, 256] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_rint(&mut rng, 128, prec);
        }
    }
}

#[test]
fn bftest_rrandom_round_matches_higher_precision() {
    for prec in [53_u64, 113, 256] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_round(&mut rng, 128, prec);
        }
    }
}

#[test]
fn bftest_rrandom_cmp_eq_lt_le() {
    for prec in [53_u64, 113, 256] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_cmp(&mut rng, 128, prec);
        }
    }
}

#[test]
fn bftest_rrandom_logic_ops() {
    for prec in [53_u64, 113, 256] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_logic(&mut rng, 128, prec);
        }
    }
}

#[test]
fn bftest_rrandom_fmod_matches_higher_precision() {
    for prec in [53_u64, 113] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_fmod(&mut rng, 128, prec);
        }
    }
}

#[test]
fn bftest_rrandom_rem_nearest_matches_higher_precision() {
    for prec in [53_u64, 113] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_rem(&mut rng, 128, prec);
        }
    }
}

#[test]
fn bftest_rrandom_can_round_property() {
    for prec in [8_u64, 53, 256] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_can_round(&mut rng, 128, prec);
        }
    }
}

#[test]
fn bftest_rrandom_decimal_add_mul_div_sqrt() {
    for seed in [1234, 1235, 1236, 1237, 1238] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_rrandom_decimal_ops(&mut rng, 128, 0);
    }
}

#[test]
fn bftest_rrandom_decimal_fmod_divrem_rint() {
    for seed in [1234, 1235, 1236, 1237, 1238] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_rrandom_decimal_fmod_divrem_rint(&mut rng, 128, 0);
    }
}

#[test]
fn bftest_parse_multi_radix_roundtrip() {
    use libbeef::parse::ParseOptions;

    for radix in [2_u8, 8, 10, 16] {
        let options = ParseOptions {
            radix,
            format: BigFormat {
                precision: libbeef::Precision::Bits(53),
                ..BigFormat::BINARY64
            },
            allow_hex_prefix: false,
            allow_binary_prefix: false,
            allow_octal_prefix: false,
            allow_nan_inf: true,
            return_radix_exponent: false,
        };

        let (nan, _) = BigFloat::parse("NaN", options).expect("parse NaN");
        assert!(nan.is_nan());

        let (inf, _) = BigFloat::parse("inf", options).expect("parse inf");
        assert!(inf.is_infinite());

        let (zero, _) = BigFloat::parse("0", options).expect("parse 0");
        assert!(zero.is_zero());
    }

    let hex_opts = ParseOptions {
        radix: 16,
        format: BigFormat {
            precision: libbeef::Precision::Bits(53),
            ..BigFormat::BINARY64
        },
        allow_hex_prefix: false,
        allow_binary_prefix: false,
        allow_octal_prefix: false,
        allow_nan_inf: true,
        return_radix_exponent: false,
    };
    let (val, _) = BigFloat::parse("1p0", hex_opts).expect("parse hex 1p0");
    assert_eq!(val, BigFloat::from_u64(1));

    let (val, _) = BigFloat::parse("1p10", hex_opts).expect("parse hex 1p10");
    assert_eq!(val, BigFloat::from_u64(1024));

    let dec_opts = ParseOptions {
        radix: 10,
        format: BigFormat {
            precision: libbeef::Precision::Bits(53),
            ..BigFormat::BINARY64
        },
        ..ParseOptions::default()
    };
    let (val, _) = BigFloat::parse("1.5", dec_opts).expect("parse 1.5");
    assert_eq!(val, BigFloat::from_f64(1.5));

    let (val, _) = BigFloat::parse("1.5e2", dec_opts).expect("parse 1.5e2");
    assert_eq!(val, BigFloat::from_f64(150.0));

    let auto_opts = ParseOptions {
        format: BigFormat {
            precision: libbeef::Precision::Bits(53),
            ..BigFormat::BINARY64
        },
        ..ParseOptions::auto_radix()
    };
    let (val, _) = BigFloat::parse("0xff", auto_opts).expect("parse 0xff");
    assert_eq!(val, BigFloat::from_u64(255));
}

#[test]
fn bftest_default_parse_rejects_prefixed_literals() {
    use libbeef::parse::ParseOptions;

    let default = ParseOptions::default();
    assert!(!default.allow_hex_prefix);
    assert!(!default.allow_binary_prefix);
    assert!(!default.allow_octal_prefix);

    let (val, n) = BigFloat::parse("0x123", default).expect("partial parse");
    assert_eq!(
        n, 1,
        "without allow_hex_prefix, 0x123 should only consume '0'"
    );
    assert!(val.is_zero());

    let (val, n) = BigFloat::parse("0b101", default).expect("partial parse");
    assert_eq!(
        n, 1,
        "without allow_binary_prefix, 0b101 should only consume '0'"
    );
    assert!(val.is_zero());

    let (val, n) = BigFloat::parse("0o777", default).expect("partial parse");
    assert_eq!(
        n, 1,
        "without allow_octal_prefix, 0o777 should only consume '0'"
    );
    assert!(val.is_zero());

    let auto = ParseOptions::auto_radix();
    let (val, n) = BigFloat::parse("0x123", auto).expect("parse 0x123");
    assert_eq!(n, 5);
    assert_eq!(val, BigFloat::from_u64(0x123));

    let (val, n) = BigFloat::parse("0b101", auto).expect("parse 0b101");
    assert_eq!(n, 5);
    assert_eq!(val, BigFloat::from_u64(0b101));

    let (val, n) = BigFloat::parse("0o777", auto).expect("parse 0o777");
    assert_eq!(n, 5);
    assert_eq!(val, BigFloat::from_u64(0o777));
}

#[test]
fn bftest_seeded_atof_roundtrip_multi_radix() {
    for seed in [1234, 1235, 1236, 1237, 1238] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_atof_roundtrip_multi_radix(&mut rng, 128, 0);
    }
}

#[test]
fn bftest_to_string_radix_power_of_two() {
    let one = BigFloat::from_u64(1);
    let s = one.to_string_radix(16, 1, Rounding::NearestEven, true);
    assert!(s.is_some());

    let three = BigFloat::from_u64(3);
    let s = three.to_string_radix(2, 2, Rounding::NearestEven, true);
    assert!(s.is_some());
}

#[test]
fn bftest_decimal_to_from_bigfloat() {
    let f = BigFormat::BINARY64;
    let fd = BigFormat::DECIMAL64;

    let dec = BigDecimal::from_scaled_i64(125, 2);
    let (bf, _status) = dec.to_bigfloat(f);
    let bf_f64 = bf.to_f64(Rounding::NearestEven);
    assert_eq!(bf_f64, 1.25);

    let bf = BigFloat::from_f64(1.25);
    let (dec2, _status2) = BigDecimal::from_bigfloat(&bf, fd);
    assert_eq!(dec2.to_decimal_string().as_deref(), Some("1.25"));
    for seed in [1234, 1235, 1236, 1237, 1238] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_decimal_to_from_bigfloat(&mut rng, 64, 0);
    }
}

#[test]
fn bftest_rrandom_transcendental_exp_log() {
    for prec in [53_u64, 113] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_exp_log(&mut rng, 32, prec);
        }
    }
}

#[test]
fn bftest_rrandom_transcendental_sincos() {
    for prec in [53_u64, 113] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_sincos(&mut rng, 32, prec);
        }
    }
}

#[test]
fn bftest_rrandom_transcendental_atan_asin_acos() {
    for prec in [53_u64, 113] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_atan_asin_acos(&mut rng, 32, prec);
        }
    }
}

#[test]
fn bftest_rrandom_transcendental_atan2_pow() {
    for prec in [53_u64, 113] {
        for seed in [1234, 1235, 1236, 1237, 1238] {
            let mut rng = Mt19937_64::new(seed);
            ops::run_rrandom_atan2_pow(&mut rng, 32, prec);
        }
    }
}

#[test]
fn bftest_div_rem_and_rem_quo_match() {
    for seed in [1234, 1235, 1236, 1237] {
        let mut rng = Mt19937_64::new(seed);
        ops::run_div_rem_and_rem_quo_match(&mut rng, 256, 0);
    }
}
