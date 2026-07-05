use beef::{
    mp_recip, mp_sqrtrem, mul_log2_radix, BigDecimal, BigFloat, BigFormat, DivRemMode, Rounding,
    Status,
};

#[allow(unused_imports)]
use super::{
    add_limbs, add_scaled_decimal_i64, add_small_test_limbs, align_scaled_decimal_i64,
    assert_correctly_rounded, bf_rrandom, bf_rrandom_int, bf_rrandom_large, cmp_test_limbs,
    digit_char, expected_decimal_digit_round, mul_limbs, pow10_i64, remainder_euclidean_i64,
    remainder_floor_i64, remainder_nearest_even_i64, round_i128_to_precision, round_rational_i64,
    shl_test_limbs, signed_zero_like_dividend, special, special_dec, trim_test_limbs, Mt19937_64,
};

// ============================================================================
// Extracted parametric test functions
//
// Each function takes `&mut Mt19937_64` (already seeded by the caller) and a
// `count` controlling how many iterations to run.  Category-B functions also
// take a `prec` parameter.  Category-A functions accept `_prec: u64` so they
// share the same signature for the runner registry.
// ============================================================================

pub fn run_mp_sqrtrem(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let n = 1 + (rng.next_u64() % 5) as usize;
        let mut a = Vec::with_capacity(2 * n);
        for _ in 0..(2 * n) {
            a.push(rng.next_u64());
        }
        a[2 * n - 1] |= 1_u64 << 62;

        let (root, rem) = mp_sqrtrem(&a);
        let mut recomposed = mul_limbs(&root, &root);
        recomposed = add_limbs(&recomposed, &rem);
        trim_test_limbs(&mut recomposed);

        let mut expected = a.clone();
        trim_test_limbs(&mut expected);
        assert_eq!(recomposed, expected);

        let mut twice_root = root.clone();
        shl_test_limbs(&mut twice_root, 1);
        assert!(cmp_test_limbs(&rem, &twice_root) != core::cmp::Ordering::Greater);
    }
}

pub fn run_mp_recip(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let n = 1 + (rng.next_u64() % 5) as usize;
        let mut a = Vec::with_capacity(n);
        for _ in 0..n {
            a.push(rng.next_u64());
        }
        a[n - 1] |= 1_u64 << 63;

        let recip = mp_recip(&a).expect("non-zero denominator");
        assert_eq!(recip.len(), n + 1);
        let product = mul_limbs(&recip, &a);
        assert!(
            product.get(2 * n).copied().unwrap_or(0) < 1,
            "reciprocal product crossed 2^(128n)"
        );

        let mut incremented = recip.clone();
        add_small_test_limbs(&mut incremented, 2);
        let product = mul_limbs(&incremented, &a);
        assert!(
            product.get(2 * n).copied().unwrap_or(0) >= 1,
            "reciprocal plus two did not cross 2^(128n)"
        );
    }
}

pub fn run_small_integer_ops(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Infinite,
        ..BigFormat::BINARY64
    };
    for _ in 0..count {
        let a = rng.small_i64();
        let mut b = rng.small_i64();
        if b == 0 {
            b = 1;
        }

        let bf_a = BigFloat::from_i64(a);
        let bf_b = BigFloat::from_i64(b);

        assert_eq!(bf_a.add(&bf_b, f), BigFloat::from_i64(a + b));
        assert_eq!(bf_a.sub(&bf_b, f), BigFloat::from_i64(a - b));
        assert_eq!(bf_a.mul(&bf_b, f), BigFloat::from_i64(a * b));

        if a % b == 0 {
            assert_eq!(bf_a.div(&bf_b, f), BigFloat::from_i64(a / b));
        }
    }
}

pub fn run_exact_dyadic_division(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Infinite,
        ..BigFormat::BINARY64
    };
    for _ in 0..count {
        let numerator = rng.small_i64();
        let shift = (rng.next_u64() % 20) as i32;
        let denominator = BigFloat::from_i64(1_i64 << shift);
        let quotient = BigFloat::from_i64(numerator).div(&denominator, f);
        let (roundtrip, status) = quotient.to_f64_status(Rounding::NearestEven);
        assert!(status.is_empty());
        assert_eq!(
            roundtrip.to_bits(),
            ((numerator as f64) / 2.0_f64.powi(shift)).to_bits(),
            "{numerator} / 2^{shift}"
        );
        assert_eq!(quotient.mul(&denominator, f), BigFloat::from_i64(numerator));
    }
}

pub fn run_large_integer_limb_ops(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Infinite,
        ..BigFormat::BINARY64
    };
    for _ in 0..count {
        let a = rng.pow2_float(140, 300);
        let b = rng.pow2_float(140, 300);

        let sum = a.add(&b, f);
        assert_eq!(sum.sub(&b, f), a);
        assert_eq!(sum.sub(&a, f), b);

        let product = a.mul(&b, f);
        let one = BigFloat::from_i64(1);
        assert_eq!(product.mul(&one, f), product);
        assert_eq!(product.div(&a, f), b);
        assert_eq!(product.div(&b, f), a);
        assert_eq!(product.rem(&a, f, DivRemMode::TowardZero), BigFloat::new());
        assert_eq!(product.rem(&b, f, DivRemMode::TowardZero), BigFloat::new());

        assert_eq!(a.round(f), a);
        assert_eq!(a.rint(Rounding::NearestEven), a);

        let square = a.mul(&a, f);
        assert_eq!(square.sqrt(f), a.abs());
    }
}

pub fn run_float64_roundtrip(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let bits = rng.finite_f64_bits();
        let value = f64::from_bits(bits);
        let bf = BigFloat::from_f64(value);
        let (roundtrip, status) = bf.to_f64_status(Rounding::NearestEven);
        assert!(
            status.is_empty(),
            "unexpected status for {bits:#x}: {status:?}"
        );
        assert_eq!(roundtrip.to_bits(), bits);
    }
}

pub fn run_float64_add_mul(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat::BINARY64;
    for _ in 0..count {
        let a = rng.moderate_f64();
        let b = rng.moderate_f64();
        let bf_a = BigFloat::from_f64(a);
        let bf_b = BigFloat::from_f64(b);

        let sum = bf_a.add(&bf_b, f).to_f64(Rounding::NearestEven);
        assert_eq!(sum.to_bits(), (a + b).to_bits(), "{a:?} + {b:?}");

        let product = bf_a.mul(&bf_b, f).to_f64(Rounding::NearestEven);
        assert_eq!(product.to_bits(), (a * b).to_bits(), "{a:?} * {b:?}");
    }
}

pub fn run_float64_div(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat::BINARY64;
    for _ in 0..count {
        let a = rng.moderate_f64();
        let mut b = rng.moderate_f64();
        if b == 0.0 {
            b = 1.0;
        }
        let bf_a = BigFloat::from_f64(a);
        let bf_b = BigFloat::from_f64(b);

        let quotient = bf_a.div(&bf_b, f).to_f64(Rounding::NearestEven);
        assert_eq!(quotient.to_bits(), (a / b).to_bits(), "{a:?} / {b:?}");
    }
}

pub fn run_float64_fmod(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat::BINARY64;
    for _ in 0..count {
        let a = rng.moderate_f64();
        let mut b = rng.moderate_f64();
        if b == 0.0 {
            b = 1.0;
        }
        let bf_a = BigFloat::from_f64(a);
        let bf_b = BigFloat::from_f64(b);

        let remainder = bf_a
            .rem(&bf_b, f, DivRemMode::TowardZero)
            .to_f64(Rounding::NearestEven);
        assert_eq!(remainder.to_bits(), (a % b).to_bits(), "{a:?} % {b:?}");
    }
}

pub fn run_remainder_nearest_even(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let exact = BigFormat {
        precision: beef::Precision::Infinite,
        ..BigFormat::BINARY64
    };
    for _ in 0..count {
        let a = rng.small_i64();
        let mut b = rng.small_i64();
        if b == 0 {
            b = 1;
        }
        let bf_a = BigFloat::from_i64(a);
        let bf_b = BigFloat::from_i64(b);
        assert_eq!(
            bf_a.rem(&bf_b, exact, DivRemMode::NearestEven),
            BigFloat::from_i64(remainder_nearest_even_i64(a, b)),
            "{a:?} rem {b:?}"
        );
    }
}

pub fn run_remainder_floor_euclidean(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Infinite,
        ..BigFormat::BINARY64
    };
    for _ in 0..count {
        let a = rng.small_i64();
        let mut b = rng.small_i64();
        if b == 0 {
            b = 1;
        }
        let bf_a = BigFloat::from_i64(a);
        let bf_b = BigFloat::from_i64(b);
        assert_eq!(
            bf_a.rem(&bf_b, f, DivRemMode::Floor),
            BigFloat::from_i64(remainder_floor_i64(a, b)),
            "{a} rem_floor {b}"
        );
        assert_eq!(
            bf_a.rem(&bf_b, f, DivRemMode::Euclidean),
            BigFloat::from_i64(remainder_euclidean_i64(a, b)),
            "{a} rem_euclidean {b}"
        );

        let a = a as f64 / 8.0;
        let b = b as f64 / 8.0;
        if b == 0.0 {
            continue;
        }
        let bf_a = BigFloat::from_f64(a);
        let bf_b = BigFloat::from_f64(b);
        let floor_rem = bf_a
            .rem(&bf_b, BigFormat::BINARY64, DivRemMode::Floor)
            .to_f64(Rounding::NearestEven);
        let euclidean_rem = bf_a
            .rem(&bf_b, BigFormat::BINARY64, DivRemMode::Euclidean)
            .to_f64(Rounding::NearestEven);
        let expected_floor = signed_zero_like_dividend(a - (a / b).floor() * b, a);
        assert_eq!(
            floor_rem.to_bits(),
            expected_floor.to_bits(),
            "{a} rem_floor {b}"
        );
        let abs_b = b.abs();
        let expected_euclidean = signed_zero_like_dividend(
            {
                let trunc = a % b;
                if trunc < 0.0 {
                    trunc + abs_b
                } else {
                    trunc
                }
            },
            a,
        );
        assert_eq!(
            euclidean_rem.to_bits(),
            expected_euclidean.to_bits(),
            "{a} rem_euclidean {b}"
        );
    }
}

pub fn run_float64_sqrt(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat::BINARY64;
    for _ in 0..count {
        let a = rng.moderate_f64().abs();
        let bf_a = BigFloat::from_f64(a);
        let root = bf_a.sqrt(f).to_f64(Rounding::NearestEven);
        assert_eq!(root.to_bits(), a.sqrt().to_bits(), "sqrt({a:?})");
    }
}

pub fn run_float64_transcendentals(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat::BINARY64;
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(200),
        ..BigFormat::BINARY64
    };
    for iter in 0..count {
        let x = ((rng.next_u64() % 2001) as i64 - 1000) as f64 / 128.0;
        let bf_x = BigFloat::from_f64(x);

        if iter < 32 {
            assert_correctly_rounded(bf_x.exp(f), bf_x.exp(f_hi), "exp");
            assert_correctly_rounded(bf_x.cos(f), bf_x.cos(f_hi), "cos");
            assert_correctly_rounded(bf_x.sin(f), bf_x.sin(f_hi), "sin");
            assert_correctly_rounded(bf_x.tan(f), bf_x.tan(f_hi), "tan");
            assert_correctly_rounded(bf_x.atan(f), bf_x.atan(f_hi), "atan");
        }

        let positive = 2.0_f64.powi(-16) + (rng.next_u64() % 10_000) as f64 / 512.0;
        let bf_positive = BigFloat::from_f64(positive);
        if iter < 32 {
            assert_correctly_rounded(bf_positive.log(f), bf_positive.log(f_hi), "log");
        }

        let unit = ((rng.next_u64() % 2001) as i64 - 1000) as f64 / 1000.0;
        let bf_unit = BigFloat::from_f64(unit);
        if iter < 32 {
            assert_correctly_rounded(bf_unit.asin(f), bf_unit.asin(f_hi), "asin");
            assert_correctly_rounded(bf_unit.acos(f), bf_unit.acos(f_hi), "acos");
        }

        let y = ((rng.next_u64() % 2001) as i64 - 1000) as f64 / 128.0;
        let bf_y = BigFloat::from_f64(y);
        if iter < 32 {
            assert_correctly_rounded(bf_y.atan2(&bf_x, f), bf_y.atan2(&bf_x, f_hi), "atan2");
        }

        let base = 2.0_f64.powi(-8) + (rng.next_u64() % 2000) as f64 / 256.0;
        let exponent = ((rng.next_u64() % 1001) as i64 - 500) as f64 / 128.0;
        let bf_base = BigFloat::from_f64(base);
        let bf_exponent = BigFloat::from_f64(exponent);
        if iter < 32 {
            assert_correctly_rounded(
                bf_base.pow(&bf_exponent, f),
                bf_base.pow(&bf_exponent, f_hi),
                "pow",
            );
        }
    }
}

pub fn run_comparisons(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let a = rng.small_i64();
        let b = rng.small_i64();
        let bf_a = BigFloat::from_i64(a);
        let bf_b = BigFloat::from_i64(b);

        assert_eq!(bf_a == bf_b, a == b);
        assert_eq!(bf_a < bf_b, a < b);
        assert_eq!(bf_a <= bf_b, a <= b);
    }
}

pub fn run_nonnegative_logic_ops(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let a = rng.small_u64();
        let b = rng.small_u64();
        let bf_a = BigFloat::from_u64(a);
        let bf_b = BigFloat::from_u64(b);

        assert_eq!(bf_a.logic_or(&bf_b), BigFloat::from_u64(a | b));
        assert_eq!(bf_a.logic_xor(&bf_b), BigFloat::from_u64(a ^ b));
        assert_eq!(bf_a.logic_and(&bf_b), BigFloat::from_u64(a & b));
    }

    for _ in 0..count / 4 {
        let a = rng.pow2_float(140, 300).abs();
        let b = rng.pow2_float(140, 300).abs();
        assert_eq!(a.logic_or(&a), a);
        assert_eq!(a.logic_xor(&a), BigFloat::new());
        assert_eq!(a.logic_and(&a), a);
        assert_eq!(a.logic_or(&b), b.logic_or(&a));
        assert_eq!(a.logic_and(&b), b.logic_and(&a));
    }
}

pub fn run_signed_logic_ops(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let a = rng.small_i64();
        let b = rng.small_i64();
        let bf_a = BigFloat::from_i64(a);
        let bf_b = BigFloat::from_i64(b);

        assert_eq!(bf_a.logic_or(&bf_b), BigFloat::from_i64(a | b));
        assert_eq!(bf_a.logic_xor(&bf_b), BigFloat::from_i64(a ^ b));
        assert_eq!(bf_a.logic_and(&bf_b), BigFloat::from_i64(a & b));
    }
}

pub fn run_fractional_logic_ops(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let a_num = rng.small_i64();
        let b_num = rng.small_i64();
        let a = a_num as f64 / 8.0;
        let b = b_num as f64 / 8.0;
        let a_int = a as i64;
        let b_int = b as i64;
        let bf_a = BigFloat::from_f64(a);
        let bf_b = BigFloat::from_f64(b);

        assert_eq!(bf_a.logic_or(&bf_b), BigFloat::from_i64(a_int | b_int));
        assert_eq!(bf_a.logic_xor(&bf_b), BigFloat::from_i64(a_int ^ b_int));
        assert_eq!(bf_a.logic_and(&bf_b), BigFloat::from_i64(a_int & b_int));
    }
}

pub fn run_decimal_integer_parse_format(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let input = rng.integer_string(10);
        let value = BigFloat::parse_integer_radix(&input, 10).expect("parse integer");
        assert_eq!(
            value.to_integer_string_radix(10).as_deref(),
            Some(input.as_str())
        );
        let reparsed =
            BigFloat::parse_integer_radix(&value.to_integer_string_radix(10).unwrap(), 10)
                .expect("reparse integer");
        assert_eq!(reparsed, value);
    }
}

pub fn run_decimal_digit_rounding(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let input = rng.integer_string(10);
        let precision = 1 + (rng.next_u64() % 30) as usize;
        let rounding = match rng.next_u64() % 7 {
            0 => Rounding::NearestEven,
            1 => Rounding::TowardZero,
            2 => Rounding::TowardNegative,
            3 => Rounding::TowardPositive,
            4 => Rounding::NearestAway,
            5 => Rounding::AwayFromZero,
            _ => Rounding::Faithful,
        };
        let (expected, inexact) = expected_decimal_digit_round(&input, precision, rounding);
        let value = BigFloat::parse_integer_radix(&input, 10).expect("parse integer");
        let (rounded, status) = value.round_status(BigFormat {
            precision: beef::Precision::Digits(precision as u64),
            rounding,
            ..BigFormat::BINARY64
        });

        assert_eq!(
            rounded.to_decimal_integer_string().as_deref(),
            Some(expected.as_str()),
            "{input} precision={precision} rounding={rounding:?}"
        );
        assert_eq!(
            status.contains(Status::INEXACT),
            inexact,
            "{input} precision={precision} rounding={rounding:?}"
        );
    }
}

pub fn run_decimal_scaled_parse_format(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let coefficient = rng.small_i64();
        let scale = (rng.next_u64() % 8) as i32;
        let value = BigDecimal::from_scaled_i64(coefficient, scale);
        let formatted = value.to_decimal_string().expect("format scaled decimal");
        let reparsed = BigDecimal::parse_decimal(&formatted).expect("reparse scaled decimal");
        assert_eq!(reparsed, value, "{formatted}");
    }
}

pub fn run_radix_integer_parse_format(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let radix = 2 + (rng.next_u64() % 35) as u8;
        let input = rng.integer_string(radix);
        let value = BigFloat::parse_integer_radix(&input, radix).expect("parse integer");
        assert_eq!(
            value.to_integer_string_radix(radix).as_deref(),
            Some(input.as_str())
        );
    }
}

pub fn run_binary64_atof_ftoa_roundtrip(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat::BINARY64;
    for _ in 0..count {
        let bits = rng.finite_f64_bits();
        let value = f64::from_bits(bits);
        let literal = value.to_string();
        let parsed = BigFloat::parse_decimal(&literal, f).expect("parse f64 decimal");
        let formatted = parsed.to_decimal_string(f).expect("format f64 decimal");
        let reparsed = BigFloat::parse_decimal(&formatted, f).expect("reparse f64 decimal");
        let (roundtrip, status) = reparsed.to_f64_status(Rounding::NearestEven);
        assert!(status.is_empty(), "unexpected status for {literal}");
        assert_eq!(roundtrip.to_bits(), bits, "{literal}");
    }
}

pub fn run_decimal_integer_ops(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat::DECIMAL64;
    for _ in 0..count {
        let a = rng.small_i64();
        let mut b = rng.small_i64();
        if b == 0 {
            b = 1;
        }

        let dec_a = BigDecimal::from_i64(a);
        let dec_b = BigDecimal::from_i64(b);

        assert_eq!(dec_a.add(&dec_b, f), BigDecimal::from_i64(a + b));
        assert_eq!(dec_a.sub(&dec_b, f), BigDecimal::from_i64(a - b));
        assert_eq!(dec_a.mul(&dec_b, f), BigDecimal::from_i64(a * b));
        if a % b == 0 {
            assert_eq!(dec_a.div(&dec_b, f), BigDecimal::from_i64(a / b));
        }
    }
}

pub fn run_decimal_scaled_ops(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat::DECIMAL64;
    for _ in 0..count {
        let a_coeff = rng.small_i64();
        let b_coeff = {
            let value = rng.small_i64();
            if value == 0 {
                1
            } else {
                value
            }
        };
        let a_scale = (rng.next_u64() % 4) as i32;
        let b_scale = (rng.next_u64() % 4) as i32;
        let dec_a = BigDecimal::from_scaled_i64(a_coeff, a_scale);
        let dec_b = BigDecimal::from_scaled_i64(b_coeff, b_scale);

        let (sum_coeff, sum_scale) = add_scaled_decimal_i64(a_coeff, a_scale, b_coeff, b_scale);
        assert_eq!(
            dec_a.add(&dec_b, f),
            BigDecimal::from_scaled_i64(sum_coeff, sum_scale)
        );

        let (diff_coeff, diff_scale) = add_scaled_decimal_i64(a_coeff, a_scale, -b_coeff, b_scale);
        assert_eq!(
            dec_a.sub(&dec_b, f),
            BigDecimal::from_scaled_i64(diff_coeff, diff_scale)
        );

        assert_eq!(
            dec_a.mul(&dec_b, f),
            BigDecimal::from_scaled_i64(a_coeff * b_coeff, a_scale + b_scale)
        );

        let product = dec_a.mul(&dec_b, f);
        assert_eq!(product.div(&dec_b, f), dec_a);

        let (lhs, rhs, scale) = align_scaled_decimal_i64(a_coeff, a_scale, b_coeff, b_scale);
        assert_eq!(
            dec_a.rem(&dec_b, f, DivRemMode::TowardZero),
            BigDecimal::from_scaled_i64(lhs % rhs, scale)
        );
        assert_eq!(
            dec_a.rem(&dec_b, f, DivRemMode::NearestEven),
            BigDecimal::from_scaled_i64(remainder_nearest_even_i64(lhs, rhs), scale)
        );
        let ((q, r), status) = dec_a.divrem_status(&dec_b, f, DivRemMode::TowardZero);
        assert!(status.is_empty());
        assert_eq!(q.mul(&dec_b, f).add(&r, f), dec_a);
        let ((q, r), status) = dec_a.divrem_status(&dec_b, f, DivRemMode::NearestEven);
        assert!(status.is_empty());
        assert_eq!(q.mul(&dec_b, f).add(&r, f), dec_a);

        for rounding in [
            Rounding::NearestEven,
            Rounding::NearestAway,
            Rounding::TowardZero,
            Rounding::TowardPositive,
            Rounding::TowardNegative,
            Rounding::AwayFromZero,
            Rounding::Faithful,
        ] {
            let (rounded, status) = dec_a.rint_status(rounding);
            assert_eq!(
                rounded,
                BigDecimal::from_i64(round_rational_i64(a_coeff, pow10_i64(a_scale), rounding)),
                "{a_coeff}e-{a_scale} with {rounding:?}"
            );
            if a_coeff % pow10_i64(a_scale) == 0 {
                assert!(status.is_empty());
            } else {
                assert!(status.contains(Status::INEXACT));
            }
        }
    }
}

pub fn run_decimal_tiny_division(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Digits(4),
        rounding: Rounding::NearestEven,
        ..BigFormat::DECIMAL64
    };
    for _ in 0..count {
        let numerator = 1 + (rng.next_u64() % 9) as i64;
        let denominator = 100_000 + (rng.next_u64() % 900_000) as i64;
        let (quotient, status) =
            BigDecimal::from_i64(numerator).div_status(&BigDecimal::from_i64(denominator), f);
        if !status.contains(Status::INEXACT) {
            // Exact result (e.g. 5/500000 = 0.00001 fits in 4 sig digits);
            // consume the sign-choice RNG call to keep the stream aligned.
            rng.next_u64();
            continue;
        }

        let actual = quotient
            .to_decimal_string()
            .expect("decimal string")
            .parse::<f64>()
            .expect("parse decimal quotient");
        let expected = numerator as f64 / denominator as f64;
        assert!(
            (actual - expected).abs() <= expected * 5e-4,
            "{numerator}/{denominator} produced {actual}, expected {expected}"
        );

        let signed_numerator = if rng.next_u64() & 1 == 0 {
            numerator
        } else {
            -numerator
        };
        let expected = signed_numerator as f64 / denominator as f64;
        let down_format = BigFormat {
            rounding: Rounding::TowardNegative,
            ..f
        };
        let up_format = BigFormat {
            rounding: Rounding::TowardPositive,
            ..f
        };
        let (down, down_status) = BigDecimal::from_i64(signed_numerator)
            .div_status(&BigDecimal::from_i64(denominator), down_format);
        let (up, up_status) = BigDecimal::from_i64(signed_numerator)
            .div_status(&BigDecimal::from_i64(denominator), up_format);
        let down = down
            .to_decimal_string()
            .expect("decimal string")
            .parse::<f64>()
            .expect("parse downward quotient");
        let up = up
            .to_decimal_string()
            .expect("decimal string")
            .parse::<f64>()
            .expect("parse upward quotient");
        assert!(down_status.contains(Status::INEXACT));
        assert!(up_status.contains(Status::INEXACT));
        assert!(
            down <= expected,
            "{signed_numerator}/{denominator} rounded down to {down}, expected {expected}"
        );
        assert!(
            up >= expected,
            "{signed_numerator}/{denominator} rounded up to {up}, expected {expected}"
        );
    }
}

pub fn run_decimal_integer_rem_divrem_rint(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat::DECIMAL64;
    for _ in 0..count {
        let a = rng.small_i64();
        let mut b = rng.small_i64();
        if b == 0 {
            b = 1;
        }

        let dec_a = BigDecimal::from_i64(a);
        let dec_b = BigDecimal::from_i64(b);
        assert_eq!(
            dec_a.rem(&dec_b, f, DivRemMode::TowardZero),
            BigDecimal::from_i64(a % b)
        );
        let ((q, r), status) = dec_a.divrem_status(&dec_b, f, DivRemMode::TowardZero);
        assert!(status.is_empty());
        assert_eq!(q, BigDecimal::from_i64(a / b));
        assert_eq!(r, BigDecimal::from_i64(a % b));
        for (mode, expected_r) in [
            (DivRemMode::NearestEven, remainder_nearest_even_i64(a, b)),
            (DivRemMode::Floor, remainder_floor_i64(a, b)),
            (DivRemMode::Euclidean, remainder_euclidean_i64(a, b)),
        ] {
            assert_eq!(
                dec_a.rem(&dec_b, f, mode),
                BigDecimal::from_i64(expected_r),
                "{a} rem {b} with {mode:?}"
            );
            let ((q, r), status) = dec_a.divrem_status(&dec_b, f, mode);
            assert!(status.is_empty());
            assert_eq!(r, BigDecimal::from_i64(expected_r));
            assert_eq!(
                q.mul(&dec_b, f).add(&r, f),
                dec_a,
                "{a} divrem {b} with {mode:?}"
            );
        }
        assert_eq!(dec_a.rint(Rounding::NearestEven), dec_a);
    }
}

pub fn run_decimal_integer_sqrt(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat::DECIMAL64;
    for _ in 0..count {
        let root = rng.small_i64().unsigned_abs() as i64;
        let square = root * root;
        assert_eq!(
            BigDecimal::from_i64(square).sqrt(f),
            BigDecimal::from_i64(root)
        );
    }
}

pub fn run_decimal_non_square_sqrt(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let mut value = (rng.next_u64() % 10_000) as i64 + 2;
        let root = (value as f64).sqrt() as i64;
        if root * root == value {
            value += 1;
        }
        let (root, status) = BigDecimal::from_i64(value).sqrt_status(BigFormat {
            precision: beef::Precision::Digits(8),
            rounding: Rounding::NearestEven,
            ..BigFormat::DECIMAL64
        });
        assert!(status.contains(Status::INEXACT));
        let actual = root
            .to_decimal_string()
            .expect("decimal string")
            .parse::<f64>()
            .expect("parse rounded sqrt");
        let expected = (value as f64).sqrt();
        assert!(
            (actual - expected).abs() <= expected * 1e-7,
            "sqrt({value}) produced {actual}, expected {expected}"
        );

        let (down, down_status) = BigDecimal::from_i64(value).sqrt_status(BigFormat {
            precision: beef::Precision::Digits(8),
            rounding: Rounding::TowardZero,
            ..BigFormat::DECIMAL64
        });
        let (up, up_status) = BigDecimal::from_i64(value).sqrt_status(BigFormat {
            precision: beef::Precision::Digits(8),
            rounding: Rounding::TowardPositive,
            ..BigFormat::DECIMAL64
        });
        let down = down
            .to_decimal_string()
            .expect("decimal string")
            .parse::<f64>()
            .expect("parse downward sqrt");
        let up = up
            .to_decimal_string()
            .expect("decimal string")
            .parse::<f64>()
            .expect("parse upward sqrt");
        assert!(down_status.contains(Status::INEXACT));
        assert!(up_status.contains(Status::INEXACT));
        assert!(down <= expected, "sqrt({value}) rounded down to {down}");
        assert!(up >= expected, "sqrt({value}) rounded up to {up}");
    }
}

pub fn run_decimal_scaled_sqrt(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat::DECIMAL64;
    for _ in 0..count {
        let root_coeff = rng.small_i64().unsigned_abs() as i64;
        let root_scale = (rng.next_u64() % 4) as i32;
        let value = BigDecimal::from_scaled_i64(root_coeff * root_coeff, root_scale * 2);
        let expected = BigDecimal::from_scaled_i64(root_coeff, root_scale);
        assert_eq!(value.sqrt(f), expected);
    }
}

pub fn run_integer_rounding(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let value = rng.small_i64() * 257;
        let precision = 3 + (rng.next_u64() % 10);
        for rounding in [
            Rounding::NearestEven,
            Rounding::TowardZero,
            Rounding::TowardPositive,
            Rounding::TowardNegative,
        ] {
            let format = BigFormat {
                precision: beef::Precision::Bits(precision),
                rounding,
                ..BigFormat::BINARY64
            };
            let rounded = BigFloat::from_i64(value).round(format);
            let expected = round_i128_to_precision(value, precision as u32, rounding);
            assert_eq!(rounded, BigFloat::from_i64(expected));
        }
    }
}

pub fn run_rint_binary_fractions(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let numerator = rng.small_i64();
        let value = BigFloat::from_f64(numerator as f64 / 8.0);
        for rounding in [
            Rounding::NearestEven,
            Rounding::NearestAway,
            Rounding::TowardZero,
            Rounding::TowardPositive,
            Rounding::TowardNegative,
            Rounding::AwayFromZero,
        ] {
            let (rounded, status) = value.rint_status(rounding);
            assert_eq!(
                rounded,
                BigFloat::from_i64(round_rational_i64(numerator, 8, rounding)),
                "{numerator}/8 with {rounding:?}"
            );
            if numerator % 8 == 0 {
                assert!(status.is_empty());
            } else {
                assert!(status.contains(Status::INEXACT));
            }
        }
    }
}

pub fn run_can_round_property(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let a = BigFloat::from_f64(rng.moderate_f64());
        let precision = 2 + (rng.next_u64() % 52);
        let k = precision + (rng.next_u64() % 16);
        let rounding = match rng.next_u64() % 4 {
            0 => Rounding::NearestEven,
            1 => Rounding::TowardZero,
            2 => Rounding::TowardPositive,
            _ => Rounding::TowardNegative,
        };
        if !a.can_round(precision, rounding, k) {
            continue;
        }

        let format = BigFormat {
            precision: beef::Precision::Bits(precision),
            rounding,
            ..BigFormat::BINARY64
        };
        let rounded = a.round(format);
        let perturb_exp = a.raw_exp() - k as i64;
        if !(-1000..=1000).contains(&perturb_exp) {
            continue;
        }
        let perturb = BigFloat::from_f64(2.0_f64.powi((perturb_exp - 1) as i32));
        for c in [perturb.clone(), perturb.neg()] {
            let changed = a.add(
                &c,
                BigFormat {
                    precision: beef::Precision::Infinite,
                    ..BigFormat::BINARY64
                },
            );
            assert_eq!(
                changed.round(format),
                rounded,
                "can_round failed for {a:?}, precision={precision}, k={k}, rounding={rounding:?}"
            );
        }
    }
}

pub fn run_mul_div_l2radix(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let a = (rng.next_u64() % 2_000_001) as i64 - 1_000_000;
        let radix = 2 + (rng.next_u64() % 35) as u8;
        for is_inverse in [false, true] {
            for is_ceil in [false, true] {
                let log2_radix = f64::from(radix).log2();
                let value = if is_inverse {
                    (a as f64) / log2_radix
                } else {
                    (a as f64) * log2_radix
                };
                let expected = if is_ceil {
                    value.ceil() as i64
                } else {
                    value.floor() as i64
                };
                let actual = mul_log2_radix(a, radix, is_inverse, is_ceil).expect("valid radix");
                assert_eq!(
                    actual, expected,
                    "a={a}, radix={radix}, inv={is_inverse}, ceil={is_ceil}"
                );
            }
        }
    }
}

pub fn run_atof_roundtrip_multi_radix(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    for _ in 0..count {
        let radix = if rng.next_u64() & 1 != 0 {
            (rng.next_u64() % 35 + 2) as u8
        } else {
            10
        };
        let len = 1 + (rng.next_u64() % 20) as usize;
        let mut s = String::new();
        if rng.next_u64() & 1 != 0 {
            s.push('-');
        }
        s.push(digit_char(
            1 + (rng.next_u64() % u64::from(radix - 1)) as u8,
        ));
        for _ in 1..len {
            s.push(digit_char((rng.next_u64() % u64::from(radix)) as u8));
        }

        let parsed = BigFloat::parse_integer_radix(&s, radix);
        assert!(parsed.is_ok(), "failed to parse {s} in radix {radix}");
        let formatted = parsed.unwrap().to_integer_string_radix(radix);
        assert_eq!(
            formatted.as_deref(),
            Some(s.as_str()),
            "roundtrip failed for radix {radix}"
        );
    }
}

pub fn run_decimal_pow_u64(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Infinite,
        ..BigFormat::DECIMAL64
    };
    for _ in 0..count {
        let base = rng.small_i64();
        let exp = rng.next_u64() % 8;
        let dec_base = BigDecimal::from_i64(base);

        let mut expected = BigDecimal::from_i64(1);
        for _ in 0..exp {
            expected = expected.mul(&dec_base, f);
        }

        assert_eq!(dec_base.pow_u64(exp), expected, "{base}^{exp}");
    }
}

pub fn run_div_rem_and_rem_quo_match(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Infinite,
        ..BigFormat::BINARY64
    };
    for _ in 0..count {
        let a = rng.small_i64();
        let mut b = rng.small_i64();
        if b == 0 {
            b = 1;
        }

        let bf_a = BigFloat::from_i64(a);
        let bf_b = BigFloat::from_i64(b);

        for mode in [
            DivRemMode::TowardZero,
            DivRemMode::NearestEven,
            DivRemMode::Floor,
        ] {
            let (q, r, _status) = bf_a.div_rem_status(&bf_b, f, mode);
            assert_eq!(
                q.mul(&bf_b, f).add(&r, f),
                bf_a,
                "{a} divrem {b} mode={mode:?}"
            );

            let (_q_val, r2, _status2) = bf_a.rem_quo_status(&bf_b, f, mode);
            assert_eq!(r, r2, "{a} remquo remainder {b} mode={mode:?}");
        }
    }
}

pub fn run_decimal_to_from_bigfloat(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat::BINARY64;
    let fd = BigFormat::DECIMAL64;
    for _ in 0..count {
        let coeff = rng.small_i64();
        let scale = (rng.next_u64() % 4) as i32;
        let dec = BigDecimal::from_scaled_i64(coeff, scale);
        let (bf, _) = dec.to_bigfloat(f);
        let (dec_back, _) = BigDecimal::from_bigfloat(&bf, fd);
        if !dec.is_zero() && !dec_back.is_zero() {
            let dec_str = dec.to_decimal_string().unwrap_or_default();
            let dec_back_str = dec_back.to_decimal_string().unwrap_or_default();
            let d1: f64 = dec_str.parse().unwrap_or(0.0);
            let d2: f64 = dec_back_str.parse().unwrap_or(0.0);
            assert!(
                (d1 - d2).abs() < d1.abs() * 1e-10 || d1 == d2,
                "decimal roundtrip: {dec_str} -> {dec_back_str}"
            );
        }
    }
}

// ============================================================================
// Category B: tests parameterized by precision
// ============================================================================

pub fn run_higher_precision_transcendentals(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY128
    };
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(200),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY128
    };
    for _ in 0..count {
        let x = ((rng.next_u64() % 2001) as i64 - 1000) as f64 / 128.0;
        let bf_x = BigFloat::from_f64(x);

        let exp_lo = bf_x.exp(f);
        let exp_hi = bf_x.exp(f_hi);
        assert_eq!(exp_lo.round(f), exp_hi.round(f), "exp({x}) at {prec} bits");

        let sin_lo = bf_x.sin(f);
        let sin_hi = bf_x.sin(f_hi);
        assert_eq!(sin_lo.round(f), sin_hi.round(f), "sin({x}) at {prec} bits");

        let cos_lo = bf_x.cos(f);
        let cos_hi = bf_x.cos(f_hi);
        assert_eq!(cos_lo.round(f), cos_hi.round(f), "cos({x}) at {prec} bits");

        if x > 0.0 {
            let log_lo = bf_x.log(f);
            let log_hi = bf_x.log(f_hi);
            assert_eq!(log_lo.round(f), log_hi.round(f), "log({x}) at {prec} bits");
        }
    }
}

pub fn run_arbitrary_precision_add(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let (a, b) = if i < 49 {
            (special(i % 7), special(i / 7))
        } else {
            (
                BigFloat::from_f64(rng.moderate_f64()),
                BigFloat::from_f64(rng.moderate_f64()),
            )
        };

        let (result, _status) = a.add_status(&b, f);
        let (result_hi, _) = a.add_status(&b, f_hi);
        let result_hi_rounded = result_hi.round(f);

        if result.is_nan() && result_hi_rounded.is_nan() {
            continue;
        }
        if result.is_nan() || result_hi_rounded.is_nan() {
            continue;
        }
        let r_f64 = result.to_f64(Rounding::NearestEven);
        let rhi_f64 = result_hi_rounded.to_f64(Rounding::NearestEven);
        assert_eq!(r_f64.to_bits(), rhi_f64.to_bits(), "add at iter {i}");
    }
}

pub fn run_arbitrary_precision_mul(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let (a, b) = if i < 49 {
            (special(i % 7), special(i / 7))
        } else {
            (
                BigFloat::from_f64(rng.moderate_f64()),
                BigFloat::from_f64(rng.moderate_f64()),
            )
        };

        let (result, _status) = a.mul_status(&b, f);
        let (result_hi, _) = a.mul_status(&b, f_hi);
        let result_hi_rounded = result_hi.round(f);

        if result.is_nan() && result_hi_rounded.is_nan() {
            continue;
        }
        if result.is_nan() || result_hi_rounded.is_nan() {
            continue;
        }
        let r_f64 = result.to_f64(Rounding::NearestEven);
        let rhi_f64 = result_hi_rounded.to_f64(Rounding::NearestEven);
        assert_eq!(r_f64.to_bits(), rhi_f64.to_bits(), "mul at iter {i}");
    }
}

pub fn run_arbitrary_precision_div(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let (a, b) = if i < 49 {
            (special(i % 7), special(i / 7))
        } else {
            (
                BigFloat::from_f64(rng.moderate_f64()),
                BigFloat::from_f64(rng.moderate_f64()),
            )
        };

        let (result, _status) = a.div_status(&b, f);
        let (result_hi, _) = a.div_status(&b, f_hi);
        let result_hi_rounded = result_hi.round(f);

        if result.is_nan() && result_hi_rounded.is_nan() {
            continue;
        }
        if result.is_nan() || result_hi_rounded.is_nan() {
            continue;
        }
        let r_f64 = result.to_f64(Rounding::NearestEven);
        let rhi_f64 = result_hi_rounded.to_f64(Rounding::NearestEven);
        assert_eq!(r_f64.to_bits(), rhi_f64.to_bits(), "div at iter {i}");
    }
}

pub fn run_arbitrary_precision_sqrt(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let a = if i < 7 {
            special(i)
        } else {
            BigFloat::from_f64(rng.moderate_f64().abs())
        };

        let (result, _status) = a.sqrt_status(f);
        let (result_hi, _) = a.sqrt_status(f_hi);
        let result_hi_rounded = result_hi.round(f);

        if result.is_nan() && result_hi_rounded.is_nan() {
            continue;
        }
        if result.is_nan() || result_hi_rounded.is_nan() {
            continue;
        }
        let r_f64 = result.to_f64(Rounding::NearestEven);
        let rhi_f64 = result_hi_rounded.to_f64(Rounding::NearestEven);
        assert_eq!(r_f64.to_bits(), rhi_f64.to_bits(), "sqrt at iter {i}");
    }
}

pub fn run_rrandom_add_sub(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let (a, b) = if i < 49 {
            (special(i % 7), special(i / 7))
        } else {
            (bf_rrandom_large(rng, prec), bf_rrandom_large(rng, prec))
        };

        let result = a.add(&b, f);
        let result_hi = a.add(&b, f_hi).round(f);
        if result.is_nan() && result_hi.is_nan() {
            continue;
        }
        assert_eq!(
            result.cmp_total(&result_hi),
            core::cmp::Ordering::Equal,
            "add prec={prec} iter={i}"
        );

        let sub_result = a.sub(&b, f);
        let sub_result_hi = a.sub(&b, f_hi).round(f);
        if sub_result.is_nan() && sub_result_hi.is_nan() {
            continue;
        }
        assert_eq!(
            sub_result.cmp_total(&sub_result_hi),
            core::cmp::Ordering::Equal,
            "sub prec={prec} iter={i}"
        );
    }
}

pub fn run_rrandom_mul(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let (a, b) = if i < 49 {
            (special(i % 7), special(i / 7))
        } else {
            (bf_rrandom_large(rng, prec), bf_rrandom_large(rng, prec))
        };

        let result = a.mul(&b, f);
        let result_hi = a.mul(&b, f_hi).round(f);
        if result.is_nan() && result_hi.is_nan() {
            continue;
        }
        assert_eq!(
            result.cmp_total(&result_hi),
            core::cmp::Ordering::Equal,
            "mul prec={prec} iter={i}"
        );
    }
}

pub fn run_rrandom_div(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let (a, b) = if i < 49 {
            (special(i % 7), special(i / 7))
        } else {
            (bf_rrandom_large(rng, prec), bf_rrandom_large(rng, prec))
        };

        let result = a.div(&b, f);
        let result_hi = a.div(&b, f_hi).round(f);
        if result.is_nan() && result_hi.is_nan() {
            continue;
        }
        assert_eq!(
            result.cmp_total(&result_hi),
            core::cmp::Ordering::Equal,
            "div prec={prec} iter={i}"
        );
    }
}

pub fn run_rrandom_sqrt(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let a = if i < 7 {
            special(i)
        } else {
            let mut v = bf_rrandom_large(rng, prec);
            if v.is_sign_negative() && !v.is_zero() {
                v = v.neg();
            }
            v
        };

        let result = a.sqrt(f);
        let result_hi = a.sqrt(f_hi).round(f);
        if result.is_nan() && result_hi.is_nan() {
            continue;
        }
        assert_eq!(
            result.cmp_total(&result_hi),
            core::cmp::Ordering::Equal,
            "sqrt prec={prec} iter={i}"
        );
    }
}

pub fn run_rrandom_rint(rng: &mut Mt19937_64, count: usize, prec: u64) {
    for i in 0..count {
        let a = if i < 7 {
            special(i)
        } else {
            let mut v = bf_rrandom_large(rng, prec);
            if !v.is_zero() {
                let prec1 = rng.next_u64() % (3 * prec) + 1;
                v.set_exp(v.raw_exp() + (prec1 as i64) / 2);
            }
            v
        };

        for rounding in [
            Rounding::NearestEven,
            Rounding::TowardZero,
            Rounding::TowardPositive,
            Rounding::TowardNegative,
        ] {
            let (result, _status) = a.rint_status(rounding);
            if !a.is_finite() {
                continue;
            }
            let result2 = a.rint(rounding);
            assert_eq!(
                result.cmp_total(&result2),
                core::cmp::Ordering::Equal,
                "rint prec={prec} iter={i} rounding={rounding:?}"
            );
        }
    }
}

pub fn run_rrandom_round(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let a = if i < 7 {
            special(i)
        } else {
            bf_rrandom_large(rng, prec)
        };

        let (result, _status) = a.round_status(f);
        let (result_hi, _status_hi) = a.round_status(f_hi);
        let result_hi_rounded = result_hi.round(f);
        if result.is_nan() && result_hi_rounded.is_nan() {
            continue;
        }
        assert_eq!(
            result.cmp_total(&result_hi_rounded),
            core::cmp::Ordering::Equal,
            "round prec={prec} iter={i}"
        );
    }
}

pub fn run_rrandom_cmp(rng: &mut Mt19937_64, count: usize, prec: u64) {
    for i in 0..count {
        let (a, b) = if i < 49 {
            (special(i % 7), special(i / 7))
        } else {
            (bf_rrandom_large(rng, prec), bf_rrandom_large(rng, prec))
        };

        let eq = a == b;
        let lt = a.cmp_num(&b) == Some(core::cmp::Ordering::Less);
        let le = a
            .cmp_num(&b)
            .is_some_and(|o| o != core::cmp::Ordering::Greater);

        if a.is_nan() || b.is_nan() {
            assert!(!eq, "NaN should not be equal");
            assert!(!lt, "NaN should not be less");
            assert!(!le, "NaN should not be less-or-equal");
        }
    }
}

pub fn run_rrandom_logic(rng: &mut Mt19937_64, count: usize, prec: u64) {
    for _ in 0..count {
        let a = bf_rrandom_int(rng, prec);
        let b = bf_rrandom_int(rng, prec);

        let or_result = a.logic_or(&b);
        let _xor_result = a.logic_xor(&b);
        let and_result = a.logic_and(&b);

        assert_eq!(a.logic_or(&a), a, "x|x == x");
        assert_eq!(a.logic_xor(&a), BigFloat::new(), "x^x == 0");
        assert_eq!(a.logic_and(&a), a, "x&x == x");
        assert_eq!(or_result, b.logic_or(&a), "or commutativity");
        assert_eq!(and_result, b.logic_and(&a), "and commutativity");
    }
}

pub fn run_rrandom_fmod(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let (a, b) = if i < 49 {
            (special(i % 7), special(i / 7))
        } else {
            (bf_rrandom_large(rng, prec), bf_rrandom_large(rng, prec))
        };

        let result = a.rem(&b, f, DivRemMode::TowardZero);
        let result_hi = a.rem(&b, f_hi, DivRemMode::TowardZero).round(f);
        if result.is_nan() && result_hi.is_nan() {
            continue;
        }
        assert_eq!(
            result.cmp_total(&result_hi),
            core::cmp::Ordering::Equal,
            "fmod prec={prec} iter={i}"
        );
    }
}

pub fn run_rrandom_rem(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let (a, b) = if i < 49 {
            (special(i % 7), special(i / 7))
        } else {
            (bf_rrandom_large(rng, prec), bf_rrandom_large(rng, prec))
        };

        let result = a.rem(&b, f, DivRemMode::NearestEven);
        let result_hi = a.rem(&b, f_hi, DivRemMode::NearestEven).round(f);
        if result.is_nan() && result_hi.is_nan() {
            continue;
        }
        assert_eq!(
            result.cmp_total(&result_hi),
            core::cmp::Ordering::Equal,
            "rem prec={prec} iter={i}"
        );
    }
}

pub fn run_rrandom_can_round(rng: &mut Mt19937_64, count: usize, prec: u64) {
    for _ in 0..count {
        let prec1 = rng.next_u64() % (3 * prec) + 2;
        let mut a = bf_rrandom(rng, prec1);
        if rng.next_u64() & 1 != 0 {
            a = a.neg();
        }

        let k = prec + (rng.next_u64() % 10);
        let rounding = match rng.next_u64() % 4 {
            0 => Rounding::NearestEven,
            1 => Rounding::TowardZero,
            2 => Rounding::TowardPositive,
            _ => Rounding::TowardNegative,
        };

        if !a.can_round(prec, rounding, k) {
            continue;
        }

        let format = BigFormat {
            precision: beef::Precision::Bits(prec),
            rounding,
            ..BigFormat::BINARY64
        };
        let a_rounded = a.round(format);

        for _ in 0..20 {
            let mut c = bf_rrandom(rng, prec1);
            if rng.next_u64() & 1 != 0 {
                c = c.neg();
            }
            if !c.is_zero() {
                c.set_exp(c.raw_exp() + a.raw_exp() - k as i64);
            }

            let b = a.add(
                &c,
                BigFormat {
                    precision: beef::Precision::Infinite,
                    ..BigFormat::BINARY64
                },
            );
            let b_rounded = b.round(format);
            assert_eq!(
                b_rounded.cmp_total(&a_rounded),
                core::cmp::Ordering::Equal,
                "can_round failed for prec={prec} k={k} rounding={rounding:?}"
            );
        }
    }
}

pub fn run_rrandom_decimal_ops(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Digits(16),
        rounding: Rounding::NearestEven,
        ..BigFormat::DECIMAL64
    };
    let f_hi = BigFormat {
        precision: beef::Precision::Digits(32),
        rounding: Rounding::NearestEven,
        ..BigFormat::DECIMAL64
    };
    for i in 0..count {
        let (a, b) = if i < 49 {
            (special_dec(i % 7), special_dec(i / 7))
        } else {
            let a_coeff = rng.small_i64();
            let b_coeff = {
                let v = rng.small_i64();
                if v == 0 {
                    1
                } else {
                    v
                }
            };
            let a_scale = (rng.next_u64() % 4) as i32;
            let b_scale = (rng.next_u64() % 4) as i32;
            (
                BigDecimal::from_scaled_i64(a_coeff, a_scale),
                BigDecimal::from_scaled_i64(b_coeff, b_scale),
            )
        };

        let sum = a.add(&b, f);
        let sum_hi = a.add(&b, f_hi).add(&BigDecimal::new(), f);
        if sum.is_nan() && sum_hi.is_nan() {
            continue;
        }
        if !sum.is_nan() && !sum_hi.is_nan() {
            assert_eq!(sum, sum_hi, "dec add iter={i}");
        }

        let product = a.mul(&b, f);
        let product_hi = a.mul(&b, f_hi).add(&BigDecimal::new(), f);
        if product.is_nan() && product_hi.is_nan() {
            continue;
        }
        if !product.is_nan() && !product_hi.is_nan() {
            assert_eq!(product, product_hi, "dec mul iter={i}");
        }

        let quotient = a.div(&b, f);
        if !quotient.is_nan() && !a.is_nan() && !b.is_nan() && !b.is_zero() {
            let quotient_hi = a.div(&b, f_hi).add(&BigDecimal::new(), f);
            if !quotient_hi.is_nan() {
                assert_eq!(quotient, quotient_hi, "dec div iter={i}");
            }
        }

        if !a.is_nan() && !a.is_infinite() && !a.is_zero() && !a.neg().is_zero() {
            let a_pos = if a.to_decimal_string().unwrap_or_default().starts_with('-') {
                a.neg()
            } else {
                a.clone()
            };
            if !a_pos
                .to_decimal_string()
                .unwrap_or_default()
                .starts_with('-')
            {
                let sqrt_result = a_pos.sqrt(f);
                let sqrt_hi = a_pos.sqrt(f_hi).add(&BigDecimal::new(), f);
                if !sqrt_result.is_nan() && !sqrt_hi.is_nan() {
                    assert_eq!(sqrt_result, sqrt_hi, "dec sqrt iter={i}");
                }
            }
        }
    }
}

pub fn run_rrandom_decimal_fmod_divrem_rint(rng: &mut Mt19937_64, count: usize, _prec: u64) {
    let f = BigFormat::DECIMAL64;
    for _ in 0..count {
        let a_coeff = rng.small_i64();
        let b_coeff = {
            let v = rng.small_i64();
            if v == 0 {
                1
            } else {
                v
            }
        };
        let a_scale = (rng.next_u64() % 4) as i32;
        let b_scale = (rng.next_u64() % 4) as i32;
        let a = BigDecimal::from_scaled_i64(a_coeff, a_scale);
        let b = BigDecimal::from_scaled_i64(b_coeff, b_scale);

        let fmod = a.rem(&b, f, DivRemMode::TowardZero);
        let ((q, r), status) = a.divrem_status(&b, f, DivRemMode::TowardZero);
        assert_eq!(fmod, r, "fmod_dec != divrem remainder");
        if status.is_empty() {
            assert_eq!(q.mul(&b, f).add(&r, f), a, "divrem roundtrip");
        }

        for rounding in [
            Rounding::NearestEven,
            Rounding::TowardZero,
            Rounding::TowardPositive,
            Rounding::TowardNegative,
        ] {
            let rint_result = a.rint(rounding);
            let rint2 = a.rint(rounding);
            assert_eq!(rint_result, rint2, "rint consistency");
        }
    }
}

pub fn run_rrandom_exp_log(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let a = if i < 7 {
            special(i)
        } else {
            let mut v = bf_rrandom_large(rng, prec);
            if !v.is_zero() {
                let prec1 = rng.next_u64() % (3 * prec) + 1;
                v.set_exp(v.raw_exp() + (prec1 as i64) / 2);
            }
            v
        };

        let exp_lo = a.exp(f);
        let exp_hi = a.exp(f_hi).round(f);
        if exp_lo.is_nan() && exp_hi.is_nan() {
            continue;
        }
        if !exp_lo.is_nan() && !exp_hi.is_nan() {
            assert_eq!(
                exp_lo.cmp_total(&exp_hi),
                core::cmp::Ordering::Equal,
                "exp prec={prec} iter={i}"
            );
        }

        let a_pos = a.abs();
        if !a_pos.is_zero() && !a_pos.is_nan() && !a_pos.is_infinite() {
            let log_lo = a_pos.log(f);
            let log_hi = a_pos.log(f_hi).round(f);
            if !log_lo.is_nan() && !log_hi.is_nan() {
                assert_eq!(
                    log_lo.cmp_total(&log_hi),
                    core::cmp::Ordering::Equal,
                    "log prec={prec} iter={i}"
                );
            }
        }
    }
}

pub fn run_rrandom_sincos(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let a = if i < 7 {
            special(i)
        } else {
            let mut v = bf_rrandom_large(rng, prec);
            if !v.is_zero() {
                v.set_exp(v.raw_exp() + 1);
            }
            v
        };

        #[allow(clippy::type_complexity)]
        let ops: &[(&str, fn(&BigFloat, BigFormat) -> BigFloat)] = &[
            ("sin", BigFloat::sin),
            ("cos", BigFloat::cos),
            ("tan", BigFloat::tan),
        ];
        for &(op_name, op) in ops {
            let result_lo = op(&a, f);
            let result_hi = op(&a, f_hi).round(f);
            if result_lo.is_nan() && result_hi.is_nan() {
                continue;
            }
            if !result_lo.is_nan() && !result_hi.is_nan() {
                assert_eq!(
                    result_lo.cmp_total(&result_hi),
                    core::cmp::Ordering::Equal,
                    "{op_name} prec={prec} iter={i}"
                );
            }
        }
    }
}

pub fn run_rrandom_atan_asin_acos(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let a = if i < 7 {
            special(i)
        } else {
            bf_rrandom_large(rng, prec)
        };

        let atan_lo = a.atan(f);
        let atan_hi = a.atan(f_hi).round(f);
        if !atan_lo.is_nan() && !atan_hi.is_nan() {
            assert_eq!(
                atan_lo.cmp_total(&atan_hi),
                core::cmp::Ordering::Equal,
                "atan prec={prec} iter={i}"
            );
        }

        if a.cmp_abs(&BigFloat::from_i64(1)) != core::cmp::Ordering::Greater {
            let asin_lo = a.asin(f);
            let asin_hi = a.asin(f_hi).round(f);
            if !asin_lo.is_nan() && !asin_hi.is_nan() {
                assert_eq!(
                    asin_lo.cmp_total(&asin_hi),
                    core::cmp::Ordering::Equal,
                    "asin prec={prec} iter={i}"
                );
            }
            let acos_lo = a.acos(f);
            let acos_hi = a.acos(f_hi).round(f);
            if !acos_lo.is_nan() && !acos_hi.is_nan() {
                assert_eq!(
                    acos_lo.cmp_total(&acos_hi),
                    core::cmp::Ordering::Equal,
                    "acos prec={prec} iter={i}"
                );
            }
        }
    }
}

pub fn run_rrandom_atan2_pow(rng: &mut Mt19937_64, count: usize, prec: u64) {
    let f = BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    let f_hi = BigFormat {
        precision: beef::Precision::Bits(prec * 2 + 64),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    };
    for i in 0..count {
        let (a, b) = if i < 49 {
            (special(i % 7), special(i / 7))
        } else {
            (bf_rrandom_large(rng, prec), bf_rrandom_large(rng, prec))
        };

        let atan2_lo = a.atan2(&b, f);
        let atan2_hi = a.atan2(&b, f_hi).round(f);
        if !atan2_lo.is_nan() && !atan2_hi.is_nan() {
            assert_eq!(
                atan2_lo.cmp_total(&atan2_hi),
                core::cmp::Ordering::Equal,
                "atan2 prec={prec} iter={i}"
            );
        }

        let a_pos = a.abs();
        if !a_pos.is_zero() && !a_pos.is_nan() && !a_pos.is_infinite() {
            let pow_lo = a_pos.pow(&b, f);
            let pow_hi = a_pos.pow(&b, f_hi).round(f);
            if !pow_lo.is_nan()
                && !pow_hi.is_nan()
                && !pow_lo.is_infinite()
                && !pow_hi.is_infinite()
                && !pow_lo.is_zero()
                && !pow_hi.is_zero()
            {
                assert_eq!(
                    pow_lo.cmp_total(&pow_hi),
                    core::cmp::Ordering::Equal,
                    "pow prec={prec} iter={i}"
                );
            }
        }
    }
}

// ============================================================================
// Test suite registry
// ============================================================================

pub struct TestSpec {
    pub name: &'static str,
    pub precisions: &'static [u64],
    pub default_count: usize,
    pub run: fn(&mut Mt19937_64, usize, u64),
}

pub const SUITE: &[TestSpec] = &[
    TestSpec {
        name: "mp_sqrtrem",
        precisions: &[0],
        default_count: 128,
        run: run_mp_sqrtrem,
    },
    TestSpec {
        name: "mp_recip",
        precisions: &[0],
        default_count: 128,
        run: run_mp_recip,
    },
    TestSpec {
        name: "small_integer_ops",
        precisions: &[0],
        default_count: 256,
        run: run_small_integer_ops,
    },
    TestSpec {
        name: "exact_dyadic_div",
        precisions: &[0],
        default_count: 512,
        run: run_exact_dyadic_division,
    },
    TestSpec {
        name: "large_limb_ops",
        precisions: &[0],
        default_count: 128,
        run: run_large_integer_limb_ops,
    },
    TestSpec {
        name: "f64_roundtrip",
        precisions: &[0],
        default_count: 512,
        run: run_float64_roundtrip,
    },
    TestSpec {
        name: "f64_add_mul",
        precisions: &[0],
        default_count: 512,
        run: run_float64_add_mul,
    },
    TestSpec {
        name: "f64_div",
        precisions: &[0],
        default_count: 512,
        run: run_float64_div,
    },
    TestSpec {
        name: "f64_fmod",
        precisions: &[0],
        default_count: 512,
        run: run_float64_fmod,
    },
    TestSpec {
        name: "rem_nearest_even",
        precisions: &[0],
        default_count: 512,
        run: run_remainder_nearest_even,
    },
    TestSpec {
        name: "rem_floor_euclidean",
        precisions: &[0],
        default_count: 512,
        run: run_remainder_floor_euclidean,
    },
    TestSpec {
        name: "f64_sqrt",
        precisions: &[0],
        default_count: 512,
        run: run_float64_sqrt,
    },
    TestSpec {
        name: "f64_transcendentals",
        precisions: &[0],
        default_count: 512,
        run: run_float64_transcendentals,
    },
    TestSpec {
        name: "comparisons",
        precisions: &[0],
        default_count: 512,
        run: run_comparisons,
    },
    TestSpec {
        name: "nonneg_logic",
        precisions: &[0],
        default_count: 512,
        run: run_nonnegative_logic_ops,
    },
    TestSpec {
        name: "signed_logic",
        precisions: &[0],
        default_count: 512,
        run: run_signed_logic_ops,
    },
    TestSpec {
        name: "frac_logic",
        precisions: &[0],
        default_count: 512,
        run: run_fractional_logic_ops,
    },
    TestSpec {
        name: "dec_int_parse_fmt",
        precisions: &[0],
        default_count: 256,
        run: run_decimal_integer_parse_format,
    },
    TestSpec {
        name: "dec_digit_round",
        precisions: &[0],
        default_count: 256,
        run: run_decimal_digit_rounding,
    },
    TestSpec {
        name: "dec_scaled_parse",
        precisions: &[0],
        default_count: 512,
        run: run_decimal_scaled_parse_format,
    },
    TestSpec {
        name: "radix_int_parse",
        precisions: &[0],
        default_count: 256,
        run: run_radix_integer_parse_format,
    },
    TestSpec {
        name: "b64_atof_ftoa",
        precisions: &[0],
        default_count: 512,
        run: run_binary64_atof_ftoa_roundtrip,
    },
    TestSpec {
        name: "dec_int_ops",
        precisions: &[0],
        default_count: 512,
        run: run_decimal_integer_ops,
    },
    TestSpec {
        name: "dec_scaled_ops",
        precisions: &[0],
        default_count: 512,
        run: run_decimal_scaled_ops,
    },
    TestSpec {
        name: "dec_tiny_div",
        precisions: &[0],
        default_count: 256,
        run: run_decimal_tiny_division,
    },
    TestSpec {
        name: "dec_rem_divrem_rint",
        precisions: &[0],
        default_count: 512,
        run: run_decimal_integer_rem_divrem_rint,
    },
    TestSpec {
        name: "dec_int_sqrt",
        precisions: &[0],
        default_count: 512,
        run: run_decimal_integer_sqrt,
    },
    TestSpec {
        name: "dec_nonsq_sqrt",
        precisions: &[0],
        default_count: 256,
        run: run_decimal_non_square_sqrt,
    },
    TestSpec {
        name: "dec_scaled_sqrt",
        precisions: &[0],
        default_count: 512,
        run: run_decimal_scaled_sqrt,
    },
    TestSpec {
        name: "int_rounding",
        precisions: &[0],
        default_count: 512,
        run: run_integer_rounding,
    },
    TestSpec {
        name: "rint_bin_frac",
        precisions: &[0],
        default_count: 512,
        run: run_rint_binary_fractions,
    },
    TestSpec {
        name: "can_round",
        precisions: &[0],
        default_count: 512,
        run: run_can_round_property,
    },
    TestSpec {
        name: "mul_div_l2radix",
        precisions: &[0],
        default_count: 512,
        run: run_mul_div_l2radix,
    },
    TestSpec {
        name: "atof_multi_radix",
        precisions: &[0],
        default_count: 128,
        run: run_atof_roundtrip_multi_radix,
    },
    TestSpec {
        name: "dec_pow_u64",
        precisions: &[0],
        default_count: 256,
        run: run_decimal_pow_u64,
    },
    TestSpec {
        name: "divrem_remquo",
        precisions: &[0],
        default_count: 256,
        run: run_div_rem_and_rem_quo_match,
    },
    TestSpec {
        name: "dec_to_from_bf",
        precisions: &[0],
        default_count: 64,
        run: run_decimal_to_from_bigfloat,
    },
    TestSpec {
        name: "hi_prec_transcend",
        precisions: &[113],
        default_count: 16,
        run: run_higher_precision_transcendentals,
    },
    TestSpec {
        name: "arb_prec_add",
        precisions: &[53],
        default_count: 128,
        run: run_arbitrary_precision_add,
    },
    TestSpec {
        name: "arb_prec_mul",
        precisions: &[53],
        default_count: 128,
        run: run_arbitrary_precision_mul,
    },
    TestSpec {
        name: "arb_prec_div",
        precisions: &[53],
        default_count: 128,
        run: run_arbitrary_precision_div,
    },
    TestSpec {
        name: "arb_prec_sqrt",
        precisions: &[53],
        default_count: 128,
        run: run_arbitrary_precision_sqrt,
    },
    TestSpec {
        name: "rr_add_sub",
        precisions: &[53, 113, 256],
        default_count: 128,
        run: run_rrandom_add_sub,
    },
    TestSpec {
        name: "rr_mul",
        precisions: &[53, 113, 256],
        default_count: 128,
        run: run_rrandom_mul,
    },
    TestSpec {
        name: "rr_div",
        precisions: &[53, 113, 256],
        default_count: 128,
        run: run_rrandom_div,
    },
    TestSpec {
        name: "rr_sqrt",
        precisions: &[53, 113, 256],
        default_count: 128,
        run: run_rrandom_sqrt,
    },
    TestSpec {
        name: "rr_rint",
        precisions: &[53, 113, 256],
        default_count: 128,
        run: run_rrandom_rint,
    },
    TestSpec {
        name: "rr_round",
        precisions: &[53, 113, 256],
        default_count: 128,
        run: run_rrandom_round,
    },
    TestSpec {
        name: "rr_cmp",
        precisions: &[53, 113, 256],
        default_count: 128,
        run: run_rrandom_cmp,
    },
    TestSpec {
        name: "rr_logic",
        precisions: &[53, 113, 256],
        default_count: 128,
        run: run_rrandom_logic,
    },
    TestSpec {
        name: "rr_fmod",
        precisions: &[53, 113],
        default_count: 128,
        run: run_rrandom_fmod,
    },
    TestSpec {
        name: "rr_rem",
        precisions: &[53, 113],
        default_count: 128,
        run: run_rrandom_rem,
    },
    TestSpec {
        name: "rr_can_round",
        precisions: &[8, 53, 256],
        default_count: 128,
        run: run_rrandom_can_round,
    },
    TestSpec {
        name: "rr_dec_ops",
        precisions: &[0],
        default_count: 128,
        run: run_rrandom_decimal_ops,
    },
    TestSpec {
        name: "rr_dec_fmod_rint",
        precisions: &[0],
        default_count: 128,
        run: run_rrandom_decimal_fmod_divrem_rint,
    },
    TestSpec {
        name: "rr_exp_log",
        precisions: &[53, 113],
        default_count: 32,
        run: run_rrandom_exp_log,
    },
    TestSpec {
        name: "rr_sincos",
        precisions: &[53, 113],
        default_count: 32,
        run: run_rrandom_sincos,
    },
    TestSpec {
        name: "rr_atan_asin_acos",
        precisions: &[53, 113],
        default_count: 32,
        run: run_rrandom_atan_asin_acos,
    },
    TestSpec {
        name: "rr_atan2_pow",
        precisions: &[53, 113],
        default_count: 32,
        run: run_rrandom_atan2_pow,
    },
];

// ============================================================================
// Continuous runner — like the original C bftest's main loop
//
// Run with:
//   cargo test --test bftest --release bftest_continuous -- --ignored --nocapture
//
// Environment variables:
//   BFTEST_SEED      starting seed (default 1234)
//   BFTEST_DURATION  milliseconds per (op, prec) combo (default 100)
//   BFTEST_SINGLE    if set, run one seed only (like -S in C bftest)
//   BFTEST_OP        filter: only run ops whose name contains this string
// ============================================================================

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore]
pub fn bftest_continuous() {
    let seed_start = env_u64("BFTEST_SEED", 1234);
    let duration_ms = env_u64("BFTEST_DURATION", 100);
    let single = std::env::var("BFTEST_SINGLE").is_ok();
    let op_filter = std::env::var("BFTEST_OP").ok();

    let suite: Vec<&TestSpec> = if let Some(ref filter) = op_filter {
        SUITE
            .iter()
            .filter(|s| s.name.contains(filter.as_str()))
            .collect()
    } else {
        SUITE.iter().collect()
    };

    eprintln!(
        "{:<24} {:>5} {:>5} {:>8} {:>12}",
        "OP", "PREC", "SEED", "COUNT", "ns/64bit"
    );

    let mut seed = seed_start;
    loop {
        for spec in &suite {
            for &prec in spec.precisions {
                let mut rng = Mt19937_64::new(seed);
                let deadline =
                    std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
                let mut total = 0_usize;
                let batch = spec.default_count;
                let start = std::time::Instant::now();
                loop {
                    (spec.run)(&mut rng, batch, prec);
                    total += batch;
                    if std::time::Instant::now() >= deadline {
                        break;
                    }
                }
                let elapsed = start.elapsed();
                let nb_limbs = if prec > 0 { prec.div_ceil(64) } else { 1 };
                let ns_per_limb = elapsed.as_nanos() as f64 / total as f64 / nb_limbs as f64;
                eprintln!(
                    "{:<24} {:>5} {:>5} {:>8} {:>12.1}",
                    spec.name, prec, seed, total, ns_per_limb
                );
            }
        }
        seed += 1;
        if single {
            break;
        }
    }
}

// ============================================================================
// Pure benchmark — measures only operation time, no verification overhead.
//
// Matches the C bftest methodology: pre-generate inputs, then time only
// the operation itself using per-call Instant measurements accumulated
// into a total, excluding setup / teardown / assertion costs.
//
// Run with:
//   cargo test --test bftest --release bftest_bench -- --ignored --nocapture
//
// Environment variables:
//   BFTEST_SEED      starting seed (default 1234)
//   BFTEST_DURATION  milliseconds per (op, prec) combo (default 100)
//   BFTEST_OP        filter: only run ops whose name contains this string
// ============================================================================

struct BenchSpec {
    name: &'static str,
    precisions: &'static [u64],
    run: fn(&mut Mt19937_64, u64, u64) -> (usize, u128),
}

fn make_format(prec: u64) -> BigFormat {
    BigFormat {
        precision: beef::Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    }
}

const BENCH_N: usize = 1024;

/// Number of pre-generated inputs. Reduced at very large precisions so that a
/// single pass over the inputs (the granularity of the deadline check) stays
/// short and input generation does not dominate wall-clock time.
fn bench_n(prec: u64) -> usize {
    if prec >= 100_000 {
        32
    } else if prec >= 20_000 {
        64
    } else {
        BENCH_N
    }
}

macro_rules! bench_unary {
    ($name:ident, $gen:expr, $op:expr) => {
        fn $name(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
            let f = make_format(prec);
            let gen_fn: fn(&mut Mt19937_64, u64) -> BigFloat = $gen;
            let inputs: Vec<BigFloat> = (0..bench_n(prec)).map(|_| gen_fn(rng, prec)).collect();
            let op_fn: fn(&BigFloat, BigFormat) -> BigFloat = $op;
            let deadline =
                std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
            let mut total = 0_usize;
            let mut accum_ns = 0_u128;
            let mut r = BigFloat::new();
            loop {
                for a in &inputs {
                    let t0 = std::time::Instant::now();
                    r = op_fn(a, f);
                    accum_ns += t0.elapsed().as_nanos();
                    total += 1;
                }
                if std::time::Instant::now() >= deadline {
                    break;
                }
            }
            std::hint::black_box(r);
            (total, accum_ns)
        }
    };
}

macro_rules! bench_binary {
    ($name:ident, $gen:expr, $op:expr) => {
        fn $name(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
            let f = make_format(prec);
            let gen_fn: fn(&mut Mt19937_64, u64) -> BigFloat = $gen;
            let inputs: Vec<(BigFloat, BigFloat)> = (0..bench_n(prec))
                .map(|_| (gen_fn(rng, prec), gen_fn(rng, prec)))
                .collect();
            let set_fn: fn(&mut BigFloat, &BigFloat, &BigFloat, BigFormat) -> beef::Status = $op;
            let deadline =
                std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
            let mut total = 0_usize;
            let mut accum_ns = 0_u128;
            let mut r = BigFloat::from_u64(1);
            r.round_assign(f);
            loop {
                for (a, b) in &inputs {
                    let t0 = std::time::Instant::now();
                    set_fn(&mut r, a, b, f);
                    accum_ns += t0.elapsed().as_nanos();
                    total += 1;
                }
                if std::time::Instant::now() >= deadline {
                    break;
                }
            }
            std::hint::black_box(r);
            (total, accum_ns)
        }
    };
}

fn gen_rrandom_large(rng: &mut Mt19937_64, prec: u64) -> BigFloat {
    loop {
        let v = bf_rrandom_large(rng, prec);
        if !v.is_zero() {
            return v;
        }
    }
}

fn gen_positive(rng: &mut Mt19937_64, prec: u64) -> BigFloat {
    let mut v = bf_rrandom_large(rng, prec);
    if v.is_sign_negative() && !v.is_zero() {
        v = v.neg();
    }
    v
}

fn gen_trig_arg(rng: &mut Mt19937_64, prec: u64) -> BigFloat {
    let mut v = bf_rrandom_large(rng, prec);
    if !v.is_zero() {
        v.set_exp(v.raw_exp() + 1);
    }
    v
}

fn gen_unit(rng: &mut Mt19937_64, prec: u64) -> BigFloat {
    bf_rrandom(rng, prec)
}

fn gen_rrandom_int(rng: &mut Mt19937_64, prec: u64) -> BigFloat {
    bf_rrandom_int(rng, prec)
}

fn bf_set_add(r: &mut BigFloat, a: &BigFloat, b: &BigFloat, f: BigFormat) -> beef::Status {
    r.set_add(a, b, f)
}
fn bf_set_sub(r: &mut BigFloat, a: &BigFloat, b: &BigFloat, f: BigFormat) -> beef::Status {
    r.set_sub(a, b, f)
}
fn bf_set_mul(r: &mut BigFloat, a: &BigFloat, b: &BigFloat, f: BigFormat) -> beef::Status {
    r.set_mul(a, b, f)
}
fn bf_round(a: &BigFloat, f: BigFormat) -> BigFloat {
    a.round(f)
}
fn bf_exp(a: &BigFloat, f: BigFormat) -> BigFloat {
    a.exp(f)
}
fn bf_log(a: &BigFloat, f: BigFormat) -> BigFloat {
    a.log(f)
}
fn bf_sin(a: &BigFloat, f: BigFormat) -> BigFloat {
    a.sin(f)
}
fn bf_cos(a: &BigFloat, f: BigFormat) -> BigFloat {
    a.cos(f)
}
fn bf_tan(a: &BigFloat, f: BigFormat) -> BigFloat {
    a.tan(f)
}
fn bf_atan_bench(a: &BigFloat, f: BigFormat) -> BigFloat {
    a.atan(f)
}
fn bf_asin_bench(a: &BigFloat, f: BigFormat) -> BigFloat {
    a.asin(f)
}
fn bf_acos_bench(a: &BigFloat, f: BigFormat) -> BigFloat {
    a.acos(f)
}
fn bf_set_atan2(r: &mut BigFloat, a: &BigFloat, b: &BigFloat, f: BigFormat) -> beef::Status {
    let (result, status) = a.atan2_status(b, f);
    *r = result;
    status
}
fn bf_set_pow(r: &mut BigFloat, a: &BigFloat, b: &BigFloat, f: BigFormat) -> beef::Status {
    let (result, status) = a.pow_status(b, f);
    *r = result;
    status
}

bench_binary!(bench_add, gen_rrandom_large, bf_set_add);
bench_binary!(bench_sub, gen_rrandom_large, bf_set_sub);
bench_binary!(bench_mul, gen_rrandom_large, bf_set_mul);
fn bench_div(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let f = make_format(prec);
    let inputs: Vec<(BigFloat, BigFloat)> = (0..bench_n(prec))
        .map(|_| {
            let a = gen_rrandom_large(rng, prec);
            let mut b = gen_rrandom_large(rng, prec);
            while b.is_zero() || b.is_nan() {
                b = gen_rrandom_large(rng, prec);
            }
            (a, b)
        })
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0_usize;
    let mut accum_ns = 0_u128;
    let mut r = BigFloat::from_u64(1);
    r.round_assign(f);
    loop {
        for (a, b) in &inputs {
            let t0 = std::time::Instant::now();
            r.set_div(a, b, f);
            accum_ns += t0.elapsed().as_nanos();
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    std::hint::black_box(r);
    (total, accum_ns)
}
fn bench_sqrt(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let f = make_format(prec);
    let inputs: Vec<BigFloat> = (0..bench_n(prec))
        .map(|_| gen_positive(rng, prec))
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0_usize;
    let mut accum_ns = 0_u128;
    let mut r = BigFloat::new();
    loop {
        for a in &inputs {
            let t0 = std::time::Instant::now();
            r.set_sqrt(a, f);
            accum_ns += t0.elapsed().as_nanos();
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    std::hint::black_box(r);
    (total, accum_ns)
}
bench_unary!(bench_round, gen_rrandom_large, bf_round);
bench_unary!(bench_exp, gen_rrandom_large, bf_exp);
bench_unary!(bench_log, gen_positive, bf_log);
bench_unary!(bench_sin, gen_trig_arg, bf_sin);
bench_unary!(bench_cos, gen_trig_arg, bf_cos);
bench_unary!(bench_tan, gen_trig_arg, bf_tan);
bench_unary!(bench_atan, gen_rrandom_large, bf_atan_bench);
bench_unary!(bench_asin, gen_unit, bf_asin_bench);
bench_unary!(bench_acos, gen_unit, bf_acos_bench);
bench_binary!(bench_atan2, gen_rrandom_large, bf_set_atan2);
bench_binary!(bench_pow, gen_positive, bf_set_pow);

fn bench_fmod(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let f = make_format(prec);
    let inputs: Vec<(BigFloat, BigFloat)> = (0..bench_n(prec))
        .map(|_| (gen_rrandom_large(rng, prec), gen_rrandom_large(rng, prec)))
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0_usize;
    let mut accum_ns = 0_u128;
    loop {
        for (a, b) in &inputs {
            let t0 = std::time::Instant::now();
            let r = a.rem(b, f, DivRemMode::TowardZero);
            accum_ns += t0.elapsed().as_nanos();
            std::hint::black_box(r);
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    (total, accum_ns)
}

fn bench_rem(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let f = make_format(prec);
    let inputs: Vec<(BigFloat, BigFloat)> = (0..bench_n(prec))
        .map(|_| (gen_rrandom_large(rng, prec), gen_rrandom_large(rng, prec)))
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0_usize;
    let mut accum_ns = 0_u128;
    loop {
        for (a, b) in &inputs {
            let t0 = std::time::Instant::now();
            let r = a.rem(b, f, DivRemMode::NearestEven);
            accum_ns += t0.elapsed().as_nanos();
            std::hint::black_box(r);
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    (total, accum_ns)
}

fn bench_rint(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let inputs: Vec<BigFloat> = (0..bench_n(prec))
        .map(|_| {
            let mut v = bf_rrandom_large(rng, prec);
            if !v.is_zero() {
                let prec1 = rng.next_u64() % (3 * prec) + 1;
                v.set_exp(v.raw_exp() + (prec1 as i64) / 2);
            }
            v
        })
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0_usize;
    let mut accum_ns = 0_u128;
    loop {
        for a in &inputs {
            let t0 = std::time::Instant::now();
            let r = a.rint(Rounding::NearestEven);
            accum_ns += t0.elapsed().as_nanos();
            std::hint::black_box(r);
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    (total, accum_ns)
}

fn bench_cmp(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let inputs: Vec<(BigFloat, BigFloat)> = (0..bench_n(prec))
        .map(|_| (gen_rrandom_large(rng, prec), gen_rrandom_large(rng, prec)))
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0_usize;
    let mut accum_ns = 0_u128;
    loop {
        for (a, b) in &inputs {
            let t0 = std::time::Instant::now();
            let eq = a == b;
            let lt = a.cmp_num(b) == Some(core::cmp::Ordering::Less);
            let le = a
                .cmp_num(b)
                .is_some_and(|o| o != core::cmp::Ordering::Greater);
            accum_ns += t0.elapsed().as_nanos();
            std::hint::black_box((eq, lt, le));
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    (total, accum_ns)
}

fn bench_logic(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let inputs: Vec<(BigFloat, BigFloat)> = (0..bench_n(prec))
        .map(|_| (gen_rrandom_int(rng, prec), gen_rrandom_int(rng, prec)))
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0_usize;
    let mut accum_ns = 0_u128;
    loop {
        for (a, b) in &inputs {
            let t0 = std::time::Instant::now();
            let r_or = a.logic_or(b);
            let r_xor = a.logic_xor(b);
            let r_and = a.logic_and(b);
            accum_ns += t0.elapsed().as_nanos();
            std::hint::black_box((r_or, r_xor, r_and));
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    (total, accum_ns)
}

fn bench_dec_add(rng: &mut Mt19937_64, _prec: u64, duration_ms: u64) -> (usize, u128) {
    let f = BigFormat {
        precision: beef::Precision::Digits(16),
        rounding: Rounding::NearestEven,
        ..BigFormat::DECIMAL64
    };
    let inputs: Vec<(BigDecimal, BigDecimal)> = (0..BENCH_N)
        .map(|_| {
            let a = BigDecimal::from_scaled_i64(rng.small_i64(), (rng.next_u64() % 4) as i32);
            let b = BigDecimal::from_scaled_i64(rng.small_i64(), (rng.next_u64() % 4) as i32);
            (a, b)
        })
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0_usize;
    let mut accum_ns = 0_u128;
    loop {
        for (a, b) in &inputs {
            let t0 = std::time::Instant::now();
            let r = a.add(b, f);
            accum_ns += t0.elapsed().as_nanos();
            std::hint::black_box(r);
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    (total, accum_ns)
}

fn bench_dec_mul(rng: &mut Mt19937_64, _prec: u64, duration_ms: u64) -> (usize, u128) {
    let f = BigFormat {
        precision: beef::Precision::Digits(16),
        rounding: Rounding::NearestEven,
        ..BigFormat::DECIMAL64
    };
    let inputs: Vec<(BigDecimal, BigDecimal)> = (0..BENCH_N)
        .map(|_| {
            let a = BigDecimal::from_scaled_i64(rng.small_i64(), (rng.next_u64() % 4) as i32);
            let b = BigDecimal::from_scaled_i64(rng.small_i64(), (rng.next_u64() % 4) as i32);
            (a, b)
        })
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0_usize;
    let mut accum_ns = 0_u128;
    loop {
        for (a, b) in &inputs {
            let t0 = std::time::Instant::now();
            let r = a.mul(b, f);
            accum_ns += t0.elapsed().as_nanos();
            std::hint::black_box(r);
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    (total, accum_ns)
}

fn bench_dec_div(rng: &mut Mt19937_64, _prec: u64, duration_ms: u64) -> (usize, u128) {
    let f = BigFormat {
        precision: beef::Precision::Digits(16),
        rounding: Rounding::NearestEven,
        ..BigFormat::DECIMAL64
    };
    let inputs: Vec<(BigDecimal, BigDecimal)> = (0..BENCH_N)
        .map(|_| {
            let a = BigDecimal::from_scaled_i64(rng.small_i64(), (rng.next_u64() % 4) as i32);
            let mut b_coeff = rng.small_i64();
            if b_coeff == 0 {
                b_coeff = 1;
            }
            let b = BigDecimal::from_scaled_i64(b_coeff, (rng.next_u64() % 4) as i32);
            (a, b)
        })
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0_usize;
    let mut accum_ns = 0_u128;
    loop {
        for (a, b) in &inputs {
            let t0 = std::time::Instant::now();
            let r = a.div(b, f);
            accum_ns += t0.elapsed().as_nanos();
            std::hint::black_box(r);
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    (total, accum_ns)
}

fn bench_dec_sqrt(rng: &mut Mt19937_64, _prec: u64, duration_ms: u64) -> (usize, u128) {
    let f = BigFormat {
        precision: beef::Precision::Digits(16),
        rounding: Rounding::NearestEven,
        ..BigFormat::DECIMAL64
    };
    let inputs: Vec<BigDecimal> = (0..BENCH_N)
        .map(|_| {
            let v = (rng.next_u64() % 10_000) as i64 + 1;
            BigDecimal::from_i64(v)
        })
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0_usize;
    let mut accum_ns = 0_u128;
    loop {
        for a in &inputs {
            let t0 = std::time::Instant::now();
            let r = a.sqrt(f);
            accum_ns += t0.elapsed().as_nanos();
            std::hint::black_box(r);
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    (total, accum_ns)
}

fn bench_dec_rint(rng: &mut Mt19937_64, _prec: u64, duration_ms: u64) -> (usize, u128) {
    let inputs: Vec<BigDecimal> = (0..BENCH_N)
        .map(|_| BigDecimal::from_scaled_i64(rng.small_i64(), (rng.next_u64() % 4) as i32))
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0_usize;
    let mut accum_ns = 0_u128;
    loop {
        for a in &inputs {
            let t0 = std::time::Instant::now();
            let r = a.rint(Rounding::NearestEven);
            accum_ns += t0.elapsed().as_nanos();
            std::hint::black_box(r);
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    (total, accum_ns)
}

const BENCH_SUITE: &[BenchSpec] = &[
    BenchSpec {
        name: "mul",
        precisions: &[53, 112, 256, 3000, 30000, 300000],
        run: bench_mul,
    },
    BenchSpec {
        name: "add",
        precisions: &[53, 112, 256, 3000, 30000, 300000],
        run: bench_add,
    },
    BenchSpec {
        name: "sub",
        precisions: &[53, 112, 256, 3000, 30000, 300000],
        run: bench_sub,
    },
    BenchSpec {
        name: "rint",
        precisions: &[53, 112, 256, 3000],
        run: bench_rint,
    },
    BenchSpec {
        name: "round",
        precisions: &[53, 112, 256, 3000],
        run: bench_round,
    },
    BenchSpec {
        name: "cmp",
        precisions: &[53, 112, 256, 3000],
        run: bench_cmp,
    },
    BenchSpec {
        name: "div",
        precisions: &[53, 112, 256, 3000, 30000, 300000],
        run: bench_div,
    },
    BenchSpec {
        name: "fmod",
        precisions: &[53, 112, 256],
        run: bench_fmod,
    },
    BenchSpec {
        name: "rem",
        precisions: &[53, 112, 256],
        run: bench_rem,
    },
    BenchSpec {
        name: "sqrt",
        precisions: &[53, 112, 256, 3000, 30000, 300000],
        run: bench_sqrt,
    },
    BenchSpec {
        name: "logic",
        precisions: &[53, 112, 256, 3000],
        run: bench_logic,
    },
    BenchSpec {
        name: "exp",
        precisions: &[53, 112, 256],
        run: bench_exp,
    },
    BenchSpec {
        name: "log",
        precisions: &[53, 112, 256],
        run: bench_log,
    },
    BenchSpec {
        name: "cos",
        precisions: &[53, 112, 256],
        run: bench_cos,
    },
    BenchSpec {
        name: "sin",
        precisions: &[53, 112, 256],
        run: bench_sin,
    },
    BenchSpec {
        name: "tan",
        precisions: &[53, 112, 256],
        run: bench_tan,
    },
    BenchSpec {
        name: "atan",
        precisions: &[53, 112, 256],
        run: bench_atan,
    },
    BenchSpec {
        name: "atan2",
        precisions: &[53, 112, 256],
        run: bench_atan2,
    },
    BenchSpec {
        name: "asin",
        precisions: &[53, 112, 256],
        run: bench_asin,
    },
    BenchSpec {
        name: "acos",
        precisions: &[53, 112, 256],
        run: bench_acos,
    },
    BenchSpec {
        name: "pow",
        precisions: &[53, 112, 256],
        run: bench_pow,
    },
    BenchSpec {
        name: "add_dec",
        precisions: &[0],
        run: bench_dec_add,
    },
    BenchSpec {
        name: "mul_dec",
        precisions: &[0],
        run: bench_dec_mul,
    },
    BenchSpec {
        name: "div_dec",
        precisions: &[0],
        run: bench_dec_div,
    },
    BenchSpec {
        name: "sqrt_dec",
        precisions: &[0],
        run: bench_dec_sqrt,
    },
    BenchSpec {
        name: "rint_dec",
        precisions: &[0],
        run: bench_dec_rint,
    },
];

#[test]
#[ignore]
pub fn bftest_bench() {
    let seed = env_u64("BFTEST_SEED", 1234);
    let duration_ms = env_u64("BFTEST_DURATION", 100);
    let op_filter = std::env::var("BFTEST_OP").ok();

    let suite: Vec<&BenchSpec> = if let Some(ref filter) = op_filter {
        BENCH_SUITE
            .iter()
            .filter(|s| s.name.contains(filter.as_str()))
            .collect()
    } else {
        BENCH_SUITE.iter().collect()
    };

    eprint!(
        "{:<20} {:>5} {:>5} {:>8} {:>10}",
        "OP", "PREC", "SEED", "CNT", "beef"
    );
    #[cfg(feature = "vs_num_bigint")]
    eprint!(" {:>10}", "num-bigint");
    #[cfg(feature = "vs_rug")]
    eprint!(" {:>10}", "rug");
    #[cfg(feature = "vs_malachite")]
    eprint!(" {:>10}", "malachite");
    eprintln!();

    for spec in &suite {
        for &prec in spec.precisions {
            let mut rng = Mt19937_64::new(seed);
            let (count, accum_ns) = (spec.run)(&mut rng, prec, duration_ms);
            let nb_limbs = if prec > 0 { prec.div_ceil(64) } else { 1 };
            let ns_per_limb = accum_ns as f64 / count as f64 / nb_limbs as f64;
            eprint!(
                "{:<20} {:>5} {:>5} {:>8} {:>10.1}",
                spec.name,
                if prec > 0 {
                    prec.to_string()
                } else {
                    "\u{2014}".to_string()
                },
                seed,
                count,
                ns_per_limb
            );

            #[cfg(feature = "vs_num_bigint")]
            {
                if let Some(bench_fn) = super::vs_num_bigint::bench_lookup(spec.name) {
                    let mut rng2 = Mt19937_64::new(seed);
                    let (c, ns) = bench_fn(&mut rng2, prec, duration_ms);
                    eprint!(" {:>10.1}", ns as f64 / c as f64 / nb_limbs as f64);
                } else {
                    eprint!(" {:>10}", "\u{2014}");
                }
            }

            #[cfg(feature = "vs_rug")]
            {
                if let Some(bench_fn) = super::vs_rug::bench_lookup(spec.name) {
                    let mut rng2 = Mt19937_64::new(seed);
                    let (c, ns) = bench_fn(&mut rng2, prec, duration_ms);
                    eprint!(" {:>10.1}", ns as f64 / c as f64 / nb_limbs as f64);
                } else {
                    eprint!(" {:>10}", "\u{2014}");
                }
            }

            #[cfg(feature = "vs_malachite")]
            {
                if let Some(bench_fn) = super::vs_malachite::bench_lookup(spec.name) {
                    let mut rng2 = Mt19937_64::new(seed);
                    let (c, ns) = bench_fn(&mut rng2, prec, duration_ms);
                    eprint!(" {:>10.1}", ns as f64 / c as f64 / nb_limbs as f64);
                } else {
                    eprint!(" {:>10}", "\u{2014}");
                }
            }

            eprintln!();
        }
    }
}
