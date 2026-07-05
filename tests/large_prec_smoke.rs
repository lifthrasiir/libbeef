// Smoke test: core ops stay consistent at very large precisions
// (div @3000+ was once dropped from the bench suite due to a bug; keep it covered).
use beef::{BigFloat, BigFormat, Precision, Rounding};

fn fmt(bits: u64) -> BigFormat {
    BigFormat {
        precision: Precision::Bits(bits),
        rounding: Rounding::NearestEven,
        exp_bits: beef::format::ExpBits::Max,
        ..BigFormat::BINARY64
    }
}

fn random_value(seed: &mut u64, bits: u64) -> BigFloat {
    // xorshift-based deterministic pseudo-random positive value with `bits` bits
    let limbs = bits.div_ceil(64);
    let mut v = BigFloat::from_u64(1);
    let f = fmt(bits + 64);
    for _ in 0..limbs {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        v.mul_u64_assign(u64::MAX, f);
        v.add_i64_assign((*seed >> 1) as i64, f);
    }
    v
}

#[test]
fn large_prec_div_mul_roundtrip() {
    for &bits in &[3000_u64, 30000, 300000] {
        let f = fmt(bits);
        let mut seed = 0x1234_5678_9abc_def0_u64 ^ bits;
        let a = random_value(&mut seed, bits);
        let b = random_value(&mut seed, bits);
        let q = a.div(&b, f);
        let back = q.mul(&b, f);
        // |back - a| / |a| <= 2^-(bits-3)
        let diff = back.sub(&a, f);
        if !diff.is_zero() {
            let rel_exp = diff.raw_exp() - a.raw_exp();
            assert!(
                rel_exp <= -(bits as i64 - 3),
                "div/mul roundtrip too inexact at {bits} bits: rel_exp={rel_exp}"
            );
        }
    }
}

#[test]
fn large_prec_sqrt_roundtrip() {
    for &bits in &[3000_u64, 30000, 300000] {
        let f = fmt(bits);
        let mut seed = 0x0fed_cba9_8765_4321_u64 ^ bits;
        let a = random_value(&mut seed, bits).abs();
        let r = a.sqrt(f);
        let back = r.mul(&r, f);
        let diff = back.sub(&a, f);
        if !diff.is_zero() {
            let rel_exp = diff.raw_exp() - a.raw_exp();
            assert!(
                rel_exp <= -(bits as i64 - 3),
                "sqrt roundtrip too inexact at {bits} bits: rel_exp={rel_exp}"
            );
        }
    }
}

#[test]
fn divnorm_large_all_ones_truncated_divisor() {
    // Regression: mp_divnorm_large truncates the divisor to n limbs and adds 1;
    // when those limbs are all ones the increment overflows and the reciprocal
    // must be taken as exactly B^n (libbf.c mp_divnorm_large "recip_done").
    // The port was missing that carry check and hit a divide-by-zero panic.
    let mut hex = String::new();
    for _ in 0..52 {
        hex.push_str("ffffffffffffffff");
    }
    for _ in 0..7 {
        hex.push_str("0000000000000000");
    }
    hex.push_str("0000000000000001"); // keep the low limb nonzero so it is not trimmed
    let b = BigFloat::parse_integer_radix(&hex, 16).unwrap();
    let a = BigFloat::from_u64(1);
    let f = fmt(3260); // precl = 51 limbs -> divnorm_large with nq < nb
    let q = a.div(&b, f);
    // 1/b: sanity-check by multiplying back
    let back = q.mul(&b, f);
    let one = BigFloat::from_u64(1);
    let diff = back.sub(&one, f);
    if !diff.is_zero() {
        assert!(
            diff.raw_exp() <= -(3260 - 3),
            "1/b roundtrip inexact: {}",
            diff.raw_exp()
        );
    }
}
