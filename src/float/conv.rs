use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::str::FromStr;

use crate::format::{mul_log2_radix, BigFormat, Precision, Rounding, BF_PREC_INF};
use crate::ftoa::{FormatStyle, ToStringOptions};
use crate::parse::{ParseFloatError, ParseOptions};
use crate::status::Status;

use super::{BigFloat, Sign};

const DIGITS_PER_LIMB_TABLE: [u8; 35] = [
    64, 40, 32, 27, 24, 22, 21, 20, 19, 18, 17, 17, 16, 16, 16, 15, 15, 15, 14, 14, 14, 14, 13, 13,
    13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12,
];

fn get_limb_radix(radix: u8) -> u64 {
    let k = DIGITS_PER_LIMB_TABLE[radix as usize - 2] as u32;
    let mut radixl = radix as u64;
    for _ in 1..k {
        radixl *= radix as u64;
    }
    radixl
}

fn ceil_log2_u64(a: u64) -> u32 {
    if a <= 1 {
        0
    } else {
        64 - (a - 1).leading_zeros()
    }
}

#[allow(dead_code)]
fn smod(a: i64, b: i64) -> u64 {
    let r = a % b;
    if r < 0 {
        (r + b) as u64
    } else {
        r as u64
    }
}

fn ceil_div_i64(a: i64, b: i64) -> i64 {
    if a >= 0 {
        (a + b - 1) / b
    } else {
        a / b
    }
}

fn to_digit(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'A'..=b'Z' => c - b'A' + 10,
        b'a'..=b'z' => c - b'a' + 10,
        _ => 36,
    }
}

fn digit_char(v: u8) -> u8 {
    if v < 10 {
        b'0' + v
    } else {
        b'a' + v - 10
    }
}

fn bf_pow_ui(base: &BigFloat, exp: u64, prec: u64, rounding: Rounding) -> (BigFloat, Status) {
    if exp == 0 {
        return (BigFloat::from_u64(1), Status::empty());
    }
    // libbf passes `BF_FLAG_EXT_EXP` whenever the precision is finite.
    let fmt = BigFormat {
        precision: Precision::Bits(prec),
        rounding,
        ..BigFormat::INTERNAL_EXT
    };
    let inf = BigFormat {
        precision: Precision::Infinite,
        ..BigFormat::INTERNAL
    };
    let actual_fmt = if prec == u64::MAX { inf } else { fmt };

    let mut r = base.clone();
    let mut status = Status::empty();
    let n_bits = 64 - exp.leading_zeros();
    for i in (0..n_bits - 1).rev() {
        let (sq, st) = r.sqr_status(actual_fmt);
        r = sq;
        status |= st;
        if (exp >> i) & 1 != 0 {
            let (m, st) = r.mul_status(base, actual_fmt);
            r = m;
            status |= st;
        }
    }
    (r, status)
}

fn bf_pow_ui_ui(a: u64, b: u64, prec: u64, rounding: Rounding) -> (BigFloat, Status) {
    if a == 10 && b <= 19 {
        static POW10: [u64; 20] = [
            1,
            10,
            100,
            1_000,
            10_000,
            100_000,
            1_000_000,
            10_000_000,
            100_000_000,
            1_000_000_000,
            10_000_000_000,
            100_000_000_000,
            1_000_000_000_000,
            10_000_000_000_000,
            100_000_000_000_000,
            1_000_000_000_000_000,
            10_000_000_000_000_000,
            100_000_000_000_000_000,
            1_000_000_000_000_000_000,
            10_000_000_000_000_000_000,
        ];
        return (BigFloat::from_u64(POW10[b as usize]), Status::empty());
    }
    let base = BigFloat::from_u64(a);
    bf_pow_ui(&base, b, prec, rounding)
}

fn bf_integer_from_radix_rec(
    tab: &[u64],
    n: u64,
    level: u32,
    n0: u64,
    radixl: u64,
    pow_tab: &mut Vec<BigFloat>,
) -> BigFloat {
    let inf = BigFormat {
        precision: Precision::Infinite,
        ..BigFormat::INTERNAL
    };
    if n == 1 {
        return BigFloat::from_u64(tab[0]);
    }

    let n2 = ((n0 * 2) >> (level + 1)).div_ceil(2);
    let n1 = n - n2;

    if level as usize >= pow_tab.len() {
        pow_tab.resize(level as usize + 1, BigFloat::new());
    }
    if pow_tab[level as usize].is_zero() && pow_tab[level as usize].limbs().is_empty() {
        pow_tab[level as usize] = bf_pow_ui_ui(radixl, n2, u64::MAX, Rounding::TowardZero).0;
    }
    let b = pow_tab[level as usize].clone();

    let r = bf_integer_from_radix_rec(&tab[n2 as usize..], n1, level + 1, n0, radixl, pow_tab);
    let r = r.mul(&b, inf);

    let t = bf_integer_from_radix_rec(tab, n2, level + 1, n0, radixl, pow_tab);
    r.add(&t, inf)
}

fn bf_integer_from_radix(tab: &[u64], n: u64, radix: u8) -> BigFloat {
    let radixl = get_limb_radix(radix);
    let pow_tab_len = ceil_log2_u64(n) as usize + 2;
    let mut pow_tab = vec![BigFloat::new(); pow_tab_len];
    bf_integer_from_radix_rec(tab, n, 0, n, radixl, &mut pow_tab)
}

#[allow(clippy::too_many_arguments)]
fn bf_integer_to_radix_rec(
    out: &mut [u64],
    a: &BigFloat,
    n: u64,
    level: u32,
    n0: u64,
    radixl: u64,
    radixl_bits: u32,
    pow_tab: &mut Vec<(BigFloat, BigFloat)>,
) {
    let inf = BigFormat {
        precision: Precision::Infinite,
        ..BigFormat::INTERNAL
    };

    if a.is_zero() {
        out[..n as usize].fill(0);
        return;
    }
    if n == 1 {
        out[0] = get_bits_from_bf(a, a.limbs().len() as i64 * 64 - a.raw_exp());
        return;
    }
    if n == 2 {
        let pos = a.limbs().len() as i64 * 64 - a.raw_exp();
        let lo = get_bits_from_bf(a, pos);
        let hi = get_bits_from_bf(a, pos + 64);
        let t = (hi as u128) << 64 | lo as u128;
        out[0] = (t % radixl as u128) as u64;
        out[1] = (t / radixl as u128) as u64;
        return;
    }

    let n2 = ((n0 * 2) >> (level + 1)).div_ceil(2);
    let n1 = n - n2;

    while pow_tab.len() <= level as usize {
        pow_tab.push((BigFloat::new(), BigFloat::new()));
    }

    if pow_tab[level as usize].0.is_zero() {
        let b = bf_pow_ui_ui(radixl, n2, u64::MAX, Rounding::TowardZero).0;
        let one = BigFloat::from_u64(1);
        let b_inv_prec = (n2 + 1) * radixl_bits as u64 + 2;
        let b_inv = one.div(
            &b,
            BigFormat {
                precision: Precision::Bits(b_inv_prec),
                rounding: Rounding::NearestEven,
                ..BigFormat::INTERNAL
            },
        );
        pow_tab[level as usize] = (b, b_inv);
    }

    let (ref b, ref b_inv) = pow_tab[level as usize].clone();

    let q_prec = n1 * radixl_bits as u64;
    let mut q = a.mul(
        b_inv,
        BigFormat {
            precision: Precision::Bits(q_prec),
            rounding: Rounding::NearestEven,
            ..BigFormat::INTERNAL
        },
    );
    q = q.rint(Rounding::TowardZero);

    let mut r = a.sub(&q.mul(b, inf), inf);

    while r.sign().is_negative() && !r.is_zero() {
        r = r.add(b, inf);
        q = q.sub(&BigFloat::from_u64(1), inf);
    }
    while r.cmp_abs(b) != core::cmp::Ordering::Less {
        r = r.sub(b, inf);
        q = q.add(&BigFloat::from_u64(1), inf);
    }

    bf_integer_to_radix_rec(
        &mut out[n2 as usize..],
        &q,
        n1,
        level + 1,
        n0,
        radixl,
        radixl_bits,
        pow_tab,
    );
    bf_integer_to_radix_rec(out, &r, n2, level + 1, n0, radixl, radixl_bits, pow_tab);
}

fn bf_integer_to_radix(a: &BigFloat, n: u64, radixl: u64) -> Vec<u64> {
    let mut out = vec![0u64; n as usize];
    let radixl_bits = ceil_log2_u64(radixl);
    let pow_tab_len = (ceil_log2_u64(n) as usize + 2) * 2;
    let mut pow_tab = Vec::with_capacity(pow_tab_len);
    bf_integer_to_radix_rec(&mut out, a, n, 0, n, radixl, radixl_bits, &mut pow_tab);
    out
}

fn get_bits_from_bf(a: &BigFloat, pos: i64) -> u64 {
    let limbs = a.limbs();
    let len = limbs.len();
    let i = pos >> 6;
    let p = (pos & 63) as u32;
    let a0 = if i >= 0 && (i as usize) < len {
        limbs[i as usize]
    } else {
        0
    };
    if p == 0 {
        a0
    } else {
        let a1 = if (i + 1) >= 0 && ((i + 1) as usize) < len {
            limbs[(i + 1) as usize]
        } else {
            0
        };
        (a0 >> p) | (a1 << (64 - p))
    }
}

fn bf_mul_pow_radix(
    t: &BigFloat,
    radix: u8,
    expn: i64,
    prec: u64,
    format: BigFormat,
) -> (BigFloat, Status) {
    if t.is_zero() || !t.is_finite() {
        return (t.clone(), Status::empty());
    }
    if expn == 0 {
        return t.round_status(format);
    }

    let e = expn.unsigned_abs();
    let expn_sign = expn < 0;

    if matches!(format.precision, Precision::Infinite) {
        // Infinite precision is only meaningful if the result is known to be
        // exact; a division that is not reports INEXACT.
        let (b, ret) = bf_pow_ui_ui(radix as u64, e, u64::MAX, Rounding::NearestEven);
        let inf = BigFormat {
            precision: Precision::Infinite,
            ..BigFormat::INTERNAL
        };
        let (r, st) = if expn_sign {
            t.div_status(
                &b,
                BigFormat {
                    precision: Precision::Bits(t.limbs().len() as u64 * 64),
                    rounding: Rounding::NearestEven,
                    ..BigFormat::INTERNAL
                },
            )
        } else {
            t.mul_status(&b, inf)
        };
        return (r, ret | st);
    }

    let mut ziv_extra_bits = 16_u64;
    loop {
        let prec1 = prec.saturating_add(ziv_extra_bits);
        let extra_bits = (ceil_log2_u64(e) * 2 + 1) as u64;
        let work_prec = prec1.saturating_add(extra_bits);
        let work_fmt = BigFormat {
            precision: Precision::Bits(work_prec),
            rounding: Rounding::NearestEven,
            ..BigFormat::INTERNAL_EXT
        };
        let (b, mut ret) = bf_pow_ui_ui(radix as u64, e, work_prec, Rounding::NearestEven);
        let overflow = !b.is_finite();
        let (r, st) = if expn_sign {
            t.div_status(&b, work_fmt)
        } else {
            t.mul_status(&b, work_fmt)
        };
        ret |= st;

        // Only an inexact intermediate result needs the rounding test.
        let need_more = ret.contains(Status::INEXACT)
            && !r.can_round(prec, format.rounding, prec1)
            && !overflow
            && r.is_finite();
        if !need_more || ziv_extra_bits > 100_000 {
            let (rounded, st) = r.round_status_owned(format);
            return (rounded, st | (ret & Status::INEXACT));
        }

        ziv_extra_bits = ziv_extra_bits + ziv_extra_bits / 2;
    }
}

fn bf_convert_to_radix(
    a: &BigFloat,
    radix: u8,
    p: u64,
    rnd_mode: Rounding,
    is_fixed_exponent: bool,
    initial_e: i64,
) -> (BigFloat, i64) {
    if a.is_zero() {
        return (BigFloat::new(), 0);
    }

    let mut e = if is_fixed_exponent {
        initial_e
    } else {
        1 + mul_log2_radix(a.raw_exp() - 1, radix, true, false).unwrap_or(0)
    };

    let _inf = BigFormat {
        precision: Precision::Infinite,
        ..BigFormat::INTERNAL
    };

    loop {
        let exp = p as i64 - e;
        let exp_sign = exp < 0;
        let abs_exp = exp.unsigned_abs();

        let prec0 = mul_log2_radix(p as i64, radix, false, true).unwrap_or(p as i64 * 4) as u64;
        let mut ziv_extra_bits = 16_u64;

        let r = loop {
            let prec = prec0 + ziv_extra_bits;
            let extra_bits = (ceil_log2_u64(abs_exp) * 2 + 1) as u64;
            let work_prec = prec + extra_bits;
            let work_fmt = BigFormat {
                precision: Precision::Bits(work_prec),
                rounding: Rounding::NearestEven,
                ..BigFormat::INTERNAL_EXT
            };

            let (pow, mut ret) =
                bf_pow_ui_ui(radix as u64, abs_exp, work_prec, Rounding::NearestEven);
            let (r, st) = if !exp_sign {
                a.mul_status(&pow, work_fmt)
            } else {
                a.div_status(&pow, work_fmt)
            };
            ret |= st;

            // If the result is not exact, check that it can be safely rounded
            // to an integer; otherwise retry with more precision. libbf has no
            // cap here: the intermediate power of the radix eventually becomes
            // exact, which makes the quotient exact as well.
            if ret.contains(Status::INEXACT)
                && !r.is_zero()
                && !r.can_round(r.raw_exp() as u64, rnd_mode, prec)
            {
                ziv_extra_bits = ziv_extra_bits + ziv_extra_bits / 2;
                continue;
            }
            break r.rint(rnd_mode);
        };

        if is_fixed_exponent {
            return (r, e);
        }

        let bound = bf_pow_ui_ui(radix as u64, p, u64::MAX, Rounding::TowardZero).0;
        if r.cmp_abs(&bound) == core::cmp::Ordering::Less {
            return (r, e);
        }
        e += 1;
    }
}

/// Port of libbf's `output_digits`. `a` must be an integer if the radix is
/// not a power of two. A dot is added before the `dot_pos` digit; `dot_pos ==
/// n_digits` does not display the dot. `n_digits >= 1`.
fn output_digits(out: &mut String, a: &BigFloat, radix: u8, n_digits: u64, dot_pos: u64) {
    let digits_per_limb: u64;
    let pos_incr: i64;
    let radix_bits: u32;
    let mut pos: i64;
    let mut first_buf_pos: u64;
    let radix_limbs: Vec<u64>;

    if radix.is_power_of_two() {
        radix_bits = radix.trailing_zeros();
        digits_per_limb = 64 / radix_bits as u64;
        pos_incr = (digits_per_limb * radix_bits as u64) as i64;
        // Digits are aligned relative to the radix point.
        pos = a.limbs().len() as i64 * 64
            + smod(a.raw_exp().wrapping_neg(), radix_bits as i64) as i64;
        first_buf_pos = 0;
        radix_limbs = Vec::new();
    } else {
        digits_per_limb = DIGITS_PER_LIMB_TABLE[radix as usize - 2] as u64;
        let radixl = get_limb_radix(radix);
        let n = n_digits.div_ceil(digits_per_limb);
        radix_limbs = bf_integer_to_radix(a, n, radixl);
        radix_bits = 0;
        pos = n as i64;
        pos_incr = 1;
        first_buf_pos = n * digits_per_limb - n_digits;
    }

    let mut buf: Vec<u8> = Vec::new();
    let mut buf_pos = digits_per_limb;
    let mut i = 0_u64;
    while i < n_digits {
        if buf_pos == digits_per_limb {
            pos -= pos_incr;
            if radix_bits == 0 {
                let v = radix_limbs.get(pos as usize).copied().unwrap_or(0);
                limb_to_a(&mut buf, v, radix, digits_per_limb as usize);
            } else {
                let v = get_bits_from_bf(a, pos);
                limb_to_a2(&mut buf, v, radix_bits, digits_per_limb as usize);
            }
            buf_pos = first_buf_pos;
            first_buf_pos = 0;
        }
        let l = if i < dot_pos {
            dot_pos
        } else {
            if i == dot_pos {
                out.push('.');
            }
            n_digits
        };
        let l = (digits_per_limb - buf_pos).min(l - i);
        for j in 0..l {
            out.push(buf[(buf_pos + j) as usize] as char);
        }
        buf_pos += l;
        i += l;
    }
}

/// Port of libbf's `limb_to_a2` (power of two radixes).
fn limb_to_a2(buf: &mut Vec<u8>, mut n: u64, radix_bits: u32, len: usize) {
    buf.resize(len, 0);
    let mask = (1_u64 << radix_bits) - 1;
    for i in (0..len).rev() {
        buf[i] = digit_char((n & mask) as u8);
        n >>= radix_bits;
    }
}

fn limb_to_a(buf: &mut Vec<u8>, mut n: u64, radix: u8, len: usize) {
    buf.resize(len, 0);
    for i in (0..len).rev() {
        let digit = (n % radix as u64) as u8;
        n /= radix as u64;
        buf[i] = digit_char(digit);
    }
}

impl BigFloat {
    /// Parses a number from a string in a given radix, returning the value and
    /// the number of bytes consumed. Corresponds to `bf_atof` in libbf.
    ///
    /// Uses a divide-and-conquer radix conversion algorithm for non-power-of-2
    /// radices; power-of-2 radices are handled by direct bit manipulation.
    ///
    /// # Exponent notation
    ///
    /// The exponent separator depends on the effective radix of the input:
    ///
    /// - **Radix 10**: `e` or `E` (e.g. `1.5e3` = 1500).
    /// - **Other radixes**: `@` (e.g. `1.5@3` in base 16 = 1.5 × 16³ = 5376).
    ///   This avoids ambiguity with `e`/`E` which are valid hex digits.
    /// - **Power-of-two radixes** additionally accept `p` or `P` as a
    ///   **binary** exponent (e.g. `1.8p4` in base 16 = 1.5 × 2⁴ = 24).
    ///
    /// The exponent value itself is always written in decimal.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat};
    /// use libbeef::parse::ParseOptions;
    ///
    /// let opts = ParseOptions::default();
    /// let (val, _consumed) = BigFloat::parse("3.14", opts).unwrap();
    /// let f = val.to_f64(libbeef::Rounding::NearestEven);
    /// assert!((f - 3.14).abs() < 1e-10);
    ///
    /// let hex_opts = ParseOptions { radix: 16, allow_hex_prefix: true, ..ParseOptions::default() };
    /// let (val, _consumed) = BigFloat::parse("0xff", hex_opts).unwrap();
    /// let f = val.to_f64(libbeef::Rounding::NearestEven);
    /// assert_eq!(f, 255.0);
    /// ```
    pub fn parse(input: &str, options: ParseOptions) -> Result<(Self, usize), ParseFloatError> {
        Self::parse_status(input, options).map(|(v, _s, n)| (v, n))
    }

    /// Parses a number from a string, returning the value, status flags, and
    /// bytes consumed. Corresponds to `bf_atof` in libbf. Returns the status
    /// flags as well.
    pub fn parse_status(
        input: &str,
        options: ParseOptions,
    ) -> Result<(Self, Status, usize), ParseFloatError> {
        let mut exponent = 0;
        bf_atof_internal(input, options, &mut exponent)
    }

    /// Parses a number from a string like [`parse_status`](Self::parse_status),
    /// additionally returning the radix exponent. Corresponds to `bf_atof2` in
    /// libbf.
    ///
    /// If [`ParseOptions::return_radix_exponent`] is set and the radix is not
    /// a power of two, the value is not scaled: the parsed number is the
    /// returned value times `radix` raised to the returned exponent, where the
    /// value is the digits read as an integer (padded with zeros to a whole
    /// number of limbs) and is not rounded. Otherwise the value is rounded as
    /// usual and the exponent is 0.
    ///
    /// The result is `(value, exponent, status, bytes consumed)`.
    ///
    /// ```
    /// use libbeef::parse::ParseOptions;
    /// use libbeef::BigFloat;
    ///
    /// let opts = ParseOptions { return_radix_exponent: true, ..ParseOptions::default() };
    /// let (mantissa, exp, _status, _n) =
    ///     BigFloat::parse_with_exponent_status("1.2345", opts).unwrap();
    /// // 1234500000000000000 * 10^-18
    /// assert_eq!(mantissa, BigFloat::from_u64(1_234_500_000_000_000_000));
    /// assert_eq!(exp, -18);
    /// ```
    pub fn parse_with_exponent_status(
        input: &str,
        options: ParseOptions,
    ) -> Result<(Self, i64, Status, usize), ParseFloatError> {
        let mut exponent = 0;
        let (v, st, n) = bf_atof_internal(input, options, &mut exponent)?;
        Ok((v, exponent, st, n))
    }

    /// Parses a floating-point literal in the given radix (2..=36), consuming
    /// the entire input string.
    ///
    /// Follows the same conventions as Rust's built-in
    /// `{integer}::from_str_radix`: the input must be fully consumed (no
    /// trailing characters), an optional leading `+` or `-` sign is accepted,
    /// and underscores are **not** accepted (unlike integer parsing).
    ///
    /// The value is parsed at infinite precision — no rounding is applied.
    /// To parse with a specific format, use [`BigFloat::parse`] with
    /// [`ParseOptions`].
    ///
    /// # Exponent notation
    ///
    /// For radix 10, the exponent separator is `e` or `E` (e.g. `1.5e3`).
    /// For radixes where `e` is a valid digit (radix > 14), the `@`
    /// character is used as the exponent separator instead (e.g.
    /// `1.5@3` in base 16 means 1.5 × 16³). For power-of-two radixes,
    /// `p` or `P` is also accepted as a **binary** exponent separator
    /// (e.g. `1.8p4` in base 16 means 1.5 × 2⁴ = 24).
    ///
    /// ```
    /// use libbeef::BigFloat;
    ///
    /// let v = BigFloat::from_str_radix("ff", 16).unwrap();
    /// assert_eq!(v.to_f64(libbeef::Rounding::NearestEven), 255.0);
    ///
    /// let v = BigFloat::from_str_radix("-1.8@1", 16).unwrap();
    /// assert_eq!(v.to_f64(libbeef::Rounding::NearestEven), -24.0);
    /// ```
    pub fn from_str_radix(input: &str, radix: u8) -> Result<Self, ParseFloatError> {
        if !(2..=36).contains(&radix) {
            return Err(ParseFloatError::new());
        }
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(ParseFloatError::new());
        }
        let options = ParseOptions {
            radix,
            format: BigFormat {
                precision: Precision::Infinite,
                ..BigFormat::INTERNAL
            },
            allow_hex_prefix: false,
            allow_binary_prefix: false,
            allow_octal_prefix: false,
            allow_nan_inf: true,
            return_radix_exponent: false,
        };
        let (val, consumed) = Self::parse(trimmed, options)?;
        if consumed != trimmed.len() {
            return Err(ParseFloatError::new());
        }
        Ok(val)
    }

    /// Converts this number to a string as described by `options`.
    /// Corresponds to `bf_ftoa` in libbf.
    ///
    /// Returns `None` if the radix is out of range, or if a free format is
    /// requested with a precision that is infinite (or too large) while the
    /// radix is not a power of two.
    ///
    /// ```
    /// use libbeef::ftoa::{FormatStyle, ToStringOptions};
    /// use libbeef::BigFloat;
    ///
    /// let x = BigFloat::from_f64(1.5e300);
    /// let shortest = ToStringOptions::default();
    /// assert_eq!(x.to_string_with(shortest).unwrap(), "1.5e300");
    ///
    /// let js = ToStringOptions { js_quirks: true, ..shortest };
    /// assert_eq!(x.to_string_with(js).unwrap(), "1.5e+300");
    ///
    /// let frac = ToStringOptions {
    ///     precision: 2,
    ///     style: FormatStyle::Fractional,
    ///     ..shortest
    /// };
    /// assert_eq!(BigFloat::from_f64(2.0 / 3.0).to_string_with(frac).unwrap(), "0.67");
    /// ```
    pub fn to_string_with(&self, options: ToStringOptions) -> Option<String> {
        bf_ftoa_internal(self, &options)
    }

    /// Converts this number to a string in the given radix (2..=36) with the
    /// specified number of significant digits. Corresponds to `bf_ftoa` in libbf.
    ///
    /// Uses Ziv's rounding technique and a divide-and-conquer conversion for
    /// non-power-of-2 radices. Returns `None` if the radix is out of range.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    ///
    /// let val = BigFloat::from_f64(255.0);
    /// let s = val.to_string_radix(10, 1, Rounding::NearestEven, true).unwrap();
    /// assert_eq!(s, "3e2");
    ///
    /// let hex = val.to_string_radix(16, 1, Rounding::NearestEven, true).unwrap();
    /// assert_eq!(hex, "1p8");
    /// ```
    pub fn to_string_radix(
        &self,
        radix: u8,
        n_digits: u64,
        rounding: Rounding,
        force_exp: bool,
    ) -> Option<String> {
        self.to_string_with(ToStringOptions {
            radix,
            precision: n_digits,
            rounding,
            style: FormatStyle::Fixed,
            force_exponent: force_exp,
            add_prefix: false,
            js_quirks: false,
        })
    }
}

/// Parses a decimal floating-point literal at infinite precision.
///
/// Accepts the same syntax as Rust's `f64` parsing: an optional sign,
/// decimal digits with an optional decimal point, and an optional `e`/`E`
/// exponent (e.g. `"3.14"`, `"-1e10"`, `"+0.5"`). Also accepts `NaN`,
/// `inf`, and `Infinity` (case-insensitive).
///
/// ```
/// use libbeef::BigFloat;
///
/// let v: BigFloat = "3.14".parse().unwrap();
/// assert!((v.to_f64(libbeef::Rounding::NearestEven) - 3.14).abs() < 1e-10);
///
/// let v: BigFloat = "-inf".parse().unwrap();
/// assert!(v.is_infinite());
/// ```
impl FromStr for BigFloat {
    type Err = ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_radix(s, 10)
    }
}

/// Parses the optional sign and digits of an exponent, like the exponent loop
/// of libbf's `bf_atof_internal`. `*p` is just after the exponent separator.
///
/// Stops accumulating once the exponent is too large for the result to be
/// anything but an overflow or underflow, returning `Err(is_negative)` with
/// `*p` at the first unconsumed digit.
fn parse_exponent(bytes: &[u8], p: &mut usize) -> Result<i64, bool> {
    let mut exp_is_neg = false;
    if *p < bytes.len() && bytes[*p] == b'+' {
        *p += 1;
    } else if *p < bytes.len() && bytes[*p] == b'-' {
        exp_is_neg = true;
        *p += 1;
    }
    let mut expn = 0_i64;
    while *p < bytes.len() {
        let c = to_digit(bytes[*p]);
        if c >= 10 {
            break;
        }
        if expn > (i64::MAX - 2 - 9) / 10 {
            return Err(exp_is_neg);
        }
        *p += 1;
        expn = expn * 10 + c as i64;
    }
    Ok(if exp_is_neg { -expn } else { expn })
}

/// The result of an exponent that overflowed in [`parse_exponent`].
fn exponent_overflow(is_neg: bool, exp_is_neg: bool, p: usize) -> (BigFloat, Status, usize) {
    let sign = Sign::from_negative(is_neg);
    if exp_is_neg {
        (BigFloat::zero(sign), Status::UNDERFLOW | Status::INEXACT, p)
    } else {
        (
            BigFloat::infinity(sign),
            Status::OVERFLOW | Status::INEXACT,
            p,
        )
    }
}

fn bf_atof_internal(
    input: &str,
    options: ParseOptions,
    pexponent: &mut i64,
) -> Result<(BigFloat, Status, usize), ParseFloatError> {
    let bytes = input.as_bytes();
    let mut pos = 0;

    let mut radix = options.radix;
    // Radix 0 selects the radix from the prefix, defaulting to 10.
    if radix == 1 || radix > 36 {
        return Err(ParseFloatError::new());
    }

    if options.allow_nan_inf
        && radix <= 16
        && bytes.len() >= 3
        && bytes[..3].eq_ignore_ascii_case(b"nan")
    {
        return Ok((BigFloat::nan(), Status::empty(), 3));
    }

    let mut is_neg = false;
    if pos < bytes.len() && bytes[pos] == b'+' {
        pos += 1;
    } else if pos < bytes.len() && bytes[pos] == b'-' {
        is_neg = true;
        pos += 1;
    }
    let p_start = pos;

    if pos < bytes.len() && bytes[pos] == b'0' && pos + 1 < bytes.len() {
        let next = bytes[pos + 1];
        // As in libbf, 0x needs radix 0 or 16 while 0o and 0b need radix 0.
        let prefix_radix = if (next == b'x' || next == b'X')
            && (radix == 0 || radix == 16)
            && options.allow_hex_prefix
        {
            16
        } else if (next == b'o' || next == b'O') && radix == 0 && options.allow_octal_prefix {
            8
        } else if (next == b'b' || next == b'B') && radix == 0 && options.allow_binary_prefix {
            2
        } else {
            0
        };
        if prefix_radix != 0 {
            radix = prefix_radix;
            pos += 2;
            // There must be a digit after the prefix.
            if pos >= bytes.len() || to_digit(bytes[pos]) >= radix {
                return Ok((BigFloat::nan(), Status::empty(), pos));
            }
        }
    } else if options.allow_nan_inf
        && radix <= 16
        && pos + 3 <= bytes.len()
        && bytes[pos..pos + 3].eq_ignore_ascii_case(b"inf")
    {
        let mut end = pos + 3;
        if end + 5 <= bytes.len() && bytes[end..end + 5].eq_ignore_ascii_case(b"inity") {
            end += 5;
        }
        return Ok((
            BigFloat::infinity(Sign::from_negative(is_neg)),
            Status::empty(),
            end,
        ));
    }

    if radix == 0 {
        radix = 10;
    }

    let is_power_of_two = radix.is_power_of_two();
    let radix_bits = if is_power_of_two {
        radix.trailing_zeros() as i32
    } else {
        0
    };

    let mut saw_leading_zero = false;
    while pos < bytes.len() && bytes[pos] == b'0' {
        saw_leading_zero = true;
        pos += 1;
    }

    if is_power_of_two {
        return parse_power_of_two(
            bytes,
            pos,
            p_start,
            radix,
            radix_bits,
            is_neg,
            options,
            saw_leading_zero,
        );
    }

    let digits_per_limb = DIGITS_PER_LIMB_TABLE[radix as usize - 2] as usize;
    let mut limb_digits: Vec<u64> = Vec::new();
    let mut cur_limb = 0_u64;
    let mut shift = digits_per_limb;
    let mut has_decpt = false;
    let mut int_len = 0_i64;
    let mut digit_count = 0_i64;

    let mut p = pos;
    loop {
        if p >= bytes.len() {
            break;
        }
        let ch = bytes[p];
        if ch == b'.'
            && !has_decpt
            && (p > p_start || (p + 1 < bytes.len() && to_digit(bytes[p + 1]) < radix))
        {
            has_decpt = true;
            int_len = digit_count;
            p += 1;
            continue;
        }
        let c = to_digit(ch);
        if c >= radix {
            break;
        }
        digit_count += 1;
        p += 1;
        cur_limb = cur_limb * radix as u64 + c as u64;
        shift -= 1;
        if shift == 0 {
            limb_digits.push(cur_limb);
            shift = digits_per_limb;
            cur_limb = 0;
        }
    }
    if !has_decpt {
        int_len = digit_count;
    }

    if shift != digits_per_limb {
        while shift != 0 {
            cur_limb *= radix as u64;
            shift -= 1;
        }
        limb_digits.push(cur_limb);
    }

    if p == p_start && !saw_leading_zero {
        return Err(ParseFloatError::new());
    }

    let mut expn = 0_i64;
    if p < bytes.len() {
        let ch = bytes[p];
        if (radix == 10 && (ch == b'e' || ch == b'E')) || (radix != 10 && ch == b'@') {
            p += 1;
            match parse_exponent(bytes, &mut p) {
                Ok(e) => expn = e,
                Err(neg) => return Ok(exponent_overflow(is_neg, neg, p)),
            }
        }
    }

    if limb_digits.is_empty() {
        return Ok((
            BigFloat::zero(Sign::from_negative(is_neg)),
            Status::empty(),
            p,
        ));
    }

    limb_digits.reverse();
    let l = limb_digits.len() as u64;
    expn = expn.saturating_sub((l as i64) * (digits_per_limb as i64) - int_len);

    let t = bf_integer_from_radix(&limb_digits, l, radix);
    // A zero keeps the sign too (`-0.0`).
    let t_signed = if t.is_zero() {
        BigFloat::zero(Sign::from_negative(is_neg))
    } else if is_neg {
        t.neg()
    } else {
        t
    };

    if options.return_radix_exponent {
        *pexponent = expn;
        return Ok((t_signed, Status::empty(), p));
    }

    let (result, status) = bf_mul_pow_radix(
        &t_signed,
        radix,
        expn,
        match options.format.precision {
            Precision::Bits(b) => b,
            Precision::Digits(d) => d * 4 + 8,
            Precision::Infinite => u64::MAX,
        },
        options.format,
    );

    Ok((result, status, p))
}

#[allow(clippy::too_many_arguments)]
fn parse_power_of_two(
    bytes: &[u8],
    mut pos: usize,
    _p_start: usize,
    radix: u8,
    radix_bits: i32,
    is_neg: bool,
    options: ParseOptions,
    saw_leading_zero: bool,
) -> Result<(BigFloat, Status, usize), ParseFloatError> {
    let mut limbs: Vec<u64> = Vec::new();
    let mut shift = 64_i32;
    let mut cur_limb = 0_u64;
    let mut has_decpt = false;
    let mut int_len = 0_i64;
    let mut digit_count = 0_i64;

    loop {
        if pos >= bytes.len() {
            break;
        }
        let ch = bytes[pos];
        if ch == b'.' && !has_decpt {
            has_decpt = true;
            int_len = digit_count;
            pos += 1;
            continue;
        }
        let c = to_digit(ch);
        if c >= radix {
            break;
        }
        digit_count += 1;
        pos += 1;
        shift -= radix_bits;
        if shift <= 0 {
            cur_limb |= (c as u64) >> (-shift);
            limbs.push(cur_limb);
            if shift < 0 {
                cur_limb = (c as u64) << (64 + shift);
            } else {
                cur_limb = 0;
            }
            shift += 64;
        } else {
            cur_limb |= (c as u64) << shift;
        }
    }
    if !has_decpt {
        int_len = digit_count;
    }

    if shift != 64 {
        limbs.push(cur_limb);
    }

    let mut expn = 0_i64;
    let mut is_bin_exp = false;
    if pos < bytes.len() {
        let ch = bytes[pos];
        if ch == b'p' || ch == b'P' || ch == b'@' {
            is_bin_exp = ch != b'@';
            pos += 1;
            match parse_exponent(bytes, &mut pos) {
                Ok(e) => expn = e,
                Err(neg) => return Ok(exponent_overflow(is_neg, neg, pos)),
            }
        }
    }

    if limbs.is_empty() {
        // The exponent has been parsed (and checked) even for a zero.
        if digit_count > 0 || saw_leading_zero {
            return Ok((
                BigFloat::zero(Sign::from_negative(is_neg)),
                Status::empty(),
                pos,
            ));
        }
        return Err(ParseFloatError::new());
    }

    limbs.reverse();

    // libbf wraps here ("XXX: may overflow"); saturate instead.
    if !is_bin_exp {
        expn = expn.saturating_mul(radix_bits as i64);
    }
    expn = expn.saturating_add(int_len * radix_bits as i64);

    let sign = Sign::from_negative(is_neg);
    let bf = BigFloat::from_raw(sign, expn, limbs);
    let (result, status) = bf.normalize_and_round(options.format);
    Ok((result, status, pos))
}

/// Port of libbf's `bf_ftoa_internal` for binary numbers.
fn bf_ftoa_internal(a: &BigFloat, options: &ToStringOptions) -> Option<String> {
    let radix = options.radix;
    let prec = options.precision;
    let rounding = options.rounding;
    let fmt = options.style;
    if !(2..=36).contains(&radix) || prec > BF_PREC_INF {
        return None;
    }

    let mut out = String::new();
    if a.is_nan() {
        return Some("NaN".into());
    }
    if a.sign().is_negative() {
        out.push('-');
    }
    if a.is_infinite() {
        out.push_str(if options.js_quirks { "Infinity" } else { "Inf" });
        return Some(out);
    }

    let radix_bits = if radix.is_power_of_two() {
        radix.trailing_zeros() as i64
    } else {
        0
    };
    let prefix = |out: &mut String| {
        if options.add_prefix {
            match radix {
                16 => out.push_str("0x"),
                8 => out.push_str("0o"),
                2 => out.push_str("0b"),
                _ => {}
            }
        }
    };
    // The exponent of zero in libbf (`BF_EXP_ZERO`).
    let a_expn = if a.is_zero() { i64::MIN } else { a.raw_exp() };
    let a1;
    let n_digits: i64;
    let n_max: i64;
    let mut n: i64;

    if fmt == FormatStyle::Fractional {
        if radix_bits != 0 {
            let format = BigFormat {
                precision: Precision::Bits(prec.saturating_mul(radix_bits as u64)),
                rounding,
                radix_point_precision: true,
                ..BigFormat::INTERNAL
            };
            let r = a.round_status(format).0;
            prefix(&mut out);
            if r.is_zero() {
                out.push('0');
                if prec > 0 {
                    out.push('.');
                    for _ in 0..prec {
                        out.push('0');
                    }
                }
            } else {
                let n = ceil_div_i64(r.raw_exp(), radix_bits);
                let n_digits = prec as i64 + n;
                if n <= 0 {
                    out.push_str("0.");
                    for _ in 0..(-n) {
                        out.push('0');
                    }
                    if n_digits > 0 {
                        output_digits(&mut out, &r, radix, n_digits as u64, n_digits as u64);
                    }
                } else {
                    output_digits(&mut out, &r, radix, n_digits as u64, n as u64);
                }
            }
        } else {
            let abs = a.abs();
            // One more digit for the rounding.
            let n = 1 + mul_log2_radix(abs_exp_or_zero(&abs), radix, true, true).unwrap_or(0);
            let n_digits = n + prec as i64;
            let (r, _) = bf_convert_to_radix(&abs, radix, n_digits as u64, rounding, true, n);
            let start = out.len();
            output_digits(&mut out, &r, radix, n_digits as u64, n as u64);
            // Remove the leading zeros because one more digit was allocated.
            let bytes = out.as_bytes();
            let mut pos = start;
            while pos + 1 < bytes.len() && bytes[pos] == b'0' && bytes[pos + 1] != b'.' {
                pos += 1;
            }
            out.replace_range(start..pos, "");
        }
        return Some(out);
    }

    if radix_bits != 0 {
        let mut r = a.clone();
        let nd;
        if fmt == FormatStyle::Fixed {
            nd = prec as i64;
            n_max = nd;
            if !r.is_zero() {
                // Align to the radix point.
                let prec_bits = (prec as i64)
                    .saturating_mul(radix_bits)
                    .saturating_sub(smod(-a_expn, radix_bits) as i64);
                let format = BigFormat {
                    precision: Precision::Bits(prec_bits.max(0) as u64),
                    rounding,
                    ..BigFormat::INTERNAL
                };
                r = r.round_status_owned(format).0;
            }
        } else {
            // Position of the digit before the most significant digit in bits.
            let pos = r.limbs().len() as i64 * 64 + smod(a_expn.wrapping_neg(), radix_bits) as i64;
            let mut d = ceil_div_i64(pos, radix_bits);
            // Remove the trailing zero digits.
            let digit_mask = (1_u64 << radix_bits) - 1;
            while d > 1 && get_bits_from_bf(&r, pos - d * radix_bits) & digit_mask == 0 {
                d -= 1;
            }
            nd = d;
            n_max = d + 4;
        }
        n_digits = nd;
        n = if r.is_zero() {
            0
        } else {
            ceil_div_i64(r.raw_exp(), radix_bits)
        };
        a1 = r;
    } else {
        let abs = a.abs();
        let mut nd;
        if fmt == FormatStyle::Fixed {
            nd = prec as i64;
            n_max = nd;
        } else {
            if prec >= BF_PREC_INF {
                return None;
            }
            nd = 1 + mul_log2_radix(prec as i64, radix, true, true).unwrap_or(0);
            // The same rule as JS, i.e. n_max = 21 for 64 bit floats in base 10.
            n_max = nd + 4;
            if fmt == FormatStyle::FreeMin {
                // Find the minimum number of digits by dichotomy.
                let mut n_digits_max = nd;
                let mut n_digits_min = 1_i64;
                let back_format = BigFormat {
                    precision: Precision::Bits(prec),
                    rounding: Rounding::NearestEven,
                    ..BigFormat::INTERNAL
                };
                while n_digits_min < n_digits_max {
                    nd = (n_digits_min + n_digits_max) / 2;
                    let (r, nn) = bf_convert_to_radix(&abs, radix, nd as u64, rounding, false, 0);
                    // Convert back to a number and compare.
                    let (b, _) = bf_mul_pow_radix(&r, radix, nn - nd, prec, back_format);
                    if b.cmp_abs(&abs) == core::cmp::Ordering::Equal {
                        n_digits_max = nd;
                    } else {
                        n_digits_min = nd + 1;
                    }
                }
                nd = n_digits_max;
            }
        }
        let (r, nn) = bf_convert_to_radix(&abs, radix, nd as u64, rounding, false, 0);
        n_digits = nd;
        n = nn;
        a1 = r;
    }

    if a1.is_zero() && fmt != FormatStyle::Fixed && !options.force_exponent {
        out.push('0');
        return Some(out);
    }

    prefix(&mut out);
    if a1.is_zero() {
        n = 1;
    }
    if options.force_exponent || n <= -6 || n > n_max {
        // Exponential notation.
        output_digits(&mut out, &a1, radix, n_digits as u64, 1);
        let (marker, exp) = if radix_bits != 0 && radix <= 16 {
            ('p', (n - 1) * radix_bits)
        } else {
            (if radix <= 10 { 'e' } else { '@' }, n - 1)
        };
        out.push(marker);
        if options.js_quirks {
            out.push_str(&alloc::format!("{exp:+}"));
        } else {
            out.push_str(&alloc::format!("{exp}"));
        }
    } else if n <= 0 {
        out.push_str("0.");
        for _ in 0..(-n) {
            out.push('0');
        }
        output_digits(&mut out, &a1, radix, n_digits as u64, n_digits as u64);
    } else if n_digits <= n {
        output_digits(&mut out, &a1, radix, n_digits as u64, n_digits as u64);
        for _ in 0..(n - n_digits) {
            out.push('0');
        }
    } else {
        output_digits(&mut out, &a1, radix, n_digits as u64, n as u64);
    }
    Some(out)
}

/// The exponent of `a` clamped to be non-negative; zero counts as 0.
fn abs_exp_or_zero(a: &BigFloat) -> i64 {
    if a.is_zero() {
        0
    } else {
        a.raw_exp().max(0)
    }
}
