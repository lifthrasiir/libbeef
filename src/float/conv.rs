use alloc::string::String;
use alloc::vec::Vec;
use core::str::FromStr;

use crate::format::{mul_log2_radix, BigFormat, Precision, Rounding};
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

fn bf_pow_ui(base: &BigFloat, exp: u64, prec: u64, rounding: Rounding) -> BigFloat {
    if exp == 0 {
        return BigFloat::from_u64(1);
    }
    let fmt = BigFormat {
        precision: Precision::Bits(prec),
        rounding,
        ..BigFormat::BINARY64
    };
    let inf = BigFormat {
        precision: Precision::Infinite,
        ..BigFormat::BINARY64
    };
    let actual_fmt = if prec == u64::MAX { inf } else { fmt };

    let mut r = base.clone();
    let n_bits = 64 - exp.leading_zeros();
    for i in (0..n_bits - 1).rev() {
        r = r.mul(&r, actual_fmt);
        if (exp >> i) & 1 != 0 {
            r = r.mul(base, actual_fmt);
        }
    }
    r
}

fn bf_pow_ui_ui(a: u64, b: u64, prec: u64, rounding: Rounding) -> BigFloat {
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
        return BigFloat::from_u64(POW10[b as usize]);
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
        ..BigFormat::BINARY64
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
        pow_tab[level as usize] = bf_pow_ui_ui(radixl, n2, u64::MAX, Rounding::TowardZero);
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
        ..BigFormat::BINARY64
    };

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
        let b = bf_pow_ui_ui(radixl, n2, u64::MAX, Rounding::TowardZero);
        let one = BigFloat::from_u64(1);
        let b_inv_prec = (n2 + 1) * radixl_bits as u64 + 2;
        let b_inv = one.div(
            &b,
            BigFormat {
                precision: Precision::Bits(b_inv_prec),
                rounding: Rounding::NearestEven,
                ..BigFormat::BINARY64
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
            ..BigFormat::BINARY64
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
        let b = bf_pow_ui_ui(radix as u64, e, u64::MAX, Rounding::NearestEven);
        let inf = BigFormat {
            precision: Precision::Infinite,
            ..BigFormat::BINARY64
        };
        if expn_sign {
            let r = t.div(
                &b,
                BigFormat {
                    precision: Precision::Bits(t.limbs().len() as u64 * 64),
                    rounding: Rounding::NearestEven,
                    ..BigFormat::BINARY64
                },
            );
            return (r, Status::empty());
        } else {
            let r = t.mul(&b, inf);
            return (r, Status::empty());
        }
    }

    let mut ziv_extra_bits = 16_u64;
    loop {
        let prec1 = prec.saturating_add(ziv_extra_bits);
        let extra_bits = (ceil_log2_u64(e) * 2 + 1) as u64;
        let work_prec = prec1.saturating_add(extra_bits);
        let work_fmt = BigFormat {
            precision: Precision::Bits(work_prec),
            rounding: Rounding::NearestEven,
            ..BigFormat::BINARY64
        };
        let b = bf_pow_ui_ui(radix as u64, e, work_prec, Rounding::NearestEven);
        let r = if expn_sign {
            t.div(&b, work_fmt)
        } else {
            t.mul(&b, work_fmt)
        };

        if r.can_round(prec, format.rounding, prec1) || !r.is_finite() {
            let (rounded, mut status) = r.round_status_owned(format);
            status |= Status::INEXACT;
            return (rounded, status);
        }

        ziv_extra_bits = ziv_extra_bits + ziv_extra_bits / 2;
        if ziv_extra_bits > 100_000 {
            return r.round_status_owned(format);
        }
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
        ..BigFormat::BINARY64
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
                ..BigFormat::BINARY64
            };

            let pow = bf_pow_ui_ui(radix as u64, abs_exp, work_prec, Rounding::NearestEven);
            let r = if !exp_sign {
                a.mul(&pow, work_fmt)
            } else {
                a.div(&pow, work_fmt)
            };

            if r.can_round(r.raw_exp() as u64, rnd_mode, prec) || r.is_zero() {
                break r.rint(rnd_mode);
            }

            ziv_extra_bits = ziv_extra_bits + ziv_extra_bits / 2;
            if ziv_extra_bits > 100_000 {
                break r.rint(rnd_mode);
            }
        };

        if is_fixed_exponent {
            return (r, e);
        }

        let bound = bf_pow_ui_ui(radix as u64, p, u64::MAX, Rounding::TowardZero);
        if r.cmp_abs(&bound) == core::cmp::Ordering::Less {
            return (r, e);
        }
        e += 1;
    }
}

fn output_digits_general(a: &BigFloat, radix: u8, n_digits: u64, dot_pos: u64) -> String {
    let mut out = String::new();

    if radix.is_power_of_two() {
        let radix_bits = radix.trailing_zeros();
        let _digits_per_limb = 64 / radix_bits;
        let _mask = ((1_u64) << radix_bits) - 1;
        let exp = a.raw_exp();

        for i in 0..n_digits {
            if i == dot_pos && i > 0 {
                out.push('.');
            }
            let bit_pos = exp - 1 - (i as i64 * radix_bits as i64);
            let mut digit_val = 0_u8;
            for b in 0..radix_bits {
                if a.get_bit_at_exp_pub(bit_pos - b as i64) {
                    digit_val |= 1 << (radix_bits - 1 - b);
                }
            }
            out.push(digit_char(digit_val) as char);
        }
    } else {
        let digits_per_limb = DIGITS_PER_LIMB_TABLE[radix as usize - 2] as u64;
        let radixl = get_limb_radix(radix);
        let n = n_digits.div_ceil(digits_per_limb);
        let radix_limbs = bf_integer_to_radix(a, n, radixl);

        let first_buf_pos = (n * digits_per_limb - n_digits) as usize;
        let mut pos = n as i64;
        let mut buf_pos = digits_per_limb as usize;
        let mut buf = Vec::new();
        let mut i = 0_u64;
        let mut first = true;

        while i < n_digits {
            if buf_pos == digits_per_limb as usize {
                pos -= 1;
                let v = if pos >= 0 && (pos as usize) < radix_limbs.len() {
                    radix_limbs[pos as usize]
                } else {
                    0
                };
                buf.clear();
                limb_to_a(&mut buf, v, radix, digits_per_limb as usize);
                buf_pos = if first { first_buf_pos } else { 0 };
                first = false;
            }

            if i == dot_pos && i > 0 {
                out.push('.');
            }

            let l = if i < dot_pos {
                dot_pos - i
            } else {
                n_digits - i
            };
            let avail = digits_per_limb as usize - buf_pos;
            let take = (l as usize).min(avail);
            for j in 0..take {
                out.push(buf[buf_pos + j] as char);
            }
            buf_pos += take;
            i += take as u64;
        }
    }
    out
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
    pub(crate) fn get_bit_at_exp_pub(&self, bit_exp: i64) -> bool {
        self.get_bit_at_exp(bit_exp)
    }

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
    /// use beef::{BigFloat, BigFormat};
    /// use beef::parse::ParseOptions;
    ///
    /// let opts = ParseOptions::default();
    /// let (val, _consumed) = BigFloat::parse("3.14", opts).unwrap();
    /// let f = val.to_f64(beef::Rounding::NearestEven);
    /// assert!((f - 3.14).abs() < 1e-10);
    ///
    /// let hex_opts = ParseOptions { allow_hex_prefix: true, ..ParseOptions::default() };
    /// let (val, _consumed) = BigFloat::parse("0xff", hex_opts).unwrap();
    /// let f = val.to_f64(beef::Rounding::NearestEven);
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
        bf_atof_internal(input, options)
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
    /// use beef::BigFloat;
    ///
    /// let v = BigFloat::from_str_radix("ff", 16).unwrap();
    /// assert_eq!(v.to_f64(beef::Rounding::NearestEven), 255.0);
    ///
    /// let v = BigFloat::from_str_radix("-1.8@1", 16).unwrap();
    /// assert_eq!(v.to_f64(beef::Rounding::NearestEven), -24.0);
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
                ..BigFormat::BINARY64
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

    /// Converts this number to a string in the given radix (2..=36) with the
    /// specified number of significant digits. Corresponds to `bf_ftoa` in libbf.
    ///
    /// Uses Ziv's rounding technique and a divide-and-conquer conversion for
    /// non-power-of-2 radices. Returns `None` if the radix is out of range.
    ///
    /// ```
    /// use beef::{BigFloat, BigFormat, Rounding};
    ///
    /// let val = BigFloat::from_f64(255.0);
    /// let s = val.to_string_radix(10, 1, Rounding::NearestEven, true).unwrap();
    /// assert_eq!(s, "3e2");
    ///
    /// let hex = val.to_string_radix(16, 1, Rounding::NearestEven, true).unwrap();
    /// assert_eq!(hex, "fp4");
    /// ```
    pub fn to_string_radix(
        &self,
        radix: u8,
        n_digits: u64,
        rounding: Rounding,
        force_exp: bool,
    ) -> Option<String> {
        if !(2..=36).contains(&radix) {
            return None;
        }
        bf_ftoa_internal(self, radix, n_digits, rounding, force_exp)
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
/// use beef::BigFloat;
///
/// let v: BigFloat = "3.14".parse().unwrap();
/// assert!((v.to_f64(beef::Rounding::NearestEven) - 3.14).abs() < 1e-10);
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

fn bf_atof_internal(
    input: &str,
    options: ParseOptions,
) -> Result<(BigFloat, Status, usize), ParseFloatError> {
    let bytes = input.as_bytes();
    let mut pos = 0;

    let mut radix = options.radix;

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
        if (next == b'x' || next == b'X') && options.allow_hex_prefix {
            radix = 16;
            pos += 2;
            if pos >= bytes.len() || to_digit(bytes[pos]) >= radix {
                return Ok((BigFloat::nan(), Status::empty(), pos));
            }
        } else if (next == b'o' || next == b'O') && options.allow_octal_prefix {
            radix = 8;
            pos += 2;
        } else if (next == b'b' || next == b'B') && options.allow_binary_prefix {
            radix = 2;
            pos += 2;
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
            let mut exp_is_neg = false;
            if p < bytes.len() && bytes[p] == b'+' {
                p += 1;
            } else if p < bytes.len() && bytes[p] == b'-' {
                exp_is_neg = true;
                p += 1;
            }
            while p < bytes.len() {
                let c = to_digit(bytes[p]);
                if c >= 10 {
                    break;
                }
                expn = expn.saturating_mul(10).saturating_add(c as i64);
                p += 1;
            }
            if exp_is_neg {
                expn = -expn;
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
    expn -= (l as i64) * (digits_per_limb as i64) - int_len;

    let t = bf_integer_from_radix(&limb_digits, l, radix);
    let mut t_signed = t;
    if is_neg && !t_signed.is_zero() {
        t_signed = t_signed.neg();
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

    if limbs.is_empty() {
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

    let mut expn = 0_i64;
    let mut is_bin_exp = false;
    if pos < bytes.len() {
        let ch = bytes[pos];
        if ch == b'p' || ch == b'P' {
            is_bin_exp = true;
            pos += 1;
            let mut exp_is_neg = false;
            if pos < bytes.len() && bytes[pos] == b'+' {
                pos += 1;
            } else if pos < bytes.len() && bytes[pos] == b'-' {
                exp_is_neg = true;
                pos += 1;
            }
            while pos < bytes.len() {
                let c = to_digit(bytes[pos]);
                if c >= 10 {
                    break;
                }
                expn = expn.saturating_mul(10).saturating_add(c as i64);
                pos += 1;
            }
            if exp_is_neg {
                expn = -expn;
            }
        } else if (radix == 10 && (ch == b'e' || ch == b'E')) || (radix != 10 && ch == b'@') {
            pos += 1;
            let mut exp_is_neg = false;
            if pos < bytes.len() && bytes[pos] == b'+' {
                pos += 1;
            } else if pos < bytes.len() && bytes[pos] == b'-' {
                exp_is_neg = true;
                pos += 1;
            }
            while pos < bytes.len() {
                let c = to_digit(bytes[pos]);
                if c >= 10 {
                    break;
                }
                expn = expn.saturating_mul(10).saturating_add(c as i64);
                pos += 1;
            }
            if exp_is_neg {
                expn = -expn;
            }
        }
    }

    if !is_bin_exp {
        expn *= radix_bits as i64;
    }
    expn += int_len * radix_bits as i64;

    let sign = Sign::from_negative(is_neg);
    let bf = BigFloat::from_raw(sign, expn, limbs);
    let (result, status) = bf.normalize_and_round(options.format);
    Ok((result, status, pos))
}

fn bf_ftoa_internal(
    a: &BigFloat,
    radix: u8,
    prec: u64,
    rounding: Rounding,
    force_exp: bool,
) -> Option<String> {
    let mut out = String::new();

    if a.is_nan() {
        return Some("NaN".into());
    }

    if a.sign().is_negative() {
        out.push('-');
    }

    if a.is_infinite() {
        out.push_str("Inf");
        return Some(out);
    }

    let is_pow2 = radix.is_power_of_two();
    let radix_bits = if is_pow2 {
        radix.trailing_zeros() as i64
    } else {
        0
    };

    if is_pow2 {
        if a.is_zero() {
            out.push('0');
            if prec > 1 {
                out.push('.');
                for _ in 1..prec {
                    out.push('0');
                }
            }
            if force_exp {
                out.push('p');
                out.push('0');
            }
            return Some(out);
        }

        let n_digits = prec;
        let n = ceil_div_i64(a.raw_exp(), radix_bits);

        let digits = output_digits_general(a, radix, n_digits, 1);
        out.push_str(&digits);

        if force_exp || n != 1 {
            out.push('p');
            out.push_str(&((n - 1) * radix_bits).to_string());
        }
    } else {
        if a.is_zero() {
            out.push('0');
            if prec > 1 {
                out.push('.');
                for _ in 1..prec {
                    out.push('0');
                }
            }
            if force_exp {
                if radix <= 10 {
                    out.push('e');
                } else {
                    out.push('@');
                }
                out.push('0');
            }
            return Some(out);
        }

        let a_abs = a.abs();
        let n_digits = prec;

        let (a1, n) = bf_convert_to_radix(&a_abs, radix, n_digits, rounding, false, 0);

        if a1.is_zero() && !force_exp {
            out.push('0');
        } else {
            let n_max = n_digits as i64 + 4;
            if force_exp || n <= -6 || n > n_max {
                let digits = output_digits_general(&a1, radix, n_digits, 1);
                out.push_str(&digits);
                if radix <= 10 {
                    out.push('e');
                } else {
                    out.push('@');
                }
                out.push_str(&(n - 1).to_string());
            } else if n <= 0 {
                out.push_str("0.");
                for _ in 0..(-n) {
                    out.push('0');
                }
                let digits = output_digits_general(&a1, radix, n_digits, n_digits);
                out.push_str(&digits);
            } else {
                let n_u = n as u64;
                if n_digits <= n_u {
                    let digits = output_digits_general(&a1, radix, n_digits, n_digits);
                    out.push_str(&digits);
                    for _ in 0..(n_u - n_digits) {
                        out.push('0');
                    }
                } else {
                    let digits = output_digits_general(&a1, radix, n_digits, n_u);
                    out.push_str(&digits);
                }
            }
        }
    }

    Some(out)
}
