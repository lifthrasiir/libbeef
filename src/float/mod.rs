mod conv;
mod mp;
mod ntt;
mod transc;

use alloc::string::String;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::marker::PhantomData;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use crate::format::Precision;
use crate::format::{BigFormat, Rounding, StaticFormat};
use crate::parse::{ParseFloatError, ParseOptions};
use crate::status::Status;

/// The unsigned integer type used for each word of the significand.
pub type Limb = u64;

/// The type used for the raw internal exponent field.
pub type RawExp = i64;

pub(crate) const RAW_EXP_ZERO: RawExp = RawExp::MIN;

pub(crate) const RAW_EXP_INF: RawExp = RawExp::MAX - 1;

pub(crate) const RAW_EXP_NAN: RawExp = RawExp::MAX;

#[doc(hidden)]
pub fn mp_sqrtrem(limbs: &[u64]) -> (Vec<u64>, Vec<u64>) {
    mp_sqrtrem_karatsuba(limbs)
}

#[doc(hidden)]
pub fn mp_recip(limbs: &[u64]) -> Option<Vec<u64>> {
    let mut denominator = limbs.to_vec();
    trim_limbs(&mut denominator);
    if denominator.is_empty() {
        return None;
    }

    let n = denominator.len();
    let mut tabr = alloc::vec![0_u64; n + 1];
    mp::mp_recip_full(&mut tabr, &denominator, n);
    Some(tabr)
}

/// Represents the sign of a floating-point number.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Sign {
    /// Positive sign (+).
    Positive = 1,
    /// Negative sign (-).
    Negative = -1,
}

impl Sign {
    /// Returns `true` if this sign is negative.
    pub const fn is_negative(self) -> bool {
        matches!(self, Self::Negative)
    }

    /// Constructs a [`Sign`] from a boolean: `true` yields [`Sign::Negative`].
    pub const fn from_negative(is_negative: bool) -> Self {
        if is_negative {
            Self::Negative
        } else {
            Self::Positive
        }
    }

    /// Returns the negated sign.
    pub const fn neg(self) -> Self {
        match self {
            Self::Positive => Self::Negative,
            Self::Negative => Self::Positive,
        }
    }

    /// Returns the negated sign if `condition` is true, otherwise returns self unchanged.
    pub const fn neg_if(self, condition: bool) -> Self {
        if condition {
            self.neg()
        } else {
            self
        }
    }
}

impl From<i64> for Sign {
    fn from(value: i64) -> Self {
        Self::from_negative(value < 0)
    }
}

impl From<f64> for Sign {
    fn from(value: f64) -> Self {
        Self::from_negative(value.is_sign_negative())
    }
}

impl core::ops::Neg for Sign {
    type Output = Self;
    fn neg(self) -> Self {
        Sign::neg(self)
    }
}

impl core::ops::Mul for Sign {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Self) -> Self {
        Self::from_negative(self.is_negative() ^ rhs.is_negative())
    }
}

/// Classification of a floating-point value.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FpCategory {
    /// Not a Number.
    Nan,
    /// Positive or negative infinity.
    Infinite,
    /// Positive or negative zero.
    Zero,
    /// A finite, non-zero value.
    Normal,
}

/// Specifies the quotient rounding mode for division-with-remainder operations.
/// Corresponds to the `mode` parameter of `bf_divrem` / `bf_remquo` in libbf.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DivRemMode {
    /// Truncate the quotient toward zero (C `%` semantics).
    TowardZero,
    /// Round the quotient to the nearest integer, ties to even (IEEE remainder).
    NearestEven,
    /// Floor the quotient (Python `%` semantics).
    Floor,
    /// Euclidean remainder (remainder is always non-negative).
    Euclidean,
}

/// An arbitrary-precision binary floating-point number.
///
/// Corresponds to `bf_t` in libbf. The value is stored as a sign, a binary
/// exponent, and a sequence of 64-bit limbs holding the normalized significand.
///
/// ```
/// use libbeef::BigFloat;
/// let a = BigFloat::from_i64(42);
/// let b = BigFloat::from_i64(8);
/// let sum = a.add(&b, libbeef::BigFormat::BINARY64);
/// assert_eq!(sum.to_f64(libbeef::Rounding::NearestEven), 50.0);
/// ```
#[derive(Clone)]
pub struct BigFloat {
    sign: Sign,
    exp: RawExp,
    limbs: Vec<Limb>,
}

impl BigFloat {
    /// Creates a new [`BigFloat`] initialized to positive zero.
    /// Corresponds to `bf_init` in libbf (which produces a zero value).
    ///
    /// ```
    /// use libbeef::BigFloat;
    /// let x = BigFloat::new();
    /// assert!(x.is_zero());
    /// ```
    pub const fn new() -> Self {
        Self::zero(Sign::Positive)
    }

    /// Returns a NaN (Not a Number) value.
    /// Corresponds to `bf_set_nan` in libbf.
    pub const fn nan() -> Self {
        Self {
            sign: Sign::Positive,
            exp: RAW_EXP_NAN,
            limbs: Vec::new(),
        }
    }

    /// Returns an infinity with the given sign.
    /// Corresponds to `bf_set_inf` in libbf.
    pub const fn infinity(sign: Sign) -> Self {
        Self {
            sign,
            exp: RAW_EXP_INF,
            limbs: Vec::new(),
        }
    }

    /// Returns a zero with the given sign.
    /// Corresponds to `bf_set_zero` in libbf.
    pub const fn zero(sign: Sign) -> Self {
        Self {
            sign,
            exp: RAW_EXP_ZERO,
            limbs: Vec::new(),
        }
    }

    /// Creates a [`BigFloat`] from an unsigned 64-bit integer.
    /// Corresponds to `bf_set_ui` in libbf.
    ///
    /// ```
    /// use libbeef::BigFloat;
    /// let x = BigFloat::from_u64(100);
    /// assert_eq!(x.to_f64(libbeef::Rounding::NearestEven), 100.0);
    /// ```
    pub fn from_u64(value: u64) -> Self {
        if value == 0 {
            return Self::new();
        }
        let shift = value.leading_zeros();
        Self {
            sign: Sign::Positive,
            exp: 64_i64 - i64::from(shift),
            limbs: alloc::vec![value << shift],
        }
    }

    /// Creates a [`BigFloat`] from a signed 64-bit integer.
    /// Corresponds to `bf_set_si` in libbf.
    ///
    /// ```
    /// use libbeef::BigFloat;
    /// let x = BigFloat::from_i64(-7);
    /// assert_eq!(x.to_f64(libbeef::Rounding::NearestEven), -7.0);
    /// ```
    pub fn from_i64(value: i64) -> Self {
        if value == 0 {
            return Self::new();
        }
        let sign = Sign::from(value);
        let magnitude = value.unsigned_abs();
        Self::from_u128_with_sign(u128::from(magnitude), sign)
    }

    /// Creates a [`BigFloat`] from an `f64`, preserving the exact value (including sign of zero).
    /// Corresponds to `bf_set_float64` in libbf.
    pub fn from_f64(value: f64) -> Self {
        if value.is_nan() {
            Self::nan()
        } else if value.is_infinite() {
            Self::infinity(Sign::from(value))
        } else if value == 0.0 {
            Self::zero(Sign::from(value))
        } else {
            let bits = value.to_bits();
            let sign = Sign::from_negative((bits >> 63) != 0);
            let exp_bits = ((bits >> 52) & 0x7ff) as i32;
            let fraction = bits & ((1_u64 << 52) - 1);

            if exp_bits == 0 {
                let bit_len = 64_i64 - i64::from(fraction.leading_zeros());
                Self::from_u128_scaled(u128::from(fraction), bit_len - 1074, sign)
            } else {
                let significand = (1_u64 << 52) | fraction;
                let unbiased_exp = exp_bits - 1023;
                Self::from_u128_scaled(u128::from(significand), i64::from(unbiased_exp + 1), sign)
            }
        }
    }

    /// Converts this value to an `f64` using the given rounding mode.
    /// Corresponds to `bf_get_float64` in libbf.
    ///
    /// ```
    /// use libbeef::{BigFloat, Rounding};
    /// let x = BigFloat::from_i64(42);
    /// let v = x.to_f64(Rounding::NearestEven);
    /// assert_eq!(v, 42.0);
    /// ```
    pub fn to_f64(&self, rounding: Rounding) -> f64 {
        self.to_f64_status(rounding).0
    }

    /// Converts this value to an `f64` using the given rounding mode.
    /// Returns the result and a [`Status`] indicating inexactness or overflow.
    /// Corresponds to `bf_get_float64` in libbf.
    pub fn to_f64_status(&self, rounding: Rounding) -> (f64, Status) {
        match self.classify() {
            FpCategory::Nan => (f64::NAN, Status::empty()),
            FpCategory::Infinite => {
                let value = if self.sign.is_negative() {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
                (value, Status::empty())
            }
            FpCategory::Zero => {
                let value = if self.sign.is_negative() { -0.0 } else { 0.0 };
                (value, Status::empty())
            }
            FpCategory::Normal => self.normal_to_f64(rounding),
        }
    }

    /// Returns the sign of this value.
    pub const fn sign(&self) -> Sign {
        self.sign
    }

    /// Returns the raw internal exponent value.
    pub const fn raw_exp(&self) -> i64 {
        self.exp
    }

    /// Returns a slice of the internal limbs of the significand.
    pub fn limbs(&self) -> &[u64] {
        &self.limbs
    }

    /// Constructs a [`BigFloat`] directly from raw sign, exponent, and limbs.
    ///
    /// A normalized `BigFloat` satisfies all of the following:
    ///
    /// - The most significant bit of the last limb is set (i.e. no leading
    ///   zero bits).
    /// - No limb is entirely zero at either end of the vector (no trailing
    ///   zero limbs and no leading zero limbs).
    /// - `exp` equals the bit position of the most significant bit, counting
    ///   from the binary point (the same convention used by [`raw_exp`]).
    ///
    /// Passing non-normalized inputs does **not** cause undefined behavior —
    /// `BigFloat` contains no raw pointers or unsafe invariants — but
    /// arithmetic and comparison operations may produce incorrect results
    /// because they assume the representation above.  If you cannot
    /// guarantee normalization at construction time, call
    /// [`normalize_and_round`] on the result.
    ///
    /// [`raw_exp`]: Self::raw_exp
    /// [`normalize_and_round`]: Self::normalize_and_round
    pub const fn from_raw(sign: Sign, exp: i64, limbs: Vec<u64>) -> Self {
        Self { sign, exp, limbs }
    }

    /// Sets the raw exponent of this value.
    pub const fn set_exp(&mut self, exp: i64) {
        self.exp = exp;
    }

    // parse() and parse_status() are in conv.rs
    // to_string_radix() is in conv.rs

    fn _dead_old_parse_status(
        input: &str,
        options: crate::parse::ParseOptions,
    ) -> Result<(Self, Status, usize), ParseFloatError> {
        let bytes = input.as_bytes();
        let mut pos = 0;

        while pos < bytes.len() && bytes[pos] == b' ' {
            pos += 1;
        }

        let mut is_neg = false;
        if pos < bytes.len() && bytes[pos] == b'+' {
            pos += 1;
        } else if pos < bytes.len() && bytes[pos] == b'-' {
            is_neg = true;
            pos += 1;
        }

        if options.allow_nan_inf {
            if bytes[pos..].len() >= 3 && bytes[pos..pos + 3].eq_ignore_ascii_case(b"nan") {
                return Ok((Self::nan(), Status::empty(), pos + 3));
            }
            if bytes[pos..].len() >= 3 && bytes[pos..pos + 3].eq_ignore_ascii_case(b"inf") {
                let mut end = pos + 3;
                if bytes[end..].len() >= 5 && bytes[end..end + 5].eq_ignore_ascii_case(b"inity") {
                    end += 5;
                }
                return Ok((
                    Self::infinity(Sign::from_negative(is_neg)),
                    Status::empty(),
                    end,
                ));
            }
        }

        let mut radix = options.radix;

        if pos < bytes.len() && bytes[pos] == b'0' && pos + 1 < bytes.len() {
            let next = bytes[pos + 1];
            if (next == b'x' || next == b'X') && options.allow_hex_prefix {
                radix = 16;
                pos += 2;
            } else if (next == b'o' || next == b'O') && options.allow_octal_prefix {
                radix = 8;
                pos += 2;
            } else if (next == b'b' || next == b'B') && options.allow_binary_prefix {
                radix = 2;
                pos += 2;
            }
        }

        let is_power_of_two = radix.is_power_of_two();
        let radix_bits = if is_power_of_two {
            radix.trailing_zeros()
        } else {
            0
        };

        let mut integer_limbs = Vec::new();
        let mut saw_digit = false;
        let mut has_decpt = false;
        let mut frac_digits = 0_i64;

        while pos < bytes.len() && bytes[pos] == b'0' {
            saw_digit = true;
            pos += 1;
        }

        while pos < bytes.len() {
            let ch = bytes[pos];
            if ch == b'.' && !has_decpt {
                has_decpt = true;
                pos += 1;
                continue;
            }
            if ch == b'_' {
                pos += 1;
                continue;
            }
            let Some(d) = digit_value(ch) else { break };
            if d >= radix {
                break;
            }
            saw_digit = true;
            mul_small_abs(&mut integer_limbs, u64::from(radix));
            add_small_abs(&mut integer_limbs, u64::from(d));
            if has_decpt {
                frac_digits += 1;
            }
            pos += 1;
        }

        if !saw_digit {
            return Err(ParseFloatError::new());
        }

        let mut exp_val = 0_i64;
        let is_bin_exp;
        if pos < bytes.len() {
            let exp_char = bytes[pos];
            if is_power_of_two && radix == 16 && (exp_char == b'p' || exp_char == b'P') {
                is_bin_exp = true;
                pos += 1;
                let (e, new_pos) = parse_exponent(bytes, pos)?;
                exp_val = e;
                pos = new_pos;
            } else if (!is_power_of_two && (exp_char == b'e' || exp_char == b'E') && radix <= 10)
                || exp_char == b'@'
            {
                is_bin_exp = false;
                pos += 1;
                let (e, new_pos) = parse_exponent(bytes, pos)?;
                exp_val = e;
                pos = new_pos;
            } else {
                is_bin_exp = is_power_of_two;
            }
        } else {
            is_bin_exp = is_power_of_two;
        }

        if integer_limbs.is_empty() {
            return Ok((
                Self::zero(Sign::from_negative(is_neg)),
                Status::empty(),
                pos,
            ));
        }

        let sign = Sign::from_negative(is_neg);
        let int_value = Self::from_abs_int_limbs(integer_limbs, sign);

        if is_power_of_two {
            let total_exp = if is_bin_exp {
                exp_val - frac_digits * radix_bits as i64
            } else {
                (exp_val - frac_digits) * radix_bits as i64
            };
            let mut result = int_value;
            if total_exp != 0 {
                result = result.mul_exp2(total_exp);
            }
            let (rounded, status) = result.round_status_owned(options.format);
            return Ok((rounded, status, pos));
        }

        let total_radix_exp = exp_val - frac_digits;
        let result = if total_radix_exp == 0 {
            int_value
        } else if total_radix_exp > 0 {
            let mut r = int_value;
            for _ in 0..total_radix_exp {
                r = r.mul(
                    &Self::from_u64(u64::from(radix)),
                    BigFormat {
                        precision: Precision::Infinite,
                        ..options.format
                    },
                );
            }
            r
        } else {
            let mut denom = Self::from_u64(1);
            for _ in 0..(-total_radix_exp) {
                denom = denom.mul(
                    &Self::from_u64(u64::from(radix)),
                    BigFormat {
                        precision: Precision::Infinite,
                        ..options.format
                    },
                );
            }
            int_value.div(&denom, options.format)
        };

        let (rounded, status) = result.round_status_owned(options.format);
        Ok((rounded, status, pos))
    }

    /// Parses a decimal integer string into a [`BigFloat`].
    /// Shorthand for [`parse_integer_radix`](Self::parse_integer_radix) with base 10.
    /// Corresponds to `bf_atof` in libbf (integer-only mode, radix 10).
    pub fn parse_decimal_integer(input: &str) -> Result<Self, ParseFloatError> {
        Self::parse_integer_radix(input, 10)
    }

    /// Parses a decimal floating-point string (e.g. `"3.14"`, `"-1.5e10"`) and rounds
    /// the result to the given format.
    /// Corresponds to `bf_atof` in libbf (radix 10 with fractional/exponent support).
    pub fn parse_decimal(input: &str, format: BigFormat) -> Result<Self, ParseFloatError> {
        let input = input.trim();
        if input.eq_ignore_ascii_case("nan") {
            return Ok(Self::nan());
        }
        if input.eq_ignore_ascii_case("inf") || input.eq_ignore_ascii_case("+inf") {
            return Ok(Self::infinity(Sign::Positive));
        }
        if input.eq_ignore_ascii_case("-inf") {
            return Ok(Self::infinity(Sign::Negative));
        }

        let mut rest = input.as_bytes();
        let mut sign = Sign::Positive;
        if let Some((&first, tail)) = rest.split_first() {
            match first {
                b'+' => rest = tail,
                b'-' => {
                    sign = Sign::Negative;
                    rest = tail;
                }
                _ => {}
            }
        }

        let mut coefficient = Vec::new();
        let mut scale = 0_i32;
        let mut saw_digit = false;
        let mut after_point = false;
        let mut index = 0;
        while index < rest.len() {
            match rest[index] {
                b'0'..=b'9' => {
                    saw_digit = true;
                    mul_small_abs(&mut coefficient, 10);
                    add_small_abs(&mut coefficient, u64::from(rest[index] - b'0'));
                    if after_point {
                        scale = scale.checked_add(1).ok_or_else(ParseFloatError::new)?;
                    }
                    index += 1;
                }
                b'.' if !after_point => {
                    after_point = true;
                    index += 1;
                }
                b'e' | b'E' => break,
                _ => return Err(ParseFloatError::new()),
            }
        }
        if !saw_digit {
            return Err(ParseFloatError::new());
        }

        if index < rest.len() {
            index += 1;
            let exp_str =
                core::str::from_utf8(&rest[index..]).map_err(|_| ParseFloatError::new())?;
            let exp = exp_str.parse::<i32>().map_err(|_| ParseFloatError::new())?;
            scale = scale.checked_sub(exp).ok_or_else(ParseFloatError::new)?;
        }

        if scale <= 0 {
            mul_pow10_abs(&mut coefficient, -scale);
            return Ok(Self::from_abs_int_limbs(coefficient, sign).round(format));
        }

        let numerator = Self::from_abs_int_limbs(coefficient, sign);
        let denominator = Self::from_abs_int_limbs(pow10_abs_limbs(scale), Sign::Positive);
        Ok(numerator.div(&denominator, format))
    }

    /// Parses an integer string in the given radix (2..=36) into a [`BigFloat`].
    /// Corresponds to `bf_atof` in libbf (integer-only mode).
    pub fn parse_integer_radix(input: &str, radix: u8) -> Result<Self, ParseFloatError> {
        if !(2..=36).contains(&radix) {
            return Err(ParseFloatError::new());
        }
        let mut chars = input.as_bytes();
        let mut sign = Sign::Positive;
        if let Some((&first, rest)) = chars.split_first() {
            match first {
                b'+' => chars = rest,
                b'-' => {
                    sign = Sign::Negative;
                    chars = rest;
                }
                _ => {}
            }
        }
        if chars.is_empty() {
            return Err(ParseFloatError::new());
        }

        let mut limbs = Vec::new();
        let mut saw_digit = false;
        for &ch in chars {
            if ch == b'_' {
                continue;
            }
            let Some(digit) = digit_value(ch) else {
                return Err(ParseFloatError::new());
            };
            if digit >= radix {
                return Err(ParseFloatError::new());
            }
            saw_digit = true;
            mul_small_abs(&mut limbs, u64::from(radix));
            add_small_abs(&mut limbs, u64::from(digit));
        }
        if !saw_digit {
            return Err(ParseFloatError::new());
        }
        Ok(Self::from_abs_int_limbs(limbs, sign))
    }

    /// Converts this value to a decimal integer string, returning `None` if the value
    /// is not an exact integer.
    /// Corresponds to `bf_ftoa` in libbf (integer mode, radix 10).
    pub fn to_decimal_integer_string(&self) -> Option<String> {
        self.to_integer_string_radix(10)
    }

    /// Converts this value to a decimal string representation, returning `None` if the
    /// value cannot be exactly represented as a finite decimal or integer.
    /// Corresponds to `bf_ftoa` in libbf (radix 10).
    pub fn to_decimal_string(&self, _format: BigFormat) -> Option<String> {
        if self.is_nan() {
            return Some("NaN".into());
        }
        if self.is_infinite() {
            return Some(if self.sign.is_negative() {
                "-inf".into()
            } else {
                "inf".into()
            });
        }
        if self.is_zero() {
            return Some(if self.sign.is_negative() {
                "-0".into()
            } else {
                "0".into()
            });
        }
        if let Some(decimal) = self.to_exact_decimal_string() {
            return Some(decimal);
        }
        self.to_decimal_integer_string()
    }

    /// Converts this value to an integer string in the given radix (2..=36),
    /// returning `None` if the value is not an exact integer or the radix is out of range.
    /// Corresponds to `bf_ftoa` in libbf (integer mode, arbitrary radix).
    pub fn to_integer_string_radix(&self, radix: u8) -> Option<String> {
        if !(2..=36).contains(&radix) {
            return None;
        }
        let limbs = self.to_abs_int_limbs()?;
        let buf = FmtRadixBuf::from_abs_limbs(limbs, radix);
        let mut out = String::new();
        if self.sign.is_negative() {
            out.push('-');
        }
        buf.write_to(false, &mut out).unwrap();
        Some(out)
    }

    // to_string_radix() is in conv.rs
    #[doc(hidden)]
    fn _old_to_string_radix(
        &self,
        radix: u8,
        n_digits: u64,
        rounding: Rounding,
        force_exp: bool,
    ) -> Option<String> {
        if !(2..=36).contains(&radix) {
            return None;
        }
        if self.is_nan() {
            return Some("NaN".into());
        }
        if self.is_infinite() {
            return Some(
                if self.sign.is_negative() {
                    "-Infinity"
                } else {
                    "Infinity"
                }
                .into(),
            );
        }
        if self.is_zero() {
            let mut s = String::new();
            if self.sign.is_negative() {
                s.push('-');
            }
            s.push('0');
            if n_digits > 1 {
                s.push('.');
                for _ in 1..n_digits {
                    s.push('0');
                }
            }
            if force_exp {
                if radix <= 10 {
                    s.push('e');
                } else {
                    s.push('@');
                }
                s.push('0');
            }
            return Some(s);
        }

        if radix.is_power_of_two() {
            return self.format_power_of_two_radix(radix, n_digits, rounding, force_exp);
        }

        self.format_general_radix(radix, n_digits, rounding, force_exp)
    }

    #[allow(dead_code)]
    fn format_power_of_two_radix(
        &self,
        radix: u8,
        n_digits: u64,
        _rounding: Rounding,
        force_exp: bool,
    ) -> Option<String> {
        let radix_bits = radix.trailing_zeros();
        let total_sig_bits = self.exp - self.lowest_bit_exp();
        let sig_radix_digits = (total_sig_bits as u64).div_ceil(radix_bits as u64);
        let out_digits = if n_digits == 0 {
            sig_radix_digits.max(1)
        } else {
            n_digits
        };

        let mut digits = Vec::new();
        for i in 0..out_digits {
            let bit_pos = self.exp - 1 - (i as i64 * radix_bits as i64);
            let mut digit_val = 0_u8;
            for b in 0..radix_bits {
                if self.get_bit_at_exp(bit_pos - b as i64) {
                    digit_val |= 1 << (radix_bits - 1 - b);
                }
            }
            digits.push(digit_char(digit_val));
        }

        let mut s = String::new();
        if self.sign.is_negative() {
            s.push('-');
        }
        s.push(digits[0]);
        if digits.len() > 1 {
            s.push('.');
            for &d in &digits[1..] {
                s.push(d);
            }
        }
        let exp = self.exp - 1;
        if force_exp || exp != 0 {
            s.push('p');
            s.push_str(&exp.to_string());
        }
        Some(s)
    }

    #[allow(dead_code)]
    fn format_general_radix(
        &self,
        radix: u8,
        n_digits: u64,
        rounding: Rounding,
        force_exp: bool,
    ) -> Option<String> {
        let fmt = BigFormat {
            precision: Precision::Bits(n_digits.saturating_mul(4).saturating_add(32).max(64)),
            rounding,
            ..BigFormat::BINARY64
        };
        let rounded = self.round(fmt);
        if let Some(int_str) = rounded.to_integer_string_radix(radix) {
            if force_exp {
                let (is_neg, abs_str) = int_str
                    .strip_prefix('-')
                    .map_or((false, int_str.as_str()), |s| (true, s));
                let mut s = String::new();
                if is_neg {
                    s.push('-');
                }
                let chars: Vec<char> = abs_str.chars().collect();
                s.push(chars[0]);
                if chars.len() > 1 || n_digits > 1 {
                    s.push('.');
                    for &c in &chars[1..] {
                        s.push(c);
                    }
                    let remaining = n_digits.saturating_sub(chars.len() as u64);
                    for _ in 0..remaining {
                        s.push('0');
                    }
                }
                if radix <= 10 {
                    s.push('e');
                } else {
                    s.push('@');
                }
                s.push_str(&((chars.len() - 1) as i64).to_string());
                return Some(s);
            }
            return Some(int_str);
        }

        self.to_decimal_string(fmt)
    }

    fn to_exact_decimal_string(&self) -> Option<String> {
        let value = self.to_scaled_signed_limbs()?;
        if value.low_exp >= 0 {
            let mut limbs = value.limbs;
            if value.low_exp > 0 {
                let shift = u32::try_from(value.low_exp).ok()?;
                shl_limbs(&mut limbs, shift);
            }
            return Some(fmt_decimal_abs_limbs(limbs, value.sign, 0));
        }

        let scale = usize::try_from(-value.low_exp).ok()?;
        let mut coefficient = value.limbs;
        for _ in 0..scale {
            mul_small_abs(&mut coefficient, 5);
        }

        let mut scale = scale;
        while scale > 0 {
            let mut quotient = coefficient.clone();
            if div_rem_small_abs(&mut quotient, 10) != 0 {
                break;
            }
            coefficient = quotient;
            scale -= 1;
        }

        Some(fmt_decimal_abs_limbs(coefficient, value.sign, scale))
    }

    /// Adds `self` and `rhs`, rounding the result to `format`.
    /// Corresponds to `bf_add` in libbf.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat};
    /// let a = BigFloat::from_i64(3);
    /// let b = BigFloat::from_i64(4);
    /// assert_eq!(a.add(&b, BigFormat::BINARY64).to_f64(libbeef::Rounding::NearestEven), 7.0);
    /// ```
    pub fn add(&self, rhs: &Self, format: BigFormat) -> Self {
        self.add_status(rhs, format).0
    }

    /// Subtracts `rhs` from `self`, rounding the result to `format`.
    /// Corresponds to `bf_sub` in libbf.
    pub fn sub(&self, rhs: &Self, format: BigFormat) -> Self {
        self.sub_status(rhs, format).0
    }

    /// Multiplies `self` by `rhs`, rounding the result to `format`.
    /// Corresponds to `bf_mul` in libbf.
    pub fn mul(&self, rhs: &Self, format: BigFormat) -> Self {
        self.mul_status(rhs, format).0
    }

    /// Divides `self` by `rhs`, rounding the result to `format`.
    /// Corresponds to `bf_div` in libbf.
    pub fn div(&self, rhs: &Self, format: BigFormat) -> Self {
        self.div_status(rhs, format).0
    }

    /// Computes the remainder of `self / rhs` using the given division mode.
    /// Corresponds to `bf_rem` in libbf.
    pub fn rem(&self, rhs: &Self, format: BigFormat, mode: DivRemMode) -> Self {
        self.rem_status(rhs, format, mode).0
    }

    /// Computes the bitwise OR of `self` and `rhs` (treating both as integers).
    /// Corresponds to `bf_logic_or` in libbf.
    pub fn logic_or(&self, rhs: &Self) -> Self {
        self.logic_or_status(rhs).0
    }

    /// Computes the bitwise XOR of `self` and `rhs` (treating both as integers).
    /// Corresponds to `bf_logic_xor` in libbf.
    pub fn logic_xor(&self, rhs: &Self) -> Self {
        self.logic_xor_status(rhs).0
    }

    /// Computes the bitwise AND of `self` and `rhs` (treating both as integers).
    /// Corresponds to `bf_logic_and` in libbf.
    pub fn logic_and(&self, rhs: &Self) -> Self {
        self.logic_and_status(rhs).0
    }

    /// Adds `self` and `rhs`, returning the result and status flags.
    /// Corresponds to `bf_add` in libbf. Returns the status flags as well.
    pub fn add_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status) {
        bf_add_internal(self, rhs, format, false)
    }

    /// Subtracts `rhs` from `self`, returning the result and status flags.
    /// Corresponds to `bf_sub` in libbf. Returns the status flags as well.
    pub fn sub_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status) {
        bf_add_internal(self, rhs, format, true)
    }

    /// Negates this value in place.
    pub fn neg_mut(&mut self) {
        self.sign = -self.sign;
    }

    /// Multiplies this value by 2^e in place.
    pub fn mul_exp2_mut(&mut self, e: i64) {
        if self.is_finite() && !self.is_zero() {
            self.exp = self.exp.saturating_add(e);
        }
    }

    /// Sets this value to the given `i64`, reusing the existing allocation.
    pub fn set_i64(&mut self, v: i64) {
        *self = Self::from_i64(v);
    }

    /// Sets this value to the given `u64`, reusing the existing allocation.
    pub fn set_u64(&mut self, v: u64) {
        *self = Self::from_u64(v);
    }

    /// In-place multiplication. Returns the status flags.
    pub fn mul_assign(&mut self, rhs: &Self, format: BigFormat) -> Status {
        let a = core::mem::take(self);
        self.set_mul(&a, rhs, format)
    }

    /// In-place addition. Returns the status flags.
    pub fn add_assign(&mut self, rhs: &Self, format: BigFormat) -> Status {
        if rhs.is_zero() && !self.is_nan() {
            return self.round_finite_to_format_mut_if_finite(format);
        }
        if self.is_zero() && !rhs.is_nan() {
            self.sign = rhs.sign;
            self.exp = rhs.exp;
            self.limbs.clear();
            self.limbs.extend_from_slice(&rhs.limbs);
            return self.normalize_and_round_mut(format);
        }
        // Swap the value out instead of cloning it (C's aliased bf_add also
        // computes into a fresh buffer, but never copies the input).
        let a = core::mem::take(self);
        bf_add_internal_into(self, &a, rhs, format, false)
    }

    /// In-place subtraction. Returns the status flags.
    pub fn sub_assign(&mut self, rhs: &Self, format: BigFormat) -> Status {
        if rhs.is_zero() && !self.is_nan() {
            return self.round_finite_to_format_mut_if_finite(format);
        }
        if self.is_zero() && !rhs.is_nan() {
            self.sign = -rhs.sign;
            self.exp = rhs.exp;
            self.limbs.clear();
            self.limbs.extend_from_slice(&rhs.limbs);
            return self.normalize_and_round_mut(format);
        }
        let a = core::mem::take(self);
        bf_add_internal_into(self, &a, rhs, format, true)
    }

    /// In-place division. Returns the status flags.
    pub fn div_assign(&mut self, rhs: &Self, format: BigFormat) -> Status {
        let a = core::mem::take(self);
        self.set_div(&a, rhs, format)
    }

    /// Sets `self` to `a * b`, reusing the existing allocation. Returns the status flags.
    pub fn set_mul(&mut self, a: &Self, b: &Self, format: BigFormat) -> Status {
        if a.is_nan() || b.is_nan() {
            *self = Self::nan();
            return Status::empty();
        }
        let sign = a.sign * b.sign;
        match (a.classify(), b.classify()) {
            (FpCategory::Infinite, FpCategory::Zero) | (FpCategory::Zero, FpCategory::Infinite) => {
                *self = Self::nan();
                return Status::INVALID_OP;
            }
            (FpCategory::Infinite, _) | (_, FpCategory::Infinite) => {
                *self = Self::infinity(sign);
                return Status::empty();
            }
            (FpCategory::Zero, _) | (_, FpCategory::Zero) => {
                *self = Self::zero(sign);
                return Status::empty();
            }
            _ => {}
        }
        bf_mul_direct_into(self, a, b, sign, format)
    }

    /// Sets `self` to `a + b`, reusing the existing allocation. Returns the status flags.
    pub fn set_add(&mut self, a: &Self, b: &Self, format: BigFormat) -> Status {
        bf_add_internal_into(self, a, b, format, false)
    }

    /// Sets `self` to `a - b`, reusing the existing allocation. Returns the status flags.
    pub fn set_sub(&mut self, a: &Self, b: &Self, format: BigFormat) -> Status {
        bf_add_internal_into(self, a, b, format, true)
    }

    /// Sets `self` to `a / b`, reusing the existing allocation. Returns the status flags.
    pub fn set_div(&mut self, a: &Self, b: &Self, format: BigFormat) -> Status {
        if a.is_nan() || b.is_nan() {
            *self = Self::nan();
            return Status::empty();
        }
        let sign = a.sign * b.sign;
        match (a.classify(), b.classify()) {
            (FpCategory::Zero, FpCategory::Zero) | (FpCategory::Infinite, FpCategory::Infinite) => {
                *self = Self::nan();
                return Status::INVALID_OP;
            }
            (_, FpCategory::Zero) => {
                *self = Self::infinity(sign);
                return Status::DIVIDE_ZERO;
            }
            (FpCategory::Infinite, _) => {
                *self = Self::infinity(sign);
                return Status::empty();
            }
            (_, FpCategory::Infinite) | (FpCategory::Zero, _) => {
                *self = Self::zero(sign);
                return Status::empty();
            }
            _ => {}
        }
        if matches!(format.precision, Precision::Infinite) {
            let (result, status) = a.div_status(b, format);
            *self = result;
            return status;
        }
        bf_div_direct_into(self, a, b, sign, format)
    }

    /// In-place multiplication by an `i64`. Returns the status flags.
    pub fn mul_i64_assign(&mut self, b: i64, format: BigFormat) -> Status {
        let (result, status) = self.mul_i64_status(b, format);
        *self = result;
        status
    }

    /// In-place multiplication by a `u64`. Returns the status flags.
    pub fn mul_u64_assign(&mut self, b: u64, format: BigFormat) -> Status {
        let bf_b = Self::from_u64(b);
        self.mul_assign(&bf_b, format)
    }

    /// In-place addition of an `i64`. Returns the status flags.
    pub fn add_i64_assign(&mut self, b: i64, format: BigFormat) -> Status {
        let bf_b = Self::from_i64(b);
        self.add_assign(&bf_b, format)
    }

    /// In-place squaring (`self = self * self`). Returns the status flags.
    ///
    /// This is the in-place counterpart of [`sqr_status`](Self::sqr_status).
    /// Needed because Rust's borrow rules prevent `self.mul_assign(&self, …)`,
    /// which would require `&mut self` and `&self` simultaneously.
    ///
    /// Corresponds to the `bf_mul(r, r, r, …)` aliasing pattern in libbf.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    ///
    /// let mut x = BigFloat::from_f64(3.0);
    /// x.sqr_assign(BigFormat::BINARY64);
    /// assert_eq!(x.to_f64(Rounding::NearestEven), 9.0);
    /// ```
    pub fn sqr_assign(&mut self, format: BigFormat) -> Status {
        let a = core::mem::take(self);
        self.set_mul(&a, &a, format)
    }

    /// Reverse subtraction: sets `self` to `lhs - self`. Returns the status flags.
    ///
    /// Needed because Rust's borrow rules prevent expressing the
    /// `bf_sub(r, a, r, …)` pattern (where the output aliases the second
    /// operand of a non-commutative operation) through the standard
    /// [`sub_assign`](Self::sub_assign), which computes `self = self - rhs`.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    ///
    /// let mut x = BigFloat::from_f64(3.0);
    /// let ten = BigFloat::from_f64(10.0);
    /// x.rsub_assign(&ten, BigFormat::BINARY64);
    /// assert_eq!(x.to_f64(Rounding::NearestEven), 7.0);
    /// ```
    pub fn rsub_assign(&mut self, lhs: &Self, format: BigFormat) -> Status {
        let b = core::mem::take(self);
        self.set_sub(lhs, &b, format)
    }

    /// Reverse division: sets `self` to `dividend / self`. Returns the status flags.
    ///
    /// Needed because Rust's borrow rules prevent expressing the
    /// `bf_div(r, a, r, …)` pattern (where the output aliases the divisor)
    /// through the standard [`div_assign`](Self::div_assign), which computes
    /// `self = self / rhs`.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    ///
    /// let mut x = BigFloat::from_f64(4.0);
    /// let twenty = BigFloat::from_f64(20.0);
    /// x.rdiv_assign(&twenty, BigFormat::BINARY64);
    /// assert_eq!(x.to_f64(Rounding::NearestEven), 5.0);
    /// ```
    pub fn rdiv_assign(&mut self, dividend: &Self, format: BigFormat) -> Status {
        let b = core::mem::take(self);
        self.set_div(dividend, &b, format)
    }

    /// In-place square root. Returns the status flags.
    pub fn sqrt_assign(&mut self, format: BigFormat) -> Status {
        let a = core::mem::take(self);
        self.set_sqrt(&a, format)
    }

    /// Rounds this value in place to the given format. Returns the status flags.
    pub fn round_assign(&mut self, format: BigFormat) -> Status {
        let (result, status) = self.round_status(format);
        *self = result;
        status
    }

    /// Multiplies `self` by `rhs`, returning the result and status flags.
    /// Corresponds to `bf_mul` in libbf. Returns the status flags as well.
    pub fn mul_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status) {
        if self.is_nan() || rhs.is_nan() {
            return (Self::nan(), Status::empty());
        }
        let sign = self.sign * rhs.sign;
        match (self.classify(), rhs.classify()) {
            (FpCategory::Infinite, FpCategory::Zero) | (FpCategory::Zero, FpCategory::Infinite) => {
                return (Self::nan(), Status::INVALID_OP);
            }
            (FpCategory::Infinite, _) | (_, FpCategory::Infinite) => {
                return (Self::infinity(sign), Status::empty());
            }
            (FpCategory::Zero, _) | (_, FpCategory::Zero) => {
                return (Self::zero(sign), Status::empty());
            }
            _ => {}
        }

        bf_mul_direct(self, rhs, sign, format)
    }

    /// Squares this value (`self * self`), returning the result.
    /// Corresponds to the `bf_mul(r, a, a, …)` pattern in libbf.
    ///
    /// This method exists because Rust's ownership rules prevent writing
    /// `a.mul(&a, format)` when `a` is behind a mutable reference — the
    /// borrow checker rejects the simultaneous `&self` and `&self` through
    /// the same `&mut`. More generally, `set_mul(&mut r, &a, &a, …)` is
    /// fine, but `r.mul_assign(&r, …)` is not when `r` is `a`.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    ///
    /// let x = BigFloat::from_f64(7.0);
    /// assert_eq!(x.sqr(BigFormat::BINARY64).to_f64(Rounding::NearestEven), 49.0);
    /// ```
    pub fn sqr(&self, format: BigFormat) -> Self {
        self.sqr_status(format).0
    }

    /// Squares this value, returning the result and status flags.
    /// Corresponds to the `bf_mul(r, a, a, …)` pattern in libbf.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding, Status};
    ///
    /// let x = BigFloat::from_f64(7.0);
    /// let (result, status) = x.sqr_status(BigFormat::BINARY64);
    /// assert_eq!(result.to_f64(Rounding::NearestEven), 49.0);
    /// assert!(status.is_empty());
    /// ```
    pub fn sqr_status(&self, format: BigFormat) -> (Self, Status) {
        self.mul_status(self, format)
    }

    /// Divides `self` by `rhs`, returning the result and status flags.
    /// Corresponds to `bf_div` in libbf. Returns the status flags as well.
    pub fn div_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status) {
        if self.is_nan() || rhs.is_nan() {
            return (Self::nan(), Status::empty());
        }
        let sign = self.sign * rhs.sign;
        match (self.classify(), rhs.classify()) {
            (FpCategory::Zero, FpCategory::Zero) | (FpCategory::Infinite, FpCategory::Infinite) => {
                return (Self::nan(), Status::INVALID_OP);
            }
            (_, FpCategory::Zero) => return (Self::infinity(sign), Status::DIVIDE_ZERO),
            (FpCategory::Infinite, _) => return (Self::infinity(sign), Status::empty()),
            (_, FpCategory::Infinite) => return (Self::zero(sign), Status::empty()),
            (FpCategory::Zero, _) => return (Self::zero(sign), Status::empty()),
            _ => {}
        }

        if matches!(format.precision, Precision::Infinite) {
            if let (Some(a), Some(b)) = (self.to_signed_int_limbs(), rhs.to_signed_int_limbs()) {
                let (quotient, remainder) = div_rem_signed_ints(&a, &b);
                if remainder.limbs.is_empty() {
                    return (Self::from_signed_int_limbs(quotient), Status::empty());
                }
            }
        }

        if matches!(format.precision, Precision::Infinite) {
            if let (Some(a), Some(b)) =
                (self.to_scaled_signed_limbs(), rhs.to_scaled_signed_limbs())
            {
                if let Some(exact) = div_scaled_signed_exact(&a, &b) {
                    return (Self::from_scaled_signed_limbs(exact), Status::empty());
                }
            }
        }

        bf_div_direct(self, rhs, sign, format)
    }

    /// Computes the remainder of `self / rhs`, returning the result and status flags.
    /// Corresponds to `bf_rem` in libbf. Returns the status flags as well.
    pub fn rem_status(&self, rhs: &Self, _format: BigFormat, mode: DivRemMode) -> (Self, Status) {
        if self.is_nan() || rhs.is_nan() || self.is_infinite() || rhs.is_zero() {
            return (Self::nan(), Status::INVALID_OP);
        }
        if self.is_zero() || rhs.is_infinite() {
            return (self.clone(), Status::empty());
        }

        if matches!(
            mode,
            DivRemMode::TowardZero
                | DivRemMode::NearestEven
                | DivRemMode::Floor
                | DivRemMode::Euclidean
        ) {
            if let (Some(a), Some(b)) =
                (self.to_scaled_signed_limbs(), rhs.to_scaled_signed_limbs())
            {
                let remainder = match mode {
                    DivRemMode::TowardZero => rem_scaled_toward_zero(&a, &b),
                    DivRemMode::NearestEven => rem_scaled_nearest_even(&a, &b),
                    DivRemMode::Floor => rem_scaled_floor(&a, &b),
                    DivRemMode::Euclidean => rem_scaled_euclidean(&a, &b),
                };
                return Self::from_scaled_signed_limbs(remainder).round_status_owned(_format);
            }
        }

        match (self.to_signed_int_limbs(), rhs.to_signed_int_limbs()) {
            (Some(a), Some(b)) => {
                let (_quotient, remainder) = div_rem_signed_ints(&a, &b);
                (Self::from_signed_int_limbs(remainder), Status::empty())
            }
            _ => unreachable!("finite BigFloat values are representable as scaled binary limbs"),
        }
    }

    /// Computes both the integer quotient and remainder of `self / rhs`.
    /// Returns `(quotient, remainder, status)`.
    /// Corresponds to `bf_divrem` in libbf. Returns the status flags as well.
    pub fn div_rem_status(
        &self,
        rhs: &Self,
        format: BigFormat,
        mode: DivRemMode,
    ) -> (Self, Self, Status) {
        if self.is_nan() || rhs.is_nan() {
            return (Self::zero(Sign::Positive), Self::nan(), Status::empty());
        }
        if self.is_infinite() || rhs.is_zero() {
            return (Self::zero(Sign::Positive), Self::nan(), Status::INVALID_OP);
        }
        if self.is_zero() {
            let (r, status) = self.round_status(format);
            return (Self::zero(Sign::Positive), r, status);
        }
        if rhs.is_infinite() {
            let (r, status) = self.round_status(format);
            return (Self::zero(Sign::Positive), r, status);
        }

        let q_sign = self.sign * rhs.sign;
        let is_rndn = matches!(mode, DivRemMode::NearestEven);
        let is_ceil = match mode {
            DivRemMode::TowardZero | DivRemMode::NearestEven => false,
            DivRemMode::Floor => q_sign.is_negative(),
            DivRemMode::Euclidean => self.sign.is_negative(),
        };

        let a1 = self.abs();
        let b1 = rhs.abs();

        let inf = BigFormat {
            precision: Precision::Infinite,
            ..format
        };

        let (mut q, mut r) = bf_tdivremu(&a1, &b1, inf);

        if !r.is_zero() {
            if is_rndn {
                let mut b_half = b1.clone();
                b_half.exp -= 1;
                let cmp = r.cmp_abs(&b_half);
                let q_low_bit = if q.is_zero() {
                    false
                } else {
                    q.get_bit_at_exp(q.lowest_bit_exp())
                };
                if cmp == Ordering::Greater || (cmp == Ordering::Equal && q_low_bit) {
                    q = q.add(&Self::from_i64(1), inf);
                    r = r.sub(&b1, inf);
                }
            } else if is_ceil {
                q = q.add(&Self::from_i64(1), inf);
                r = r.sub(&b1, inf);
            }
        }

        if !r.is_zero() {
            r.sign = r.sign * self.sign;
        }
        q.sign = if q.is_zero() { Sign::Positive } else { q_sign };
        let (r, status) = r.round_status_owned(format);
        (q, r, status)
    }

    /// Computes the remainder and an `i64` quotient of `self / rhs`.
    /// Returns `(quotient_i64, remainder, status)`.
    /// Corresponds to `bf_remquo` in libbf. Returns the status flags as well.
    pub fn rem_quo_status(
        &self,
        rhs: &Self,
        format: BigFormat,
        mode: DivRemMode,
    ) -> (i64, Self, Status) {
        let (q, r, status) = self.div_rem_status(rhs, format, mode);
        let q_val = q.to_f64(Rounding::TowardZero) as i64;
        (q_val, r, status)
    }

    /// Bitwise OR of `self` and `rhs`. Returns the status flags as well.
    /// Corresponds to `bf_logic_or` in libbf.
    pub fn logic_or_status(&self, rhs: &Self) -> (Self, Status) {
        self.logic_status(rhs, |a, b| a | b)
    }

    /// Bitwise XOR of `self` and `rhs`. Returns the status flags as well.
    /// Corresponds to `bf_logic_xor` in libbf.
    pub fn logic_xor_status(&self, rhs: &Self) -> (Self, Status) {
        self.logic_status(rhs, |a, b| a ^ b)
    }

    /// Bitwise AND of `self` and `rhs`. Returns the status flags as well.
    /// Corresponds to `bf_logic_and` in libbf.
    pub fn logic_and_status(&self, rhs: &Self) -> (Self, Status) {
        self.logic_status(rhs, |a, b| a & b)
    }

    fn logic_status<F: Fn(u64, u64) -> u64>(&self, rhs: &Self, op: F) -> (Self, Status) {
        if self.is_nan() || rhs.is_nan() || self.is_infinite() || rhs.is_infinite() {
            return (Self::zero(Sign::Positive), Status::INVALID_OP);
        }
        (bf_logic_op_direct(self, rhs, op), Status::empty())
    }

    /// Rounds this value to the given format.
    /// Corresponds to `bf_round` in libbf.
    pub fn round(&self, format: BigFormat) -> Self {
        self.round_status(format).0
    }

    /// Rounds this value to the given format, returning the status flags as well.
    /// Corresponds to `bf_round` in libbf.
    pub fn round_status(&self, format: BigFormat) -> (Self, Status) {
        if !self.is_finite() {
            return (self.clone(), Status::empty());
        }
        self.round_finite_to_format(format)
    }

    fn round_status_owned(mut self, format: BigFormat) -> (Self, Status) {
        if !self.is_finite() {
            return (self, Status::empty());
        }
        let status = self.round_finite_to_format_mut(format);
        (self, status)
    }

    /// Normalizes the internal representation and then rounds to `format`.
    /// Corresponds to `bf_normalize_and_round` in libbf.
    pub fn normalize_and_round(mut self, format: BigFormat) -> (Self, Status) {
        let status = self.normalize_and_round_mut(format);
        (self, status)
    }

    fn normalize_and_round_mut(&mut self, format: BigFormat) -> Status {
        let old_len = self.limbs.len();
        trim_limbs(&mut self.limbs);
        if self.limbs.is_empty() {
            self.exp = RAW_EXP_ZERO;
            return Status::empty();
        }
        self.exp -= ((old_len - self.limbs.len()) as i64) * 64;
        let l = self.limbs.len();
        let top = self.limbs[l - 1];
        let shift = top.leading_zeros();
        if shift != 0 {
            // In-place left shift without allocating — shift < 64
            let mut carry = 0_u64;
            for limb in self.limbs.iter_mut() {
                let next = *limb >> (64 - shift);
                *limb = (*limb << shift) | carry;
                carry = next;
            }
            self.exp -= i64::from(shift);
        }
        self.round_finite_to_format_mut(format)
    }

    /// Rounds this value to an integer using the given rounding mode.
    /// Corresponds to `bf_rint` in libbf.
    pub fn rint(&self, rounding: Rounding) -> Self {
        self.rint_status(rounding).0
    }

    /// Rounds this value to an integer, returning the status flags as well.
    /// Corresponds to `bf_rint` in libbf.
    pub fn rint_status(&self, rounding: Rounding) -> (Self, Status) {
        if !self.is_finite() {
            return (self.clone(), Status::empty());
        }
        // rint = round to integer = round with radix-point precision 0
        use crate::format::ExpBits;
        let format = BigFormat {
            precision: Precision::Bits(0),
            rounding,
            radix_point_precision: true,
            exp_bits: ExpBits::Max,
            subnormal: false,
        };
        let mut r = self.clone();
        let status = r.round_finite_to_format_mut(format);
        (r, status)
    }

    /// Computes the integer square root and remainder: returns `(root, remainder, status)`
    /// such that `root^2 + remainder == self` when `self` is a non-negative integer.
    /// Corresponds to `bf_sqrtrem` in libbf.
    pub fn sqrtrem(&self) -> (Self, Self, Status) {
        if self.is_nan() {
            return (Self::nan(), Self::zero(Sign::Positive), Status::empty());
        }
        if self.is_infinite() {
            if self.sign.is_negative() {
                return (Self::nan(), Self::zero(Sign::Positive), Status::INVALID_OP);
            }
            return (self.clone(), Self::zero(Sign::Positive), Status::empty());
        }
        if self.is_zero() {
            return (self.clone(), Self::zero(Sign::Positive), Status::empty());
        }
        if self.sign.is_negative() {
            return (Self::nan(), Self::zero(Sign::Positive), Status::INVALID_OP);
        }

        let inf = BigFormat {
            precision: Precision::Infinite,
            ..BigFormat::BINARY64
        };
        let prec = ((self.exp + 1) / 2).max(2) as u64;
        let root = self.sqrt(BigFormat {
            precision: Precision::Bits(prec),
            rounding: Rounding::TowardZero,
            ..BigFormat::BINARY64
        });
        let root = root.rint(Rounding::TowardZero);
        let root_sq = root.mul(&root, inf);
        let rem = self.sub(&root_sq, inf);
        let status = if rem.is_zero() {
            Status::empty()
        } else {
            Status::INEXACT
        };
        (root, rem, status)
    }

    /// Attempts to convert this value to an `i64`, returning `None` if the value
    /// is not an integer or is out of range.
    /// Corresponds to `bf_get_int64` in libbf.
    pub fn get_int64(&self) -> Option<i64> {
        if self.is_zero() {
            return Some(0);
        }
        if !self.is_finite() {
            return None;
        }
        let (f, status) = self.to_f64_status(Rounding::TowardZero);
        if status.contains(Status::OVERFLOW) {
            return None;
        }
        let val = f as i64;
        if val as f64 != f {
            return None;
        }
        Some(val)
    }

    /// Attempts to convert this value to a `u64`, returning `None` if the value
    /// is not a non-negative integer or is out of range.
    /// Corresponds to `bf_get_uint64` in libbf.
    pub fn get_uint64(&self) -> Option<u64> {
        if self.is_zero() {
            return Some(0);
        }
        if !self.is_finite() || self.sign.is_negative() {
            return None;
        }
        let abs_limbs = self.to_abs_int_limbs()?;
        if abs_limbs.len() > 1 {
            return None;
        }
        abs_limbs.first().copied().or(Some(0))
    }

    /// Computes the square root of this value, rounded to `format`.
    /// Corresponds to `bf_sqrt` in libbf. Uses Newton iteration via integer square root.
    pub fn sqrt(&self, format: BigFormat) -> Self {
        self.sqrt_status(format).0
    }

    /// Computes the square root, returning the result and status flags.
    /// Corresponds to `bf_sqrt` in libbf. Uses Newton iteration via integer square root.
    pub fn sqrt_status(&self, format: BigFormat) -> (Self, Status) {
        if self.is_nan() {
            return (Self::nan(), Status::empty());
        }
        if self.is_infinite() {
            if self.sign.is_negative() {
                return (Self::nan(), Status::INVALID_OP);
            }
            return (self.clone(), Status::empty());
        }
        if self.sign.is_negative() && !self.is_zero() {
            return (Self::nan(), Status::INVALID_OP);
        }
        if self.is_zero() {
            return (self.clone(), Status::empty());
        }

        let precision = match format.precision {
            Precision::Bits(p) => p,
            Precision::Digits(d) => decimal_digits_to_binary_precision(d),
            Precision::Infinite => {
                // Infinite precision: only return exact results
                match self.to_abs_int_limbs() {
                    Some(n) => {
                        let (root, rem) = sqrt_rem_abs_limbs(&n);
                        if rem.is_empty() {
                            return (
                                Self::from_abs_int_limbs(root, Sign::Positive),
                                Status::empty(),
                            );
                        }
                        return (Self::nan(), Status::INVALID_OP);
                    }
                    None => return (Self::nan(), Status::INVALID_OP),
                }
            }
        };

        bf_sqrt_direct(
            self,
            BigFormat {
                precision: Precision::Bits(precision),
                ..format
            },
            format,
        )
    }

    /// Sets `self` to `sqrt(a)`, reusing the existing allocation.
    /// Returns the status flags. In-place counterpart of
    /// [`sqrt_status`](Self::sqrt_status); corresponds to `bf_sqrt(r, a, …)`
    /// in libbf where `r` is a distinct variable.
    pub fn set_sqrt(&mut self, a: &Self, format: BigFormat) -> Status {
        if a.is_nan() {
            *self = Self::nan();
            return Status::empty();
        }
        if a.is_infinite() {
            if a.sign.is_negative() {
                *self = Self::nan();
                return Status::INVALID_OP;
            }
            *self = a.clone();
            return Status::empty();
        }
        if a.sign.is_negative() && !a.is_zero() {
            *self = Self::nan();
            return Status::INVALID_OP;
        }
        if a.is_zero() {
            *self = a.clone();
            return Status::empty();
        }

        let precision = match format.precision {
            Precision::Bits(p) => p,
            Precision::Digits(d) => decimal_digits_to_binary_precision(d),
            Precision::Infinite => {
                let (result, status) = a.sqrt_status(format);
                *self = result;
                return status;
            }
        };

        bf_sqrt_direct_into(
            self,
            a,
            BigFormat {
                precision: Precision::Bits(precision),
                ..format
            },
            format,
        )
    }

    // exp, log, sin, cos, tan, atan, asin, acos, atan2, pow
    // are implemented in transc.rs

    /// Returns the negation of this value.
    /// Corresponds to `bf_neg` in libbf.
    pub fn neg(&self) -> Self {
        let mut value = self.clone();
        value.sign = -value.sign;
        value
    }

    /// Returns the absolute value.
    pub fn abs(&self) -> Self {
        let mut value = self.clone();
        value.sign = Sign::Positive;
        value
    }

    /// Returns `true` if this value is finite (not NaN or infinity).
    /// Corresponds to `bf_is_finite` in libbf.
    pub const fn is_finite(&self) -> bool {
        self.exp < RAW_EXP_INF
    }

    /// Returns `true` if this value is NaN.
    /// Corresponds to `bf_is_nan` in libbf.
    pub const fn is_nan(&self) -> bool {
        self.exp == RAW_EXP_NAN
    }

    /// Returns `true` if this value is positive or negative infinity.
    /// No direct libbf equivalent (libbf checks via the exponent field).
    pub const fn is_infinite(&self) -> bool {
        self.exp == RAW_EXP_INF
    }

    /// Returns `true` if this value is positive or negative zero.
    /// Corresponds to `bf_is_zero` in libbf.
    pub const fn is_zero(&self) -> bool {
        self.exp == RAW_EXP_ZERO
    }

    /// Returns `true` if the sign of this value is negative.
    /// No direct libbf equivalent.
    pub const fn is_sign_negative(&self) -> bool {
        self.sign.is_negative()
    }

    /// Classifies this value into one of the [`FpCategory`] variants.
    /// No direct libbf equivalent.
    pub const fn classify(&self) -> FpCategory {
        match self.exp {
            RAW_EXP_NAN => FpCategory::Nan,
            RAW_EXP_INF => FpCategory::Infinite,
            RAW_EXP_ZERO => FpCategory::Zero,
            _ => FpCategory::Normal,
        }
    }

    /// Compares the absolute values of `self` and `rhs`.
    /// Corresponds to `bf_cmpu` in libbf.
    pub fn cmp_abs(&self, rhs: &Self) -> Ordering {
        self.cmp_magnitude(rhs)
    }

    /// Compares `self` and `rhs` numerically, returning `None` if either is NaN.
    /// Corresponds to `bf_cmp` in libbf.
    pub fn cmp_num(&self, rhs: &Self) -> Option<Ordering> {
        if self.is_nan() || rhs.is_nan() {
            return None;
        }
        if self.is_zero() && rhs.is_zero() {
            return Some(Ordering::Equal);
        }
        match (self.sign, rhs.sign) {
            (Sign::Negative, Sign::Positive) => Some(Ordering::Less),
            (Sign::Positive, Sign::Negative) => Some(Ordering::Greater),
            (Sign::Positive, Sign::Positive) => Some(self.cmp_magnitude(rhs)),
            (Sign::Negative, Sign::Negative) => Some(self.cmp_magnitude(rhs).reverse()),
        }
    }

    /// Total ordering comparison that orders all values including NaN.
    /// Corresponds to `bf_cmp_full` in libbf.
    pub fn cmp_total(&self, rhs: &Self) -> Ordering {
        self.exp
            .cmp(&rhs.exp)
            .then_with(|| self.sign.is_negative().cmp(&rhs.sign.is_negative()))
            .then_with(|| self.limbs.cmp(&rhs.limbs))
    }

    /// Alias for [`cmp_total`](Self::cmp_total).
    /// Corresponds to `bf_cmp_full` in libbf.
    pub fn cmp_full(&self, rhs: &Self) -> Ordering {
        self.cmp_total(rhs)
    }

    /// Tests whether this value can be correctly rounded to `precision` bits with the
    /// given rounding mode, assuming at least `k` correct bits are available.
    /// Corresponds to `bf_can_round` in libbf.
    pub fn can_round(&self, precision: u64, rounding: Rounding, k: u64) -> bool {
        if !self.is_finite() || self.is_zero() {
            return false;
        }
        if rounding == Rounding::Faithful {
            return k >= precision.saturating_add(1);
        }
        if k < precision.saturating_add(2) {
            return false;
        }

        let precision = i128::from(precision);
        let k = i128::from(k);
        let bit_pos = (self.limbs.len() as i128) * 64 - 1 - precision;
        let mut n = k - precision;
        let is_nearest = matches!(rounding, Rounding::NearestEven | Rounding::NearestAway);
        let mut bit = get_abs_bit_signed(&self.limbs, bit_pos);
        if is_nearest {
            bit = !bit;
        }
        let mut bit_pos = bit_pos - 1;
        n -= 1;
        while n != 0 {
            if get_abs_bit_signed(&self.limbs, bit_pos) != bit {
                return true;
            }
            bit_pos -= 1;
            n -= 1;
        }
        false
    }

    fn cmp_magnitude(&self, rhs: &Self) -> Ordering {
        match (self.classify(), rhs.classify()) {
            (FpCategory::Nan, FpCategory::Nan) => Ordering::Equal,
            (FpCategory::Nan, _) => Ordering::Greater,
            (_, FpCategory::Nan) => Ordering::Less,
            (FpCategory::Infinite, FpCategory::Infinite) => Ordering::Equal,
            (FpCategory::Infinite, _) => Ordering::Greater,
            (_, FpCategory::Infinite) => Ordering::Less,
            (FpCategory::Zero, FpCategory::Zero) => Ordering::Equal,
            (FpCategory::Zero, _) => Ordering::Less,
            (_, FpCategory::Zero) => Ordering::Greater,
            (FpCategory::Normal, FpCategory::Normal) => self
                .exp
                .cmp(&rhs.exp)
                .then_with(|| self.limbs.cmp(&rhs.limbs)),
        }
    }

    fn from_u128_with_sign(magnitude: u128, sign: Sign) -> Self {
        if magnitude == 0 {
            return Self::zero(sign);
        }

        let bit_len = 128_u32 - magnitude.leading_zeros();
        Self::from_u128_scaled(magnitude, i64::from(bit_len), sign)
    }

    fn from_u128_scaled(magnitude: u128, exp: RawExp, sign: Sign) -> Self {
        if magnitude == 0 {
            return Self::zero(sign);
        }

        let bit_len = 128_u32 - magnitude.leading_zeros();
        let limb_count = bit_len.div_ceil(64) as usize;
        let total_bits = (limb_count as u32) * 64;
        let shift = total_bits - bit_len;
        let normalized = magnitude << shift;

        let mut limbs = Vec::with_capacity(limb_count);
        limbs.push(normalized as u64);
        if limb_count == 2 {
            limbs.push((normalized >> 64) as u64);
        }

        Self { sign, exp, limbs }
    }

    pub(crate) fn to_signed_int_limbs(&self) -> Option<SignedInt> {
        let magnitude = self.to_abs_int_limbs()?;
        Some(SignedInt {
            sign: if magnitude.is_empty() {
                Sign::Positive
            } else {
                self.sign
            },
            limbs: magnitude,
        })
    }

    fn to_scaled_signed_limbs(&self) -> Option<ScaledSignedInt> {
        if self.is_zero() {
            return Some(ScaledSignedInt {
                sign: Sign::Positive,
                limbs: Vec::new(),
                low_exp: 0,
            });
        }
        if !self.is_finite() || self.limbs.is_empty() {
            return None;
        }
        let mut limbs = self.limbs.clone();
        trim_limbs(&mut limbs);
        Some(ScaledSignedInt {
            sign: self.sign,
            low_exp: self.exp - (self.limbs.len() as i64) * 64,
            limbs,
        })
    }

    fn to_abs_int_limbs(&self) -> Option<Vec<u64>> {
        if self.is_zero() {
            return Some(Vec::new());
        }
        if !self.is_finite() || self.exp < 0 || self.limbs.is_empty() {
            return None;
        }

        let total_bits = (self.limbs.len() as i64) * 64;
        let shift = total_bits - self.exp;
        let mut limbs = self.limbs.clone();

        if shift > 0 {
            if !shr_limbs_exact(&mut limbs, shift as u32) {
                return None;
            }
        } else if shift < 0 {
            shl_limbs(&mut limbs, (-shift) as u32);
        }

        trim_limbs(&mut limbs);
        Some(limbs)
    }

    fn from_signed_int_limbs(value: SignedInt) -> Self {
        Self::from_abs_int_limbs(value.limbs, value.sign)
    }

    fn from_scaled_signed_limbs(value: ScaledSignedInt) -> Self {
        Self::from_abs_int_limbs_scaled(value.limbs, value.sign, value.low_exp)
    }

    fn from_abs_int_limbs(limbs: Vec<u64>, sign: Sign) -> Self {
        Self::from_abs_int_limbs_scaled(limbs, sign, 0)
    }

    fn from_abs_int_limbs_scaled(mut limbs: Vec<u64>, sign: Sign, low_exp: i64) -> Self {
        trim_limbs(&mut limbs);
        if limbs.is_empty() {
            return Self::zero(sign);
        }

        let top = *limbs.last().expect("non-empty limbs");
        let bit_len = ((limbs.len() - 1) as i64) * 64 + (64_i64 - i64::from(top.leading_zeros()));
        let shift = top.leading_zeros();
        if shift != 0 {
            shl_limbs(&mut limbs, shift);
        }
        remove_low_zero_limbs(&mut limbs);

        Self {
            sign,
            exp: bit_len + low_exp,
            limbs,
        }
    }

    #[inline]
    fn round_finite_to_format_mut_if_finite(&mut self, format: BigFormat) -> Status {
        if self.is_finite() {
            self.round_finite_to_format_mut(format)
        } else {
            Status::empty()
        }
    }

    fn round_finite_to_format(&self, format: BigFormat) -> (Self, Status) {
        let mut r = self.clone();
        let status = r.round_finite_to_format_mut(format);
        (r, status)
    }

    fn round_finite_to_format_mut(&mut self, format: BigFormat) -> Status {
        let prec1 = match format.precision {
            Precision::Infinite => {
                remove_low_zero_limbs(&mut self.limbs);
                return Status::empty();
            }
            Precision::Bits(bits) => bits,
            Precision::Digits(digits) => {
                remove_low_zero_limbs(&mut self.limbs);
                let (r, s) = self.round_to_decimal_digits(digits, format.rounding);
                *self = r;
                return s;
            }
        };
        let prec1_i64 = prec1 as i64;
        if prec1_i64 < 0 {
            remove_low_zero_limbs(&mut self.limbs);
            return Status::empty();
        }

        let l = self.limbs.len() as i64;
        if l == 0 {
            return Status::empty();
        }

        let (e_min, e_max) = exp_range_from_format(&format);

        let prec = if format.radix_point_precision {
            prec1_i64.saturating_add(self.exp)
        } else if self.exp < e_min && format.subnormal {
            prec1_i64 - (e_min - self.exp)
        } else {
            prec1_i64
        };

        let significant_bits = l * 64;
        if prec > 0 && significant_bits <= prec {
            if e_max < i64::MAX && self.exp > e_max {
                let (r, s) = self.set_overflow(prec1, format);
                *self = r;
                return s;
            }
            remove_low_zero_limbs(&mut self.limbs);
            return Status::empty();
        }

        bf_round_in_place(self, l, prec, prec1, e_min, e_max, format)
    }

    fn set_overflow(&self, prec: u64, format: BigFormat) -> (Self, Status) {
        let rnd_mode = format.rounding;
        let sign = self.sign;
        let rounds_to_inf = matches!(
            rnd_mode,
            Rounding::NearestEven | Rounding::NearestAway | Rounding::AwayFromZero
        ) || (rnd_mode == Rounding::TowardNegative && sign.is_negative())
            || (rnd_mode == Rounding::TowardPositive && !sign.is_negative())
            || matches!(format.precision, Precision::Infinite);

        if rounds_to_inf {
            (Self::infinity(sign), Status::OVERFLOW | Status::INEXACT)
        } else {
            let (_e_min, e_max) = exp_range_from_format(&format);
            let l = prec.div_ceil(64) as usize;
            let mut limbs = alloc::vec![u64::MAX; l];
            let mask_bits = (prec % 64) as u32;
            if mask_bits != 0 {
                let mask = u64::MAX << (64 - mask_bits);
                limbs[0] = mask;
            }
            let result = Self::from_abs_int_limbs_scaled(limbs, sign, e_max - prec as i64);
            (result, Status::OVERFLOW | Status::INEXACT)
        }
    }

    fn round_to_decimal_digits(&self, digits: u64, rounding: Rounding) -> (Self, Status) {
        if digits == 0 {
            return (self.clone(), Status::empty());
        }

        let Some(decimal) = self.to_exact_decimal_string() else {
            return (self.clone(), Status::empty());
        };
        if decimal == "0" || decimal == "-0" {
            return (self.clone(), Status::empty());
        }

        let (is_negative, abs_decimal) = decimal
            .strip_prefix('-')
            .map_or((false, decimal.as_str()), |rest| (true, rest));
        let Some(digits_usize) = usize::try_from(digits).ok() else {
            return (self.clone(), Status::empty());
        };

        let point_pos = abs_decimal.find('.').unwrap_or(abs_decimal.len());
        let mut all_digits = String::with_capacity(abs_decimal.len());
        for byte in abs_decimal.bytes() {
            if byte != b'.' {
                all_digits.push(char::from(byte));
            }
        }
        let first_sig = all_digits
            .as_bytes()
            .iter()
            .position(|&digit| digit != b'0');
        let Some(first_sig) = first_sig else {
            return (self.clone(), Status::empty());
        };
        let significant_len = all_digits.len() - first_sig;
        if significant_len <= digits_usize {
            return (self.clone(), Status::empty());
        }

        let keep_len = first_sig + digits_usize;
        let drop_len = all_digits.len() - keep_len;
        let kept = &all_digits[..keep_len];
        let discarded = &all_digits[keep_len..];
        let discarded_nonzero = discarded.as_bytes().iter().any(|&digit| digit != b'0');
        if !discarded_nonzero {
            return (self.clone(), Status::empty());
        }

        let increment = should_increment_decimal_round(kept, discarded, is_negative, rounding);
        let mut rounded_abs = if increment {
            increment_decimal_digits(kept)
        } else {
            kept.to_string()
        };
        for _ in 0..drop_len {
            rounded_abs.push('0');
        }

        let scale = all_digits.len().saturating_sub(point_pos);
        let mut rounded_with_point = if scale == 0 {
            rounded_abs
        } else if rounded_abs.len() <= scale {
            let mut literal = String::with_capacity(scale + 2);
            literal.push('0');
            literal.push('.');
            for _ in 0..(scale - rounded_abs.len()) {
                literal.push('0');
            }
            literal.push_str(&rounded_abs);
            literal
        } else {
            rounded_abs.insert(rounded_abs.len() - scale, '.');
            rounded_abs
        };

        let rounded_literal = if is_negative {
            let mut literal = String::with_capacity(rounded_with_point.len() + 1);
            literal.push('-');
            literal.push_str(&rounded_with_point);
            literal
        } else {
            core::mem::take(&mut rounded_with_point)
        };
        let rounded = if scale == 0 {
            Self::parse_integer_radix(&rounded_literal, 10)
                .expect("internally generated decimal integer")
        } else {
            Self::parse_decimal(
                &rounded_literal,
                BigFormat {
                    precision: Precision::Bits(decimal_digits_to_binary_precision(digits)),
                    ..BigFormat::BINARY64
                },
            )
            .expect("internally generated decimal literal")
        };
        (rounded, Status::INEXACT)
    }

    fn normal_to_f64(&self, rounding: Rounding) -> (f64, Status) {
        if self.exp > 1024 {
            let value = if self.sign.is_negative() {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            return (value, Status::OVERFLOW | Status::INEXACT);
        }
        if self.exp < -1073 {
            return (
                if self.sign.is_negative() { -0.0 } else { 0.0 },
                Status::UNDERFLOW | Status::INEXACT,
            );
        }

        let precision = if self.exp >= -1021 {
            53_i64
        } else {
            self.exp + 1074
        };
        if precision <= 0 {
            return (
                if self.sign.is_negative() { -0.0 } else { 0.0 },
                Status::UNDERFLOW | Status::INEXACT,
            );
        }

        let (mant, inexact) = self.rounded_mantissa_for_precision(precision as u32, rounding);
        let mut exp = self.exp;
        let mut mant = mant;
        if mant >= (1_u128 << precision) {
            mant >>= 1;
            exp += 1;
        }
        if exp > 1024 {
            let value = if self.sign.is_negative() {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            return (value, Status::OVERFLOW | Status::INEXACT);
        }

        let sign_bit = u64::from(self.sign.is_negative()) << 63;
        let bits = if exp >= -1021 {
            let exp_bits = ((exp + 1022) as u64) << 52;
            let frac = (mant as u64) & ((1_u64 << 52) - 1);
            sign_bit | exp_bits | frac
        } else {
            sign_bit | mant as u64
        };

        let mut status = Status::empty();
        if inexact {
            status |= Status::INEXACT;
        }
        if inexact && exp < -1021 {
            status |= Status::UNDERFLOW;
        }
        (f64::from_bits(bits), status)
    }

    fn rounded_mantissa_for_precision(&self, precision: u32, rounding: Rounding) -> (u128, bool) {
        debug_assert!(precision <= 64);
        let keep_start = self.exp - i64::from(precision);
        let mut mant = 0_u128;
        for i in 0..precision {
            let bit_exp = keep_start + i64::from(i);
            if self.get_bit_at_exp(bit_exp) {
                mant |= 1_u128 << i;
            }
        }

        let mut inexact = false;
        if keep_start > self.lowest_bit_exp() {
            for bit_exp in self.lowest_bit_exp()..keep_start {
                if self.get_bit_at_exp(bit_exp) {
                    inexact = true;
                    break;
                }
            }
        }
        if !inexact {
            return (mant, false);
        }

        let guard = keep_start > self.lowest_bit_exp() && self.get_bit_at_exp(keep_start - 1);
        let sticky = if keep_start > self.lowest_bit_exp() + 1 {
            (self.lowest_bit_exp()..(keep_start - 1)).any(|bit_exp| self.get_bit_at_exp(bit_exp))
        } else {
            false
        };

        let add_one = match rounding {
            Rounding::NearestEven => guard && (sticky || (mant & 1) != 0),
            Rounding::NearestAway => guard,
            Rounding::TowardZero => false,
            Rounding::TowardPositive => !self.sign.is_negative(),
            Rounding::TowardNegative => self.sign.is_negative(),
            Rounding::AwayFromZero => true,
            Rounding::Faithful => false,
        };
        if add_one {
            mant += 1;
        }
        (mant, true)
    }

    fn lowest_bit_exp(&self) -> i64 {
        self.exp - (self.limbs.len() as i64) * 64
    }

    fn get_bit_at_exp(&self, bit_exp: i64) -> bool {
        let offset = bit_exp - self.lowest_bit_exp();
        if offset < 0 {
            return false;
        }
        let offset = offset as usize;
        self.limbs
            .get(offset / 64)
            .is_some_and(|limb| ((limb >> (offset % 64)) & 1) != 0)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SignedInt {
    pub(crate) sign: Sign,
    pub(crate) limbs: Vec<u64>,
}

/// In-place rounding, equivalent to libbf's `__bf_round`.
/// `r` must be finite and non-zero with `l` limbs. `prec` is the adjusted
/// precision (may be <= 0 for subnormals). `prec1` is the original precision.
fn bf_round_in_place(
    r: &mut BigFloat,
    l: i64,
    prec: i64,
    prec1: u64,
    e_min: i64,
    e_max: i64,
    format: BigFormat,
) -> Status {
    let l_usize = l as usize;

    // Compute guard and sticky bits directly from the existing limbs.
    // bit_pos is the position (from the LSB of limbs[0]) of the bit at
    // index `prec` counting from the MSB (which is at position l*64-1).
    // guard bit = bit at position `prec` from MSB (0-indexed)
    // sticky bits = all bits below the guard bit
    let guard_pos = l * 64 - 1 - prec; // bit position of guard bit

    let (add_one, inexact) = if format.rounding == Rounding::Faithful {
        // Faithful rounding: just use the guard bit as add_one, always "inexact"
        let guard = if prec >= l * 64 || prec < 0 {
            false
        } else {
            get_abs_bit(&r.limbs, guard_pos as usize)
        };
        (guard, guard)
    } else {
        // Sticky: any bits set below the guard bit position
        let sticky_scan_pos = (prec + 1).max(0);
        let sticky = if sticky_scan_pos >= l * 64 {
            false
        } else {
            scan_bit_nz(&r.limbs, l * 64 - 1 - sticky_scan_pos)
        };

        let guard = if prec < 0 || prec >= l * 64 {
            false
        } else {
            get_abs_bit(&r.limbs, guard_pos as usize)
        };

        let inexact = guard || sticky;
        let add_one = inexact
            && match format.rounding {
                Rounding::NearestEven => {
                    if guard {
                        if sticky {
                            true
                        } else {
                            // round to even: check the bit at prec-1
                            prec >= 1 && get_abs_bit(&r.limbs, (l * 64 - prec) as usize)
                        }
                    } else {
                        false
                    }
                }
                Rounding::NearestAway => guard,
                Rounding::TowardZero => false,
                Rounding::TowardPositive => !r.sign.is_negative(),
                Rounding::TowardNegative => r.sign.is_negative(),
                Rounding::AwayFromZero => true,
                Rounding::Faithful => unreachable!(),
            };
        (add_one, inexact)
    };

    if prec <= 0 {
        let mut status = Status::empty();
        if inexact {
            status |= Status::INEXACT;
        }
        if add_one {
            r.limbs.clear();
            r.limbs.push(1_u64 << 63);
            r.exp += 1 - prec;
            if r.exp < e_min {
                status |= Status::UNDERFLOW;
            }
            return status;
        } else {
            *r = BigFloat::zero(r.sign);
            if e_min > i64::MIN / 2 {
                status |= Status::UNDERFLOW;
            }
            return status;
        }
    }

    if add_one {
        // Add one at the bit position `prec - 1` from the MSB.
        // That bit position from LSB of limbs[0] is: l*64 - 1 - (prec - 1) = l*64 - prec
        let bit_pos = l * 64 - prec;
        let pos = (bit_pos >> 6) as usize;
        let mut carry = 1_u64 << (bit_pos & 63);

        let mut i = pos;
        while i < l_usize {
            let v = r.limbs[i].wrapping_add(carry);
            carry = u64::from(v < carry);
            r.limbs[i] = v;
            if carry == 0 {
                break;
            }
            i += 1;
        }
        if carry != 0 {
            // Overflow: shift right by 1
            let mut prev = 1_u64;
            for i in (pos..l_usize).rev() {
                let a = r.limbs[i];
                r.limbs[i] = (a >> 1) | (prev << 63);
                prev = a;
            }
            r.exp += 1;
        }
    }

    // Check underflow
    if r.exp < e_min {
        if format.subnormal {
            if inexact {
                let status = Status::INEXACT | Status::UNDERFLOW;
                // Clear low bits and remove trailing zeros below
                bf_round_clear_low_bits(r, l, prec);
                return status;
            }
        } else {
            *r = BigFloat::zero(r.sign);
            return Status::UNDERFLOW | Status::INEXACT;
        }
    }

    // Check overflow
    if e_max < i64::MAX && r.exp > e_max {
        let (result, status) = r.set_overflow(prec1, format);
        *r = result;
        return status;
    }

    // Clear low bits: mask off everything below `prec`
    bf_round_clear_low_bits(r, l, prec);

    let mut status = Status::empty();
    if inexact {
        status |= Status::INEXACT;
    }
    status
}

/// Clear the low bits after rounding and remove trailing zero limbs.
#[inline]
fn bf_round_clear_low_bits(r: &mut BigFloat, l: i64, prec: i64) {
    let bit_pos = l * 64 - prec;
    let mut i = (bit_pos >> 6) as usize;
    let l_usize = l as usize;

    if i < l_usize {
        let shift = (bit_pos & 63) as u32;
        if shift != 0 {
            let mask = u64::MAX << shift;
            r.limbs[i] &= mask;
        }
    } else {
        i = 0;
    }

    // Remove trailing zero limbs
    while i < l_usize && r.limbs[i] == 0 {
        i += 1;
    }
    if i > 0 {
        r.limbs.drain(..i);
    }
}

/// Scan for any non-zero bit at or below `bit_pos` (inclusive, from LSB of limbs[0]).
#[inline]
fn scan_bit_nz(limbs: &[u64], bit_pos: i64) -> bool {
    if bit_pos < 0 {
        return false;
    }
    let pos = (bit_pos >> 6) as usize;
    if pos >= limbs.len() {
        return false;
    }
    let mask = (bit_pos & 63) as u32;
    let v = limbs[pos] & ((2_u64 << mask).wrapping_sub(1));
    if v != 0 {
        return true;
    }
    limbs[..pos].iter().any(|&limb| limb != 0)
}

#[inline]
fn bf_cmpu(a: &BigFloat, b: &BigFloat) -> Ordering {
    if a.exp != b.exp {
        return a.exp.cmp(&b.exp);
    }
    let len = a.limbs.len().max(b.limbs.len());
    for i in (0..len).rev() {
        let v1 = bf_get_limbz(a, (a.limbs.len() as i64) - (len as i64) + (i as i64));
        let v2 = bf_get_limbz(b, (b.limbs.len() as i64) - (len as i64) + (i as i64));
        match v1.cmp(&v2) {
            Ordering::Equal => {}
            order => return order,
        }
    }
    Ordering::Equal
}

#[inline(always)]
fn bf_get_limbz(a: &BigFloat, idx: i64) -> u64 {
    if idx < 0 || idx as usize >= a.limbs.len() {
        0
    } else {
        a.limbs[idx as usize]
    }
}

#[inline(always)]
fn bf_get_bits(tab: &[u64], pos: i64) -> u64 {
    let i = pos >> 6;
    let p = (pos & 63) as u32;
    let a0 = if i < 0 || i as usize >= tab.len() {
        0
    } else {
        tab[i as usize]
    };
    if p == 0 {
        a0
    } else {
        let a1 = if (i + 1) < 0 || (i + 1) as usize >= tab.len() {
            0
        } else {
            tab[(i + 1) as usize]
        };
        (a0 >> p) | (a1 << (64 - p))
    }
}

#[inline]
fn bf_count_cancelled_bits(a: &BigFloat, b: &BigFloat) -> i64 {
    let mut bit_offset = (a.limbs.len() as i64) * 64 - 1;
    let b_offset = ((b.limbs.len() as i64) - (a.limbs.len() as i64)) * 64 - 63 + a.exp - b.exp;
    let mut n = 0_i64;

    let (v1, v2) = loop {
        let v1 = bf_get_limbz(a, bit_offset >> 6);
        let v2 = bf_get_bits(&b.limbs, bit_offset + b_offset);
        if v1 != v2 {
            break (v1, v2);
        }
        n += 64;
        bit_offset -= 64;
    };

    let p = (v1 ^ v2).leading_zeros() as i32 + 1;
    n += i64::from(p);
    let trailing_bits = 64 - p;
    if trailing_bits > 0 {
        let mask = limb_low_mask(trailing_bits);
        let p1 = ((v1 & mask)
            .leading_zeros()
            .min((!v2 & mask).leading_zeros()) as i32)
            - (64 - trailing_bits);
        n += i64::from(p1);
        if p1 != trailing_bits {
            return n;
        }
    }

    bit_offset -= 64;
    loop {
        let v1 = bf_get_limbz(a, bit_offset >> 6);
        let v2 = bf_get_bits(&b.limbs, bit_offset + b_offset);
        if v1 != 0 || v2 != u64::MAX {
            n += i64::from(v1.leading_zeros().min((!v2).leading_zeros()));
            break;
        }
        n += 64;
        bit_offset -= 64;
    }
    n
}

fn bf_add_internal(
    a: &BigFloat,
    b: &BigFloat,
    format: BigFormat,
    b_neg: bool,
) -> (BigFloat, Status) {
    let mut result = BigFloat::new();
    let status = bf_add_internal_into(&mut result, a, b, format, b_neg);
    (result, status)
}

fn bf_add_internal_into(
    r: &mut BigFloat,
    a: &BigFloat,
    b: &BigFloat,
    format: BigFormat,
    b_neg: bool,
) -> Status {
    let a_sign = a.sign;
    let b_sign = if b_neg { -b.sign } else { b.sign };
    let is_sub = a_sign != b_sign;

    if a.is_nan() || b.is_nan() {
        *r = BigFloat::nan();
        return Status::empty();
    }
    if a.exp == RAW_EXP_INF || b.exp == RAW_EXP_INF {
        if a.exp == RAW_EXP_INF && b.exp == RAW_EXP_INF {
            if is_sub {
                *r = BigFloat::nan();
                return Status::INVALID_OP;
            }
            *r = BigFloat::infinity(a_sign);
            return Status::empty();
        }
        *r = if a.exp == RAW_EXP_INF {
            BigFloat::infinity(a_sign)
        } else {
            BigFloat::infinity(b_sign)
        };
        return Status::empty();
    }

    let cmp_res = bf_cmpu(a, b);
    let (a, b, r_sign) = match cmp_res {
        Ordering::Less => (b, a, b_sign),
        _ => (a, b, a_sign),
    };

    if cmp_res == Ordering::Equal && is_sub && a.is_finite() {
        let sign = Sign::from_negative(format.rounding == Rounding::TowardNegative);
        *r = BigFloat::zero(sign);
        return Status::empty();
    }

    if a.limbs.is_empty() || b.limbs.is_empty() {
        if a.limbs.is_empty() {
            *r = BigFloat::zero(r_sign);
            return Status::empty();
        }
        r.sign = r_sign;
        r.exp = a.exp;
        r.limbs.clear();
        r.limbs.extend_from_slice(&a.limbs);
        return r.normalize_and_round_mut(format);
    }

    let d = a.exp - b.exp;
    let tot_len = (a.limbs.len() as i64).max((b.limbs.len() as i64) + (d + 63) / 64);

    let r_len = match format.precision {
        Precision::Bits(prec) => {
            let prec_i64 = prec as i64;
            let cancelled_bits = if is_sub {
                if d <= 1 {
                    bf_count_cancelled_bits(a, b)
                } else {
                    1
                }
            } else {
                0
            };
            let precl = (cancelled_bits + prec_i64 + 2 + 63) / 64;
            precl.min(tot_len).max(0)
        }
        _ => tot_len,
    };
    let r_len_usize = r_len as usize;

    let a_offset = (a.limbs.len() as i64) - r_len;
    let b_bit_offset = ((b.limbs.len() as i64) - r_len) * 64 + d;
    let mut carry = u64::from(is_sub);
    let mut z = 0_u64;
    let sub_mask = if is_sub { u64::MAX } else { 0 };

    let mut i = r_len - tot_len;
    while i < 0 {
        let ap = a_offset + i;
        let bp = b_bit_offset + i * 64;
        let mut inflag = false;
        let v1 = if ap >= 0 && (ap as usize) < a.limbs.len() {
            inflag = true;
            a.limbs[ap as usize]
        } else {
            0
        };
        let v2 = if bp + 64 > 0 && bp < (b.limbs.len() as i64) * 64 {
            inflag = true;
            bf_get_bits(&b.limbs, bp)
        } else {
            0
        } ^ sub_mask;

        if !inflag {
            i = 0;
            if ap < 0 {
                i = i.min(-a_offset);
            }
            if bp + 64 <= 0 {
                i = i.min((-b_bit_offset) >> 6);
            }
        } else {
            i += 1;
        }

        let (sum, carry1) = v1.overflowing_add(v2);
        let (sum, carry2) = sum.overflowing_add(carry);
        carry = u64::from(carry1 || carry2);
        z |= sum;
    }

    r.limbs.clear();
    r.limbs.resize(r_len_usize, 0);
    for i in 0..r_len_usize {
        let v1 = bf_get_limbz(a, a_offset + i as i64);
        let v2 = bf_get_bits(&b.limbs, b_bit_offset + (i as i64) * 64) ^ sub_mask;
        let (sum, carry1) = v1.overflowing_add(v2);
        let (sum, carry2) = sum.overflowing_add(carry);
        carry = u64::from(carry1 || carry2);
        r.limbs[i] = sum;
    }

    if r_len_usize > 0 {
        r.limbs[0] |= u64::from(z != 0);
    }

    r.exp = a.exp;
    r.sign = r_sign;
    if !is_sub && carry != 0 {
        r.limbs.push(1);
        r.exp += 64;
    }

    r.normalize_and_round_mut(format)
}

fn bf_mul_direct(a: &BigFloat, b: &BigFloat, sign: Sign, format: BigFormat) -> (BigFloat, Status) {
    let mut result = BigFloat::new();
    let status = bf_mul_direct_into(&mut result, a, b, sign, format);
    (result, status)
}

/// Multiply into `r`, reusing its allocation (mirrors C `bf_mul` writing into
/// `r->tab`). `r` must not alias `a` or `b`.
fn bf_mul_direct_into(
    r: &mut BigFloat,
    a: &BigFloat,
    b: &BigFloat,
    sign: Sign,
    format: BigFormat,
) -> Status {
    let (a, b) = if a.limbs.len() < b.limbs.len() {
        (b, a)
    } else {
        (a, b)
    };

    let mut a_len = a.limbs.len();
    let mut b_len = b.limbs.len();

    if format.rounding == Rounding::Faithful {
        if let Precision::Bits(prec) = format.precision {
            let precl = (prec + 2).div_ceil(64) as usize;
            a_len = a_len.min(precl);
            b_len = b_len.min(precl);
        }
    }

    let a_tab = &a.limbs[a.limbs.len() - a_len..];
    let b_tab = &b.limbs[b.limbs.len() - b_len..];

    ntt::mp_mul_into_vec(&mut r.limbs, a_tab, b_tab);
    r.sign = sign;
    r.exp = a.exp + b.exp;
    r.normalize_and_round_mut(format)
}

/// Direct sqrt on BigFloat, mirroring libbf's `bf_sqrt`.
/// `internal_format` has binary precision for the sqrt computation.
/// `round_format` is what the final result is rounded to.
fn bf_sqrt_direct(
    a: &BigFloat,
    internal_format: BigFormat,
    round_format: BigFormat,
) -> (BigFloat, Status) {
    let mut result = BigFloat::new();
    let status = bf_sqrt_direct_into(&mut result, a, internal_format, round_format);
    (result, status)
}

/// Compute sqrt into `r`, reusing its allocation. `r` must not alias `a`.
fn bf_sqrt_direct_into(
    r: &mut BigFloat,
    a: &BigFloat,
    internal_format: BigFormat,
    round_format: BigFormat,
) -> Status {
    let precision = match internal_format.precision {
        Precision::Bits(p) => p,
        _ => unreachable!(),
    };
    let format = round_format;

    // n = number of limbs for the result
    let n = (2 * (precision + 2)).div_ceil(2 * 64) as usize;
    if n == 0 {
        *r = BigFloat::zero(Sign::Positive);
        return Status::empty();
    }

    // Build a 2n-limb integer from the mantissa (+ 1 extra for remainder high limb).
    // Stack buffer covers up to 256-bit precision (n <= 4, 2n+1 <= 9).
    let mut a1_buf = [0_u64; 9];
    let mut a1_vec;
    let a1: &mut [u64] = if 2 * n < a1_buf.len() {
        &mut a1_buf[..2 * n + 1]
    } else {
        a1_vec = alloc::vec![0_u64; 2 * n + 1];
        &mut a1_vec
    };
    let n1 = (2 * n).min(a.limbs.len());
    a1[2 * n - n1..2 * n].copy_from_slice(&a.limbs[a.limbs.len() - n1..]);

    // If exponent is odd, shift right by 1
    let mut has_remainder = false;
    if a.exp & 1 != 0 {
        let mut carry = 0_u64;
        for i in (0..2 * n).rev() {
            let next = a1[i] << 63;
            a1[i] = (a1[i] >> 1) | carry;
            carry = next;
        }
        has_remainder = carry != 0;
    }

    r.limbs.clear();
    r.limbs.resize(n, 0);
    mp::mp_sqrtrem_full(&mut r.limbs, a1, n);

    if !has_remainder {
        has_remainder = a1[..n + 1].iter().any(|&x| x != 0);
    }
    if !has_remainder && n1 < a.limbs.len() {
        has_remainder = a.limbs[..a.limbs.len() - n1].iter().any(|&x| x != 0);
    }

    if has_remainder {
        r.limbs[0] |= 1;
    }

    r.sign = Sign::Positive;
    r.exp = (a.exp + 1) >> 1;
    r.normalize_and_round_mut(format)
}

fn bf_div_direct(a: &BigFloat, b: &BigFloat, sign: Sign, format: BigFormat) -> (BigFloat, Status) {
    let precision = match format.precision {
        Precision::Bits(bits) => bits,
        Precision::Digits(digits) => decimal_digits_to_binary_precision(digits),
        Precision::Infinite => {
            // Infinite precision non-exact division: fall back to ScaledSignedInt path
            let a_ssi = a.to_scaled_signed_limbs().unwrap();
            let b_ssi = b.to_scaled_signed_limbs().unwrap();
            return div_scaled_signed(&a_ssi, &b_ssi, format)
                .map(|exact| BigFloat::from_scaled_signed_limbs(exact).round_status_owned(format))
                .unwrap_or_else(|| (BigFloat::nan(), Status::INVALID_OP));
        }
    };

    let mut result = BigFloat::new();
    let status = bf_div_finite_into(&mut result, a, b, sign, format, precision);
    (result, status)
}

/// Divide two finite non-zero values into `r`, reusing its allocation
/// (mirrors C `__bf_div` writing the quotient into `r->tab`).
/// `r` must not alias `a` or `b`.
fn bf_div_direct_into(
    r: &mut BigFloat,
    a: &BigFloat,
    b: &BigFloat,
    sign: Sign,
    format: BigFormat,
) -> Status {
    let precision = match format.precision {
        Precision::Bits(bits) => bits,
        Precision::Digits(digits) => decimal_digits_to_binary_precision(digits),
        Precision::Infinite => unreachable!("handled by set_div"),
    };
    bf_div_finite_into(r, a, b, sign, format, precision)
}

fn bf_div_finite_into(
    r: &mut BigFloat,
    a: &BigFloat,
    b: &BigFloat,
    sign: Sign,
    format: BigFormat,
    precision: u64,
) -> Status {
    let precl = (precision + 2).div_ceil(64) as usize;
    let nb = b.limbs.len();
    let n = a.limbs.len().max(precl);

    let na = n + nb;
    let d = na - a.limbs.len();

    let exp = a.exp - b.exp + 64;

    // Stack-allocate the shifted numerator for small divisions
    // (covers BINARY64/BINARY128/256-bit and smaller).
    if na <= 9 {
        let mut taba_buf = [0_u64; 10];
        taba_buf[d..d + a.limbs.len()].copy_from_slice(&a.limbs);
        r.limbs.clear();
        r.limbs.resize(n + 1, 0);
        mp::mp_divnorm(&mut r.limbs, &mut taba_buf, na, &b.limbs, nb);
        if taba_buf[..nb].iter().any(|&x| x != 0) {
            r.limbs[0] |= 1;
        }
        r.sign = sign;
        r.exp = exp;
        return r.normalize_and_round_mut(format);
    }

    // Build the shifted numerator: a.limbs padded with zeros on the low end
    let mut taba = alloc::vec![0_u64; na + 1];
    taba[d..d + a.limbs.len()].copy_from_slice(&a.limbs);

    // BigFloat significands are always normalized (MSB set), so shift == 0.
    // mp_divnorm requires the divisor's MSB to be set, which it always is.
    r.limbs.clear();
    r.limbs.resize(n + 1, 0);
    mp::mp_divnorm(&mut r.limbs, &mut taba, na, &b.limbs, nb);

    // Check if remainder is non-zero (sticky bit)
    if taba[..nb].iter().any(|&x| x != 0) {
        r.limbs[0] |= 1;
    }

    r.sign = sign;
    r.exp = exp;
    r.normalize_and_round_mut(format)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ScaledSignedInt {
    sign: Sign,
    limbs: Vec<u64>,
    low_exp: i64,
}

fn add_signed_ints(a: &SignedInt, b: &SignedInt) -> SignedInt {
    if a.limbs.is_empty() {
        return b.clone();
    }
    if b.limbs.is_empty() {
        return a.clone();
    }
    if a.sign == b.sign {
        SignedInt {
            sign: a.sign,
            limbs: add_abs_limbs(&a.limbs, &b.limbs),
        }
    } else {
        match cmp_abs_limbs(&a.limbs, &b.limbs) {
            Ordering::Greater => SignedInt {
                sign: a.sign,
                limbs: sub_abs_limbs(&a.limbs, &b.limbs),
            },
            Ordering::Less => SignedInt {
                sign: b.sign,
                limbs: sub_abs_limbs(&b.limbs, &a.limbs),
            },
            Ordering::Equal => SignedInt {
                sign: Sign::Positive,
                limbs: Vec::new(),
            },
        }
    }
}

fn add_scaled_signed(a: &ScaledSignedInt, b: &ScaledSignedInt) -> ScaledSignedInt {
    let low_exp = a.low_exp.min(b.low_exp);
    let a_int = SignedInt {
        sign: a.sign,
        limbs: shift_for_low_exp(&a.limbs, a.low_exp, low_exp),
    };
    let b_int = SignedInt {
        sign: b.sign,
        limbs: shift_for_low_exp(&b.limbs, b.low_exp, low_exp),
    };
    let sum = add_signed_ints(&a_int, &b_int);
    ScaledSignedInt {
        sign: sum.sign,
        limbs: sum.limbs,
        low_exp,
    }
}

#[allow(dead_code)]
fn add_scaled_signed_libbf(
    a: &ScaledSignedInt,
    b: &ScaledSignedInt,
    format: BigFormat,
) -> (BigFloat, Status) {
    let Precision::Bits(precision) = format.precision else {
        let exact = add_scaled_signed(a, b);
        return BigFloat::from_scaled_signed_limbs(exact).round_status_owned(format);
    };
    let Some(precision_i64) = i64::try_from(precision).ok() else {
        let exact = add_scaled_signed(a, b);
        return BigFloat::from_scaled_signed_limbs(exact).round_status_owned(format);
    };
    let Some(precl_base) = precision_i64.checked_add(2) else {
        let exact = add_scaled_signed(a, b);
        return BigFloat::from_scaled_signed_limbs(exact).round_status_owned(format);
    };

    if a.limbs.is_empty() || b.limbs.is_empty() {
        let value = if a.limbs.is_empty() { b } else { a };
        return BigFloat::from_scaled_signed_limbs(value.clone()).round_status_owned(format);
    }

    let is_sub = a.sign != b.sign;
    let (a, b, result_sign) = match cmp_scaled_abs(a, b) {
        Ordering::Greater => (a, b, a.sign),
        Ordering::Less => (b, a, b.sign),
        Ordering::Equal if is_sub => {
            let sign = Sign::from_negative(format.rounding == Rounding::TowardNegative);
            return (BigFloat::zero(sign), Status::empty());
        }
        Ordering::Equal => (a, b, a.sign),
    };

    let a_exp = scaled_exp(a);
    let b_exp = scaled_exp(b);
    let d = a_exp - b_exp;
    let cancelled_bits = if is_sub {
        if d <= 1 {
            count_cancelled_bits_scaled(a, b)
        } else {
            1
        }
    } else {
        0
    };

    let Some(precl_bits) = precl_base.checked_add(cancelled_bits) else {
        let exact = add_scaled_signed(a, b);
        return BigFloat::from_scaled_signed_limbs(exact).round_status_owned(format);
    };
    let precl = precl_bits.saturating_add(63) / 64;
    let tot_len = (a.limbs.len() as i64).max((b.limbs.len() as i64) + div_ceil_i64(d, 64));
    let r_len = precl.min(tot_len).max(0);
    let Some(r_len_usize) = usize::try_from(r_len).ok() else {
        let exact = add_scaled_signed(a, b);
        return BigFloat::from_scaled_signed_limbs(exact).round_status_owned(format);
    };

    let a_offset = a.limbs.len() as i64 - r_len;
    let b_bit_offset = (b.limbs.len() as i64 - r_len) * 64 + d;
    let mut carry = u64::from(is_sub);
    let mut z = 0_u64;
    let sub_mask = if is_sub { u64::MAX } else { 0 };

    let mut i = r_len - tot_len;
    while i < 0 {
        let ap = a_offset + i;
        let bp = b_bit_offset + i * 64;
        let mut inflag = false;
        let v1 = if ap >= 0 && (ap as usize) < a.limbs.len() {
            inflag = true;
            a.limbs[ap as usize]
        } else {
            0
        };
        let v2 = if bp + 64 > 0 && bp < (b.limbs.len() as i64) * 64 {
            inflag = true;
            get_bits_abs(&b.limbs, bp)
        } else {
            0
        } ^ sub_mask;

        if !inflag {
            i = 0;
            if ap < 0 {
                i = i.min(-a_offset);
            }
            if bp + 64 <= 0 {
                i = i.min((-b_bit_offset) >> 6);
            }
        } else {
            i += 1;
        }

        let (sum, carry1) = v1.overflowing_add(v2);
        let (sum, carry2) = sum.overflowing_add(carry);
        carry = u64::from(carry1 || carry2);
        z |= sum;
    }

    let mut limbs = Vec::with_capacity(r_len_usize + 1);
    for i in 0..r_len {
        let v1 = get_limbz_abs(&a.limbs, a_offset + i);
        let v2 = get_bits_abs(&b.limbs, b_bit_offset + i * 64) ^ sub_mask;
        let (sum, carry1) = v1.overflowing_add(v2);
        let (sum, carry2) = sum.overflowing_add(carry);
        carry = u64::from(carry1 || carry2);
        limbs.push(sum);
    }

    if let Some(first) = limbs.first_mut() {
        *first |= u64::from(z != 0);
    }
    if !is_sub && carry != 0 {
        limbs.push(1);
    }

    let low_exp = a_exp - r_len * 64;
    BigFloat::from_scaled_signed_limbs(ScaledSignedInt {
        sign: result_sign,
        limbs,
        low_exp,
    })
    .round_status_owned(format)
}

#[allow(dead_code)]
fn mul_scaled_signed(a: &ScaledSignedInt, b: &ScaledSignedInt) -> ScaledSignedInt {
    let sign = a.sign * b.sign;
    let limbs = mul_abs_limbs(&a.limbs, &b.limbs);
    ScaledSignedInt {
        sign,
        limbs,
        low_exp: a.low_exp + b.low_exp,
    }
}

#[allow(dead_code)]
fn faithful_mul_operands(
    mut a: ScaledSignedInt,
    mut b: ScaledSignedInt,
    format: BigFormat,
) -> (ScaledSignedInt, ScaledSignedInt) {
    if format.rounding != Rounding::Faithful {
        return (a, b);
    }
    let Precision::Bits(precision) = format.precision else {
        return (a, b);
    };
    let Some(precl) = precision
        .checked_add(2)
        .and_then(|value| value.checked_add(63))
        .map(|value| value / 64)
        .and_then(|value| usize::try_from(value).ok())
    else {
        return (a, b);
    };
    truncate_scaled_high_limbs(&mut a, precl);
    truncate_scaled_high_limbs(&mut b, precl);
    (a, b)
}

#[allow(dead_code)]
fn truncate_scaled_high_limbs(value: &mut ScaledSignedInt, keep: usize) {
    if keep == 0 || value.limbs.len() <= keep {
        return;
    }
    let drop = value.limbs.len() - keep;
    value.limbs.drain(..drop);
    value.low_exp += (drop as i64) * 64;
}

#[allow(dead_code)]
fn scaled_exp(value: &ScaledSignedInt) -> i64 {
    value.low_exp + (value.limbs.len() as i64) * 64
}

#[allow(dead_code)]
fn cmp_scaled_abs(a: &ScaledSignedInt, b: &ScaledSignedInt) -> Ordering {
    let a_bits = abs_bit_len(&a.limbs);
    let b_bits = abs_bit_len(&b.limbs);
    let a_exp = a.low_exp + a_bits as i64;
    let b_exp = b.low_exp + b_bits as i64;
    match a_exp.cmp(&b_exp) {
        Ordering::Equal => {}
        order => return order,
    }

    let low = a.low_exp.min(b.low_exp);
    for bit_exp in (low..a_exp).rev() {
        match get_scaled_abs_bit(a, bit_exp).cmp(&get_scaled_abs_bit(b, bit_exp)) {
            Ordering::Equal => {}
            order => return order,
        }
    }
    Ordering::Equal
}

#[allow(dead_code)]
fn get_scaled_abs_bit(value: &ScaledSignedInt, bit_exp: i64) -> bool {
    let offset = bit_exp - value.low_exp;
    if offset < 0 {
        return false;
    }
    let offset = offset as usize;
    value
        .limbs
        .get(offset / 64)
        .is_some_and(|limb| ((limb >> (offset % 64)) & 1) != 0)
}

#[allow(dead_code)]
fn get_limbz_abs(limbs: &[u64], idx: i64) -> u64 {
    if idx < 0 {
        0
    } else {
        limbs.get(idx as usize).copied().unwrap_or(0)
    }
}

#[allow(dead_code)]
fn get_bits_abs(limbs: &[u64], pos: i64) -> u64 {
    let i = pos >> 6;
    let p = (pos & 63) as u32;
    let a0 = get_limbz_abs(limbs, i);
    if p == 0 {
        a0
    } else {
        let a1 = get_limbz_abs(limbs, i + 1);
        (a0 >> p) | (a1 << (64 - p))
    }
}

#[inline]
fn limb_low_mask(bits: i32) -> u64 {
    if bits <= 0 {
        0
    } else if bits >= 64 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    }
}

#[allow(dead_code)]
fn count_cancelled_bits_scaled(a: &ScaledSignedInt, b: &ScaledSignedInt) -> i64 {
    debug_assert!(cmp_scaled_abs(a, b) != Ordering::Less);
    let mut bit_offset = (a.limbs.len() as i64) * 64 - 1;
    let b_offset =
        (b.limbs.len() as i64 - a.limbs.len() as i64) * 64 - 63 + scaled_exp(a) - scaled_exp(b);
    let mut n = 0_i64;

    let (v1, v2) = loop {
        let v1 = get_limbz_abs(&a.limbs, bit_offset >> 6);
        let v2 = get_bits_abs(&b.limbs, bit_offset + b_offset);
        if v1 != v2 {
            break (v1, v2);
        }
        n += 64;
        bit_offset -= 64;
    };

    let p = (v1 ^ v2).leading_zeros() as i32 + 1;
    n += i64::from(p);
    let trailing_bits = 64 - p;
    if trailing_bits > 0 {
        let mask = limb_low_mask(trailing_bits);
        let p1 = ((v1 & mask)
            .leading_zeros()
            .min((!v2 & mask).leading_zeros()) as i32)
            - (64 - trailing_bits);
        n += i64::from(p1);
        if p1 != trailing_bits {
            return n;
        }
    }

    bit_offset -= 64;
    loop {
        let v1 = get_limbz_abs(&a.limbs, bit_offset >> 6);
        let v2 = get_bits_abs(&b.limbs, bit_offset + b_offset);
        if v1 != 0 || v2 != u64::MAX {
            n += i64::from(v1.leading_zeros().min((!v2).leading_zeros()));
            break;
        }
        n += 64;
        bit_offset -= 64;
    }
    n
}

#[allow(dead_code)]
fn div_ceil_i64(value: i64, divisor: i64) -> i64 {
    debug_assert!(divisor > 0);
    if value >= 0 {
        (value + divisor - 1) / divisor
    } else {
        value / divisor
    }
}

fn bf_tdivremu(a: &BigFloat, b: &BigFloat, inf: BigFormat) -> (BigFloat, BigFloat) {
    debug_assert!(!a.sign.is_negative());
    debug_assert!(!b.sign.is_negative());
    if a.cmp_abs(b) == Ordering::Less {
        return (BigFloat::from_u64(0), a.clone());
    }
    let prec = (a.exp - b.exp + 1).max(2) as u64;
    let q_fmt = BigFormat {
        precision: Precision::Bits(prec),
        rounding: Rounding::TowardZero,
        ..inf
    };
    let q = a.div(b, q_fmt);
    let q = q.rint(Rounding::TowardZero);
    let r = a.sub(&q.mul(b, inf), inf);
    (q, r)
}

#[inline]
fn exp_range_from_format(format: &BigFormat) -> (i64, i64) {
    use crate::format::ExpBits;
    match format.exp_bits {
        ExpBits::Bits(bits) if bits > 0 && bits < 64 => {
            let e_range = 1_i64 << (bits - 1);
            let e_min = -e_range + 3;
            let e_max = e_range;
            (e_min, e_max)
        }
        ExpBits::Max | ExpBits::Extended | ExpBits::Bits(_) => (i64::MIN / 2, i64::MAX / 2),
    }
}

#[inline]
fn decimal_digits_to_binary_precision(digits: u64) -> u64 {
    digits.saturating_mul(4).saturating_add(8).max(2)
}

const SMALL_POW10_TABLE: [u64; 20] = {
    let mut table = [0_u64; 20];
    table[0] = 1;
    let mut i = 1;
    while i < 20 {
        table[i] = table[i - 1] * 10;
        i += 1;
    }
    table
};

pub(crate) fn pow_integer_status(
    base: &BigFloat,
    exponent: &SignedInt,
    format: BigFormat,
) -> (BigFloat, Status) {
    if !exponent.sign.is_negative() && exponent.limbs.len() <= 1 {
        let exp_val = exponent.limbs.first().copied().unwrap_or(0);
        if let Some(base_val) = base.get_uint64() {
            if base_val == 10 && (exp_val as usize) < SMALL_POW10_TABLE.len() {
                let result = BigFloat::from_u64(SMALL_POW10_TABLE[exp_val as usize]);
                return result.round_status_owned(format);
            }
        }
    }

    let exact = BigFormat {
        precision: Precision::Infinite,
        ..format
    };
    let mut result = BigFloat::from_i64(1);
    let mut factor = base.clone();
    let mut status = Status::empty();
    let exp_bits = abs_bit_len(&exponent.limbs);

    for bit in 0..exp_bits {
        if get_abs_bit(&exponent.limbs, bit) {
            let (next, mul_status) = result.mul_status(&factor, exact);
            result = next;
            status |= mul_status;
        }
        if bit + 1 != exp_bits {
            let (next, mul_status) = factor.mul_status(&factor, exact);
            factor = next;
            status |= mul_status;
        }
    }

    if exponent.sign.is_negative() && !exponent.limbs.is_empty() {
        let (reciprocal, div_status) = BigFloat::from_i64(1).div_status(&result, format);
        status |= div_status;
        (reciprocal, status)
    } else {
        let (rounded, round_status) = result.round_status_owned(format);
        status |= round_status;
        (rounded, status)
    }
}

fn div_scaled_signed(
    a: &ScaledSignedInt,
    b: &ScaledSignedInt,
    format: BigFormat,
) -> Option<ScaledSignedInt> {
    if b.limbs.is_empty() {
        return None;
    }
    let sign = a.sign * b.sign;
    if a.limbs.is_empty() {
        return Some(ScaledSignedInt {
            sign,
            limbs: Vec::new(),
            low_exp: 0,
        });
    }

    let (integer_quotient, integer_remainder) = div_rem_abs_limbs(&a.limbs, &b.limbs);
    if integer_remainder.is_empty() {
        return Some(ScaledSignedInt {
            sign,
            limbs: integer_quotient,
            low_exp: a.low_exp - b.low_exp,
        });
    }

    let precision = match format.precision {
        Precision::Bits(bits) => bits,
        Precision::Digits(digits) => decimal_digits_to_binary_precision(digits),
        Precision::Infinite => return None,
    };
    let precl = precision
        .checked_add(2)?
        .checked_add(63)?
        .checked_div(64)
        .and_then(|value| usize::try_from(value).ok())?;
    let nb = b.limbs.len();
    let n = a.limbs.len().max(precl);
    let shift_limbs = n.checked_add(nb)?.checked_sub(a.limbs.len())?;
    let shift = shift_limbs.checked_mul(64)?;
    let shift = u32::try_from(shift).ok()?;

    let mut numerator = a.limbs.clone();
    if shift != 0 {
        shl_limbs(&mut numerator, shift);
    }
    let (mut quotient, remainder) = div_rem_abs_limbs(&numerator, &b.limbs);
    if !remainder.is_empty() {
        if quotient.is_empty() {
            quotient.push(1);
        } else {
            quotient[0] |= 1;
        }
    }

    Some(ScaledSignedInt {
        sign,
        limbs: quotient,
        low_exp: a.low_exp - b.low_exp - i64::from(shift),
    })
}

fn div_scaled_signed_exact(a: &ScaledSignedInt, b: &ScaledSignedInt) -> Option<ScaledSignedInt> {
    if b.limbs.is_empty() {
        return None;
    }
    let sign = a.sign * b.sign;
    if a.limbs.is_empty() {
        return Some(ScaledSignedInt {
            sign,
            limbs: Vec::new(),
            low_exp: 0,
        });
    }

    let mut divisor = b.limbs.clone();
    let twos = remove_low_zero_bits(&mut divisor);
    if divisor != [1] {
        let (quotient, remainder) = div_rem_abs_limbs(&a.limbs, &b.limbs);
        if remainder.is_empty() {
            return Some(ScaledSignedInt {
                sign,
                limbs: quotient,
                low_exp: a.low_exp - b.low_exp,
            });
        }
        return None;
    }

    Some(ScaledSignedInt {
        sign,
        limbs: a.limbs.clone(),
        low_exp: a.low_exp - b.low_exp - i64::from(twos),
    })
}

fn rem_scaled_toward_zero(a: &ScaledSignedInt, b: &ScaledSignedInt) -> ScaledSignedInt {
    if a.limbs.is_empty() || b.limbs.is_empty() {
        return ScaledSignedInt {
            sign: a.sign,
            limbs: Vec::new(),
            low_exp: 0,
        };
    }
    let low_exp = a.low_exp.min(b.low_exp);
    let a_limbs = shift_for_low_exp(&a.limbs, a.low_exp, low_exp);
    // One of the two shifts is always zero; avoid copying the divisor then.
    let b_storage;
    let b_limbs: &[u64] = if b.low_exp == low_exp {
        &b.limbs
    } else {
        b_storage = shift_for_low_exp(&b.limbs, b.low_exp, low_exp);
        &b_storage
    };
    let (_q, r) = div_rem_abs_limbs_owned(a_limbs, b_limbs);
    ScaledSignedInt {
        sign: a.sign,
        limbs: r,
        low_exp,
    }
}

fn rem_scaled_nearest_even(a: &ScaledSignedInt, b: &ScaledSignedInt) -> ScaledSignedInt {
    if a.limbs.is_empty() || b.limbs.is_empty() {
        return ScaledSignedInt {
            sign: a.sign,
            limbs: Vec::new(),
            low_exp: 0,
        };
    }

    let low_exp = a.low_exp.min(b.low_exp);
    let a_limbs = shift_for_low_exp(&a.limbs, a.low_exp, low_exp);
    let b_storage;
    let b_limbs: &[u64] = if b.low_exp == low_exp {
        &b.limbs
    } else {
        b_storage = shift_for_low_exp(&b.limbs, b.low_exp, low_exp);
        &b_storage
    };
    let (q, r) = div_rem_abs_limbs_owned(a_limbs, b_limbs);
    if r.is_empty() {
        return ScaledSignedInt {
            sign: a.sign,
            limbs: r,
            low_exp,
        };
    }

    let mut twice_r = r.clone();
    shl_limbs(&mut twice_r, 1);
    let q_is_odd = q.first().is_some_and(|limb| (limb & 1) != 0);
    let adjust = match cmp_abs_limbs(&twice_r, b_limbs) {
        Ordering::Greater => true,
        Ordering::Equal => q_is_odd,
        Ordering::Less => false,
    };

    if adjust {
        ScaledSignedInt {
            sign: -a.sign,
            limbs: sub_abs_limbs(b_limbs, &r),
            low_exp,
        }
    } else {
        ScaledSignedInt {
            sign: a.sign,
            limbs: r,
            low_exp,
        }
    }
}

fn rem_scaled_floor(a: &ScaledSignedInt, b: &ScaledSignedInt) -> ScaledSignedInt {
    let remainder = rem_scaled_toward_zero(a, b);
    if remainder.limbs.is_empty() || a.sign == b.sign {
        return remainder;
    }
    add_scaled_signed(&remainder, b)
}

fn rem_scaled_euclidean(a: &ScaledSignedInt, b: &ScaledSignedInt) -> ScaledSignedInt {
    let remainder = rem_scaled_toward_zero(a, b);
    if remainder.limbs.is_empty() || !remainder.sign.is_negative() {
        return remainder;
    }
    let abs_b = ScaledSignedInt {
        sign: Sign::Positive,
        limbs: b.limbs.clone(),
        low_exp: b.low_exp,
    };
    add_scaled_signed(&remainder, &abs_b)
}

#[allow(dead_code)]
fn rint_scaled_to_integer(value: &ScaledSignedInt, rounding: Rounding) -> (ScaledSignedInt, bool) {
    if value.limbs.is_empty() || value.low_exp >= 0 {
        return (
            ScaledSignedInt {
                sign: value.sign,
                limbs: value.limbs.clone(),
                low_exp: value.low_exp,
            },
            false,
        );
    }

    let shift = value.low_exp.unsigned_abs();
    let mut quotient = value.limbs.clone();
    if shift > u64::from(u32::MAX) {
        quotient.clear();
    } else {
        shr_limbs_trunc(&mut quotient, shift as u32);
    }

    let inexact = any_abs_bits_below(&value.limbs, shift);
    if !inexact {
        return (
            ScaledSignedInt {
                sign: value.sign,
                limbs: quotient,
                low_exp: 0,
            },
            false,
        );
    }

    let guard = shift > 0
        && usize::try_from(shift - 1)
            .ok()
            .is_some_and(|bit| get_abs_bit(&value.limbs, bit));
    let sticky = shift > 1 && any_abs_bits_below(&value.limbs, shift - 1);
    let q_is_odd = quotient.first().is_some_and(|limb| (limb & 1) != 0);
    let add_one = match rounding {
        Rounding::NearestEven => guard && (sticky || q_is_odd),
        Rounding::NearestAway => guard,
        Rounding::TowardZero => false,
        Rounding::TowardPositive => !value.sign.is_negative(),
        Rounding::TowardNegative => value.sign.is_negative(),
        Rounding::AwayFromZero => true,
        Rounding::Faithful => false,
    };
    if add_one {
        add_small_abs(&mut quotient, 1);
    }

    (
        ScaledSignedInt {
            sign: value.sign,
            limbs: quotient,
            low_exp: 0,
        },
        true,
    )
}

#[allow(dead_code)]
fn sqrt_scaled_binary(value: &ScaledSignedInt, precision: u64) -> (ScaledSignedInt, bool) {
    if value.limbs.is_empty() {
        return (
            ScaledSignedInt {
                sign: Sign::Positive,
                limbs: Vec::new(),
                low_exp: 0,
            },
            false,
        );
    }

    let a_limbs = &value.limbs;
    let a_len = a_limbs.len();
    let a_expn = value.low_exp + (a_len as i64) * 64;

    let n = (2 * (precision + 2)).div_ceil(2 * 64) as usize;
    if n == 0 {
        return (
            ScaledSignedInt {
                sign: Sign::Positive,
                limbs: Vec::new(),
                low_exp: 0,
            },
            false,
        );
    }

    let mut a1 = alloc::vec![0_u64; 2 * n + 1];
    let n1 = (2 * n).min(a_len);
    a1[2 * n - n1..2 * n].copy_from_slice(&a_limbs[a_len - n1..]);

    let mut res = 0_u64;
    if a_expn & 1 != 0 {
        let mut carry = 0_u64;
        for limb in a1[..2 * n].iter_mut().rev() {
            let new_carry = *limb & 1;
            *limb = (*limb >> 1) | (carry << 63);
            carry = new_carry;
        }
        res = carry;
    }

    let mut tabs = alloc::vec![0_u64; n];
    mp::mp_sqrtrem_full(&mut tabs, &mut a1, n);

    if res == 0 {
        res = if a1[..n + 1].iter().any(|&x| x != 0) {
            1
        } else {
            0
        };
    }
    if res == 0 && n1 < a_len {
        res = if a_limbs[..a_len - n1].iter().any(|&x| x != 0) {
            1
        } else {
            0
        };
    }
    let inexact = res != 0;
    if inexact {
        tabs[0] |= 1;
    }

    trim_limbs(&mut tabs);

    let r_expn = (a_expn + 1) >> 1;
    let r_low_exp = r_expn - (n as i64) * 64;

    (
        ScaledSignedInt {
            sign: Sign::Positive,
            limbs: tabs,
            low_exp: r_low_exp,
        },
        inexact,
    )
}

fn shift_for_low_exp(limbs: &[u64], current_low_exp: i64, target_low_exp: i64) -> Vec<u64> {
    debug_assert!(current_low_exp >= target_low_exp);
    let mut out = limbs.to_vec();
    let shift = current_low_exp - target_low_exp;
    if shift != 0 {
        shl_limbs(&mut out, shift as u32);
    }
    out
}

fn div_rem_signed_ints(a: &SignedInt, b: &SignedInt) -> (SignedInt, SignedInt) {
    assert!(!b.limbs.is_empty(), "division by zero");
    let (quotient, remainder) = div_rem_abs_limbs(&a.limbs, &b.limbs);
    (
        SignedInt {
            sign: a.sign * b.sign,
            limbs: quotient,
        },
        SignedInt {
            sign: a.sign,
            limbs: remainder,
        },
    )
}

/// Direct logic op on BigFloat values, mirroring libbf's `bf_logic_op`.
/// Reads the truncated-integer limbs of both operands straight out of their
/// mantissas (via `bf_get_bits`) and streams the two's complement conversion
/// of negative operands with a carry, so the only allocation is the result
/// buffer — matching C's zero-copy handling of positive operands.
fn bf_logic_op_direct<F: Fn(u64, u64) -> u64>(a: &BigFloat, b: &BigFloat, op: F) -> BigFloat {
    // Number of integer bits of |trunc(x)| (0 for zero, non-finite, |x| < 1)
    let a_bits = if a.is_finite() && a.exp > 0 { a.exp } else { 0 };
    let b_bits = if b.is_finite() && b.exp > 0 { b.exp } else { 0 };
    let a_neg = a_bits > 0 && a.sign.is_negative();
    let b_neg = b_bits > 0 && b.sign.is_negative();
    let r_neg = op(u64::from(a_neg), u64::from(b_neg)) & 1 != 0;

    // Result length in bits, following C bf_logic_op: AND with a non-negative
    // result cannot be longer than its non-negative operand(s); everything
    // else needs max(len). (Extra high limbs come out as the sign fill and
    // are trimmed by normalization.)
    let and_like = op(u64::MAX, 0) == 0 && op(0, u64::MAX) == 0 && op(0, 0) == 0;
    let l_bits = if and_like && !r_neg {
        if !a_neg && !b_neg {
            a_bits.min(b_bits)
        } else if !a_neg {
            a_bits
        } else {
            b_bits
        }
    } else {
        a_bits.max(b_bits)
    };
    let len = (l_bits.max(1) as u64).div_ceil(64) as usize;

    // Bit offset of the integer LSB inside the mantissa (like C's
    // a->len * LIMB_BITS - a->expn); reads outside the mantissa yield 0.
    let a_off = (a.limbs.len() as i64) * 64 - a_bits;
    let b_off = (b.limbs.len() as i64) * 64 - b_bits;

    let mut result = alloc::vec![0_u64; len];
    // Streaming two's complement: -x == !x + 1, with the +1 carried up
    // through the low zero limbs of the magnitude.
    let mut a_carry = a_neg;
    let mut b_carry = b_neg;
    for (i, out) in result.iter_mut().enumerate() {
        let ma = bf_get_bits(&a.limbs, a_off + (i as i64) * 64);
        let mb = bf_get_bits(&b.limbs, b_off + (i as i64) * 64);
        let va = if a_neg {
            let v = (!ma).wrapping_add(u64::from(a_carry));
            a_carry = a_carry && ma == 0;
            v
        } else {
            ma
        };
        let vb = if b_neg {
            let v = (!mb).wrapping_add(u64::from(b_carry));
            b_carry = b_carry && mb == 0;
            v
        } else {
            mb
        };
        *out = op(va, vb);
    }

    // Convert back from two's complement
    let sign = Sign::from_negative(r_neg);
    if r_neg {
        // Negate: invert + add 1
        for limb in result.iter_mut() {
            *limb = !*limb;
        }
        add_small_abs(&mut result, 1);
    }
    trim_limbs(&mut result);

    if result.is_empty() {
        return BigFloat::zero(Sign::Positive);
    }

    BigFloat::from_abs_int_limbs(result, sign)
}

#[allow(dead_code)]
fn logic_signed_ints(
    a: &SignedInt,
    b: &SignedInt,
    op: fn(&[u64], &[u64]) -> Vec<u64>,
) -> SignedInt {
    let sign = Sign::from_negative(op_sign(a.sign, b.sign, op));
    let bits = abs_bit_len(&a.limbs).max(abs_bit_len(&b.limbs)) + 2;
    let len = bits.div_ceil(64).max(1);
    let a_tc = to_twos_complement_limbs(a, len);
    let b_tc = to_twos_complement_limbs(b, len);
    let mut result = op(&a_tc, &b_tc);
    result.resize(len, if sign.is_negative() { u64::MAX } else { 0 });
    from_twos_complement_limbs(result, sign)
}

#[allow(dead_code)]
fn op_sign(a: Sign, b: Sign, op: fn(&[u64], &[u64]) -> Vec<u64>) -> bool {
    let av = if a.is_negative() { [u64::MAX] } else { [0] };
    let bv = if b.is_negative() { [u64::MAX] } else { [0] };
    op(&av, &bv).first().copied().unwrap_or(0) == u64::MAX
}

#[allow(dead_code)]
fn to_twos_complement_limbs(value: &SignedInt, len: usize) -> Vec<u64> {
    let mut out = value.limbs.clone();
    out.resize(len, 0);
    if value.sign.is_negative() && !value.limbs.is_empty() {
        for limb in &mut out {
            *limb = !*limb;
        }
        add_small_abs(&mut out, 1);
        out.truncate(len);
    }
    out
}

#[allow(dead_code)]
fn from_twos_complement_limbs(mut limbs: Vec<u64>, sign: Sign) -> SignedInt {
    if sign.is_negative() {
        for limb in &mut limbs {
            *limb = !*limb;
        }
        add_small_abs(&mut limbs, 1);
    }
    trim_limbs(&mut limbs);
    SignedInt {
        sign: if limbs.is_empty() {
            Sign::Positive
        } else {
            sign
        },
        limbs,
    }
}

fn cmp_abs_limbs(a: &[u64], b: &[u64]) -> Ordering {
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
                Ordering::Equal => {}
                order => return order,
            }
        }
        Ordering::Equal
    })
}

fn add_abs_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = long.to_vec();
    let mut carry = if short.is_empty() {
        0
    } else {
        mp_add_abs(&mut out[..short.len()], long, short, 0)
    };
    let mut index = short.len();
    while carry != 0 && index < out.len() {
        let value = out[index].wrapping_add(carry);
        carry = u64::from(value < carry);
        out[index] = value;
        index += 1;
    }
    if carry != 0 {
        out.push(carry);
    }
    trim_limbs(&mut out);
    out
}

fn sub_abs_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    debug_assert!(cmp_abs_limbs(a, b) != Ordering::Less);
    let mut out = a.to_vec();
    let mut borrow = if b.is_empty() {
        0
    } else {
        mp_sub_abs(&mut out[..b.len()], a, b, 0)
    };
    let mut index = b.len();
    while borrow != 0 && index < out.len() {
        let value = out[index].wrapping_sub(borrow);
        borrow = u64::from(value > out[index]);
        out[index] = value;
        index += 1;
    }
    debug_assert_eq!(borrow, 0);
    trim_limbs(&mut out);
    out
}

#[inline]
fn mp_add_abs(out: &mut [u64], op1: &[u64], op2: &[u64], carry: u64) -> u64 {
    debug_assert!(op1.len() >= out.len());
    debug_assert!(op2.len() >= out.len());
    let mut carry = carry;
    for i in 0..out.len() {
        let value = op1[i];
        let sum = value.wrapping_add(op2[i]);
        let carry1 = u64::from(sum < value);
        let sum_with_carry = sum.wrapping_add(carry);
        carry = u64::from(sum_with_carry < carry) | carry1;
        out[i] = sum_with_carry;
    }
    carry
}

#[inline]
fn mp_sub_abs(out: &mut [u64], op1: &[u64], op2: &[u64], borrow: u64) -> u64 {
    debug_assert!(op1.len() >= out.len());
    debug_assert!(op2.len() >= out.len());
    let mut borrow = borrow;
    for i in 0..out.len() {
        let value = op1[i];
        let difference = value.wrapping_sub(op2[i]);
        let borrow1 = u64::from(difference > value);
        let difference_with_borrow = difference.wrapping_sub(borrow);
        borrow = u64::from(difference_with_borrow > difference) | borrow1;
        out[i] = difference_with_borrow;
    }
    borrow
}

#[allow(dead_code)]
fn mul_abs_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    ntt::mp_mul(a, a.len(), b, b.len())
}

#[inline]
fn mp_mul1_abs(out: &mut [u64], input: &[u64], multiplier: u64, carry: u64) -> u64 {
    debug_assert!(out.len() >= input.len());
    let mut carry = u128::from(carry);
    for (dst, &value) in out.iter_mut().zip(input) {
        let product = u128::from(value) * u128::from(multiplier) + carry;
        *dst = product as u64;
        carry = product >> 64;
    }
    carry as u64
}

#[inline]
fn mp_add_mul1_abs(out: &mut [u64], input: &[u64], multiplier: u64) -> u64 {
    debug_assert!(out.len() >= input.len());
    let mut carry = 0_u128;
    for (dst, &value) in out.iter_mut().zip(input) {
        let product = u128::from(value) * u128::from(multiplier) + u128::from(*dst) + carry;
        *dst = product as u64;
        carry = product >> 64;
    }
    carry as u64
}

fn mp_mul_basecase_abs(out: &mut [u64], op1: &[u64], op2: &[u64]) {
    debug_assert_eq!(out.len(), op1.len() + op2.len());
    debug_assert!(!op1.is_empty());
    debug_assert!(!op2.is_empty());

    out[op1.len()] = mp_mul1_abs(out, op1, op2[0], 0);
    for i in 1..op2.len() {
        let carry = mp_add_mul1_abs(&mut out[i..], op1, op2[i]);
        out[i + op1.len()] = carry;
    }
}

#[allow(dead_code)]
fn logic_or_abs_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(a.get(i).copied().unwrap_or(0) | b.get(i).copied().unwrap_or(0));
    }
    trim_limbs(&mut out);
    out
}

#[allow(dead_code)]
fn logic_xor_abs_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0));
    }
    trim_limbs(&mut out);
    out
}

#[allow(dead_code)]
fn logic_and_abs_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    let n = a.len().min(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(a[i] & b[i]);
    }
    trim_limbs(&mut out);
    out
}

pub(crate) fn div_rem_abs_limbs(a: &[u64], b: &[u64]) -> (Vec<u64>, Vec<u64>) {
    div_rem_abs_limbs_owned(a.to_vec(), b)
}

/// Like [`div_rem_abs_limbs`] but consumes the numerator, reusing its buffer
/// for the shifted numerator and the remainder (no defensive copies).
pub(crate) fn div_rem_abs_limbs_owned(mut a: Vec<u64>, b: &[u64]) -> (Vec<u64>, Vec<u64>) {
    assert!(!b.is_empty(), "division by zero");
    trim_limbs(&mut a);
    let nb = b.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1);
    let b_trimmed = &b[..nb];
    if nb == 0 {
        panic!("division by zero");
    }
    if a.is_empty() || cmp_abs_limbs(&a, b_trimmed) == Ordering::Less {
        return (Vec::new(), a);
    }

    let na = a.len();

    if nb == 1 {
        let rem = div_rem_small_abs(&mut a, b_trimmed[0]);
        trim_limbs(&mut a);
        return (
            a,
            if rem == 0 {
                Vec::new()
            } else {
                alloc::vec![rem]
            },
        );
    }

    let b_top = b_trimmed[nb - 1];
    let shift = b_top.leading_zeros();
    // Shift the numerator left in place (one extra high limb for the carry)
    a.push(0);
    let taba = &mut a;
    let mut b_shifted_storage;
    let tabb: &[u64];

    if shift != 0 {
        let mut carry = 0_u64;
        for item in taba[..na].iter_mut() {
            let v = *item;
            *item = (v << shift) | carry;
            carry = v >> (64 - shift);
        }
        taba[na] = carry;

        b_shifted_storage = b_trimmed.to_vec();
        carry = 0;
        for item in b_shifted_storage.iter_mut() {
            let v = *item;
            *item = (v << shift) | carry;
            carry = v >> (64 - shift);
        }
        tabb = &b_shifted_storage;
    } else {
        tabb = b_trimmed;
    }

    let actual_na = if taba[na] != 0 { na + 1 } else { na };
    let nq = actual_na - nb;
    let mut tabq = alloc::vec![0_u64; nq + 1];

    mp::mp_divnorm(&mut tabq, taba, actual_na, tabb, nb);

    if shift != 0 {
        let mut carry = 0_u64;
        for i in (0..nb).rev() {
            let v = taba[i];
            taba[i] = (v >> shift) | carry;
            carry = v << (64 - shift);
        }
    }

    // Reuse the numerator buffer as the remainder
    a.truncate(nb);
    trim_limbs(&mut tabq);
    trim_limbs(&mut a);
    (tabq, a)
}

fn mp_sqrtrem_karatsuba(input: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let mut value: Vec<u64> = input.to_vec();
    trim_limbs(&mut value);
    if value.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let total = value.len();
    if total <= 2 {
        return sqrt_rem_abs_limbs(&value);
    }

    let n = total.div_ceil(2);
    let mut a = alloc::vec![0_u64; 2 * n + 1];
    let n1 = (2 * n).min(total);
    a[2 * n - n1..2 * n].copy_from_slice(&value[total - n1..]);

    if a[2 * n - 1] < (1_u64 << 62) {
        return sqrt_rem_abs_limbs(input);
    }

    let mut tabs = alloc::vec![0_u64; n];
    mp::mp_sqrtrem_full(&mut tabs, &mut a, n);
    trim_limbs(&mut tabs);
    let rh = a[n];
    let mut rem = a[..n].to_vec();
    if rh != 0 {
        rem.push(rh);
    }
    trim_limbs(&mut rem);
    (tabs, rem)
}

pub(crate) fn sqrt_rem_abs_limbs(n: &[u64]) -> (Vec<u64>, Vec<u64>) {
    if n.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let bit_len = abs_bit_len(n);
    let pair_count = bit_len.div_ceil(2);
    let mut root = Vec::new();
    let mut rem = Vec::new();

    for pair in (0..pair_count).rev() {
        shl_limbs(&mut rem, 2);
        let high_bit = pair * 2 + 1;
        let low_bit = pair * 2;
        let pair_value =
            (u64::from(get_abs_bit(n, high_bit)) << 1) | u64::from(get_abs_bit(n, low_bit));
        add_small_abs(&mut rem, pair_value);

        let mut candidate = root.clone();
        shl_limbs(&mut candidate, 2);
        add_small_abs(&mut candidate, 1);

        if cmp_abs_limbs(&rem, &candidate) != Ordering::Less {
            rem = sub_abs_limbs(&rem, &candidate);
            shl_limbs(&mut root, 1);
            add_small_abs(&mut root, 1);
        } else {
            shl_limbs(&mut root, 1);
        }
    }

    trim_limbs(&mut root);
    trim_limbs(&mut rem);
    (root, rem)
}

#[inline]
fn abs_bit_len(limbs: &[u64]) -> usize {
    let mut len = limbs.len();
    while len > 0 && limbs[len - 1] == 0 {
        len -= 1;
    }
    if len == 0 {
        0
    } else {
        (len - 1) * 64 + (64 - limbs[len - 1].leading_zeros() as usize)
    }
}

#[inline]
fn get_abs_bit(limbs: &[u64], bit: usize) -> bool {
    limbs
        .get(bit / 64)
        .is_some_and(|limb| ((limb >> (bit % 64)) & 1) != 0)
}

#[inline]
fn get_abs_bit_signed(limbs: &[u64], bit: i128) -> bool {
    if bit < 0 {
        return false;
    }
    usize::try_from(bit)
        .ok()
        .is_some_and(|bit| get_abs_bit(limbs, bit))
}

#[allow(dead_code)]
fn any_abs_bits_below(limbs: &[u64], bit: u64) -> bool {
    if bit == 0 {
        return false;
    }
    let full_words = (bit / 64).min(limbs.len() as u64) as usize;
    if limbs[..full_words].iter().any(|&limb| limb != 0) {
        return true;
    }
    let rem_bits = (bit % 64) as u32;
    rem_bits != 0
        && limbs
            .get(full_words)
            .is_some_and(|limb| (limb & ((1_u64 << rem_bits) - 1)) != 0)
}

#[inline]
fn add_small_abs(limbs: &mut Vec<u64>, value: u64) {
    if value == 0 {
        return;
    }
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

#[inline]
fn mul_small_abs(limbs: &mut Vec<u64>, value: u64) {
    if limbs.is_empty() || value == 1 {
        return;
    }
    if value == 0 {
        limbs.clear();
        return;
    }
    let mut carry = 0_u128;
    for limb in limbs.iter_mut() {
        let product = u128::from(*limb) * u128::from(value) + carry;
        *limb = product as u64;
        carry = product >> 64;
    }
    if carry != 0 {
        limbs.push(carry as u64);
    }
}

fn pow10_abs_limbs(exp: i32) -> Vec<u64> {
    debug_assert!(exp >= 0);
    let mut limbs = alloc::vec![1];
    mul_pow10_abs(&mut limbs, exp);
    limbs
}

fn mul_pow10_abs(limbs: &mut Vec<u64>, exp: i32) {
    debug_assert!(exp >= 0);
    if exp == 0 {
        return;
    }
    // For small exponents, single-limb multiply is faster than building a power
    const SMALL_POW10: [u64; 20] = {
        let mut table = [0_u64; 20];
        table[0] = 1;
        let mut i = 1;
        while i < 20 {
            table[i] = table[i - 1] * 10;
            i += 1;
        }
        table
    };

    let mut remaining = exp;
    while remaining > 0 {
        let chunk = remaining.min(19) as usize;
        mul_small_abs(limbs, SMALL_POW10[chunk]);
        remaining -= chunk as i32;
    }
}

fn div_rem_small_abs(limbs: &mut Vec<u64>, divisor: u64) -> u64 {
    debug_assert!(divisor != 0);
    let mut rem = 0_u128;
    for limb in limbs.iter_mut().rev() {
        let value = (rem << 64) | u128::from(*limb);
        *limb = (value / u128::from(divisor)) as u64;
        rem = value % u128::from(divisor);
    }
    trim_limbs(limbs);
    rem as u64
}

#[allow(dead_code)]
fn parse_exponent(bytes: &[u8], mut pos: usize) -> Result<(i64, usize), ParseFloatError> {
    let mut neg = false;
    if pos < bytes.len() {
        if bytes[pos] == b'+' {
            pos += 1;
        } else if bytes[pos] == b'-' {
            neg = true;
            pos += 1;
        }
    }
    if pos >= bytes.len() || !bytes[pos].is_ascii_digit() {
        return Err(ParseFloatError::new());
    }
    let mut val = 0_i64;
    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        val = val
            .saturating_mul(10)
            .saturating_add(i64::from(bytes[pos] - b'0'));
        pos += 1;
    }
    if neg {
        val = -val;
    }
    Ok((val, pos))
}

fn digit_value(ch: u8) -> Option<u8> {
    match ch {
        b'0'..=b'9' => Some(ch - b'0'),
        b'a'..=b'z' => Some(ch - b'a' + 10),
        b'A'..=b'Z' => Some(ch - b'A' + 10),
        _ => None,
    }
}

fn fmt_decimal_abs_limbs_to_write(
    mut limbs: Vec<u64>,
    sign: Sign,
    scale: usize,
    w: &mut impl fmt::Write,
) -> fmt::Result {
    trim_limbs(&mut limbs);
    let mut digits = Vec::new();
    while !limbs.is_empty() {
        let rem = div_rem_small_abs(&mut limbs, 10);
        digits.push(digit_char(rem as u8));
    }
    if digits.is_empty() {
        digits.push('0');
    }
    digits.reverse();

    if sign.is_negative() {
        w.write_char('-')?;
    }
    if scale != 0 {
        if digits.len() <= scale {
            w.write_str("0.")?;
            for _ in 0..(scale - digits.len()) {
                w.write_char('0')?;
            }
            for &d in &digits {
                w.write_char(d)?;
            }
        } else {
            let int_len = digits.len() - scale;
            for &d in &digits[..int_len] {
                w.write_char(d)?;
            }
            w.write_char('.')?;
            for &d in &digits[int_len..] {
                w.write_char(d)?;
            }
        }
    } else {
        for &d in &digits {
            w.write_char(d)?;
        }
    }
    Ok(())
}

fn fmt_decimal_abs_limbs(limbs: Vec<u64>, sign: Sign, scale: usize) -> String {
    let mut out = String::new();
    fmt_decimal_abs_limbs_to_write(limbs, sign, scale, &mut out).unwrap();
    out
}

fn should_increment_decimal_round(
    kept: &str,
    discarded: &str,
    is_negative: bool,
    rounding: Rounding,
) -> bool {
    match rounding {
        Rounding::TowardZero | Rounding::Faithful => false,
        Rounding::TowardPositive => !is_negative,
        Rounding::TowardNegative => is_negative,
        Rounding::AwayFromZero => true,
        Rounding::NearestAway | Rounding::NearestEven => {
            let bytes = discarded.as_bytes();
            let first = bytes[0];
            if first > b'5' {
                return true;
            }
            if first < b'5' {
                return false;
            }
            let above_half = bytes[1..].iter().any(|&digit| digit != b'0');
            if above_half || matches!(rounding, Rounding::NearestAway) {
                return true;
            }
            kept.as_bytes()
                .last()
                .is_some_and(|digit| !(digit - b'0').is_multiple_of(2))
        }
    }
}

fn increment_decimal_digits(digits: &str) -> String {
    let mut bytes = digits.as_bytes().to_vec();
    for idx in (0..bytes.len()).rev() {
        if bytes[idx] != b'9' {
            bytes[idx] += 1;
            for digit in &mut bytes[idx + 1..] {
                *digit = b'0';
            }
            return String::from_utf8(bytes).expect("decimal digits are utf8");
        }
    }

    let mut out = String::with_capacity(digits.len() + 1);
    out.push('1');
    for _ in 0..digits.len() {
        out.push('0');
    }
    out
}

fn digit_char(value: u8) -> char {
    debug_assert!(value < 36);
    if value < 10 {
        char::from(b'0' + value)
    } else {
        char::from(b'a' + value - 10)
    }
}

fn digit_char_case(value: u8, uppercase: bool) -> char {
    debug_assert!(value < 36);
    if value < 10 {
        char::from(b'0' + value)
    } else if uppercase {
        char::from(b'A' + value - 10)
    } else {
        char::from(b'a' + value - 10)
    }
}

struct FmtRadixBuf {
    digits: Vec<u8>,
}

impl FmtRadixBuf {
    fn from_abs_limbs(mut limbs: Vec<u64>, radix: u8) -> Self {
        trim_limbs(&mut limbs);
        let mut digits = Vec::new();
        if limbs.is_empty() {
            digits.push(0);
        } else {
            while !limbs.is_empty() {
                let rem = div_rem_small_abs(&mut limbs, u64::from(radix));
                digits.push(rem as u8);
            }
            digits.reverse();
        }
        Self { digits }
    }

    fn write_to(&self, uppercase: bool, w: &mut impl fmt::Write) -> fmt::Result {
        for &d in &self.digits {
            w.write_char(digit_char_case(d, uppercase))?;
        }
        Ok(())
    }

    fn write_to_str(&self, uppercase: bool) -> String {
        let mut s = String::with_capacity(self.digits.len());
        self.write_to(uppercase, &mut s).unwrap();
        s
    }
}

fn fmt_integer_radix(
    f: &mut fmt::Formatter<'_>,
    abs_limbs: Vec<u64>,
    is_negative: bool,
    radix: u8,
    uppercase: bool,
    prefix: &str,
) -> fmt::Result {
    let buf = FmtRadixBuf::from_abs_limbs(abs_limbs, radix);
    let formatted = buf.write_to_str(uppercase);
    f.pad_integral(!is_negative, prefix, &formatted)
}

fn fmt_exact_decimal(bf: &BigFloat, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match bf.classify() {
        FpCategory::Nan => return f.pad("NaN"),
        FpCategory::Infinite => {
            if bf.sign.is_negative() {
                return f.pad("-inf");
            } else {
                return f.pad("inf");
            }
        }
        FpCategory::Zero => {
            if bf.sign.is_negative() {
                return f.pad("-0");
            } else {
                return f.pad("0");
            }
        }
        FpCategory::Normal => {}
    }

    if let Some(value) = bf.to_scaled_signed_limbs() {
        if value.low_exp >= 0 {
            let mut limbs = value.limbs;
            if value.low_exp > 0 {
                if let Ok(shift) = u32::try_from(value.low_exp) {
                    shl_limbs(&mut limbs, shift);
                } else {
                    return fmt_integer_radix(f, limbs, value.sign.is_negative(), 10, false, "");
                }
            }
            return fmt_decimal_abs_limbs_to_fmt(limbs, value.sign, 0, f);
        }

        if let Ok(scale) = usize::try_from(-value.low_exp) {
            let mut coefficient = value.limbs;
            for _ in 0..scale {
                mul_small_abs(&mut coefficient, 5);
            }

            let mut scale = scale;
            while scale > 0 {
                let mut quotient = coefficient.clone();
                if div_rem_small_abs(&mut quotient, 10) != 0 {
                    break;
                }
                coefficient = quotient;
                scale -= 1;
            }

            return fmt_decimal_abs_limbs_to_fmt(coefficient, value.sign, scale, f);
        }
    }

    if let Some(limbs) = bf.to_abs_int_limbs() {
        return fmt_integer_radix(f, limbs, bf.sign.is_negative(), 10, false, "");
    }

    f.pad("NaN")
}

fn fmt_decimal_abs_limbs_to_fmt(
    mut limbs: Vec<u64>,
    sign: Sign,
    scale: usize,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    trim_limbs(&mut limbs);
    let mut digits = Vec::new();
    while !limbs.is_empty() {
        let rem = div_rem_small_abs(&mut limbs, 10);
        digits.push(digit_char(rem as u8));
    }
    if digits.is_empty() {
        digits.push('0');
    }
    digits.reverse();

    let mut buf = String::with_capacity(digits.len() + scale + 3);
    if sign.is_negative() {
        buf.push('-');
    }
    if scale != 0 {
        if digits.len() <= scale {
            buf.push('0');
            buf.push('.');
            for _ in 0..(scale - digits.len()) {
                buf.push('0');
            }
            for &d in &digits {
                buf.push(d);
            }
        } else {
            let int_len = digits.len() - scale;
            for &d in &digits[..int_len] {
                buf.push(d);
            }
            buf.push('.');
            for &d in &digits[int_len..] {
                buf.push(d);
            }
        }
    } else {
        for &d in &digits {
            buf.push(d);
        }
    }

    f.pad(&buf)
}

/// Left-shift in place for normalization: shift < 64, no word shift, no
/// trailing zero removal needed (caller handles it). This avoids the
/// overhead of the general `shl_limbs` for the normalize-and-round path.
#[allow(dead_code)]
fn normalize_shl_in_place(limbs: &mut Vec<u64>, shift: u32) {
    debug_assert!(shift > 0 && shift < 64);
    let mut carry = 0_u64;
    for limb in limbs.iter_mut() {
        let next = *limb >> (64 - shift);
        *limb = (*limb << shift) | carry;
        carry = next;
    }
    // For normalization, carry should be zero since we computed the shift
    // from the leading zeros of the top limb. But if not, push it.
    if carry != 0 {
        limbs.push(carry);
    }
}

fn shl_limbs(limbs: &mut Vec<u64>, shift: u32) {
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

fn shr_limbs_exact(limbs: &mut Vec<u64>, shift: u32) -> bool {
    if limbs.is_empty() || shift == 0 {
        return true;
    }
    let word_shift = (shift / 64) as usize;
    let bit_shift = shift % 64;
    if word_shift >= limbs.len() {
        let exact = limbs.iter().all(|&limb| limb == 0);
        limbs.clear();
        return exact;
    }
    if limbs[..word_shift].iter().any(|&limb| limb != 0) {
        return false;
    }
    limbs.drain(..word_shift);
    if bit_shift != 0 {
        let mask = (1_u64 << bit_shift) - 1;
        if limbs.first().is_some_and(|limb| limb & mask != 0) {
            return false;
        }
        let mut carry = 0_u64;
        for limb in limbs.iter_mut().rev() {
            let next = *limb << (64 - bit_shift);
            *limb = (*limb >> bit_shift) | carry;
            carry = next;
        }
    }
    trim_limbs(limbs);
    true
}

fn shr_limbs_trunc(limbs: &mut Vec<u64>, shift: u32) {
    if limbs.is_empty() || shift == 0 {
        return;
    }
    let word_shift = (shift / 64) as usize;
    let bit_shift = shift % 64;
    if word_shift >= limbs.len() {
        limbs.clear();
        return;
    }
    limbs.drain(..word_shift);
    if bit_shift != 0 {
        let mut carry = 0_u64;
        for limb in limbs.iter_mut().rev() {
            let next = *limb << (64 - bit_shift);
            *limb = (*limb >> bit_shift) | carry;
            carry = next;
        }
    }
    trim_limbs(limbs);
}

#[inline]
fn trim_limbs(limbs: &mut Vec<u64>) {
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
}

#[inline]
fn remove_low_zero_limbs(limbs: &mut Vec<u64>) {
    let first_non_zero = limbs
        .iter()
        .position(|&limb| limb != 0)
        .unwrap_or(limbs.len());
    if first_non_zero > 0 {
        limbs.drain(..first_non_zero);
    }
}

fn remove_low_zero_bits(limbs: &mut Vec<u64>) -> u32 {
    trim_limbs(limbs);
    if limbs.is_empty() {
        return 0;
    }
    let first_non_zero = limbs
        .iter()
        .position(|&limb| limb != 0)
        .unwrap_or(limbs.len());
    if first_non_zero > 0 {
        limbs.drain(..first_non_zero);
    }
    let count = (first_non_zero as u32) * 64;
    if limbs.is_empty() {
        return count;
    }
    let shift = limbs[0].trailing_zeros();
    if shift != 0 {
        shr_limbs_trunc(limbs, shift);
        count + shift
    } else {
        count
    }
}

impl Default for BigFloat {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.classify() {
            FpCategory::Nan => f.write_str("BigFloat(NaN)"),
            FpCategory::Infinite => {
                if self.sign.is_negative() {
                    f.write_str("BigFloat(-Inf)")
                } else {
                    f.write_str("BigFloat(Inf)")
                }
            }
            FpCategory::Zero => {
                if self.sign.is_negative() {
                    f.write_str("BigFloat(-0)")
                } else {
                    f.write_str("BigFloat(0)")
                }
            }
            FpCategory::Normal => f
                .debug_struct("BigFloat")
                .field("sign", &self.sign)
                .field("exp", &self.exp)
                .field("limbs", &self.limbs)
                .finish(),
        }
    }
}

impl fmt::Display for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_exact_decimal(self, f)
    }
}

impl fmt::Binary for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_abs_int_limbs() {
            Some(limbs) => fmt_integer_radix(f, limbs, self.sign.is_negative(), 2, false, "0b"),
            None => fmt_exact_decimal(self, f),
        }
    }
}

impl fmt::Octal for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_abs_int_limbs() {
            Some(limbs) => fmt_integer_radix(f, limbs, self.sign.is_negative(), 8, false, "0o"),
            None => fmt_exact_decimal(self, f),
        }
    }
}

impl fmt::LowerHex for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_abs_int_limbs() {
            Some(limbs) => fmt_integer_radix(f, limbs, self.sign.is_negative(), 16, false, "0x"),
            None => fmt_exact_decimal(self, f),
        }
    }
}

impl fmt::UpperHex for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_abs_int_limbs() {
            Some(limbs) => fmt_integer_radix(f, limbs, self.sign.is_negative(), 16, true, "0x"),
            None => fmt_exact_decimal(self, f),
        }
    }
}

impl fmt::LowerExp for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n_digits = match f.precision() {
            Some(p) => p as u64 + 1,
            None => 0,
        };
        match self.to_string_radix(10, n_digits, Rounding::NearestEven, true) {
            Some(s) => f.pad(&s),
            None => f.pad("NaN"),
        }
    }
}

impl fmt::UpperExp for BigFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n_digits = match f.precision() {
            Some(p) => p as u64 + 1,
            None => 0,
        };
        match self.to_string_radix(10, n_digits, Rounding::NearestEven, true) {
            Some(mut s) => {
                if let Some(pos) = s.find('e') {
                    s.replace_range(pos..pos + 1, "E");
                }
                f.pad(&s)
            }
            None => f.pad("NaN"),
        }
    }
}

impl PartialEq for BigFloat {
    fn eq(&self, other: &Self) -> bool {
        self.cmp_num(other) == Some(Ordering::Equal)
    }
}

impl PartialOrd for BigFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cmp_num(other)
    }
}

impl From<u64> for BigFloat {
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl From<i64> for BigFloat {
    fn from(value: i64) -> Self {
        Self::from_i64(value)
    }
}

impl From<f64> for BigFloat {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl From<u8> for BigFloat {
    fn from(value: u8) -> Self {
        Self::from_u64(u64::from(value))
    }
}

impl From<u16> for BigFloat {
    fn from(value: u16) -> Self {
        Self::from_u64(u64::from(value))
    }
}

impl From<u32> for BigFloat {
    fn from(value: u32) -> Self {
        Self::from_u64(u64::from(value))
    }
}

impl From<i8> for BigFloat {
    fn from(value: i8) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl From<i16> for BigFloat {
    fn from(value: i16) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl From<i32> for BigFloat {
    fn from(value: i32) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl From<f32> for BigFloat {
    fn from(value: f32) -> Self {
        Self::from_f64(f64::from(value))
    }
}

impl From<u128> for BigFloat {
    fn from(value: u128) -> Self {
        Self::from_u128_with_sign(value, Sign::Positive)
    }
}

impl From<i128> for BigFloat {
    fn from(value: i128) -> Self {
        let sign = Sign::from_negative(value < 0);
        Self::from_u128_with_sign(value.unsigned_abs(), sign)
    }
}

/// A floating-point number with a statically-known format `F`.
///
/// Wraps a [`BigFloat`] and uses the format `F` (which implements [`StaticFormat`])
/// to supply rounding parameters for arithmetic operations automatically.
///
/// ```
/// use libbeef::formats::Binary64;
/// use libbeef::Float;
/// let a = Float::<Binary64>::from_big(libbeef::BigFloat::from_f64(2.5));
/// let b = Float::<Binary64>::from_big(libbeef::BigFloat::from_f64(1.5));
/// let sum = a.add(&b);
/// assert_eq!(sum.as_big().to_f64(libbeef::Rounding::NearestEven), 4.0);
/// ```
pub struct Float<F> {
    value: BigFloat,
    _format: PhantomData<F>,
}

impl<F> Clone for Float<F> {
    fn clone(&self) -> Self {
        Self::from_big(self.value.clone())
    }
}

impl<F> Float<F> {
    /// Creates a new [`Float`] initialized to positive zero.
    pub const fn new() -> Self {
        Self::from_big(BigFloat::new())
    }

    /// Returns a NaN value in this format.
    pub const fn nan() -> Self {
        Self::from_big(BigFloat::nan())
    }

    /// Returns an infinity with the given sign in this format.
    pub const fn infinity(sign: Sign) -> Self {
        Self::from_big(BigFloat::infinity(sign))
    }

    /// Returns a zero with the given sign in this format.
    pub const fn zero(sign: Sign) -> Self {
        Self::from_big(BigFloat::zero(sign))
    }

    /// Unwraps this [`Float`] into its underlying [`BigFloat`].
    pub fn into_big(self) -> BigFloat {
        self.value
    }

    /// Returns a reference to the underlying [`BigFloat`].
    pub const fn as_big(&self) -> &BigFloat {
        &self.value
    }

    /// Returns a mutable reference to the underlying [`BigFloat`].
    pub const fn as_big_mut(&mut self) -> &mut BigFloat {
        &mut self.value
    }

    /// Wraps a [`BigFloat`] in this format type without performing any rounding.
    pub const fn from_big(value: BigFloat) -> Self {
        Self {
            value,
            _format: PhantomData,
        }
    }

    /// Reinterprets this value as a [`Float`] with a different format `G`
    /// without changing the underlying bits.
    pub fn into_format<G>(self) -> Float<G> {
        Float::from_big(self.value)
    }

    /// Compares the absolute values of `self` and `rhs`.
    /// Delegates to [`BigFloat::cmp_abs`]. Corresponds to `bf_cmpu` in libbf.
    pub fn cmp_abs(&self, rhs: &Self) -> Ordering {
        self.value.cmp_abs(&rhs.value)
    }

    /// Compares `self` and `rhs` numerically, returning `None` if either is NaN.
    /// Delegates to [`BigFloat::cmp_num`]. Corresponds to `bf_cmp` in libbf.
    pub fn cmp_num(&self, rhs: &Self) -> Option<Ordering> {
        self.value.cmp_num(&rhs.value)
    }

    /// Total ordering comparison that orders all values including NaN.
    /// Delegates to [`BigFloat::cmp_total`]. Corresponds to `bf_cmp_full` in libbf.
    pub fn cmp_total(&self, rhs: &Self) -> Ordering {
        self.value.cmp_total(&rhs.value)
    }

    /// Alias for [`cmp_total`](Self::cmp_total).
    /// Corresponds to `bf_cmp_full` in libbf.
    pub fn cmp_full(&self, rhs: &Self) -> Ordering {
        self.cmp_total(rhs)
    }
}

impl<F: StaticFormat> Float<F> {
    /// Adds `self` and `rhs` using the format's rounding.
    /// Corresponds to `bf_add` in libbf.
    pub fn add(&self, rhs: &Self) -> Self {
        self.add_status(rhs).0
    }

    /// Adds `self` and `rhs`, returning the status flags as well.
    /// Corresponds to `bf_add` in libbf.
    pub fn add_status(&self, rhs: &Self) -> (Self, Status) {
        let (value, status) = self.value.add_status(&rhs.value, F::FORMAT);
        (Self::from_big(value), status)
    }

    /// Subtracts `rhs` from `self` using the format's rounding.
    /// Corresponds to `bf_sub` in libbf.
    pub fn sub(&self, rhs: &Self) -> Self {
        self.sub_status(rhs).0
    }

    /// Subtracts `rhs` from `self`, returning the status flags as well.
    /// Corresponds to `bf_sub` in libbf.
    pub fn sub_status(&self, rhs: &Self) -> (Self, Status) {
        let (value, status) = self.value.sub_status(&rhs.value, F::FORMAT);
        (Self::from_big(value), status)
    }

    /// Multiplies `self` by `rhs` using the format's rounding.
    /// Corresponds to `bf_mul` in libbf.
    pub fn mul(&self, rhs: &Self) -> Self {
        self.mul_status(rhs).0
    }

    /// Multiplies `self` by `rhs`, returning the status flags as well.
    /// Corresponds to `bf_mul` in libbf.
    pub fn mul_status(&self, rhs: &Self) -> (Self, Status) {
        let (value, status) = self.value.mul_status(&rhs.value, F::FORMAT);
        (Self::from_big(value), status)
    }

    /// Divides `self` by `rhs` using the format's rounding.
    /// Corresponds to `bf_div` in libbf.
    pub fn div(&self, rhs: &Self) -> Self {
        self.div_status(rhs).0
    }

    /// Divides `self` by `rhs`, returning the status flags as well.
    /// Corresponds to `bf_div` in libbf.
    pub fn div_status(&self, rhs: &Self) -> (Self, Status) {
        let (value, status) = self.value.div_status(&rhs.value, F::FORMAT);
        (Self::from_big(value), status)
    }

    /// Computes the remainder of `self / rhs` using the format's rounding.
    /// Corresponds to `bf_rem` in libbf.
    pub fn rem(&self, rhs: &Self, mode: DivRemMode) -> Self {
        self.rem_status(rhs, mode).0
    }

    /// Computes the remainder, returning the status flags as well.
    /// Corresponds to `bf_rem` in libbf.
    pub fn rem_status(&self, rhs: &Self, mode: DivRemMode) -> (Self, Status) {
        let (value, status) = self.value.rem_status(&rhs.value, F::FORMAT, mode);
        (Self::from_big(value), status)
    }

    /// Squares this value (`self * self`) using the format's rounding.
    ///
    /// See [`BigFloat::sqr`] for why this method exists.
    ///
    /// ```
    /// use libbeef::Float;
    /// use libbeef::formats::Binary64;
    ///
    /// let x = Float::<Binary64>::from_f64(7.0);
    /// assert_eq!(x.sqr().as_big().to_f64(libbeef::Rounding::NearestEven), 49.0);
    /// ```
    pub fn sqr(&self) -> Self {
        self.sqr_status().0
    }

    /// Squares this value, returning the status flags as well.
    ///
    /// See [`BigFloat::sqr_status`] for why this method exists.
    ///
    /// ```
    /// use libbeef::Float;
    /// use libbeef::formats::Binary64;
    ///
    /// let x = Float::<Binary64>::from_f64(7.0);
    /// let (result, status) = x.sqr_status();
    /// assert_eq!(result.as_big().to_f64(libbeef::Rounding::NearestEven), 49.0);
    /// ```
    pub fn sqr_status(&self) -> (Self, Status) {
        let (value, status) = self.value.sqr_status(F::FORMAT);
        (Self::from_big(value), status)
    }

    /// Computes the square root using the format's rounding.
    /// Corresponds to `bf_sqrt` in libbf. Uses Newton iteration via integer square root.
    pub fn sqrt(&self) -> Self {
        self.sqrt_status().0
    }

    /// Computes the square root, returning the status flags as well.
    /// Corresponds to `bf_sqrt` in libbf.
    pub fn sqrt_status(&self) -> (Self, Status) {
        let (value, status) = self.value.sqrt_status(F::FORMAT);
        (Self::from_big(value), status)
    }

    /// Computes e raised to the power of `self`.
    /// Corresponds to `bf_exp` in libbf. Uses argument reduction and Taylor series.
    pub fn exp(&self) -> Self {
        Self::from_big(self.value.exp(F::FORMAT))
    }

    /// Computes the natural logarithm of `self`.
    /// Corresponds to `bf_log` in libbf. Uses AGM (arithmetic-geometric mean) iteration.
    pub fn log(&self) -> Self {
        Self::from_big(self.value.log(F::FORMAT))
    }

    /// Computes the cosine of `self` (in radians).
    /// Corresponds to `bf_cos` in libbf. Uses argument reduction and Taylor series.
    pub fn cos(&self) -> Self {
        Self::from_big(self.value.cos(F::FORMAT))
    }

    /// Computes the sine of `self` (in radians).
    /// Corresponds to `bf_sin` in libbf. Uses argument reduction and Taylor series.
    pub fn sin(&self) -> Self {
        Self::from_big(self.value.sin(F::FORMAT))
    }

    /// Computes the tangent of `self` (in radians).
    /// Corresponds to `bf_tan` in libbf. Uses sin/cos internally.
    pub fn tan(&self) -> Self {
        Self::from_big(self.value.tan(F::FORMAT))
    }

    /// Computes the arctangent of `self`.
    /// Corresponds to `bf_atan` in libbf. Uses argument reduction and Taylor series.
    pub fn atan(&self) -> Self {
        Self::from_big(self.value.atan(F::FORMAT))
    }

    /// Computes the arcsine of `self`.
    /// Corresponds to `bf_asin` in libbf. Uses atan-based identity.
    pub fn asin(&self) -> Self {
        Self::from_big(self.value.asin(F::FORMAT))
    }

    /// Computes the arccosine of `self`.
    /// Corresponds to `bf_acos` in libbf. Uses atan-based identity.
    pub fn acos(&self) -> Self {
        Self::from_big(self.value.acos(F::FORMAT))
    }

    /// Computes the two-argument arctangent of `self` and `rhs`.
    /// Corresponds to `bf_atan2` in libbf.
    pub fn atan2(&self, rhs: &Self) -> Self {
        Self::from_big(self.value.atan2(&rhs.value, F::FORMAT))
    }

    /// Raises `self` to the power `rhs`.
    /// Corresponds to `bf_pow` in libbf. Uses exp(rhs * log(self)) for non-integer exponents.
    pub fn pow(&self, rhs: &Self) -> Self {
        Self::from_big(self.value.pow(&rhs.value, F::FORMAT))
    }

    /// Creates a [`Float`] from an unsigned 64-bit integer, rounded to this format.
    ///
    /// Delegates to [`BigFloat::from_u64`] followed by rounding.
    ///
    /// ```
    /// use libbeef::Float;
    /// use libbeef::formats::Binary64;
    ///
    /// let x = Float::<Binary64>::from_u64(100);
    /// assert_eq!(x.as_big().to_f64(libbeef::Rounding::NearestEven), 100.0);
    /// ```
    pub fn from_u64(value: u64) -> Self {
        Self::from_big(BigFloat::from_u64(value).round(F::FORMAT))
    }

    /// Creates a [`Float`] from a signed 64-bit integer, rounded to this format.
    ///
    /// Delegates to [`BigFloat::from_i64`] followed by rounding.
    ///
    /// ```
    /// use libbeef::Float;
    /// use libbeef::formats::Binary64;
    ///
    /// let x = Float::<Binary64>::from_i64(-42);
    /// assert_eq!(x.as_big().to_f64(libbeef::Rounding::NearestEven), -42.0);
    /// ```
    pub fn from_i64(value: i64) -> Self {
        Self::from_big(BigFloat::from_i64(value).round(F::FORMAT))
    }

    /// Creates a [`Float`] from an `f64`, rounded to this format.
    ///
    /// Delegates to [`BigFloat::from_f64`] followed by rounding. Preserves
    /// the exact value of the `f64` input (including sign of zero, infinities,
    /// and NaN) before rounding to the target format.
    ///
    /// ```
    /// use libbeef::Float;
    /// use libbeef::formats::Binary64;
    ///
    /// let x = Float::<Binary64>::from_f64(3.14);
    /// assert_eq!(x.as_big().to_f64(libbeef::Rounding::NearestEven), 3.14);
    /// ```
    pub fn from_f64(value: f64) -> Self {
        Self::from_big(BigFloat::from_f64(value).round(F::FORMAT))
    }

    /// Parses a floating-point literal in the given radix (2..=36), consuming
    /// the entire input string, and rounds the result to this format.
    ///
    /// See [`BigFloat::from_str_radix`] for the accepted syntax and exponent
    /// notation.
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
            format: F::FORMAT,
            allow_hex_prefix: false,
            allow_binary_prefix: false,
            allow_octal_prefix: false,
            allow_nan_inf: true,
            return_radix_exponent: false,
        };
        let (val, consumed) = BigFloat::parse(trimmed, options)?;
        if consumed != trimmed.len() {
            return Err(ParseFloatError::new());
        }
        Ok(Self::from_big(val))
    }
}

impl<F: StaticFormat> core::str::FromStr for Float<F> {
    type Err = ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_radix(s, 10)
    }
}

impl<F> Default for Float<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> fmt::Debug for Float<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Float").field(&self.value).finish()
    }
}

impl<F> fmt::Display for Float<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)
    }
}

impl<F> fmt::Binary for Float<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Binary::fmt(&self.value, f)
    }
}

impl<F> fmt::Octal for Float<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Octal::fmt(&self.value, f)
    }
}

impl<F> fmt::LowerHex for Float<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.value, f)
    }
}

impl<F> fmt::UpperHex for Float<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperHex::fmt(&self.value, f)
    }
}

impl<F> fmt::LowerExp for Float<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerExp::fmt(&self.value, f)
    }
}

impl<F> fmt::UpperExp for Float<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperExp::fmt(&self.value, f)
    }
}

impl<F> From<BigFloat> for Float<F> {
    fn from(value: BigFloat) -> Self {
        Self::from_big(value)
    }
}

impl<F> From<Float<F>> for BigFloat {
    fn from(value: Float<F>) -> Self {
        value.into_big()
    }
}

impl<F: StaticFormat> From<u8> for Float<F> {
    fn from(value: u8) -> Self {
        Self::from_u64(u64::from(value))
    }
}

impl<F: StaticFormat> From<u16> for Float<F> {
    fn from(value: u16) -> Self {
        Self::from_u64(u64::from(value))
    }
}

impl<F: StaticFormat> From<u32> for Float<F> {
    fn from(value: u32) -> Self {
        Self::from_u64(u64::from(value))
    }
}

impl<F: StaticFormat> From<u64> for Float<F> {
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl<F: StaticFormat> From<i8> for Float<F> {
    fn from(value: i8) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl<F: StaticFormat> From<i16> for Float<F> {
    fn from(value: i16) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl<F: StaticFormat> From<i32> for Float<F> {
    fn from(value: i32) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl<F: StaticFormat> From<i64> for Float<F> {
    fn from(value: i64) -> Self {
        Self::from_i64(value)
    }
}

impl<F: StaticFormat> From<f32> for Float<F> {
    fn from(value: f32) -> Self {
        Self::from_f64(f64::from(value))
    }
}

impl<F: StaticFormat> From<f64> for Float<F> {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}

impl<F: StaticFormat> From<u128> for Float<F> {
    fn from(value: u128) -> Self {
        Self::from_big(BigFloat::from(value).round(F::FORMAT))
    }
}

impl<F: StaticFormat> From<i128> for Float<F> {
    fn from(value: i128) -> Self {
        Self::from_big(BigFloat::from(value).round(F::FORMAT))
    }
}

impl<F> PartialEq for Float<F> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<F> PartialOrd for Float<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<F: StaticFormat> Add for Float<F> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.value.add_assign(&rhs.value, F::FORMAT);
        self
    }
}

impl<'a, F: StaticFormat> Add<&'a Float<F>> for Float<F> {
    type Output = Self;

    fn add(mut self, rhs: &'a Float<F>) -> Self::Output {
        self.value.add_assign(&rhs.value, F::FORMAT);
        self
    }
}

impl<F: StaticFormat> AddAssign for Float<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.value.add_assign(&rhs.value, F::FORMAT);
    }
}

impl<F: StaticFormat> Sub for Float<F> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self.value.sub_assign(&rhs.value, F::FORMAT);
        self
    }
}

impl<F: StaticFormat> SubAssign for Float<F> {
    fn sub_assign(&mut self, rhs: Self) {
        self.value.sub_assign(&rhs.value, F::FORMAT);
    }
}

impl<F: StaticFormat> Mul for Float<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Float::mul(&self, &rhs)
    }
}

impl<F: StaticFormat> MulAssign for Float<F> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Float::mul(self, &rhs);
    }
}

impl<F: StaticFormat> Div for Float<F> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Float::div(&self, &rhs)
    }
}

impl<F: StaticFormat> DivAssign for Float<F> {
    fn div_assign(&mut self, rhs: Self) {
        *self = Float::div(self, &rhs);
    }
}

impl<F: StaticFormat> Rem for Float<F> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Float::rem(&self, &rhs, DivRemMode::TowardZero)
    }
}

impl<F: StaticFormat> RemAssign for Float<F> {
    fn rem_assign(&mut self, rhs: Self) {
        *self = self.clone() % rhs;
    }
}

impl<F> Neg for Float<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_big(self.value.neg())
    }
}

/// An arbitrary-precision integer backed by [`BigFloat`].
///
/// No direct libbf equivalent — this is a Rust-specific newtype that provides
/// type-safe integer operations. All arithmetic is exact (infinite precision).
///
/// ```
/// use libbeef::Integer;
///
/// let a = Integer::from_i64(42);
/// let b = Integer::from_i64(8);
/// let (q, r) = a.div_rem(&b);
/// assert_eq!(q.to_i64(), Some(5));
/// assert_eq!(r.to_i64(), Some(2));
/// ```
#[derive(Clone, Debug, Default, PartialEq, PartialOrd)]
pub struct Integer(BigFloat);

impl Integer {
    /// Unwraps this [`Integer`] into its underlying [`BigFloat`].
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// let n = Integer::from_i64(7);
    /// let bf = n.into_big();
    /// assert_eq!(bf.get_int64(), Some(7));
    /// ```
    pub fn into_big(self) -> BigFloat {
        self.0
    }

    /// Returns a reference to the underlying [`BigFloat`].
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// let n = Integer::from_i64(7);
    /// assert_eq!(n.as_big().get_int64(), Some(7));
    /// ```
    pub const fn as_big(&self) -> &BigFloat {
        &self.0
    }

    /// Creates the integer zero.
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// assert!(Integer::zero().is_zero());
    /// ```
    pub const fn zero() -> Self {
        Self(BigFloat::new())
    }

    /// Creates an [`Integer`] from an `i64`.
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// let n = Integer::from_i64(-99);
    /// assert_eq!(n.to_i64(), Some(-99));
    /// ```
    pub fn from_i64(value: i64) -> Self {
        Self(BigFloat::from_i64(value))
    }

    /// Creates an [`Integer`] from a `u64`.
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// let n = Integer::from_u64(1_000_000);
    /// assert_eq!(n.to_u64(), Some(1_000_000));
    /// ```
    pub fn from_u64(value: u64) -> Self {
        Self(BigFloat::from_u64(value))
    }

    /// Tries to convert a [`BigFloat`] into an [`Integer`].
    ///
    /// Returns `None` if `bf` is NaN, infinite, or has a fractional part.
    ///
    /// ```
    /// use libbeef::{BigFloat, Integer};
    ///
    /// assert!(Integer::from_bigfloat(BigFloat::from_i64(5)).is_some());
    /// assert!(Integer::from_bigfloat(BigFloat::from_f64(2.5)).is_none());
    /// assert!(Integer::from_bigfloat(BigFloat::nan()).is_none());
    /// ```
    pub fn from_bigfloat(bf: BigFloat) -> Option<Self> {
        if bf.is_nan() || bf.is_infinite() {
            return None;
        }
        if bf.is_zero() {
            return Some(Self::zero());
        }
        let rounded = bf.rint(Rounding::TowardZero);
        if rounded.cmp_num(&bf) == Some(Ordering::Equal) {
            Some(Self(bf))
        } else {
            None
        }
    }

    /// Converts to `i64`, returning `None` if the value doesn't fit.
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// assert_eq!(Integer::from_i64(-42).to_i64(), Some(-42));
    /// ```
    pub fn to_i64(&self) -> Option<i64> {
        self.0.get_int64()
    }

    /// Converts to `u64`, returning `None` if the value is negative or doesn't fit.
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// assert_eq!(Integer::from_u64(42).to_u64(), Some(42));
    /// assert_eq!(Integer::from_i64(-1).to_u64(), None);
    /// ```
    pub fn to_u64(&self) -> Option<u64> {
        self.0.get_uint64()
    }

    /// Converts to `f64` (may lose precision for large values).
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// assert_eq!(Integer::from_i64(42).to_f64(), 42.0);
    /// ```
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64_status(Rounding::NearestEven).0
    }

    /// Returns `true` if this integer is zero.
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// assert!(Integer::zero().is_zero());
    /// assert!(!Integer::from_i64(1).is_zero());
    /// ```
    pub const fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// Returns `true` if this integer is negative.
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// assert!(Integer::from_i64(-1).is_negative());
    /// assert!(!Integer::from_i64(1).is_negative());
    /// ```
    pub const fn is_negative(&self) -> bool {
        self.0.is_sign_negative() && !self.0.is_zero()
    }

    /// Returns the sign of this integer.
    ///
    /// ```
    /// use libbeef::{Integer, Sign};
    ///
    /// assert_eq!(Integer::from_i64(-5).sign(), Sign::Negative);
    /// assert_eq!(Integer::from_i64(5).sign(), Sign::Positive);
    /// ```
    pub const fn sign(&self) -> Sign {
        self.0.sign()
    }

    /// Exact addition (infinite precision).
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// let sum = Integer::from_i64(100).add(&Integer::from_i64(23));
    /// assert_eq!(sum.to_i64(), Some(123));
    /// ```
    pub fn add(&self, rhs: &Self) -> Self {
        Self(self.0.add(&rhs.0, Self::inf_format()))
    }

    /// Exact subtraction (infinite precision).
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// let diff = Integer::from_i64(100).sub(&Integer::from_i64(42));
    /// assert_eq!(diff.to_i64(), Some(58));
    /// ```
    pub fn sub(&self, rhs: &Self) -> Self {
        Self(self.0.sub(&rhs.0, Self::inf_format()))
    }

    /// Exact multiplication (infinite precision).
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// let prod = Integer::from_i64(6).mul(&Integer::from_i64(7));
    /// assert_eq!(prod.to_i64(), Some(42));
    /// ```
    pub fn mul(&self, rhs: &Self) -> Self {
        Self(self.0.mul(&rhs.0, Self::inf_format()))
    }

    /// Integer division with remainder (truncating toward zero).
    ///
    /// # Panics
    ///
    /// Panics if `rhs` is zero.
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// let (q, r) = Integer::from_i64(17).div_rem(&Integer::from_i64(5));
    /// assert_eq!(q.to_i64(), Some(3));
    /// assert_eq!(r.to_i64(), Some(2));
    ///
    /// let (q, r) = Integer::from_i64(-17).div_rem(&Integer::from_i64(5));
    /// assert_eq!(q.to_i64(), Some(-3));
    /// assert_eq!(r.to_i64(), Some(-2));
    /// ```
    pub fn div_rem(&self, rhs: &Self) -> (Self, Self) {
        assert!(!rhs.is_zero(), "division by zero");
        let (q, r, _status) =
            self.0
                .div_rem_status(&rhs.0, Self::inf_format(), DivRemMode::TowardZero);
        (Self(q), Self(r))
    }

    /// Negates this integer.
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// assert_eq!(Integer::from_i64(5).neg().to_i64(), Some(-5));
    /// ```
    pub fn neg(&self) -> Self {
        Self(self.0.neg())
    }

    /// Returns the absolute value.
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// assert_eq!(Integer::from_i64(-5).abs().to_i64(), Some(5));
    /// ```
    pub fn abs(&self) -> Self {
        Self(self.0.abs())
    }

    /// Exact squaring (`self * self`).
    ///
    /// See [`BigFloat::sqr`] for why this method exists.
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// assert_eq!(Integer::from_i64(7).sqr().to_i64(), Some(49));
    /// assert_eq!(Integer::from_i64(-5).sqr().to_i64(), Some(25));
    /// ```
    pub fn sqr(&self) -> Self {
        Self(self.0.mul(&self.0, Self::inf_format()))
    }

    /// Exact integer exponentiation.
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// assert_eq!(Integer::from_i64(2).pow(10).to_i64(), Some(1024));
    /// assert_eq!(Integer::from_i64(3).pow(0).to_i64(), Some(1));
    /// ```
    pub fn pow(&self, exp: u64) -> Self {
        if exp == 0 {
            return Self::from_i64(1);
        }
        let exponent = SignedInt {
            sign: Sign::Positive,
            limbs: alloc::vec![exp],
        };
        let (result, _status) = pow_integer_status(&self.0, &exponent, Self::inf_format());
        Self(result)
    }

    /// Compares two integers, returning `None` only if either is NaN
    /// (which cannot happen for valid `Integer` values).
    ///
    /// ```
    /// use libbeef::Integer;
    /// use core::cmp::Ordering;
    ///
    /// assert_eq!(Integer::from_i64(5).cmp(&Integer::from_i64(3)), Some(Ordering::Greater));
    /// assert_eq!(Integer::from_i64(3).cmp(&Integer::from_i64(3)), Some(Ordering::Equal));
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn cmp(&self, rhs: &Self) -> Option<Ordering> {
        self.0.cmp_num(&rhs.0)
    }

    /// Parses a decimal integer string (e.g. `"12345"`, `"-99"`) into an [`Integer`].
    ///
    /// Delegates to [`BigFloat::parse_decimal_integer`].
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// let n = Integer::parse_decimal_integer("12345").unwrap();
    /// assert_eq!(n.to_i64(), Some(12345));
    ///
    /// let neg = Integer::parse_decimal_integer("-99").unwrap();
    /// assert_eq!(neg.to_i64(), Some(-99));
    /// ```
    pub fn parse_decimal_integer(input: &str) -> Result<Self, ParseFloatError> {
        BigFloat::parse_decimal_integer(input).map(Self)
    }

    /// Parses an integer string in the given radix (2..=36) into an [`Integer`].
    ///
    /// Delegates to [`BigFloat::parse_integer_radix`].
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// let hex = Integer::parse_integer_radix("ff", 16).unwrap();
    /// assert_eq!(hex.to_i64(), Some(255));
    ///
    /// let bin = Integer::parse_integer_radix("1010", 2).unwrap();
    /// assert_eq!(bin.to_i64(), Some(10));
    /// ```
    pub fn parse_integer_radix(input: &str, radix: u8) -> Result<Self, ParseFloatError> {
        BigFloat::parse_integer_radix(input, radix).map(Self)
    }

    fn inf_format() -> BigFormat {
        BigFormat {
            precision: Precision::Infinite,
            ..BigFormat::BINARY64
        }
    }
}

impl fmt::Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let limbs = self.0.to_abs_int_limbs().unwrap_or_default();
        fmt_integer_radix(f, limbs, self.is_negative(), 10, false, "")
    }
}

impl fmt::Binary for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let limbs = self.0.to_abs_int_limbs().unwrap_or_default();
        fmt_integer_radix(f, limbs, self.is_negative(), 2, false, "0b")
    }
}

impl fmt::Octal for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let limbs = self.0.to_abs_int_limbs().unwrap_or_default();
        fmt_integer_radix(f, limbs, self.is_negative(), 8, false, "0o")
    }
}

impl fmt::LowerHex for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let limbs = self.0.to_abs_int_limbs().unwrap_or_default();
        fmt_integer_radix(f, limbs, self.is_negative(), 16, false, "0x")
    }
}

impl fmt::UpperHex for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let limbs = self.0.to_abs_int_limbs().unwrap_or_default();
        fmt_integer_radix(f, limbs, self.is_negative(), 16, true, "0x")
    }
}

impl From<i64> for Integer {
    fn from(value: i64) -> Self {
        Self::from_i64(value)
    }
}

impl From<u64> for Integer {
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl From<u8> for Integer {
    fn from(value: u8) -> Self {
        Self::from_u64(u64::from(value))
    }
}

impl From<u16> for Integer {
    fn from(value: u16) -> Self {
        Self::from_u64(u64::from(value))
    }
}

impl From<u32> for Integer {
    fn from(value: u32) -> Self {
        Self::from_u64(u64::from(value))
    }
}

impl From<i8> for Integer {
    fn from(value: i8) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl From<i16> for Integer {
    fn from(value: i16) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl From<i32> for Integer {
    fn from(value: i32) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl From<u128> for Integer {
    fn from(value: u128) -> Self {
        Self(BigFloat::from(value))
    }
}

impl From<i128> for Integer {
    fn from(value: i128) -> Self {
        Self(BigFloat::from(value))
    }
}

impl Add for Integer {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Integer::add(&self, &rhs)
    }
}

impl Sub for Integer {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Integer::sub(&self, &rhs)
    }
}

impl Mul for Integer {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Integer::mul(&self, &rhs)
    }
}

impl Neg for Integer {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Integer::neg(&self)
    }
}

impl core::str::FromStr for Integer {
    type Err = ParseFloatError;

    /// Parses a decimal integer string into an [`Integer`].
    ///
    /// ```
    /// use libbeef::Integer;
    ///
    /// let n: Integer = "12345".parse().unwrap();
    /// assert_eq!(n.to_i64(), Some(12345));
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_decimal_integer(s)
    }
}
