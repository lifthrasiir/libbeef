use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use core::str::FromStr;

use crate::float::{
    div_rem_abs_limbs, sqrt_rem_abs_limbs, DivRemMode, RawExp, Sign, RAW_EXP_INF, RAW_EXP_NAN,
    RAW_EXP_ZERO,
};
use crate::format::{BigFormat, Precision, Rounding, StaticFormat};
use crate::parse::ParseFloatError;
use crate::status::Status;

/// The unsigned integer type used for each digit-group in the decimal representation.
pub type Limb = u64;

/// Arbitrary-precision decimal floating-point number.
///
/// Corresponds to `bfdec_t` in libbf. Stores values as a scaled decimal
/// integer with sign, exponent, and scale fields. Supports NaN, infinities,
/// and signed zeros.
///
/// ```
/// use libbeef::BigDecimal;
///
/// let x = BigDecimal::parse_decimal("3.14").unwrap();
/// let s = x.to_decimal_string().unwrap();
/// assert_eq!(s, "3.14");
/// ```
#[derive(Clone)]
pub struct BigDecimal {
    sign: Sign,
    exp: RawExp,
    limbs: Vec<Limb>,
    scale: i32,
}

impl BigDecimal {
    /// Creates a new `BigDecimal` initialized to positive zero.
    ///
    /// Corresponds to `bfdec_init` in libbf.
    ///
    /// ```
    /// use libbeef::BigDecimal;
    ///
    /// let x = BigDecimal::new();
    /// assert!(x.is_zero());
    /// ```
    pub const fn new() -> Self {
        Self::zero(Sign::Positive)
    }

    /// Creates a `BigDecimal` representing NaN (Not a Number).
    ///
    /// Corresponds to `bfdec_set_nan` in libbf.
    pub const fn nan() -> Self {
        Self {
            sign: Sign::Positive,
            exp: RAW_EXP_NAN,
            limbs: Vec::new(),
            scale: 0,
        }
    }

    /// Creates a `BigDecimal` representing positive or negative infinity.
    ///
    /// Corresponds to `bfdec_set_inf` in libbf.
    pub const fn infinity(sign: Sign) -> Self {
        Self {
            sign,
            exp: RAW_EXP_INF,
            limbs: Vec::new(),
            scale: 0,
        }
    }

    /// Creates a `BigDecimal` representing positive or negative zero.
    ///
    /// Corresponds to `bfdec_set_zero` in libbf.
    pub const fn zero(sign: Sign) -> Self {
        Self {
            sign,
            exp: RAW_EXP_ZERO,
            limbs: Vec::new(),
            scale: 0,
        }
    }

    /// Creates a `BigDecimal` from a 64-bit signed integer.
    ///
    /// Corresponds to `bfdec_set_si` in libbf.
    ///
    /// ```
    /// use libbeef::BigDecimal;
    ///
    /// let x = BigDecimal::from_i64(42);
    /// assert_eq!(x.to_decimal_string().unwrap(), "42");
    /// ```
    pub fn from_i64(value: i64) -> Self {
        if value == 0 {
            return Self::new();
        }
        let sign = Sign::from(value);
        Self::from_binary_abs_limbs(alloc::vec![value.unsigned_abs()], sign)
    }

    /// Creates a `BigDecimal` from a scaled integer: `coefficient * 10^(-scale)`.
    /// Exposes the internal scaled representation.
    ///
    /// ```
    /// use libbeef::BigDecimal;
    ///
    /// let x = BigDecimal::from_scaled_i64(314, 2);
    /// assert_eq!(x.to_decimal_string().unwrap(), "3.14");
    /// ```
    pub fn from_scaled_i64(coefficient: i64, scale: i32) -> Self {
        if coefficient == 0 {
            return Self::new();
        }
        let sign = Sign::from(coefficient);
        Self::from_scaled_decimal_limbs(ScaledDecimalLimbs {
            sign,
            limbs: binary_abs_to_decimal_limbs(alloc::vec![coefficient.unsigned_abs()]),
            scale: scale.max(0),
        })
    }

    /// Parses a decimal string into a `BigDecimal`.
    ///
    /// Accepts integers, decimals with `.`, and scientific notation with `e`/`E`.
    /// Also accepts `nan`, `inf`, `+inf`, `-inf` (case-insensitive).
    /// Corresponds to `bfdec_atof` in libbf.
    ///
    /// ```
    /// use libbeef::BigDecimal;
    ///
    /// let x = BigDecimal::parse_decimal("3.14e2").unwrap();
    /// assert_eq!(x.to_decimal_string().unwrap(), "314");
    /// ```
    pub fn parse_decimal(input: &str) -> Result<Self, ParseFloatError> {
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
                    push_dec_digit(&mut coefficient, u64::from(rest[index] - b'0'));
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
        if scale < 0 {
            mul_pow10_dec_limbs(&mut coefficient, -scale);
            scale = 0;
        }
        Ok(Self::from_scaled_decimal_limbs(ScaledDecimalLimbs {
            sign,
            limbs: coefficient,
            scale,
        }))
    }

    /// Converts this `BigDecimal` to a decimal string representation.
    ///
    /// Returns `None` for non-representable internal states. NaN and infinities
    /// produce `"NaN"`, `"inf"`, or `"-inf"`.
    /// Corresponds to `bfdec_ftoa` in libbf.
    ///
    /// ```
    /// use libbeef::BigDecimal;
    ///
    /// let x = BigDecimal::from_i64(-7);
    /// assert_eq!(x.to_decimal_string().unwrap(), "-7");
    /// ```
    pub fn to_decimal_string(&self) -> Option<String> {
        if self.is_nan() {
            return Some("NaN".into());
        }
        if self.is_infinite() {
            return Some(
                if self.sign.is_negative() {
                    "-inf"
                } else {
                    "inf"
                }
                .into(),
            );
        }
        let value = self.to_scaled_decimal_limbs()?;
        if value.limbs.is_empty() {
            return Some("0".into());
        }
        let mut digits = decimal_limbs_to_string(&value.limbs);
        if value.scale > 0 {
            let scale = value.scale as usize;
            if digits.len() <= scale {
                let mut padded = String::from("0.");
                for _ in 0..(scale - digits.len()) {
                    padded.push('0');
                }
                padded.push_str(&digits);
                digits = padded;
            } else {
                digits.insert(digits.len() - scale, '.');
            }
        }
        if value.sign.is_negative() {
            digits.insert(0, '-');
        }
        Some(digits)
    }

    /// Adds two `BigDecimal` values, rounding the result according to `format`.
    ///
    /// Corresponds to `bfdec_add` in libbf.
    ///
    /// ```
    /// use libbeef::{BigDecimal, BigFormat};
    ///
    /// let a = BigDecimal::parse_decimal("1.5").unwrap();
    /// let b = BigDecimal::parse_decimal("2.5").unwrap();
    /// let sum = a.add(&b, BigFormat::DECIMAL64);
    /// assert_eq!(sum.to_decimal_string().unwrap(), "4");
    /// ```
    pub fn add(&self, rhs: &Self, format: BigFormat) -> Self {
        self.add_status(rhs, format).0
    }

    /// Adds two `BigDecimal` values. Returns the status flags as well.
    ///
    /// Corresponds to `bfdec_add` in libbf.
    pub fn add_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status) {
        if self.is_nan() || rhs.is_nan() {
            return (Self::nan(), Status::empty());
        }
        if self.is_infinite() || rhs.is_infinite() {
            return match (self.is_infinite(), rhs.is_infinite()) {
                (true, true) if self.sign != rhs.sign => (Self::nan(), Status::INVALID_OP),
                (true, _) => (self.clone(), Status::empty()),
                (_, true) => (rhs.clone(), Status::empty()),
                _ => unreachable!(),
            };
        }
        let lhs = self
            .to_scaled_decimal_limbs()
            .expect("finite BigDecimal values are representable as scaled decimal limbs");
        let rhs = rhs
            .to_scaled_decimal_limbs()
            .expect("finite BigDecimal values are representable as scaled decimal limbs");
        let sum = add_scaled_decimal_limbs(&lhs, &rhs);
        Self::from_scaled_decimal_limbs(sum).round_to_format(format)
    }

    /// Adds a 64-bit signed integer to this `BigDecimal`.
    ///
    /// Corresponds to `bfdec_add_si` in libbf.
    pub fn add_i64(&self, b: i64, format: BigFormat) -> Self {
        self.add(&Self::from_i64(b), format)
    }

    /// Subtracts `rhs` from this `BigDecimal`, rounding the result according to `format`.
    ///
    /// Corresponds to `bfdec_sub` in libbf.
    pub fn sub(&self, rhs: &Self, format: BigFormat) -> Self {
        self.sub_status(rhs, format).0
    }

    /// Subtracts `rhs` from this `BigDecimal`. Returns the status flags as well.
    ///
    /// Corresponds to `bfdec_sub` in libbf.
    pub fn sub_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status) {
        self.add_status(&rhs.neg(), format)
    }

    /// Multiplies two `BigDecimal` values, rounding the result according to `format`.
    ///
    /// Corresponds to `bfdec_mul` in libbf.
    pub fn mul(&self, rhs: &Self, format: BigFormat) -> Self {
        self.mul_status(rhs, format).0
    }

    /// Multiplies two `BigDecimal` values. Returns the status flags as well.
    ///
    /// Corresponds to `bfdec_mul` in libbf.
    pub fn mul_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status) {
        if self.is_nan() || rhs.is_nan() {
            return (Self::nan(), Status::empty());
        }
        let sign = self.sign * rhs.sign;
        match (self.classify(), rhs.classify()) {
            (DecCategory::Infinite, DecCategory::Zero)
            | (DecCategory::Zero, DecCategory::Infinite) => {
                return (Self::nan(), Status::INVALID_OP);
            }
            (DecCategory::Infinite, _) | (_, DecCategory::Infinite) => {
                return (Self::infinity(sign), Status::empty());
            }
            (DecCategory::Zero, _) | (_, DecCategory::Zero) => {
                return (Self::zero(sign), Status::empty());
            }
            _ => {}
        }
        let lhs = self
            .to_scaled_decimal_limbs()
            .expect("finite BigDecimal values are representable as scaled decimal limbs");
        let rhs = rhs
            .to_scaled_decimal_limbs()
            .expect("finite BigDecimal values are representable as scaled decimal limbs");
        let product = mul_scaled_decimal_limbs(&lhs, &rhs);
        Self::from_scaled_decimal_limbs(product).round_to_format(format)
    }

    /// Squares this value (`self * self`), returning the result.
    ///
    /// See [`BigFloat::sqr`](crate::BigFloat::sqr) for why this method exists.
    ///
    /// ```
    /// use libbeef::{BigDecimal, BigFormat};
    ///
    /// let x = BigDecimal::from_i64(7);
    /// let format = BigFormat::DECIMAL64;
    /// assert_eq!(x.sqr(format).get_int32(), Some(49));
    /// ```
    pub fn sqr(&self, format: BigFormat) -> Self {
        self.sqr_status(format).0
    }

    /// Squares this value, returning the result and status flags.
    ///
    /// See [`BigFloat::sqr_status`](crate::BigFloat::sqr_status) for why this method exists.
    ///
    /// ```
    /// use libbeef::{BigDecimal, BigFormat};
    ///
    /// let x = BigDecimal::from_i64(7);
    /// let (result, status) = x.sqr_status(BigFormat::DECIMAL64);
    /// assert_eq!(result.get_int32(), Some(49));
    /// ```
    pub fn sqr_status(&self, format: BigFormat) -> (Self, Status) {
        self.mul_status(self, format)
    }

    /// Multiplies this `BigDecimal` by a 64-bit signed integer.
    ///
    /// Corresponds to `bfdec_mul_si` in libbf.
    pub fn mul_i64(&self, b: i64, format: BigFormat) -> Self {
        self.mul(&Self::from_i64(b), format)
    }

    /// Attempts to convert this `BigDecimal` to a 32-bit signed integer.
    ///
    /// Returns `None` if the value is not an integer, is out of `i32` range,
    /// or is NaN/infinite.
    /// Corresponds to `bfdec_get_int32` in libbf.
    ///
    /// ```
    /// use libbeef::BigDecimal;
    ///
    /// let x = BigDecimal::from_i64(42);
    /// assert_eq!(x.get_int32(), Some(42));
    /// ```
    pub fn get_int32(&self) -> Option<i32> {
        if self.is_zero() {
            return Some(0);
        }
        if !matches!(self.classify(), DecCategory::Normal) {
            return None;
        }
        if self.scale > 0 {
            return None;
        }
        let s = self.to_decimal_string()?;
        s.parse::<i32>().ok()
    }

    /// Divides this `BigDecimal` by `rhs`, rounding the result according to `format`.
    ///
    /// Corresponds to `bfdec_div` in libbf.
    pub fn div(&self, rhs: &Self, format: BigFormat) -> Self {
        self.div_status(rhs, format).0
    }

    /// Divides this `BigDecimal` by `rhs`. Returns the status flags as well.
    ///
    /// Corresponds to `bfdec_div` in libbf.
    pub fn div_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status) {
        if self.is_nan() || rhs.is_nan() {
            return (Self::nan(), Status::empty());
        }
        let sign = self.sign * rhs.sign;
        match (self.classify(), rhs.classify()) {
            (DecCategory::Zero, DecCategory::Zero)
            | (DecCategory::Infinite, DecCategory::Infinite) => {
                return (Self::nan(), Status::INVALID_OP);
            }
            (_, DecCategory::Zero) => return (Self::infinity(sign), Status::DIVIDE_ZERO),
            (DecCategory::Infinite, _) => return (Self::infinity(sign), Status::empty()),
            (_, DecCategory::Infinite) => return (Self::zero(sign), Status::empty()),
            (DecCategory::Zero, _) => return (Self::zero(sign), Status::empty()),
            _ => {}
        }
        let lhs = self
            .to_scaled_decimal_limbs()
            .expect("finite BigDecimal values are representable as scaled decimal limbs");
        let rhs = rhs
            .to_scaled_decimal_limbs()
            .expect("finite BigDecimal values are representable as scaled decimal limbs");
        div_scaled_decimal_limbs(&lhs, &rhs, format)
    }

    /// Computes the remainder of dividing this `BigDecimal` by `rhs`.
    ///
    /// Corresponds to `bfdec_rem` in libbf.
    pub fn rem(&self, rhs: &Self, format: BigFormat, mode: DivRemMode) -> Self {
        self.rem_status(rhs, format, mode).0
    }

    /// Computes the remainder of dividing this `BigDecimal` by `rhs`. Returns the status flags as well.
    ///
    /// Corresponds to `bfdec_rem` in libbf.
    pub fn rem_status(&self, rhs: &Self, format: BigFormat, mode: DivRemMode) -> (Self, Status) {
        if self.is_nan() || rhs.is_nan() || self.is_infinite() || rhs.is_zero() {
            return (Self::nan(), Status::INVALID_OP);
        }
        if self.is_zero() || rhs.is_infinite() {
            return (self.clone(), Status::empty());
        }

        let lhs = self
            .to_scaled_decimal_limbs()
            .expect("finite BigDecimal values are representable as scaled decimal limbs");
        let rhs = rhs
            .to_scaled_decimal_limbs()
            .expect("finite BigDecimal values are representable as scaled decimal limbs");
        let (_, remainder) = divrem_scaled_decimal_limbs(&lhs, &rhs, mode);
        Self::from_scaled_decimal_limbs(remainder).round_to_format(format)
    }

    /// Computes both the quotient and remainder of dividing this `BigDecimal` by `rhs`.
    /// Returns the status flags as well.
    ///
    /// Corresponds to `bfdec_divrem` in libbf.
    pub fn divrem_status(
        &self,
        rhs: &Self,
        format: BigFormat,
        mode: DivRemMode,
    ) -> ((Self, Self), Status) {
        if self.is_nan() || rhs.is_nan() || self.is_infinite() || rhs.is_zero() {
            return ((Self::nan(), Self::nan()), Status::INVALID_OP);
        }
        if self.is_zero() || rhs.is_infinite() {
            return ((Self::zero(Sign::Positive), self.clone()), Status::empty());
        }

        let lhs = self
            .to_scaled_decimal_limbs()
            .expect("finite BigDecimal values are representable as scaled decimal limbs");
        let rhs = rhs
            .to_scaled_decimal_limbs()
            .expect("finite BigDecimal values are representable as scaled decimal limbs");
        let (quotient, remainder) = divrem_scaled_decimal_limbs(&lhs, &rhs, mode);
        let (remainder, status) =
            Self::from_scaled_decimal_limbs(remainder).round_to_format(format);
        (
            (Self::from_scaled_decimal_limbs(quotient), remainder),
            status,
        )
    }

    /// Rounds this `BigDecimal` to the nearest integer according to the given rounding mode.
    ///
    /// Corresponds to `bfdec_rint` in libbf.
    pub fn rint(&self, rounding: Rounding) -> Self {
        self.rint_status(rounding).0
    }

    /// Rounds this `BigDecimal` to the nearest integer. Returns the status flags as well.
    ///
    /// Corresponds to `bfdec_rint` in libbf.
    pub fn rint_status(&self, rounding: Rounding) -> (Self, Status) {
        if !matches!(self.classify(), DecCategory::Normal) || self.scale == 0 {
            return (self.clone(), Status::empty());
        }
        let value = self
            .to_scaled_decimal_limbs()
            .expect("finite BigDecimal values are representable as scaled decimal limbs");
        let (rounded, inexact) = rint_scaled_decimal_limbs(value, rounding);
        let status = if inexact {
            Status::INEXACT
        } else {
            Status::empty()
        };
        (Self::from_scaled_decimal_limbs(rounded), status)
    }

    /// Computes the square root of this `BigDecimal`, rounding according to `format`.
    ///
    /// Uses integer square root on the scaled decimal representation.
    /// Corresponds to `bfdec_sqrt` in libbf.
    pub fn sqrt(&self, format: BigFormat) -> Self {
        self.sqrt_status(format).0
    }

    /// Computes the square root of this `BigDecimal`. Returns the status flags as well.
    ///
    /// Corresponds to `bfdec_sqrt` in libbf.
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
        if matches!(format.precision, Precision::Infinite) {
            return (Self::nan(), Status::INVALID_OP);
        }

        let value = self
            .to_scaled_decimal_limbs()
            .expect("finite BigDecimal values are representable as scaled decimal limbs");
        sqrt_scaled_decimal_limbs(&value, format)
    }

    /// Raises this `BigDecimal` to the power of an unsigned 64-bit integer.
    ///
    /// Uses repeated squaring. Corresponds to `bfdec_pow_ui` in libbf.
    ///
    /// ```
    /// use libbeef::BigDecimal;
    ///
    /// let x = BigDecimal::from_i64(3);
    /// let y = x.pow_u64(4);
    /// assert_eq!(y.to_decimal_string().unwrap(), "81");
    /// ```
    pub fn pow_u64(&self, exp: u64) -> Self {
        self.pow_u64_status(exp).0
    }

    /// Raises this `BigDecimal` to the power of an unsigned integer. Returns the status flags as well.
    ///
    /// Corresponds to `bfdec_pow_ui` in libbf.
    pub fn pow_u64_status(&self, exp: u64) -> (Self, Status) {
        if self.is_nan() {
            return (Self::nan(), Status::empty());
        }
        if exp == 0 {
            return (Self::from_i64(1), Status::empty());
        }
        if self.is_zero() {
            return (self.clone(), Status::empty());
        }
        if self.is_infinite() {
            if self.sign.is_negative() && !exp.is_multiple_of(2) {
                return (Self::infinity(Sign::Negative), Status::empty());
            }
            return (Self::infinity(Sign::Positive), Status::empty());
        }
        let inf = BigFormat {
            precision: Precision::Infinite,
            ..BigFormat::DECIMAL64
        };
        let mut result = Self::from_i64(1);
        let mut base = self.clone();
        let mut e = exp;
        while e > 0 {
            if e & 1 != 0 {
                result = result.mul(&base, inf);
            }
            e >>= 1;
            if e > 0 {
                base = base.mul(&base, inf);
            }
        }
        (result, Status::empty())
    }

    /// Converts this `BigDecimal` to a [`BigFloat`](crate::BigFloat), rounding according to `format`.
    ///
    /// Converts via the decimal string representation.
    /// Corresponds to `bfdec_to_f` in libbf.
    // NOTE: questionable pub visibility
    pub fn to_bigfloat(
        &self,
        format: crate::format::BigFormat,
    ) -> (crate::float::BigFloat, crate::status::Status) {
        use crate::float::BigFloat;
        if self.is_nan() {
            return (BigFloat::nan(), Status::empty());
        }
        if self.is_infinite() {
            return (BigFloat::infinity(self.sign), Status::empty());
        }
        if self.is_zero() {
            return (BigFloat::zero(self.sign), Status::empty());
        }
        let decimal_str = self.to_decimal_string().unwrap_or_else(|| "0".into());
        match BigFloat::parse_decimal(&decimal_str, format) {
            Ok(bf) => bf.round_status(format),
            Err(_) => (BigFloat::nan(), Status::INVALID_OP),
        }
    }

    /// Creates a `BigDecimal` from a [`BigFloat`](crate::BigFloat), rounding according to `format`.
    ///
    /// Converts via the decimal string representation.
    /// Corresponds to `bfdec_from_f` in libbf.
    // NOTE: questionable pub visibility
    pub fn from_bigfloat(
        bf: &crate::float::BigFloat,
        format: crate::format::BigFormat,
    ) -> (Self, crate::status::Status) {
        if bf.is_nan() {
            return (Self::nan(), Status::empty());
        }
        if bf.is_infinite() {
            return (Self::infinity(bf.sign()), Status::empty());
        }
        if bf.is_zero() {
            return (Self::zero(bf.sign()), Status::empty());
        }
        let decimal_str = bf.to_decimal_string(format).unwrap_or_else(|| "0".into());
        match Self::parse_decimal(&decimal_str) {
            Ok(dec) => dec.round_to_format(format),
            Err(_) => (Self::nan(), Status::INVALID_OP),
        }
    }

    /// Returns this `BigDecimal` with the sign flipped.
    ///
    /// Corresponds to `bfdec_neg` in libbf.
    pub fn neg(&self) -> Self {
        let mut value = self.clone();
        value.sign = -value.sign;
        value
    }

    /// Negates this value in place.
    pub fn neg_mut(&mut self) {
        self.sign = -self.sign;
    }

    /// Sets this value to the given `i64`, reusing the existing allocation.
    pub fn set_i64(&mut self, v: i64) {
        *self = Self::from_i64(v);
    }

    /// In-place addition. Returns the status flags.
    pub fn add_assign(&mut self, rhs: &Self, format: BigFormat) -> Status {
        let (result, status) = self.add_status(rhs, format);
        *self = result;
        status
    }

    /// In-place subtraction. Returns the status flags.
    pub fn sub_assign(&mut self, rhs: &Self, format: BigFormat) -> Status {
        let (result, status) = self.sub_status(rhs, format);
        *self = result;
        status
    }

    /// In-place multiplication. Returns the status flags.
    pub fn mul_assign(&mut self, rhs: &Self, format: BigFormat) -> Status {
        let (result, status) = self.mul_status(rhs, format);
        *self = result;
        status
    }

    /// In-place division. Returns the status flags.
    pub fn div_assign(&mut self, rhs: &Self, format: BigFormat) -> Status {
        let (result, status) = self.div_status(rhs, format);
        *self = result;
        status
    }

    /// Sets `self` to `a + b`, reusing the existing allocation. Returns the status flags.
    pub fn set_add(&mut self, a: &Self, b: &Self, format: BigFormat) -> Status {
        let (result, status) = a.add_status(b, format);
        *self = result;
        status
    }

    /// Sets `self` to `a - b`, reusing the existing allocation. Returns the status flags.
    pub fn set_sub(&mut self, a: &Self, b: &Self, format: BigFormat) -> Status {
        let (result, status) = a.sub_status(b, format);
        *self = result;
        status
    }

    /// Sets `self` to `a * b`, reusing the existing allocation. Returns the status flags.
    pub fn set_mul(&mut self, a: &Self, b: &Self, format: BigFormat) -> Status {
        let (result, status) = a.mul_status(b, format);
        *self = result;
        status
    }

    /// Sets `self` to `a / b`, reusing the existing allocation. Returns the status flags.
    pub fn set_div(&mut self, a: &Self, b: &Self, format: BigFormat) -> Status {
        let (result, status) = a.div_status(b, format);
        *self = result;
        status
    }

    /// In-place multiplication by an `i64`. Returns the status flags.
    pub fn mul_i64_assign(&mut self, b: i64, format: BigFormat) -> Status {
        let bf_b = Self::from_i64(b);
        self.mul_assign(&bf_b, format)
    }

    /// In-place addition of an `i64`. Returns the status flags.
    pub fn add_i64_assign(&mut self, b: i64, format: BigFormat) -> Status {
        let bf_b = Self::from_i64(b);
        self.add_assign(&bf_b, format)
    }

    /// In-place squaring (`self = self * self`). Returns the status flags.
    ///
    /// See [`BigFloat::sqr_assign`](crate::BigFloat::sqr_assign) for why this method exists.
    ///
    /// ```
    /// use libbeef::{BigDecimal, BigFormat};
    ///
    /// let mut x = BigDecimal::from_i64(7);
    /// x.sqr_assign(BigFormat::DECIMAL64);
    /// assert_eq!(x.get_int32(), Some(49));
    /// ```
    pub fn sqr_assign(&mut self, format: BigFormat) -> Status {
        let (result, status) = self.sqr_status(format);
        *self = result;
        status
    }

    /// Reverse subtraction: sets `self` to `lhs - self`. Returns the status flags.
    ///
    /// See [`BigFloat::rsub_assign`](crate::BigFloat::rsub_assign) for why this method exists.
    ///
    /// ```
    /// use libbeef::{BigDecimal, BigFormat};
    ///
    /// let mut x = BigDecimal::from_i64(3);
    /// let ten = BigDecimal::from_i64(10);
    /// x.rsub_assign(&ten, BigFormat::DECIMAL64);
    /// assert_eq!(x.get_int32(), Some(7));
    /// ```
    pub fn rsub_assign(&mut self, lhs: &Self, format: BigFormat) -> Status {
        let (result, status) = lhs.sub_status(self, format);
        *self = result;
        status
    }

    /// Reverse division: sets `self` to `dividend / self`. Returns the status flags.
    ///
    /// See [`BigFloat::rdiv_assign`](crate::BigFloat::rdiv_assign) for why this method exists.
    ///
    /// ```
    /// use libbeef::{BigDecimal, BigFormat};
    ///
    /// let mut x = BigDecimal::from_i64(4);
    /// let twenty = BigDecimal::from_i64(20);
    /// x.rdiv_assign(&twenty, BigFormat::DECIMAL64);
    /// assert_eq!(x.get_int32(), Some(5));
    /// ```
    pub fn rdiv_assign(&mut self, dividend: &Self, format: BigFormat) -> Status {
        let (result, status) = dividend.div_status(self, format);
        *self = result;
        status
    }

    /// In-place square root. Returns the status flags.
    pub fn sqrt_assign(&mut self, format: BigFormat) -> Status {
        let (result, status) = self.sqrt_status(format);
        *self = result;
        status
    }

    /// Returns `true` if this value is NaN.
    ///
    /// Corresponds to `bfdec_is_nan` in libbf.
    pub const fn is_nan(&self) -> bool {
        self.exp == RAW_EXP_NAN
    }

    /// Returns `true` if this value is positive or negative infinity.
    ///
    /// No direct libbf equivalent (libbf uses `bfdec_is_finite` for the inverse check).
    pub const fn is_infinite(&self) -> bool {
        self.exp == RAW_EXP_INF
    }

    /// Returns `true` if this value is positive or negative zero.
    ///
    /// Corresponds to `bfdec_is_zero` in libbf.
    pub const fn is_zero(&self) -> bool {
        self.exp == RAW_EXP_ZERO
    }

    fn classify(&self) -> DecCategory {
        match self.exp {
            RAW_EXP_NAN => DecCategory::Nan,
            RAW_EXP_INF => DecCategory::Infinite,
            RAW_EXP_ZERO => DecCategory::Zero,
            _ => DecCategory::Normal,
        }
    }

    fn from_scaled_decimal_limbs(mut value: ScaledDecimalLimbs) -> Self {
        normalize_scaled_decimal_limbs(&mut value);
        if value.limbs.is_empty() {
            return Self::new();
        }
        let binary = decimal_limbs_to_binary_abs(&value.limbs);
        let mut out = Self::from_binary_abs_limbs(binary, value.sign);
        out.scale = value.scale.max(0);
        out
    }

    fn from_binary_abs_limbs(mut limbs: Vec<u64>, sign: Sign) -> Self {
        trim_binary_limbs(&mut limbs);
        if limbs.is_empty() {
            return Self::zero(sign);
        }
        let top = *limbs.last().expect("non-empty limbs");
        let bit_len = ((limbs.len() - 1) as i64) * 64 + (64_i64 - i64::from(top.leading_zeros()));
        let shift = top.leading_zeros();
        if shift != 0 {
            shl_binary_limbs(&mut limbs, shift);
        }
        remove_low_zero_binary_limbs(&mut limbs);
        Self {
            sign,
            exp: bit_len,
            limbs,
            scale: 0,
        }
    }

    fn to_scaled_decimal_limbs(&self) -> Option<ScaledDecimalLimbs> {
        if self.is_zero() {
            return Some(ScaledDecimalLimbs {
                sign: Sign::Positive,
                limbs: Vec::new(),
                scale: 0,
            });
        }
        if self.is_nan() || self.is_infinite() {
            return None;
        }
        let binary = self.to_binary_abs_int_limbs()?;
        let limbs = binary_abs_to_decimal_limbs(binary);
        Some(ScaledDecimalLimbs {
            sign: self.sign,
            limbs,
            scale: self.scale,
        })
    }

    fn to_binary_abs_int_limbs(&self) -> Option<Vec<u64>> {
        if self.is_zero() {
            return Some(Vec::new());
        }
        if self.is_nan() || self.is_infinite() || self.exp < 0 || self.limbs.is_empty() {
            return None;
        }

        let total_bits = (self.limbs.len() as i64) * 64;
        let shift = total_bits - self.exp;
        let mut limbs = self.limbs.clone();
        if shift > 0 {
            if !shr_binary_limbs_exact(&mut limbs, shift as u32) {
                return None;
            }
        } else if shift < 0 {
            shl_binary_limbs(&mut limbs, (-shift) as u32);
        }
        trim_binary_limbs(&mut limbs);
        Some(limbs)
    }

    fn round_to_format(self, format: BigFormat) -> (Self, Status) {
        let Precision::Digits(digits) = format.precision else {
            return (self, Status::empty());
        };
        let value = self
            .to_scaled_decimal_limbs()
            .expect("finite BigDecimal values are representable as scaled decimal limbs");
        let (rounded, inexact) = round_scaled_decimal_limbs(value, digits, format.rounding);
        let status = if inexact {
            Status::INEXACT
        } else {
            Status::empty()
        };
        (Self::from_scaled_decimal_limbs(rounded), status)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecCategory {
    Nan,
    Infinite,
    Zero,
    Normal,
}

const DEC_LIMB_DIGITS: i32 = 19;
const DEC_BASE: u64 = 10_000_000_000_000_000_000;
const DEC_POW10: [u64; 20] = [
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct ScaledDecimalLimbs {
    sign: Sign,
    limbs: Vec<u64>,
    scale: i32,
}

fn add_scaled_decimal_limbs(a: &ScaledDecimalLimbs, b: &ScaledDecimalLimbs) -> ScaledDecimalLimbs {
    let scale = a.scale.max(b.scale);
    let mut lhs = a.limbs.clone();
    let mut rhs = b.limbs.clone();
    mul_pow10_dec_limbs(&mut lhs, scale - a.scale);
    mul_pow10_dec_limbs(&mut rhs, scale - b.scale);

    let (sign, limbs) = if a.sign == b.sign {
        (a.sign, add_abs_dec_limbs(&lhs, &rhs))
    } else {
        match cmp_abs_dec_limbs(&lhs, &rhs) {
            core::cmp::Ordering::Greater => (a.sign, sub_abs_dec_limbs(&lhs, &rhs)),
            core::cmp::Ordering::Less => (b.sign, sub_abs_dec_limbs(&rhs, &lhs)),
            core::cmp::Ordering::Equal => (Sign::Positive, Vec::new()),
        }
    };
    let mut out = ScaledDecimalLimbs { sign, limbs, scale };
    normalize_scaled_decimal_limbs(&mut out);
    out
}

fn mul_scaled_decimal_limbs(a: &ScaledDecimalLimbs, b: &ScaledDecimalLimbs) -> ScaledDecimalLimbs {
    let mut out = ScaledDecimalLimbs {
        sign: a.sign * b.sign,
        limbs: mul_abs_dec_limbs(&a.limbs, &b.limbs),
        scale: a.scale.saturating_add(b.scale),
    };
    normalize_scaled_decimal_limbs(&mut out);
    out
}

fn div_scaled_decimal_limbs(
    a: &ScaledDecimalLimbs,
    b: &ScaledDecimalLimbs,
    format: BigFormat,
) -> (BigDecimal, Status) {
    assert!(!b.limbs.is_empty(), "division by zero");
    let sign = a.sign * b.sign;

    if matches!(format.precision, Precision::Infinite) {
        let denominator = scaled_decimal_div_denominator(b, a.scale);
        let denominator_bin = decimal_limbs_to_binary_abs(&denominator);
        let max_scale = dec_digit_count_limbs(&denominator)
            .saturating_add(u64::try_from(a.scale.max(0)).unwrap_or(0))
            .saturating_add(1);
        for result_scale in 0..=max_scale {
            let Some(scale_shift) = i32::try_from(result_scale)
                .ok()
                .and_then(|scale| b.scale.checked_add(scale))
            else {
                break;
            };
            let numerator = scaled_decimal_div_numerator(a, scale_shift);
            let (q_bin, r_bin) =
                div_rem_abs_limbs(&decimal_limbs_to_binary_abs(&numerator), &denominator_bin);
            if r_bin.is_empty() {
                let Some(scale) = i32::try_from(result_scale).ok() else {
                    return (BigDecimal::nan(), Status::INVALID_OP);
                };
                return (
                    BigDecimal::from_scaled_decimal_limbs(ScaledDecimalLimbs {
                        sign,
                        limbs: binary_abs_to_decimal_limbs(q_bin),
                        scale,
                    }),
                    Status::empty(),
                );
            }
        }
        return (BigDecimal::nan(), Status::INVALID_OP);
    }

    let Precision::Digits(digits) = format.precision else {
        return (BigDecimal::nan(), Status::INVALID_OP);
    };
    if digits == 0 {
        return (BigDecimal::zero(sign), Status::INEXACT);
    }

    let target_digits = digits.saturating_add(2);
    let Some(mut result_scale) = i32::try_from(target_digits).ok() else {
        return (BigDecimal::nan(), Status::INVALID_OP);
    };
    let denominator = scaled_decimal_div_denominator(b, a.scale);
    let denominator_bin = decimal_limbs_to_binary_abs(&denominator);

    loop {
        let Some(scale_shift) = b.scale.checked_add(result_scale) else {
            return (BigDecimal::nan(), Status::INVALID_OP);
        };
        let numerator = scaled_decimal_div_numerator(a, scale_shift);
        let (q_bin, r_bin) =
            div_rem_abs_limbs(&decimal_limbs_to_binary_abs(&numerator), &denominator_bin);
        let quotient = binary_abs_to_decimal_limbs(q_bin);
        let quotient_digits = dec_digit_count_limbs(&quotient);
        if !quotient.is_empty() && quotient_digits >= target_digits {
            let (rounded, round_inexact) = round_scaled_decimal_limbs_with_sticky(
                ScaledDecimalLimbs {
                    sign,
                    limbs: quotient,
                    scale: result_scale,
                },
                digits,
                format.rounding,
                !r_bin.is_empty(),
            );
            let status = if round_inexact || !r_bin.is_empty() {
                Status::INEXACT
            } else {
                Status::empty()
            };
            return (BigDecimal::from_scaled_decimal_limbs(rounded), status);
        }
        let Some(next_scale) = result_scale.checked_add(1) else {
            return (BigDecimal::nan(), Status::INVALID_OP);
        };
        result_scale = next_scale;
    }
}

fn sqrt_scaled_decimal_limbs(
    value: &ScaledDecimalLimbs,
    format: BigFormat,
) -> (BigDecimal, Status) {
    debug_assert!(!value.sign.is_negative());
    let Precision::Digits(digits) = format.precision else {
        return (BigDecimal::nan(), Status::INVALID_OP);
    };

    // Single computation at working precision. sqrt(v * 10^(2m)) ==
    // sqrt(v) * 10^m exactly, so the higher-scale remainder is empty iff the
    // square root is exact — no separate exact-precision pass is needed.
    let exact_scale = (value.scale + 1) / 2;
    let Some(digits_i32) = i32::try_from(digits).ok() else {
        return (BigDecimal::nan(), Status::INVALID_OP);
    };
    let Some(out_scale) = exact_scale
        .checked_add(digits_i32)
        .and_then(|scale| scale.checked_add(2))
    else {
        return (BigDecimal::nan(), Status::INVALID_OP);
    };
    let Some(adjust) = out_scale
        .checked_mul(2)
        .and_then(|scale| scale.checked_sub(value.scale))
    else {
        return (BigDecimal::nan(), Status::INVALID_OP);
    };
    let mut input = value.limbs.clone();
    mul_pow10_dec_limbs(&mut input, adjust);
    let (root, rem) = sqrt_decimal_abs_limbs(&input);
    let (rounded, status) = BigDecimal::from_scaled_decimal_limbs(ScaledDecimalLimbs {
        sign: Sign::Positive,
        limbs: root,
        scale: out_scale,
    })
    .round_to_format(format);
    if rem.is_empty() {
        (rounded, status)
    } else {
        (rounded, status | Status::INEXACT)
    }
}

fn sqrt_decimal_abs_limbs(limbs: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let input = decimal_limbs_to_binary_abs(limbs);
    let (root, rem) = sqrt_rem_abs_limbs(&input);
    (
        binary_abs_to_decimal_limbs(root),
        binary_abs_to_decimal_limbs(rem),
    )
}

fn scaled_decimal_div_numerator(value: &ScaledDecimalLimbs, extra_scale: i32) -> Vec<u64> {
    let mut out = value.limbs.clone();
    mul_pow10_dec_limbs(&mut out, extra_scale);
    out
}

fn scaled_decimal_div_denominator(value: &ScaledDecimalLimbs, extra_scale: i32) -> Vec<u64> {
    let mut out = value.limbs.clone();
    mul_pow10_dec_limbs(&mut out, extra_scale);
    out
}

fn divrem_scaled_decimal_limbs(
    a: &ScaledDecimalLimbs,
    b: &ScaledDecimalLimbs,
    mode: DivRemMode,
) -> (ScaledDecimalLimbs, ScaledDecimalLimbs) {
    assert!(!b.limbs.is_empty(), "division by zero");

    let scale = a.scale.max(b.scale);
    let mut lhs = a.limbs.clone();
    let mut rhs = b.limbs.clone();
    mul_pow10_dec_limbs(&mut lhs, scale - a.scale);
    mul_pow10_dec_limbs(&mut rhs, scale - b.scale);
    trim_dec_limbs(&mut lhs);
    trim_dec_limbs(&mut rhs);

    let lhs_bin = decimal_limbs_to_binary_abs(&lhs);
    let rhs_bin = decimal_limbs_to_binary_abs(&rhs);
    let (q_bin, r_bin) = div_rem_abs_limbs(&lhs_bin, &rhs_bin);
    let mut q_abs = binary_abs_to_decimal_limbs(q_bin);
    let mut r_abs = binary_abs_to_decimal_limbs(r_bin);

    let q_negative = a.sign.is_negative() ^ b.sign.is_negative();
    let mut q_sign = Sign::from_negative(q_negative);
    let mut r_sign = a.sign;

    if !r_abs.is_empty() {
        let is_ceil = match mode {
            DivRemMode::TowardZero => false,
            DivRemMode::NearestEven => {
                let twice_r = mul_abs_dec_limbs(&r_abs, &[2]);
                match cmp_abs_dec_limbs(&twice_r, &rhs) {
                    core::cmp::Ordering::Greater => true,
                    core::cmp::Ordering::Equal => q_abs.first().is_some_and(|limb| limb & 1 != 0),
                    core::cmp::Ordering::Less => false,
                }
            }
            DivRemMode::Floor => q_negative,
            DivRemMode::Euclidean => a.sign.is_negative(),
        };

        if is_ceil {
            add_small_dec_limbs(&mut q_abs, 1);
            r_abs = sub_abs_dec_limbs(&rhs, &r_abs);
            r_sign = if r_sign.is_negative() {
                Sign::Positive
            } else {
                Sign::Negative
            };
        }
    }

    trim_dec_limbs(&mut q_abs);
    trim_dec_limbs(&mut r_abs);
    if q_abs.is_empty() {
        q_sign = Sign::Positive;
    }
    if r_abs.is_empty() {
        r_sign = Sign::Positive;
    }

    (
        ScaledDecimalLimbs {
            sign: q_sign,
            limbs: q_abs,
            scale: 0,
        },
        ScaledDecimalLimbs {
            sign: r_sign,
            limbs: r_abs,
            scale,
        },
    )
}

fn rint_scaled_decimal_limbs(
    mut value: ScaledDecimalLimbs,
    rounding: Rounding,
) -> (ScaledDecimalLimbs, bool) {
    normalize_scaled_decimal_limbs(&mut value);
    if value.limbs.is_empty() || value.scale <= 0 {
        value.scale = 0;
        return (value, false);
    }

    let drop_digits = value.scale as u64;
    let guard_pos = drop_digits - 1;
    let guard = get_dec_digit(&value.limbs, guard_pos);
    let sticky = guard_pos > 0 && scan_dec_digit_nz(&value.limbs, guard_pos - 1);
    let retained_odd = get_dec_digit(&value.limbs, drop_digits) & 1 != 0;
    let inexact = guard != 0 || sticky;

    let increment = inexact
        && match rounding {
            Rounding::NearestEven => guard > 5 || (guard == 5 && (sticky || retained_odd)),
            Rounding::NearestAway => guard >= 5,
            Rounding::TowardZero | Rounding::Faithful => false,
            Rounding::TowardPositive => !value.sign.is_negative(),
            Rounding::TowardNegative => value.sign.is_negative(),
            Rounding::AwayFromZero => true,
        };

    div_pow10_dec_limbs(&mut value.limbs, drop_digits);
    if increment {
        add_small_dec_limbs(&mut value.limbs, 1);
    }
    value.scale = 0;
    normalize_scaled_decimal_limbs(&mut value);
    (value, inexact)
}

fn round_scaled_decimal_limbs(
    value: ScaledDecimalLimbs,
    digits: u64,
    rounding: Rounding,
) -> (ScaledDecimalLimbs, bool) {
    round_scaled_decimal_limbs_with_sticky(value, digits, rounding, false)
}

fn round_scaled_decimal_limbs_with_sticky(
    mut value: ScaledDecimalLimbs,
    digits: u64,
    rounding: Rounding,
    sticky_below: bool,
) -> (ScaledDecimalLimbs, bool) {
    normalize_scaled_decimal_limbs(&mut value);
    if value.limbs.is_empty() || digits == 0 {
        return (value, sticky_below);
    }

    let digit_count = dec_digit_count_limbs(&value.limbs);
    if digit_count <= digits {
        if !sticky_below {
            return (value, false);
        }
        let increment = match rounding {
            Rounding::TowardZero
            | Rounding::Faithful
            | Rounding::NearestEven
            | Rounding::NearestAway => false,
            Rounding::TowardPositive => !value.sign.is_negative(),
            Rounding::TowardNegative => value.sign.is_negative(),
            Rounding::AwayFromZero => true,
        };
        if increment {
            add_small_dec_limbs(&mut value.limbs, 1);
        }
        normalize_scaled_decimal_limbs(&mut value);
        return (value, true);
    }

    let drop_digits = digit_count - digits;
    let guard_pos = drop_digits - 1;
    let guard = get_dec_digit(&value.limbs, guard_pos);
    let sticky = sticky_below || (guard_pos > 0 && scan_dec_digit_nz(&value.limbs, guard_pos - 1));
    let retained_odd = get_dec_digit(&value.limbs, drop_digits) & 1 != 0;
    let inexact = guard != 0 || sticky;
    if !inexact {
        return (value, false);
    }

    let increment = match rounding {
        Rounding::NearestEven => guard > 5 || (guard == 5 && (sticky || retained_odd)),
        Rounding::NearestAway => guard >= 5,
        Rounding::TowardZero | Rounding::Faithful => false,
        Rounding::TowardPositive => !value.sign.is_negative(),
        Rounding::TowardNegative => value.sign.is_negative(),
        Rounding::AwayFromZero => true,
    };

    div_pow10_dec_limbs(&mut value.limbs, drop_digits);
    if increment {
        add_small_dec_limbs(&mut value.limbs, 1);
    }

    if let Ok(drop_i32) = i32::try_from(drop_digits) {
        if value.scale >= drop_i32 {
            value.scale -= drop_i32;
        } else {
            let mul = drop_i32 - value.scale;
            mul_pow10_dec_limbs(&mut value.limbs, mul);
            value.scale = 0;
        }
    } else {
        let mul = drop_digits.saturating_sub(u64::try_from(value.scale.max(0)).unwrap_or(0));
        mul_pow10_dec_limbs_u64(&mut value.limbs, mul);
        value.scale = 0;
    }
    normalize_scaled_decimal_limbs(&mut value);
    (value, true)
}

fn normalize_scaled_decimal_limbs(value: &mut ScaledDecimalLimbs) {
    trim_dec_limbs(&mut value.limbs);
    if value.limbs.is_empty() {
        value.sign = Sign::Positive;
        value.scale = 0;
        return;
    }
    while value.scale > 0 {
        let Some(first) = value.limbs.first_mut() else {
            break;
        };
        if *first % 10 != 0 {
            break;
        }
        let mut rem = 0_u128;
        for limb in value.limbs.iter_mut().rev() {
            let current = rem * u128::from(DEC_BASE) + u128::from(*limb);
            *limb = (current / 10) as u64;
            rem = current % 10;
        }
        debug_assert_eq!(rem, 0);
        value.scale -= 1;
        trim_dec_limbs(&mut value.limbs);
    }
}

fn dec_digit_count_limbs(limbs: &[u64]) -> u64 {
    let mut len = limbs.len();
    while len > 0 && limbs[len - 1] == 0 {
        len -= 1;
    }
    if len == 0 {
        return 0;
    }
    ((len - 1) as u64) * (DEC_LIMB_DIGITS as u64) + dec_limb_digit_count(limbs[len - 1])
}

fn dec_limb_digit_count(value: u64) -> u64 {
    debug_assert!(value < DEC_BASE);
    if value == 0 {
        0
    } else {
        value.ilog10() as u64 + 1
    }
}

fn get_dec_digit(limbs: &[u64], pos: u64) -> u64 {
    let limb = (pos / DEC_LIMB_DIGITS as u64) as usize;
    let shift = (pos % DEC_LIMB_DIGITS as u64) as usize;
    limbs
        .get(limb)
        .map_or(0, |value| (value / DEC_POW10[shift]) % 10)
}

fn scan_dec_digit_nz(limbs: &[u64], pos: u64) -> bool {
    let limb = (pos / DEC_LIMB_DIGITS as u64) as usize;
    let shift = (pos % DEC_LIMB_DIGITS as u64) as usize;
    if let Some(&value) = limbs.get(limb) {
        if value % DEC_POW10[shift + 1] != 0 {
            return true;
        }
    }
    limbs.iter().take(limb).any(|&value| value != 0)
}

fn div_pow10_dec_limbs(limbs: &mut Vec<u64>, digits: u64) {
    if limbs.is_empty() || digits == 0 {
        return;
    }
    let whole = (digits / DEC_LIMB_DIGITS as u64) as usize;
    let rem = (digits % DEC_LIMB_DIGITS as u64) as usize;
    if whole > 0 {
        if whole >= limbs.len() {
            limbs.clear();
            return;
        }
        limbs.drain(..whole);
    }
    if rem > 0 {
        div_rem_dec_small(limbs, DEC_POW10[rem]);
    }
    trim_dec_limbs(limbs);
}

fn div_rem_dec_small(limbs: &mut Vec<u64>, divisor: u64) -> u64 {
    debug_assert!(divisor > 0 && divisor <= DEC_BASE);
    let mut rem = 0_u128;
    for limb in limbs.iter_mut().rev() {
        let current = rem * u128::from(DEC_BASE) + u128::from(*limb);
        *limb = (current / u128::from(divisor)) as u64;
        rem = current % u128::from(divisor);
    }
    trim_dec_limbs(limbs);
    rem as u64
}

fn add_small_dec_limbs(limbs: &mut Vec<u64>, value: u64) {
    if value == 0 {
        return;
    }
    let mut carry = value;
    for limb in limbs.iter_mut() {
        let sum = *limb + carry;
        if sum >= DEC_BASE {
            *limb = sum - DEC_BASE;
            carry = 1;
        } else {
            *limb = sum;
            return;
        }
    }
    limbs.push(carry);
}

fn push_dec_digit(limbs: &mut Vec<u64>, digit: u64) {
    debug_assert!(digit < 10);
    if limbs.is_empty() && digit == 0 {
        return;
    }
    mul_pow10_dec_limbs(limbs, 1);
    add_small_dec_limbs(limbs, digit);
}

fn cmp_abs_dec_limbs(a: &[u64], b: &[u64]) -> core::cmp::Ordering {
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

fn add_abs_dec_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = long.to_vec();
    let mut carry = if short.is_empty() {
        0
    } else {
        mp_add_dec(&mut out[..short.len()], long, short, 0)
    };
    let mut index = short.len();
    while carry != 0 && index < out.len() {
        let value = out[index] + carry;
        if value >= DEC_BASE {
            out[index] = value - DEC_BASE;
            carry = 1;
            index += 1;
        } else {
            out[index] = value;
            carry = 0;
        }
    }
    if carry != 0 {
        out.push(carry);
    }
    trim_dec_limbs(&mut out);
    out
}

fn sub_abs_dec_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    debug_assert!(cmp_abs_dec_limbs(a, b) != core::cmp::Ordering::Less);
    let mut out = a.to_vec();
    let mut borrow = if b.is_empty() {
        0
    } else {
        mp_sub_dec(&mut out[..b.len()], a, b, 0)
    };
    let mut index = b.len();
    while borrow != 0 && index < out.len() {
        if out[index] == 0 {
            out[index] = DEC_BASE - 1;
            borrow = 1;
            index += 1;
        } else {
            out[index] -= 1;
            borrow = 0;
        }
    }
    debug_assert_eq!(borrow, 0);
    trim_dec_limbs(&mut out);
    out
}

fn mp_add_dec(out: &mut [u64], op1: &[u64], op2: &[u64], carry: u64) -> u64 {
    debug_assert_eq!(out.len(), op2.len());
    debug_assert!(op1.len() >= op2.len());
    let mut carry = carry;
    for i in 0..op2.len() {
        let sum = u128::from(op1[i]) + u128::from(op2[i]) + u128::from(carry);
        if sum >= u128::from(DEC_BASE) {
            out[i] = (sum - u128::from(DEC_BASE)) as u64;
            carry = 1;
        } else {
            out[i] = sum as u64;
            carry = 0;
        }
    }
    carry
}

fn mp_sub_dec(out: &mut [u64], op1: &[u64], op2: &[u64], borrow: u64) -> u64 {
    debug_assert_eq!(out.len(), op2.len());
    debug_assert!(op1.len() >= op2.len());
    let mut borrow = borrow;
    for i in 0..op2.len() {
        let rhs = op2[i] + borrow;
        if op1[i] < rhs {
            out[i] = (u128::from(op1[i]) + u128::from(DEC_BASE) - u128::from(rhs)) as u64;
            borrow = 1;
        } else {
            out[i] = op1[i] - rhs;
            borrow = 0;
        }
    }
    borrow
}

fn mul_abs_dec_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let (op1, op2) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut out = alloc::vec![0; op1.len() + op2.len()];
    mp_mul_basecase_dec(&mut out, op1, op2);
    trim_dec_limbs(&mut out);
    out
}

fn mp_mul_basecase_dec(out: &mut [u64], op1: &[u64], op2: &[u64]) {
    debug_assert_eq!(out.len(), op1.len() + op2.len());
    out[op1.len()] = mp_mul1_dec(&mut out[..op1.len()], op1, op2[0], 0);
    for i in 1..op2.len() {
        let carry = mp_add_mul1_dec(&mut out[i..], op1, op2[i]);
        out[i + op1.len()] = carry;
    }
}

fn mp_mul1_dec(out: &mut [u64], input: &[u64], multiplier: u64, carry: u64) -> u64 {
    debug_assert_eq!(out.len(), input.len());
    let mut carry = u128::from(carry);
    for (slot, &limb) in out.iter_mut().zip(input) {
        let product = u128::from(limb) * u128::from(multiplier) + carry;
        *slot = (product % u128::from(DEC_BASE)) as u64;
        carry = product / u128::from(DEC_BASE);
    }
    carry as u64
}

fn mp_add_mul1_dec(out: &mut [u64], input: &[u64], multiplier: u64) -> u64 {
    debug_assert!(out.len() >= input.len());
    let mut carry = 0_u128;
    for i in 0..input.len() {
        let product = u128::from(input[i]) * u128::from(multiplier) + u128::from(out[i]) + carry;
        out[i] = (product % u128::from(DEC_BASE)) as u64;
        carry = product / u128::from(DEC_BASE);
    }
    carry as u64
}

fn mul_pow10_dec_limbs(limbs: &mut Vec<u64>, exp: i32) {
    if limbs.is_empty() || exp <= 0 {
        return;
    }
    let whole = (exp / DEC_LIMB_DIGITS) as usize;
    let rem = (exp % DEC_LIMB_DIGITS) as usize;
    if whole > 0 {
        limbs.splice(0..0, core::iter::repeat_n(0, whole));
    }
    if rem > 0 {
        let multiplier = DEC_POW10[rem];
        let input = limbs.clone();
        let carry = mp_mul1_dec(limbs, &input, multiplier, 0);
        if carry != 0 {
            limbs.push(carry);
        }
    }
    trim_dec_limbs(limbs);
}

fn mul_pow10_dec_limbs_u64(limbs: &mut Vec<u64>, exp: u64) {
    if limbs.is_empty() || exp == 0 {
        return;
    }
    let whole = (exp / u64::try_from(DEC_LIMB_DIGITS).expect("fits")) as usize;
    let rem = (exp % u64::try_from(DEC_LIMB_DIGITS).expect("fits")) as usize;
    if whole > 0 {
        limbs.splice(0..0, core::iter::repeat_n(0, whole));
    }
    if rem > 0 {
        let multiplier = DEC_POW10[rem];
        let input = limbs.clone();
        let carry = mp_mul1_dec(limbs, &input, multiplier, 0);
        if carry != 0 {
            limbs.push(carry);
        }
    }
    trim_dec_limbs(limbs);
}

fn binary_abs_to_decimal_limbs(mut binary: Vec<u64>) -> Vec<u64> {
    trim_binary_limbs(&mut binary);
    if binary.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    while !binary.is_empty() {
        let rem = div_rem_binary_small(&mut binary, DEC_BASE);
        out.push(rem);
    }
    trim_dec_limbs(&mut out);
    out
}

fn decimal_limbs_to_binary_abs(limbs: &[u64]) -> Vec<u64> {
    let mut out = Vec::new();
    for &limb in limbs.iter().rev() {
        mul_binary_small(&mut out, DEC_BASE);
        add_binary_small(&mut out, limb);
    }
    trim_binary_limbs(&mut out);
    out
}

fn decimal_limbs_to_string(limbs: &[u64]) -> String {
    let mut len = limbs.len();
    while len > 0 && limbs[len - 1] == 0 {
        len -= 1;
    }
    if len == 0 {
        return "0".into();
    }

    let mut out = limbs[len - 1].to_string();
    for &limb in limbs[..len - 1].iter().rev() {
        push_padded_dec_limb(&mut out, limb);
    }
    out
}

fn push_padded_dec_limb(out: &mut String, limb: u64) {
    debug_assert!(limb < DEC_BASE);
    let digits = limb.to_string();
    for _ in digits.len()..(DEC_LIMB_DIGITS as usize) {
        out.push('0');
    }
    out.push_str(&digits);
}

fn trim_dec_limbs(limbs: &mut Vec<u64>) {
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
}

fn mul_binary_small(limbs: &mut Vec<u64>, multiplier: u64) {
    if limbs.is_empty() || multiplier == 1 {
        return;
    }
    if multiplier == 0 {
        limbs.clear();
        return;
    }
    let mut carry = 0_u128;
    for limb in limbs.iter_mut() {
        let product = u128::from(*limb) * u128::from(multiplier) + carry;
        *limb = product as u64;
        carry = product >> 64;
    }
    if carry != 0 {
        limbs.push(carry as u64);
    }
}

fn add_binary_small(limbs: &mut Vec<u64>, value: u64) {
    if value == 0 {
        return;
    }
    let mut carry = value;
    for limb in limbs.iter_mut() {
        let (sum, overflow) = limb.overflowing_add(carry);
        *limb = sum;
        if overflow {
            carry = 1;
        } else {
            return;
        }
    }
    limbs.push(carry);
}

fn div_rem_binary_small(limbs: &mut Vec<u64>, divisor: u64) -> u64 {
    debug_assert_ne!(divisor, 0);
    let mut rem = 0_u128;
    for limb in limbs.iter_mut().rev() {
        let current = (rem << 64) | u128::from(*limb);
        *limb = (current / u128::from(divisor)) as u64;
        rem = current % u128::from(divisor);
    }
    trim_binary_limbs(limbs);
    rem as u64
}

fn shl_binary_limbs(limbs: &mut Vec<u64>, shift: u32) {
    if limbs.is_empty() || shift == 0 {
        return;
    }
    let word_shift = (shift / 64) as usize;
    let bit_shift = shift % 64;
    if word_shift > 0 {
        limbs.splice(0..0, core::iter::repeat_n(0, word_shift));
    }
    if bit_shift == 0 {
        return;
    }
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

fn shr_binary_limbs_exact(limbs: &mut Vec<u64>, shift: u32) -> bool {
    if limbs.is_empty() || shift == 0 {
        return true;
    }
    let word_shift = (shift / 64) as usize;
    let bit_shift = shift % 64;
    if word_shift > 0 {
        if limbs.iter().take(word_shift).any(|&limb| limb != 0) {
            return false;
        }
        limbs.drain(..word_shift);
    }
    if bit_shift != 0 {
        let mask = (1_u64 << bit_shift) - 1;
        if limbs.first().is_some_and(|limb| (limb & mask) != 0) {
            return false;
        }
        let mut carry = 0_u64;
        for limb in limbs.iter_mut().rev() {
            let next = *limb << (64 - bit_shift);
            *limb = (*limb >> bit_shift) | carry;
            carry = next;
        }
    }
    trim_binary_limbs(limbs);
    true
}

fn trim_binary_limbs(limbs: &mut Vec<u64>) {
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
}

fn remove_low_zero_binary_limbs(limbs: &mut Vec<u64>) {
    let first_non_zero = limbs
        .iter()
        .position(|&limb| limb != 0)
        .unwrap_or(limbs.len());
    if first_non_zero > 0 {
        limbs.drain(..first_non_zero);
    }
}

impl fmt::Display for BigDecimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nan() {
            return f.pad("NaN");
        }
        if self.is_infinite() {
            return f.pad(if self.sign.is_negative() {
                "-inf"
            } else {
                "inf"
            });
        }
        let Some(value) = self.to_scaled_decimal_limbs() else {
            return f.pad("0");
        };
        if value.limbs.is_empty() {
            return f.pad("0");
        }
        let digits = decimal_limbs_to_string(&value.limbs);
        let mut buf = String::new();
        if value.sign.is_negative() {
            buf.push('-');
        }
        if value.scale > 0 {
            let scale = value.scale as usize;
            if digits.len() <= scale {
                buf.push_str("0.");
                for _ in 0..(scale - digits.len()) {
                    buf.push('0');
                }
                buf.push_str(&digits);
            } else {
                buf.push_str(&digits[..digits.len() - scale]);
                buf.push('.');
                buf.push_str(&digits[digits.len() - scale..]);
            }
        } else {
            buf.push_str(&digits);
        }
        f.pad(&buf)
    }
}

/// Parses a decimal floating-point literal.
///
/// Accepts the same syntax as [`BigDecimal::parse_decimal`]: integers,
/// decimals with `.`, scientific notation with `e`/`E`, and the special
/// values `nan`, `inf`, `+inf`, `-inf` (case-insensitive).
///
/// ```
/// use libbeef::BigDecimal;
///
/// let v: BigDecimal = "3.14".parse().unwrap();
/// assert_eq!(v.to_decimal_string().unwrap(), "3.14");
/// ```
impl FromStr for BigDecimal {
    type Err = ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_decimal(s)
    }
}

impl From<i8> for BigDecimal {
    fn from(value: i8) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl From<i16> for BigDecimal {
    fn from(value: i16) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl From<i32> for BigDecimal {
    fn from(value: i32) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl From<i64> for BigDecimal {
    fn from(value: i64) -> Self {
        Self::from_i64(value)
    }
}

impl From<u8> for BigDecimal {
    fn from(value: u8) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl From<u16> for BigDecimal {
    fn from(value: u16) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl From<u32> for BigDecimal {
    fn from(value: u32) -> Self {
        Self::from_i64(i64::from(value))
    }
}

impl From<i128> for BigDecimal {
    fn from(value: i128) -> Self {
        if value == 0 {
            return Self::new();
        }
        let sign = Sign::from_negative(value < 0);
        let mag = value.unsigned_abs();
        let lo = mag as u64;
        let hi = (mag >> 64) as u64;
        let mut limbs = alloc::vec![lo];
        if hi != 0 {
            limbs.push(hi);
        }
        Self::from_binary_abs_limbs(limbs, sign)
    }
}

impl From<u128> for BigDecimal {
    fn from(value: u128) -> Self {
        if value == 0 {
            return Self::new();
        }
        let lo = value as u64;
        let hi = (value >> 64) as u64;
        let mut limbs = alloc::vec![lo];
        if hi != 0 {
            limbs.push(hi);
        }
        Self::from_binary_abs_limbs(limbs, Sign::Positive)
    }
}

impl Default for BigDecimal {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BigDecimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.exp {
            RAW_EXP_NAN => f.write_str("BigDecimal(NaN)"),
            RAW_EXP_INF => {
                if self.sign.is_negative() {
                    f.write_str("BigDecimal(-Inf)")
                } else {
                    f.write_str("BigDecimal(Inf)")
                }
            }
            RAW_EXP_ZERO => {
                if self.sign.is_negative() {
                    f.write_str("BigDecimal(-0)")
                } else {
                    f.write_str("BigDecimal(0)")
                }
            }
            _ => f
                .debug_struct("BigDecimal")
                .field("sign", &self.sign)
                .field("exp", &self.exp)
                .field("limbs", &self.limbs)
                .field("scale", &self.scale)
                .finish(),
        }
    }
}

impl PartialEq for BigDecimal {
    fn eq(&self, other: &Self) -> bool {
        if self.is_nan() || other.is_nan() {
            return false;
        }
        if self.is_zero() && other.is_zero() {
            return true;
        }
        self.sign == other.sign
            && self.exp == other.exp
            && self.limbs == other.limbs
            && self.scale == other.scale
    }
}

/// A decimal floating-point number with a format pinned at the type level.
///
/// Analogous to [`Float<F>`](crate::Float) for binary floats. The type
/// parameter `F` implements [`StaticFormat`] and determines precision and
/// rounding for arithmetic operations.
///
/// ```
/// use libbeef::decimal::Decimal;
/// use libbeef::formats::Decimal64;
/// use libbeef::BigDecimal;
///
/// let x = Decimal::<Decimal64>::from_big(BigDecimal::from_i64(10));
/// let y = Decimal::<Decimal64>::from_big(BigDecimal::from_i64(3));
/// let q = x.div(&y);
/// ```
pub struct Decimal<F> {
    value: BigDecimal,
    _format: PhantomData<F>,
}

impl<F> Clone for Decimal<F> {
    fn clone(&self) -> Self {
        Self::from_big(self.value.clone())
    }
}

impl<F> Decimal<F> {
    /// Creates a new `Decimal<F>` initialized to positive zero.
    pub const fn new() -> Self {
        Self::from_big(BigDecimal::new())
    }

    /// Wraps a [`BigDecimal`] in the typed `Decimal<F>` wrapper.
    pub const fn from_big(value: BigDecimal) -> Self {
        Self {
            value,
            _format: PhantomData,
        }
    }

    /// Unwraps this `Decimal<F>` into the underlying [`BigDecimal`].
    pub fn into_big(self) -> BigDecimal {
        self.value
    }

    /// Returns a reference to the underlying [`BigDecimal`].
    pub const fn as_big(&self) -> &BigDecimal {
        &self.value
    }

    /// Returns a NaN value in this format.
    ///
    /// Delegates to [`BigDecimal::nan`].
    ///
    /// ```
    /// use libbeef::Decimal;
    /// use libbeef::formats::Decimal64;
    ///
    /// let x = Decimal::<Decimal64>::nan();
    /// assert!(x.as_big().is_nan());
    /// ```
    pub const fn nan() -> Self {
        Self::from_big(BigDecimal::nan())
    }

    /// Returns an infinity with the given sign in this format.
    ///
    /// Delegates to [`BigDecimal::infinity`].
    ///
    /// ```
    /// use libbeef::{Decimal, Sign};
    /// use libbeef::formats::Decimal64;
    ///
    /// let x = Decimal::<Decimal64>::infinity(Sign::Positive);
    /// assert!(x.as_big().is_infinite());
    /// ```
    pub const fn infinity(sign: Sign) -> Self {
        Self::from_big(BigDecimal::infinity(sign))
    }

    /// Returns a zero with the given sign in this format.
    ///
    /// Delegates to [`BigDecimal::zero`].
    ///
    /// ```
    /// use libbeef::{Decimal, Sign};
    /// use libbeef::formats::Decimal64;
    ///
    /// let x = Decimal::<Decimal64>::zero(Sign::Negative);
    /// assert!(x.as_big().is_zero());
    /// ```
    pub const fn zero(sign: Sign) -> Self {
        Self::from_big(BigDecimal::zero(sign))
    }

    /// Creates a [`Decimal`] from a signed 64-bit integer.
    ///
    /// Delegates to [`BigDecimal::from_i64`].
    ///
    /// ```
    /// use libbeef::Decimal;
    /// use libbeef::formats::Decimal64;
    ///
    /// let x = Decimal::<Decimal64>::from_i64(42);
    /// assert_eq!(x.as_big().get_int32(), Some(42));
    /// ```
    pub fn from_i64(value: i64) -> Self {
        Self::from_big(BigDecimal::from_i64(value))
    }

    /// Creates a [`Decimal`] from a scaled signed 64-bit integer.
    ///
    /// The value represented is `coefficient * 10^(-scale)`. For example,
    /// `from_scaled_i64(314, 2)` produces `3.14`.
    ///
    /// Delegates to [`BigDecimal::from_scaled_i64`].
    ///
    /// ```
    /// use libbeef::Decimal;
    /// use libbeef::formats::Decimal64;
    ///
    /// let x = Decimal::<Decimal64>::from_scaled_i64(314, 2);
    /// assert_eq!(x.as_big().to_decimal_string(), Some("3.14".into()));
    /// ```
    pub fn from_scaled_i64(coefficient: i64, scale: i32) -> Self {
        Self::from_big(BigDecimal::from_scaled_i64(coefficient, scale))
    }

    /// Converts this `Decimal<F>` into a `Decimal<G>` with a different format type.
    /// Does not re-round the value.
    pub fn into_format<G>(self) -> Decimal<G> {
        Decimal::from_big(self.value)
    }
}

impl<F: StaticFormat> Decimal<F> {
    /// Adds two decimal values using the format's precision and rounding.
    ///
    /// Delegates to [`BigDecimal::add`]. Corresponds to `bfdec_add` in libbf.
    pub fn add(&self, rhs: &Self) -> Self {
        self.add_status(rhs).0
    }

    /// Adds two decimal values. Returns the status flags as well.
    ///
    /// Delegates to [`BigDecimal::add_status`]. Corresponds to `bfdec_add` in libbf.
    pub fn add_status(&self, rhs: &Self) -> (Self, Status) {
        let (value, status) = self.value.add_status(&rhs.value, F::FORMAT);
        (Self::from_big(value), status)
    }

    /// Subtracts `rhs` from this value using the format's precision and rounding.
    ///
    /// Delegates to [`BigDecimal::sub`]. Corresponds to `bfdec_sub` in libbf.
    pub fn sub(&self, rhs: &Self) -> Self {
        self.sub_status(rhs).0
    }

    /// Subtracts `rhs` from this value. Returns the status flags as well.
    ///
    /// Delegates to [`BigDecimal::sub_status`]. Corresponds to `bfdec_sub` in libbf.
    pub fn sub_status(&self, rhs: &Self) -> (Self, Status) {
        let (value, status) = self.value.sub_status(&rhs.value, F::FORMAT);
        (Self::from_big(value), status)
    }

    /// Multiplies two decimal values using the format's precision and rounding.
    ///
    /// Delegates to [`BigDecimal::mul`]. Corresponds to `bfdec_mul` in libbf.
    pub fn mul(&self, rhs: &Self) -> Self {
        self.mul_status(rhs).0
    }

    /// Multiplies two decimal values. Returns the status flags as well.
    ///
    /// Delegates to [`BigDecimal::mul_status`]. Corresponds to `bfdec_mul` in libbf.
    pub fn mul_status(&self, rhs: &Self) -> (Self, Status) {
        let (value, status) = self.value.mul_status(&rhs.value, F::FORMAT);
        (Self::from_big(value), status)
    }

    /// Divides this value by `rhs` using the format's precision and rounding.
    ///
    /// Delegates to [`BigDecimal::div`]. Corresponds to `bfdec_div` in libbf.
    pub fn div(&self, rhs: &Self) -> Self {
        self.div_status(rhs).0
    }

    /// Divides this value by `rhs`. Returns the status flags as well.
    ///
    /// Delegates to [`BigDecimal::div_status`]. Corresponds to `bfdec_div` in libbf.
    pub fn div_status(&self, rhs: &Self) -> (Self, Status) {
        let (value, status) = self.value.div_status(&rhs.value, F::FORMAT);
        (Self::from_big(value), status)
    }

    /// Computes the remainder of dividing this value by `rhs`.
    ///
    /// Delegates to [`BigDecimal::rem`]. Corresponds to `bfdec_rem` in libbf.
    pub fn rem(&self, rhs: &Self, mode: DivRemMode) -> Self {
        self.rem_status(rhs, mode).0
    }

    /// Computes the remainder of dividing this value by `rhs`. Returns the status flags as well.
    ///
    /// Delegates to [`BigDecimal::rem_status`]. Corresponds to `bfdec_rem` in libbf.
    pub fn rem_status(&self, rhs: &Self, mode: DivRemMode) -> (Self, Status) {
        let (value, status) = self.value.rem_status(&rhs.value, F::FORMAT, mode);
        (Self::from_big(value), status)
    }

    /// Rounds this value to the nearest integer according to the given rounding mode.
    ///
    /// Delegates to [`BigDecimal::rint`]. Corresponds to `bfdec_rint` in libbf.
    pub fn rint(&self, rounding: Rounding) -> Self {
        self.rint_status(rounding).0
    }

    /// Rounds this value to the nearest integer. Returns the status flags as well.
    ///
    /// Delegates to [`BigDecimal::rint_status`]. Corresponds to `bfdec_rint` in libbf.
    pub fn rint_status(&self, rounding: Rounding) -> (Self, Status) {
        let (value, status) = self.value.rint_status(rounding);
        (Self::from_big(value), status)
    }

    /// Squares this value (`self * self`) using the format's precision and rounding.
    ///
    /// See [`BigFloat::sqr`](crate::BigFloat::sqr) for why this method exists.
    ///
    /// ```
    /// use libbeef::Decimal;
    /// use libbeef::formats::Decimal64;
    ///
    /// let x = Decimal::<Decimal64>::from_i64(7);
    /// assert_eq!(x.sqr().as_big().get_int32(), Some(49));
    /// ```
    pub fn sqr(&self) -> Self {
        self.sqr_status().0
    }

    /// Squares this value, returning the status flags as well.
    ///
    /// See [`BigFloat::sqr_status`](crate::BigFloat::sqr_status) for why this method exists.
    ///
    /// ```
    /// use libbeef::Decimal;
    /// use libbeef::formats::Decimal64;
    ///
    /// let x = Decimal::<Decimal64>::from_i64(7);
    /// let (result, status) = x.sqr_status();
    /// assert_eq!(result.as_big().get_int32(), Some(49));
    /// ```
    pub fn sqr_status(&self) -> (Self, Status) {
        let (value, status) = self.value.sqr_status(F::FORMAT);
        (Self::from_big(value), status)
    }

    /// Computes the square root using the format's precision and rounding.
    ///
    /// Delegates to [`BigDecimal::sqrt`]. Corresponds to `bfdec_sqrt` in libbf.
    pub fn sqrt(&self) -> Self {
        self.sqrt_status().0
    }

    /// Computes the square root. Returns the status flags as well.
    ///
    /// Delegates to [`BigDecimal::sqrt_status`]. Corresponds to `bfdec_sqrt` in libbf.
    pub fn sqrt_status(&self) -> (Self, Status) {
        let (value, status) = self.value.sqrt_status(F::FORMAT);
        (Self::from_big(value), status)
    }
}

impl<F: StaticFormat> FromStr for Decimal<F> {
    type Err = ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let big = BigDecimal::parse_decimal(s)?;
        let (rounded, _status) = big.round_to_format(F::FORMAT);
        Ok(Self::from_big(rounded))
    }
}

impl<F> Default for Decimal<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> fmt::Debug for Decimal<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Decimal").field(&self.value).finish()
    }
}

impl<F> fmt::Display for Decimal<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.value, f)
    }
}

impl<F> From<BigDecimal> for Decimal<F> {
    fn from(value: BigDecimal) -> Self {
        Self::from_big(value)
    }
}

impl<F> From<Decimal<F>> for BigDecimal {
    fn from(value: Decimal<F>) -> Self {
        value.into_big()
    }
}

impl<F: StaticFormat> Add for Decimal<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Decimal::add(&self, &rhs)
    }
}

impl<F: StaticFormat> AddAssign for Decimal<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = Decimal::add(self, &rhs);
    }
}

impl<F: StaticFormat> Sub for Decimal<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Decimal::sub(&self, &rhs)
    }
}

impl<F: StaticFormat> SubAssign for Decimal<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Decimal::sub(self, &rhs);
    }
}

impl<F: StaticFormat> Mul for Decimal<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Decimal::mul(&self, &rhs)
    }
}

impl<F: StaticFormat> MulAssign for Decimal<F> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Decimal::mul(self, &rhs);
    }
}

impl<F: StaticFormat> Div for Decimal<F> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Decimal::div(&self, &rhs)
    }
}

impl<F: StaticFormat> DivAssign for Decimal<F> {
    fn div_assign(&mut self, rhs: Self) {
        *self = Decimal::div(self, &rhs);
    }
}

impl<F: StaticFormat> Rem for Decimal<F> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Decimal::rem(&self, &rhs, DivRemMode::TowardZero)
    }
}

impl<F: StaticFormat> RemAssign for Decimal<F> {
    fn rem_assign(&mut self, rhs: Self) {
        *self = Decimal::rem(self, &rhs, DivRemMode::TowardZero);
    }
}

impl<F> From<i8> for Decimal<F> {
    fn from(value: i8) -> Self {
        Self::from_big(BigDecimal::from_i64(i64::from(value)))
    }
}

impl<F> From<i16> for Decimal<F> {
    fn from(value: i16) -> Self {
        Self::from_big(BigDecimal::from_i64(i64::from(value)))
    }
}

impl<F> From<i32> for Decimal<F> {
    fn from(value: i32) -> Self {
        Self::from_big(BigDecimal::from_i64(i64::from(value)))
    }
}

impl<F> From<i64> for Decimal<F> {
    fn from(value: i64) -> Self {
        Self::from_big(BigDecimal::from_i64(value))
    }
}

impl<F> From<u8> for Decimal<F> {
    fn from(value: u8) -> Self {
        Self::from_big(BigDecimal::from_i64(i64::from(value)))
    }
}

impl<F> From<u16> for Decimal<F> {
    fn from(value: u16) -> Self {
        Self::from_big(BigDecimal::from_i64(i64::from(value)))
    }
}

impl<F> From<u32> for Decimal<F> {
    fn from(value: u32) -> Self {
        Self::from_big(BigDecimal::from_i64(i64::from(value)))
    }
}

impl<F> From<i128> for Decimal<F> {
    fn from(value: i128) -> Self {
        Self::from_big(BigDecimal::from(value))
    }
}

impl<F> From<u128> for Decimal<F> {
    fn from(value: u128) -> Self {
        Self::from_big(BigDecimal::from(value))
    }
}

impl<F> Neg for Decimal<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_big(self.value.neg())
    }
}
