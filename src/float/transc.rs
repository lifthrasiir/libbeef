use crate::format::{BigFormat, Precision, Rounding};
use crate::status::Status;

use super::{BigFloat, DivRemMode, Sign};

#[cfg(feature = "std")]
struct ConstCache {
    prec: u64,
    val: BigFloat,
}

#[cfg(feature = "std")]
impl ConstCache {
    const fn new() -> Self {
        Self {
            prec: 0,
            val: BigFloat::zero(Sign::Positive),
        }
    }
}

#[cfg(feature = "std")]
fn cached_const(cache: &mut ConstCache, prec: u64, compute: fn(u64) -> BigFloat) -> BigFloat {
    if cache.prec < prec {
        cache.val = compute(prec);
        cache.prec = prec;
    }
    cache.val.clone()
}

#[cfg(feature = "std")]
fn cached_log2(prec: u64) -> BigFloat {
    use std::cell::RefCell;
    thread_local! {
        static CACHE: RefCell<ConstCache> = const { RefCell::new(ConstCache::new()) };
    }
    CACHE.with(|c| cached_const(&mut c.borrow_mut(), prec, const_log2_internal))
}

#[cfg(not(feature = "std"))]
fn cached_log2(prec: u64) -> BigFloat {
    const_log2_internal(prec)
}

#[cfg(feature = "std")]
fn cached_pi(prec: u64) -> BigFloat {
    use std::cell::RefCell;
    thread_local! {
        static CACHE: RefCell<ConstCache> = const { RefCell::new(ConstCache::new()) };
    }
    CACHE.with(|c| cached_const(&mut c.borrow_mut(), prec, const_pi_internal))
}

#[cfg(not(feature = "std"))]
fn cached_pi(prec: u64) -> BigFloat {
    const_pi_internal(prec)
}

fn isqrt(a: u64) -> u64 {
    if a == 0 {
        return 0;
    }
    // Newton's iteration from an initial overestimate; avoids `f64::sqrt`,
    // which is unavailable in `no_std`.
    let mut s = 1u64 << (64 - a.leading_zeros()).div_ceil(2);
    loop {
        let t = (s + a / s) / 2;
        if t >= s {
            return s;
        }
        s = t;
    }
}

impl BigFloat {
    /// Multiplies `self` by 2^e by adjusting the exponent.
    ///
    /// Corresponds to `bf_mul_2exp` in libbf.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat};
    /// let x = BigFloat::from_i64(3);
    /// let y = x.mul_exp2(4); // 3 * 16 = 48
    /// let f = y.to_f64(libbeef::Rounding::NearestEven);
    /// assert_eq!(f, 48.0);
    /// ```
    pub fn mul_exp2(&self, e: i64) -> Self {
        if !self.is_finite() || self.is_zero() {
            return self.clone();
        }
        let mut r = self.clone();
        r.exp = r.exp.saturating_add(e);
        r
    }

    /// Multiplies `self` by a signed 64-bit integer.
    ///
    /// Corresponds to `bf_mul_si` in libbf.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat};
    /// let x = BigFloat::from_i64(7);
    /// let y = x.mul_i64(-3, BigFormat::BINARY64);
    /// let f = y.to_f64(libbeef::Rounding::NearestEven);
    /// assert_eq!(f, -21.0);
    /// ```
    pub fn mul_i64(&self, b: i64, format: BigFormat) -> Self {
        self.mul_i64_status(b, format).0
    }

    /// Multiplies `self` by a signed 64-bit integer, returning status flags as well.
    ///
    /// Corresponds to `bf_mul_si` in libbf.
    pub fn mul_i64_status(&self, b: i64, format: BigFormat) -> (Self, Status) {
        let bf_b = Self::from_i64(b);
        self.mul_status(&bf_b, format)
    }

    /// Multiplies `self` by an unsigned 64-bit integer.
    ///
    /// Corresponds to `bf_mul_ui` in libbf.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat};
    /// let x = BigFloat::from_i64(5);
    /// let y = x.mul_u64(4, BigFormat::BINARY64);
    /// let f = y.to_f64(libbeef::Rounding::NearestEven);
    /// assert_eq!(f, 20.0);
    /// ```
    pub fn mul_u64(&self, b: u64, format: BigFormat) -> Self {
        let bf_b = Self::from_u64(b);
        self.mul(&bf_b, format)
    }

    /// Adds a signed 64-bit integer to `self`.
    ///
    /// Corresponds to `bf_add_si` in libbf.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat};
    /// let x = BigFloat::from_i64(10);
    /// let y = x.add_i64(-3, BigFormat::BINARY64);
    /// let f = y.to_f64(libbeef::Rounding::NearestEven);
    /// assert_eq!(f, 7.0);
    /// ```
    pub fn add_i64(&self, b: i64, format: BigFormat) -> Self {
        let bf_b = Self::from_i64(b);
        self.add(&bf_b, format)
    }

    /// Computes the remainder and low-order bits of the quotient of `self / rhs`.
    ///
    /// Corresponds to `bf_remquo` in libbf. Returns `(quotient_low_bits, remainder)`.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    /// let x = BigFloat::from_i64(7);
    /// let y = BigFloat::from_i64(3);
    /// let (q, r) = x.remquo(&y, BigFormat::BINARY64, Rounding::NearestEven);
    /// let rf = r.to_f64(Rounding::NearestEven);
    /// assert_eq!(q, 2);
    /// assert_eq!(rf, 1.0);
    /// ```
    pub fn remquo(&self, rhs: &Self, format: BigFormat, rnd_mode: Rounding) -> (i64, Self) {
        let rnd_mode_div = match rnd_mode {
            Rounding::NearestEven => DivRemMode::NearestEven,
            Rounding::TowardZero => DivRemMode::TowardZero,
            Rounding::TowardNegative => DivRemMode::Floor,
            _ => DivRemMode::NearestEven,
        };
        let (q, r, _status) = self.div_rem_status(rhs, format, rnd_mode_div);
        let q_i64 = q.to_i64_truncate();
        (q_i64, r)
    }

    fn to_i64_truncate(&self) -> i64 {
        if self.is_zero() || !self.is_finite() {
            return 0;
        }
        self.to_f64(Rounding::TowardZero) as i64
    }

    /// Computes the constant pi to the precision specified in `format`.
    ///
    /// Corresponds to `bf_const_pi` in libbf. Uses the Chudnovsky algorithm
    /// with binary splitting for fast convergence.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat};
    /// let pi = BigFloat::pi(BigFormat::BINARY64);
    /// let f = pi.to_f64(libbeef::Rounding::NearestEven);
    /// assert!((f - std::f64::consts::PI).abs() < 1e-15);
    /// ```
    pub fn pi(format: BigFormat) -> Self {
        let precision = match format.precision {
            Precision::Bits(bits) => bits,
            Precision::Digits(digits) => digits.saturating_mul(4).saturating_add(8).max(2),
            Precision::Infinite => return Self::nan(),
        };
        let prec1 = precision + 64;
        let pi = cached_pi(prec1);
        pi.round(format)
    }

    /// Computes the constant ln(2) to the precision specified in `format`.
    ///
    /// Corresponds to `bf_const_log2` in libbf. Uses binary splitting for
    /// efficient high-precision computation.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat};
    /// let ln2 = BigFloat::log2(BigFormat::BINARY64);
    /// let f = ln2.to_f64(libbeef::Rounding::NearestEven);
    /// assert!((f - std::f64::consts::LN_2).abs() < 1e-15);
    /// ```
    pub fn log2(format: BigFormat) -> Self {
        let precision = match format.precision {
            Precision::Bits(bits) => bits,
            Precision::Digits(digits) => digits.saturating_mul(4).saturating_add(8).max(2),
            Precision::Infinite => return Self::nan(),
        };
        let prec1 = precision + 64;
        let log2 = cached_log2(prec1);
        log2.round(format)
    }

    /// Computes e^self (the exponential function), returning status flags as well.
    ///
    /// Corresponds to `bf_exp` in libbf. Uses argument reduction with ln(2) and
    /// a Taylor series, with Ziv's rounding technique for correct rounding.
    pub fn exp_status(&self, format: BigFormat) -> (Self, Status) {
        if self.is_nan() {
            return (Self::nan(), Status::empty());
        }
        if self.is_infinite() {
            if self.sign.is_negative() {
                return (Self::zero(Sign::Positive), Status::empty());
            } else {
                return (Self::infinity(Sign::Positive), Status::empty());
            }
        }
        if self.is_zero() {
            return Self::from_i64(1).round_status_owned(format);
        }

        let precision = match format.precision {
            Precision::Bits(bits) => bits,
            Precision::Digits(digits) => digits.saturating_mul(4).saturating_add(8).max(2),
            Precision::Infinite => return (Self::nan(), Status::INVALID_OP),
        };

        if self.exp < 0 && (-self.exp) >= (precision as i64 + 2) {
            let r = Self::from_i64(1);
            let (result, status) =
                add_epsilon(&r, -(precision as i64 + 2), self.sign.is_negative(), format);
            return (result, status);
        }

        if self.exp > 0 {
            if let Some((result, status)) = check_exp_overflow_underflow(self, precision, format) {
                return (result, status);
            }
        }

        ziv_rounding(self, precision, format, exp_internal)
    }

    /// Computes e^self (the exponential function).
    ///
    /// Corresponds to `bf_exp` in libbf. Uses argument reduction with ln(2)
    /// and a Taylor series.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat};
    /// let one = BigFloat::from_i64(1);
    /// let e = one.exp(BigFormat::BINARY64);
    /// let f = e.to_f64(libbeef::Rounding::NearestEven);
    /// assert!((f - std::f64::consts::E).abs() < 1e-15);
    /// ```
    pub fn exp(&self, format: BigFormat) -> Self {
        self.exp_status(format).0
    }

    /// Computes the natural logarithm of `self`, returning status flags as well.
    ///
    /// Corresponds to `bf_log` in libbf. Uses argument reduction and an
    /// AGM-like series, with Ziv's rounding technique for correct rounding.
    pub fn log_status(&self, format: BigFormat) -> (Self, Status) {
        if self.is_nan() {
            return (Self::nan(), Status::empty());
        }
        if self.is_infinite() {
            if self.sign.is_negative() {
                return (Self::nan(), Status::INVALID_OP);
            } else {
                return (Self::infinity(Sign::Positive), Status::empty());
            }
        }
        if self.is_zero() {
            return (Self::infinity(Sign::Negative), Status::empty());
        }
        if self.sign.is_negative() {
            return (Self::nan(), Status::INVALID_OP);
        }
        if self.cmp_num(&Self::from_i64(1)) == Some(core::cmp::Ordering::Equal) {
            return (Self::zero(Sign::Positive), Status::empty());
        }

        let precision = match format.precision {
            Precision::Bits(bits) => bits,
            Precision::Digits(digits) => digits.saturating_mul(4).saturating_add(8).max(2),
            Precision::Infinite => return (Self::nan(), Status::INVALID_OP),
        };

        ziv_rounding(self, precision, format, log_internal)
    }

    /// Computes the natural logarithm of `self`.
    ///
    /// Corresponds to `bf_log` in libbf. Uses argument reduction and an
    /// AGM-like series.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    /// let e = BigFloat::from_i64(1).exp(BigFormat::BINARY64);
    /// let ln_e = e.log(BigFormat::BINARY64);
    /// let f = ln_e.to_f64(Rounding::NearestEven);
    /// assert!((f - 1.0).abs() < 1e-14);
    /// ```
    pub fn log(&self, format: BigFormat) -> Self {
        self.log_status(format).0
    }

    /// Computes the cosine of `self` (in radians), returning status flags as well.
    ///
    /// Corresponds to `bf_cos` in libbf. Uses argument reduction with pi and a
    /// Taylor series, with Ziv's rounding technique for correct rounding.
    pub fn cos_status(&self, format: BigFormat) -> (Self, Status) {
        if self.is_nan() {
            return (Self::nan(), Status::empty());
        }
        if self.is_infinite() {
            return (Self::nan(), Status::INVALID_OP);
        }
        if self.is_zero() {
            return Self::from_i64(1).round_status_owned(format);
        }

        let precision = match format.precision {
            Precision::Bits(bits) => bits,
            Precision::Digits(digits) => digits.saturating_mul(4).saturating_add(8).max(2),
            Precision::Infinite => return (Self::nan(), Status::INVALID_OP),
        };

        if self.exp < 0 {
            let e = 2 * self.exp - 1;
            if e < -(precision as i64 + 2) {
                let r = Self::from_i64(1);
                let (result, status) = add_epsilon(&r, e, true, format);
                return (result, status);
            }
        }

        ziv_rounding(self, precision, format, |a, prec| {
            let (_s, c) = sincos_select(a, prec, false, true);
            let c = c.unwrap();
            (c, Status::INEXACT)
        })
    }

    /// Computes the cosine of `self` (in radians).
    ///
    /// Corresponds to `bf_cos` in libbf. Uses argument reduction with pi and
    /// a Taylor series.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    /// let zero = BigFloat::from_i64(0);
    /// let cos0 = zero.cos(BigFormat::BINARY64);
    /// let f = cos0.to_f64(Rounding::NearestEven);
    /// assert_eq!(f, 1.0);
    /// ```
    pub fn cos(&self, format: BigFormat) -> Self {
        self.cos_status(format).0
    }

    /// Computes the sine of `self` (in radians), returning status flags as well.
    ///
    /// Corresponds to `bf_sin` in libbf. Uses the same argument reduction and
    /// Taylor series approach as `cos`, with Ziv's rounding technique.
    pub fn sin_status(&self, format: BigFormat) -> (Self, Status) {
        if self.is_nan() {
            return (Self::nan(), Status::empty());
        }
        if self.is_infinite() {
            return (Self::nan(), Status::INVALID_OP);
        }
        if self.is_zero() {
            return (self.clone(), Status::empty());
        }

        let precision = match format.precision {
            Precision::Bits(bits) => bits,
            Precision::Digits(digits) => digits.saturating_mul(4).saturating_add(8).max(2),
            Precision::Infinite => return (Self::nan(), Status::INVALID_OP),
        };

        if self.exp < 0 {
            let e = sat_add(2 * self.exp, self.exp - 2);
            let len_bits = (self.limbs.len() as i64) * 64 + 2;
            if e < self.exp - (precision as i64 + 2).max(len_bits) {
                let (result, status) = add_epsilon(self, e, !self.sign.is_negative(), format);
                return (result, status);
            }
        }

        ziv_rounding(self, precision, format, |a, prec| {
            let (s, _c) = sincos_select(a, prec, true, false);
            let s = s.unwrap();
            (s, Status::INEXACT)
        })
    }

    /// Computes the sine of `self` (in radians).
    ///
    /// Corresponds to `bf_sin` in libbf. Uses argument reduction with pi and
    /// a Taylor series.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    /// let half_pi = BigFloat::pi(BigFormat::BINARY64)
    ///     .mul_exp2(-1);
    /// let sin_val = half_pi.sin(BigFormat::BINARY64);
    /// let f = sin_val.to_f64(Rounding::NearestEven);
    /// assert!((f - 1.0).abs() < 1e-15);
    /// ```
    pub fn sin(&self, format: BigFormat) -> Self {
        self.sin_status(format).0
    }

    /// Computes the tangent of `self` (in radians), returning status flags as well.
    ///
    /// Corresponds to `bf_tan` in libbf. Computed as sin/cos, with Ziv's
    /// rounding technique for correct rounding.
    pub fn tan_status(&self, format: BigFormat) -> (Self, Status) {
        if self.is_nan() {
            return (Self::nan(), Status::empty());
        }
        if self.is_infinite() {
            return (Self::nan(), Status::INVALID_OP);
        }
        if self.is_zero() {
            return (self.clone(), Status::empty());
        }

        let precision = match format.precision {
            Precision::Bits(bits) => bits,
            Precision::Digits(digits) => digits.saturating_mul(4).saturating_add(8).max(2),
            Precision::Infinite => return (Self::nan(), Status::INVALID_OP),
        };

        if self.exp < 0 {
            let e = sat_add(2 * self.exp, self.exp - 1);
            let len_bits = (self.limbs.len() as i64) * 64 + 2;
            if e < self.exp - (precision as i64 + 2).max(len_bits) {
                let (result, status) = add_epsilon(self, e, self.sign.is_negative(), format);
                return (result, status);
            }
        }

        ziv_rounding(self, precision, format, tan_internal)
    }

    /// Computes the tangent of `self` (in radians).
    ///
    /// Corresponds to `bf_tan` in libbf. Computed as sin/cos.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    /// let zero = BigFloat::from_i64(0);
    /// let tan0 = zero.tan(BigFormat::BINARY64);
    /// let f = tan0.to_f64(Rounding::NearestEven);
    /// assert_eq!(f, 0.0);
    /// ```
    pub fn tan(&self, format: BigFormat) -> Self {
        self.tan_status(format).0
    }

    /// Computes the arctangent of `self` (result in radians), returning status flags as well.
    ///
    /// Corresponds to `bf_atan` in libbf. Uses argument reduction and a
    /// Taylor series, with Ziv's rounding technique for correct rounding.
    pub fn atan_status(&self, format: BigFormat) -> (Self, Status) {
        if self.is_nan() {
            return (Self::nan(), Status::empty());
        }
        if self.is_infinite() {
            let precision = match format.precision {
                Precision::Bits(bits) => bits,
                Precision::Digits(digits) => digits.saturating_mul(4).saturating_add(8).max(2),
                Precision::Infinite => return (Self::nan(), Status::INVALID_OP),
            };
            let pi = cached_pi(precision + 32);
            let half_pi = pi.mul_exp2(-1);
            let mut result = half_pi;
            if self.sign.is_negative() {
                result = result.neg();
            }
            return result.round_status_owned(format);
        }
        if self.is_zero() {
            return (self.clone(), Status::empty());
        }

        let precision = match format.precision {
            Precision::Bits(bits) => bits,
            Precision::Digits(digits) => digits.saturating_mul(4).saturating_add(8).max(2),
            Precision::Infinite => return (Self::nan(), Status::INVALID_OP),
        };

        if self.cmp_abs(&Self::from_i64(1)) == core::cmp::Ordering::Equal {
            let pi = cached_pi(precision + 32);
            let quarter_pi = pi.mul_exp2(-2);
            let mut result = quarter_pi;
            if self.sign.is_negative() {
                result = result.neg();
            }
            return result.round_status_owned(format);
        }

        if self.exp < 0 {
            let e = sat_add(2 * self.exp, self.exp - 1);
            let len_bits = (self.limbs.len() as i64) * 64 + 2;
            if e < self.exp - (precision as i64 + 2).max(len_bits) {
                let (result, status) = add_epsilon(self, e, !self.sign.is_negative(), format);
                return (result, status);
            }
        }

        ziv_rounding(self, precision, format, |a, prec| {
            atan_internal(a, prec, false)
        })
    }

    /// Computes the arctangent of `self` (result in radians).
    ///
    /// Corresponds to `bf_atan` in libbf. Uses argument reduction and a
    /// Taylor series.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    /// let one = BigFloat::from_i64(1);
    /// let atan1 = one.atan(BigFormat::BINARY64);
    /// let f = atan1.to_f64(Rounding::NearestEven);
    /// assert!((f - std::f64::consts::FRAC_PI_4).abs() < 1e-15);
    /// ```
    pub fn atan(&self, format: BigFormat) -> Self {
        self.atan_status(format).0
    }

    /// Computes `atan2(self, x)` -- the angle (in radians) of the point (x, self),
    /// returning status flags as well.
    ///
    /// Corresponds to `bf_atan2` in libbf. Uses Ziv's rounding technique for
    /// correct rounding.
    pub fn atan2_status(&self, x: &Self, format: BigFormat) -> (Self, Status) {
        let precision = match format.precision {
            Precision::Bits(bits) => bits,
            Precision::Digits(digits) => digits.saturating_mul(4).saturating_add(8).max(2),
            Precision::Infinite => return (Self::nan(), Status::INVALID_OP),
        };

        if self.is_nan() || x.is_nan() {
            return (Self::nan(), Status::empty());
        }

        let y = self;
        ziv_rounding(y, precision, format, |y, prec| atan2_internal(y, x, prec))
    }

    /// Computes `atan2(self, x)` -- the angle (in radians) of the point (x, self).
    ///
    /// Corresponds to `bf_atan2` in libbf.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    /// let one = BigFloat::from_i64(1);
    /// let zero = BigFloat::from_i64(0);
    /// let angle = one.atan2(&zero, BigFormat::BINARY64);
    /// let f = angle.to_f64(Rounding::NearestEven);
    /// assert!((f - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    /// ```
    pub fn atan2(&self, x: &Self, format: BigFormat) -> Self {
        self.atan2_status(x, format).0
    }

    /// Computes the arcsine of `self` (result in radians), returning status flags as well.
    ///
    /// Corresponds to `bf_asin` in libbf. Computed via `atan(x / sqrt(1 - x^2))`,
    /// with Ziv's rounding technique for correct rounding.
    pub fn asin_status(&self, format: BigFormat) -> (Self, Status) {
        if self.is_nan() {
            return (Self::nan(), Status::empty());
        }
        if self.is_infinite() {
            return (Self::nan(), Status::INVALID_OP);
        }
        if self.is_zero() {
            return (self.clone(), Status::empty());
        }

        let precision = match format.precision {
            Precision::Bits(bits) => bits,
            Precision::Digits(digits) => digits.saturating_mul(4).saturating_add(8).max(2),
            Precision::Infinite => return (Self::nan(), Status::INVALID_OP),
        };

        if self.cmp_abs(&Self::from_i64(1)) == core::cmp::Ordering::Greater {
            return (Self::nan(), Status::INVALID_OP);
        }

        if self.exp < 0 {
            let e = sat_add(2 * self.exp, self.exp - 2);
            let len_bits = (self.limbs.len() as i64) * 64 + 2;
            if e < self.exp - (precision as i64 + 2).max(len_bits) {
                let (result, status) = add_epsilon(self, e, self.sign.is_negative(), format);
                return (result, status);
            }
        }

        ziv_rounding(self, precision, format, |a, prec| {
            asin_internal(a, prec, false)
        })
    }

    /// Computes the arcsine of `self` (result in radians).
    ///
    /// Corresponds to `bf_asin` in libbf. Computed via atan.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    /// let one = BigFloat::from_i64(1);
    /// let asin1 = one.asin(BigFormat::BINARY64);
    /// let f = asin1.to_f64(Rounding::NearestEven);
    /// assert!((f - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    /// ```
    pub fn asin(&self, format: BigFormat) -> Self {
        self.asin_status(format).0
    }

    /// Computes the arccosine of `self` (result in radians), returning status flags as well.
    ///
    /// Corresponds to `bf_acos` in libbf. Computed via atan, with Ziv's
    /// rounding technique for correct rounding.
    pub fn acos_status(&self, format: BigFormat) -> (Self, Status) {
        if self.is_nan() {
            return (Self::nan(), Status::empty());
        }
        if self.is_infinite() {
            return (Self::nan(), Status::INVALID_OP);
        }
        if self.is_zero() {
            let precision = match format.precision {
                Precision::Bits(bits) => bits,
                Precision::Digits(digits) => digits.saturating_mul(4).saturating_add(8).max(2),
                Precision::Infinite => return (Self::nan(), Status::INVALID_OP),
            };
            let pi = cached_pi(precision + 32);
            let half_pi = pi.mul_exp2(-1);
            return half_pi.round_status_owned(format);
        }

        let precision = match format.precision {
            Precision::Bits(bits) => bits,
            Precision::Digits(digits) => digits.saturating_mul(4).saturating_add(8).max(2),
            Precision::Infinite => return (Self::nan(), Status::INVALID_OP),
        };

        if self.cmp_abs(&Self::from_i64(1)) == core::cmp::Ordering::Greater {
            return (Self::nan(), Status::INVALID_OP);
        }

        if !self.sign.is_negative()
            && self.cmp_num(&Self::from_i64(1)) == Some(core::cmp::Ordering::Equal)
        {
            return (Self::zero(Sign::Positive), Status::empty());
        }

        ziv_rounding(self, precision, format, |a, prec| {
            asin_internal(a, prec, true)
        })
    }

    /// Computes the arccosine of `self` (result in radians).
    ///
    /// Corresponds to `bf_acos` in libbf. Computed via atan.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    /// let zero = BigFloat::from_i64(0);
    /// let acos0 = zero.acos(BigFormat::BINARY64);
    /// let f = acos0.to_f64(Rounding::NearestEven);
    /// assert!((f - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    /// ```
    pub fn acos(&self, format: BigFormat) -> Self {
        self.acos_status(format).0
    }

    /// Computes `self` raised to the power `rhs`, returning status flags as well.
    ///
    /// Corresponds to `bf_pow` in libbf. For non-integer exponents, computed as
    /// `exp(rhs * log(self))`. Integer exponents use direct exponentiation.
    /// Uses Ziv's rounding technique for correct rounding.
    pub fn pow_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status) {
        if rhs.is_zero() {
            return Self::from_i64(1).round_status_owned(format);
        }
        if self.is_nan() || rhs.is_nan() {
            if rhs.is_nan()
                && !self.is_nan()
                && self.cmp_abs(&Self::from_i64(1)) == core::cmp::Ordering::Equal
                && !self.sign.is_negative()
            {
                return Self::from_i64(1).round_status_owned(format);
            }
            return (Self::nan(), Status::empty());
        }

        if self.is_zero() || self.is_infinite() || rhs.is_infinite() {
            return self.pow_special_cases(rhs, format);
        }

        let precision = match format.precision {
            Precision::Bits(bits) => bits,
            Precision::Digits(digits) => digits.saturating_mul(4).saturating_add(8).max(2),
            Precision::Infinite => {
                if let Some(exponent) = rhs.to_signed_int_limbs() {
                    return super::pow_integer_status(self, &exponent, format);
                }
                return (Self::nan(), Status::INVALID_OP);
            }
        };

        if let Some(exponent) = rhs.to_signed_int_limbs() {
            return super::pow_integer_status(self, &exponent, format);
        }

        if self.sign.is_negative() {
            let y_emin = rhs.get_exp_min();
            if y_emin < 0 {
                return (Self::nan(), Status::INVALID_OP);
            }
        }

        if self.cmp_num(&Self::from_i64(1)) == Some(core::cmp::Ordering::Equal) {
            return Self::from_i64(1).round_status_owned(format);
        }
        if self.sign.is_negative()
            && self.cmp_num(&Self::from_i64(-1)) == Some(core::cmp::Ordering::Equal)
        {
            return Self::from_i64(1).round_status_owned(format);
        }

        let x_abs = self.abs();
        let r_sign = if self.sign.is_negative() {
            let y_emin = rhs.get_exp_min();
            y_emin == 0
        } else {
            false
        };

        let mut adjusted_format = format;
        if r_sign {
            adjusted_format.rounding = match adjusted_format.rounding {
                Rounding::TowardNegative => Rounding::TowardPositive,
                Rounding::TowardPositive => Rounding::TowardNegative,
                other => other,
            };
        }

        let x_bits = x_abs.exp - x_abs.get_exp_min();
        if x_bits == 1 {
            let b = x_abs.exp - 1;
            let e_bf = rhs.mul_i64(b, fmt_bits(64));
            let e = e_bf.to_f64(Rounding::TowardZero) as i64;
            let result = Self::from_i64(1);
            let (mut result, status) = result.mul_exp2(e).round_status_owned(adjusted_format);
            if r_sign && result.is_finite() && !result.is_zero() {
                result.sign = Sign::Negative;
            }
            return (result, status);
        }

        let rhs_clone = rhs.clone();
        let (mut result, status) = ziv_rounding(&x_abs, precision, adjusted_format, |a, prec| {
            pow_generic(a, &rhs_clone, prec)
        });
        if r_sign && result.is_finite() && !result.is_zero() {
            result.sign = Sign::Negative;
        }
        (result, status)
    }

    /// Computes `self` raised to the power `rhs`.
    ///
    /// Corresponds to `bf_pow` in libbf. Uses `exp(rhs * log(self))` for
    /// non-integer exponents.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Rounding};
    /// let two = BigFloat::from_i64(2);
    /// let ten = BigFloat::from_i64(10);
    /// let result = two.pow(&ten, BigFormat::BINARY64);
    /// let f = result.to_f64(Rounding::NearestEven);
    /// assert_eq!(f, 1024.0);
    /// ```
    pub fn pow(&self, rhs: &Self, format: BigFormat) -> Self {
        self.pow_status(rhs, format).0
    }

    fn pow_special_cases(&self, rhs: &Self, _format: BigFormat) -> (Self, Status) {
        let one = Self::from_i64(1);
        if self.cmp_abs(&one) == core::cmp::Ordering::Equal && !self.sign.is_negative() {
            return (one, Status::empty());
        }
        if rhs.is_infinite() {
            let cmp = self.cmp_abs(&one);
            if cmp == core::cmp::Ordering::Equal && !self.sign.is_negative() {
                return (one, Status::empty());
            }
            if self.sign.is_negative() && self.cmp_abs(&one) == core::cmp::Ordering::Equal {
                return (one, Status::empty());
            }
            if rhs.sign.is_negative() == (cmp == core::cmp::Ordering::Greater) {
                return (Self::zero(Sign::Positive), Status::empty());
            } else {
                return (Self::infinity(Sign::Positive), Status::empty());
            }
        }

        let y_emin = rhs.get_exp_min();
        let y_is_odd = y_emin == 0;

        if self.is_infinite() {
            if rhs.sign.is_negative() {
                return (
                    Self::zero(Sign::from_negative(y_is_odd && self.sign.is_negative())),
                    Status::empty(),
                );
            } else {
                return (
                    Self::infinity(Sign::from_negative(y_is_odd && self.sign.is_negative())),
                    Status::empty(),
                );
            }
        }

        if self.is_zero() {
            if rhs.sign.is_negative() {
                let result =
                    Self::infinity(Sign::from_negative(y_is_odd && self.sign.is_negative()));
                return (result, Status::DIVIDE_ZERO);
            } else {
                return (
                    Self::zero(Sign::from_negative(y_is_odd && self.sign.is_negative())),
                    Status::empty(),
                );
            }
        }

        (Self::nan(), Status::INVALID_OP)
    }

    fn get_exp_min(&self) -> i64 {
        if self.is_zero() || !self.is_finite() || self.limbs.is_empty() {
            return 0;
        }
        let total_bits = (self.limbs.len() as i64) * 64;
        let mut min_exp = self.exp - total_bits;
        for (i, &limb) in self.limbs.iter().enumerate() {
            if limb != 0 {
                min_exp = self.exp - total_bits + (i as i64) * 64 + limb.trailing_zeros() as i64;
                break;
            }
        }
        min_exp
    }
}

fn fmt_bits(prec: u64) -> BigFormat {
    BigFormat {
        precision: Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::BINARY64
    }
}

fn fmt_faithful(prec: u64) -> BigFormat {
    BigFormat {
        precision: Precision::Bits(prec),
        rounding: Rounding::Faithful,
        ..BigFormat::BINARY64
    }
}

fn fmt_inf() -> BigFormat {
    BigFormat {
        precision: Precision::Infinite,
        ..BigFormat::BINARY64
    }
}

fn sat_add(a: i64, b: i64) -> i64 {
    a.saturating_add(b)
}

fn check_exp_overflow_underflow(
    a: &BigFloat,
    precision: u64,
    format: BigFormat,
) -> Option<(BigFloat, Status)> {
    let exp_bits = match format.exp_bits {
        crate::format::ExpBits::Bits(b) => b as u64,
        crate::format::ExpBits::Max => return None,
        crate::format::ExpBits::Extended => return None,
    };
    let e_max = 1_i64 << (exp_bits - 1);
    let e_min = -e_max + 3;

    let log2 = cached_log2(64);

    let upper = log2.mul_u64(e_max as u64, fmt_bits(64));
    if a.cmp_num(&upper) == Some(core::cmp::Ordering::Greater) {
        let result = BigFloat::infinity(Sign::Positive);
        return Some((result, Status::OVERFLOW | Status::INEXACT));
    }

    let e_min_sub = if format.subnormal {
        e_min - precision as i64 + 1
    } else {
        e_min
    };
    let lower = log2.mul_i64(e_min_sub - 2, fmt_bits(64));
    if a.cmp_num(&lower) == Some(core::cmp::Ordering::Less) {
        let rnd_mode = format.rounding;
        let result = if rnd_mode == Rounding::TowardPositive {
            let mut r = BigFloat::from_i64(1);
            r.exp = e_min_sub;
            r
        } else {
            BigFloat::zero(Sign::Positive)
        };
        return Some((result, Status::UNDERFLOW | Status::INEXACT));
    }

    None
}

fn add_epsilon(a: &BigFloat, e: i64, e_sign: bool, format: BigFormat) -> (BigFloat, Status) {
    let mut t = BigFloat::from_i64(1);
    t.sign = Sign::from_negative(e_sign);
    t.exp = t.exp.saturating_add(e);
    let r = a.add(&t, format);
    let status = Status::INEXACT;
    (r, status)
}

fn ziv_rounding(
    a: &BigFloat,
    precision: u64,
    format: BigFormat,
    f: impl Fn(&BigFloat, u64) -> (BigFloat, Status),
) -> (BigFloat, Status) {
    let rnd_mode = format.rounding;
    if rnd_mode == Rounding::Faithful {
        let (r, _) = f(a, precision);
        return r.round_status_owned(format);
    }

    let mut ziv_extra_bits = 32_u64;
    loop {
        let prec1 = precision.saturating_add(ziv_extra_bits);
        let (r, ret) = f(a, prec1);

        if !(ret & (Status::OVERFLOW | Status::UNDERFLOW)).is_empty() {
            return r.round_status_owned(format);
        }

        if !ret.contains(Status::INEXACT) {
            return r.round_status_owned(format);
        }

        if r.can_round(precision, rnd_mode, prec1) {
            let (rounded, mut status) = r.round_status_owned(format);
            status |= Status::INEXACT;
            return (rounded, status);
        }
        ziv_extra_bits = ziv_extra_bits.saturating_mul(2);
        if ziv_extra_bits > 100_000 {
            panic!(
                "ziv_rounding did not converge: precision={}, ziv_extra_bits={}",
                precision, ziv_extra_bits
            );
        }
    }
}

fn const_log2_rec(n1: u64, n2: u64, need_p: bool) -> (BigFloat, BigFloat, BigFloat) {
    if n2 - n1 == 1 {
        let p = if n1 == 0 {
            BigFloat::from_i64(3)
        } else {
            let mut p = BigFloat::from_u64(n1);
            p.sign = Sign::Negative;
            p
        };
        let mut q = BigFloat::from_u64(2 * n1 + 1);
        q.exp += 2;
        let t = p.clone();
        (t, p, q)
    } else {
        let m = n1 + ((n2 - n1) >> 1);
        let inf = fmt_inf();
        let (t, p, q) = const_log2_rec(n1, m, true);
        let (t1, p1, q1) = const_log2_rec(m, n2, need_p);

        let t1p = t1.mul(&p, inf);
        let mut new_t = t.mul(&q1, inf);
        new_t.add_assign(&t1p, inf);
        let new_p = if need_p { p.mul(&p1, inf) } else { p };
        let new_q = q.mul(&q1, inf);
        (new_t, new_p, new_q)
    }
}

fn const_log2_internal(prec: u64) -> BigFloat {
    let w = prec + 15;
    let n = w / 3 + 1;
    let (t, _p, q) = const_log2_rec(0, n, false);
    t.div(&q, fmt_bits(prec))
}

const CHUD_A: u64 = 13591409;
const CHUD_B: u64 = 545140134;
const CHUD_C: u64 = 640320;
const CHUD_BITS_PER_TERM: u64 = 47;

fn chud_bs(a: i64, b: i64, need_g: bool) -> (BigFloat, BigFloat, BigFloat) {
    let inf = fmt_inf();

    if a == b - 1 {
        let mut g = BigFloat::from_u64((2 * b as u64).wrapping_sub(1));
        g.mul_u64_assign(6 * b as u64 - 1, inf);
        g.mul_u64_assign(6 * b as u64 - 5, inf);

        let mut t0 = BigFloat::from_u64(CHUD_B);
        t0.mul_u64_assign(b as u64, inf);
        t0.add_assign(&BigFloat::from_u64(CHUD_A), inf);
        let mut p = g.mul(&t0, inf);
        if b & 1 != 0 {
            p.sign = Sign::Negative;
        }

        let mut q = BigFloat::from_u64(b as u64);
        q.mul_u64_assign(b as u64, inf);
        q.mul_u64_assign(b as u64, inf);
        q.mul_u64_assign(CHUD_C * CHUD_C * CHUD_C / 24, inf);

        (p, q, g)
    } else {
        let c = (a + b) / 2;
        let (p1, q1, g1) = chud_bs(a, c, true);
        let (p2, q2, _g2) = chud_bs(c, b, need_g);

        let p2g = p2.mul(&g1, inf);
        let mut new_p = p1.mul(&q2, inf);
        new_p.add_assign(&p2g, inf);
        let new_q = q1.mul(&q2, inf);
        let new_g = if need_g { g1.mul(&_g2, inf) } else { g1 };
        (new_p, new_q, new_g)
    }
}

fn const_pi_internal(prec: u64) -> BigFloat {
    let n = (prec / CHUD_BITS_PER_TERM + 1) as i64;
    let prec1 = prec + 32;

    let (p, q, _g) = chud_bs(0, n, false);

    let ga = q.mul_u64(CHUD_A, fmt_bits(prec1));
    let p_total = ga.add(&p, fmt_bits(prec1));
    let q_div = q.div(&p_total, fmt_faithful(prec1));

    let c_bf = BigFloat::from_u64(CHUD_C);
    let c_sqrt = c_bf.sqrt(fmt_faithful(prec1));
    let c_scaled = c_sqrt.mul_u64(CHUD_C / 12, fmt_faithful(prec1));
    q_div.mul(&c_scaled, fmt_bits(prec))
}

fn exp_internal(a: &BigFloat, prec: u64) -> (BigFloat, Status) {
    let n: i64 = if a.exp <= -1 {
        if a.sign.is_negative() {
            -1
        } else {
            0
        }
    } else {
        let log2 = cached_log2(64);
        let q = a.div(&log2, fmt_bits(64));
        let f = q.to_f64(Rounding::TowardNegative);
        // `f64::floor` is unavailable in `no_std`.
        let n = f as i64;
        if (n as f64) > f {
            n.saturating_sub(1)
        } else {
            n
        }
    };

    let k = isqrt(prec.div_ceil(2));
    let l = (prec.saturating_sub(1)) / k + 1;
    let mut prec1 = prec + (k + 2 * l + 18) + k + 8;
    if a.exp > 0 {
        prec1 += a.exp as u64;
    }

    let fmt1 = fmt_bits(prec1);

    let log2 = cached_log2(prec1);
    let mut t = log2.mul_i64(n, fmt1);
    t.rsub_assign(a, fmt1);
    t.mul_exp2_mut(-(k as i64));

    let mut r = BigFloat::from_i64(1);
    let mut u = BigFloat::new();
    let one = BigFloat::from_i64(1);
    for i in (1..=l).rev() {
        u.set_u64(i);
        u.rdiv_assign(&t, fmt1);
        r.mul_assign(&u, fmt1);
        r.add_assign(&one, fmt1);
    }

    for _ in 0..k {
        r.sqr_assign(fmt1);
    }

    r.mul_exp2_mut(n);
    (r, Status::INEXACT)
}

fn log_internal(a: &BigFloat, prec: u64) -> (BigFloat, Status) {
    let mut t = a.clone();
    let mut n = t.exp;
    t.exp = 0;

    let mut two_thirds = BigFloat::from_u64(0xAAAAAAAA);
    two_thirds.exp = 0;
    if t.cmp_num(&two_thirds) == Some(core::cmp::Ordering::Less) {
        t.exp += 1;
        n -= 1;
    }

    let k = isqrt(prec.div_ceil(2));
    let l = prec / (2 * k) + 1;
    let prec1 = prec + k + 2 * l + 32;
    let fmt1 = fmt_bits(prec1);

    t.add_i64_assign(-1, fmt_inf());

    let mut u = BigFloat::new();
    for _ in 0..k {
        u.set_add(&t, &BigFloat::from_i64(1), fmt1);
        u.sqrt_assign(fmt_faithful(prec1));
        u.add_i64_assign(1, fmt1);
        t.div_assign(&u, fmt1);
    }

    u.set_add(&t, &BigFloat::from_i64(2), fmt1);
    let y = t.div(&u, fmt1);
    let y2 = y.mul(&y, fmt1);

    let mut r = BigFloat::from_i64(0);
    let mut frac = BigFloat::new();
    for i in (1..=l).rev() {
        u.set_u64(2 * i + 1);
        frac.set_i64(1);
        frac.div_assign(&u, fmt1);
        r.add_assign(&frac, fmt1);
        r.mul_assign(&y2, fmt1);
    }
    r.add_i64_assign(1, fmt1);
    r.mul_assign(&y, fmt1);

    r.mul_exp2_mut(k as i64 + 1);

    let log2 = cached_log2(prec1);
    u = log2.mul_i64(n, fmt1);
    r.add_assign(&u, fmt1);

    (r, Status::INEXACT)
}

fn sqrt_sin(cosm1: &BigFloat, prec1: u64) -> BigFloat {
    let fmt1 = fmt_bits(prec1);
    let mut sum = cosm1.mul(cosm1, fmt1);
    sum.neg_mut();
    let mut tmp = cosm1.mul_exp2(1);
    tmp.neg_mut();
    sum.add_assign(&tmp, fmt1);
    sum.sqrt_assign(fmt_faithful(prec1));
    sum
}

fn sincos(a: &BigFloat, prec: u64) -> (BigFloat, BigFloat) {
    let (s, c) = sincos_select(a, prec, true, true);
    (s.unwrap(), c.unwrap())
}

/// Like C `bf_sincos` with nullable outputs: compute only the requested
/// results, skipping the expensive `sqrt_sin` combine step for the other.
fn sincos_select(
    a: &BigFloat,
    prec: u64,
    need_sin: bool,
    need_cos: bool,
) -> (Option<BigFloat>, Option<BigFloat>) {
    let inf = fmt_inf();
    let k = isqrt(prec / 2);
    let l = prec / (2 * k) + 1;
    let prec1 = prec + 2 * k + l + 8;
    let fmt1 = fmt_bits(prec1);

    let (t, modv) = if a.exp <= -1 {
        (a.clone(), 0_i64)
    } else {
        let mut cancel = 0_i64;
        loop {
            let prec2 = prec1 as i64 + a.exp + cancel;
            let prec2 = prec2.max(64) as u64;
            let pi = cached_pi(prec2);
            let half_pi = pi.mul_exp2(-1);
            let (q, rem) = a.remquo(&half_pi, fmt_bits(prec2), Rounding::NearestEven);
            if q == 0 || (!rem.is_zero() && (rem.exp + prec2 as i64) >= (prec1 as i64 - 1)) {
                break (rem, q & 3);
            }
            cancel = if rem.is_zero() {
                (cancel + 1) * 3 / 2
            } else {
                (-rem.exp).max((cancel + 1) * 3 / 2)
            };
        }
    };

    let is_neg = t.sign.is_negative();

    let mut t_reduced = t.mul(&t, fmt1);
    t_reduced.mul_exp2_mut(-2 * k as i64);

    let mut r = BigFloat::from_i64(1);
    let mut denom = BigFloat::new();
    let mut u = BigFloat::new();
    for i in (1..=l).rev() {
        denom.set_u64(2 * i - 1);
        denom.mul_u64_assign(2 * i, inf);
        u.set_div(&t_reduced, &denom, fmt1);
        r.mul_assign(&u, fmt1);
        r.neg_mut();
        if i != 1 {
            r.add_i64_assign(1, fmt1);
        }
    }

    for _ in 0..k {
        u.set_mul(&r, &r, fmt1);
        r.mul_exp2_mut(1);
        r.add_assign(&u, fmt1);
        r.mul_exp2_mut(1);
    }

    let modv = ((modv % 4) + 4) % 4;

    let c = if !need_cos {
        None
    } else if (modv & 1) == 0 {
        let mut c = r.add_i64(1, fmt1);
        c.sign = c.sign.neg_if((modv >> 1) & 1 != 0);
        Some(c)
    } else {
        let mut c = sqrt_sin(&r, prec1);
        c.sign = Sign::from_negative(!is_neg);
        c.sign = c.sign.neg_if((modv >> 1) & 1 != 0);
        Some(c)
    };

    let s = if !need_sin {
        None
    } else if (modv & 1) == 0 {
        let mut s = sqrt_sin(&r, prec1);
        s.sign = Sign::from_negative(is_neg);
        s.sign = s.sign.neg_if((modv >> 1) & 1 != 0);
        Some(s)
    } else {
        let mut s = r.add_i64(1, fmt1);
        s.sign = s.sign.neg_if((modv >> 1) & 1 != 0);
        Some(s)
    };

    (s, c)
}

fn atan_internal(a: &BigFloat, prec: u64, add_pi2: bool) -> (BigFloat, Status) {
    if a.is_zero() && !add_pi2 {
        return (a.clone(), Status::empty());
    }
    let k = isqrt(prec.div_ceil(2));
    let l = prec / (2 * k) + 1;
    let prec1 = prec + k + 2 * l + 32;
    let fmt1 = fmt_bits(prec1);

    let cmp_1 = a.exp >= 1;
    let mut t = if cmp_1 {
        BigFloat::from_i64(1).div(a, fmt1)
    } else {
        a.clone()
    };

    let mut u = BigFloat::new();
    for _ in 0..k {
        u.set_mul(&t, &t, fmt1);
        u.add_i64_assign(1, fmt1);
        u.sqrt_assign(fmt1);
        u.add_i64_assign(1, fmt1);
        t.div_assign(&u, fmt1);
    }

    let mut x2 = BigFloat::new();
    x2.set_mul(&t, &t, fmt1);
    let mut r = BigFloat::from_i64(0);
    let mut frac = BigFloat::new();
    for i in (1..=l).rev() {
        u.set_u64(2 * i + 1);
        frac.set_i64(1);
        frac.div_assign(&u, fmt1);
        r.neg_mut();
        r.add_assign(&frac, fmt1);
        r.mul_assign(&x2, fmt1);
    }
    r.neg_mut();
    r.add_i64_assign(1, fmt1);
    r.mul_assign(&t, fmt1);

    r.mul_exp2_mut(k as i64);

    let mut i = if add_pi2 { 1_i64 } else { 0 };
    if cmp_1 {
        r.neg_mut();
        i += 1 - 2 * (if a.sign.is_negative() { 1 } else { 0 });
    }

    if i != 0 {
        let mut pi_part = cached_pi(prec1);
        if i != 2 && i != -2 {
            pi_part.mul_exp2_mut(-1);
        }
        if i < 0 {
            pi_part.neg_mut();
        }
        r.add_assign(&pi_part, fmt1);
    }

    (r, Status::INEXACT)
}

fn atan2_internal(y: &BigFloat, x: &BigFloat, prec: u64) -> (BigFloat, Status) {
    let prec1 = prec + 32;
    let fmt1 = fmt_bits(prec1);

    if y.is_nan() || x.is_nan() {
        return (BigFloat::nan(), Status::empty());
    }

    let t = if y.is_infinite() && x.is_infinite() {
        let mut t = BigFloat::from_i64(1);
        t.sign = y.sign * x.sign;
        t
    } else if y.is_zero() && x.is_zero() {
        BigFloat::zero(y.sign * x.sign)
    } else {
        y.div(x, fmt_faithful(prec1))
    };

    let (mut r, mut ret) = atan_internal(&t, prec1, false);

    if x.sign.is_negative() {
        let mut pi = cached_pi(prec1);
        pi.sign = y.sign;
        r.add_assign(&pi, fmt1);
        ret |= Status::INEXACT;
    }

    (r, ret)
}

fn tan_internal(a: &BigFloat, prec: u64) -> (BigFloat, Status) {
    let prec1 = prec + 8;
    let (s, c) = sincos(a, prec1);
    let r = s.div(&c, fmt_faithful(prec1));
    (r, Status::INEXACT)
}

fn asin_internal(a: &BigFloat, prec: u64, is_acos: bool) -> (BigFloat, Status) {
    let prec1 = prec + 8;
    let (fmt1, fmt2) = if a.exp >= 0 {
        (fmt_bits(prec1), fmt_inf())
    } else {
        (fmt_bits(prec1), fmt_bits(prec1))
    };

    let mut t = a.mul(a, fmt2);
    t.neg_mut();
    t.add_i64_assign(1, fmt2);
    t.sqrt_assign(fmt1);
    t.rdiv_assign(a, fmt1);
    if is_acos {
        t.neg_mut();
    }
    let (r, _) = atan_internal(&t, prec1, is_acos);
    (r, Status::INEXACT)
}

fn pow_generic(x: &BigFloat, y: &BigFloat, prec: u64) -> (BigFloat, Status) {
    let prec1 = prec + 32;

    let mut t = x.log(fmt_faithful(prec1));
    t.mul_assign(y, fmt_faithful(prec1));
    if t.is_nan() {
        return (BigFloat::nan(), Status::INEXACT);
    }
    let (r, _) = exp_internal(&t, prec1);
    (r, Status::INEXACT)
}
