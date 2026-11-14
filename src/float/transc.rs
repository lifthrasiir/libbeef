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
    /// Multiplies `self` by 2^e, with infinite precision and the maximal exponent range.
    ///
    /// Corresponds to `bf_mul_2exp` in libbf called with `BF_PREC_INF` and the
    /// default exponent width: `e` is clamped to +-(2^62 - 1), and a result
    /// beyond the exponent range overflows to infinity or underflows to zero.
    /// Use [`mul_exp2_status`](Self::mul_exp2_status) to choose the format.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat};
    /// let x = BigFloat::from_i64(3);
    /// let y = x.mul_exp2(4); // 3 * 16 = 48
    /// let f = y.to_f64(libbeef::Rounding::NearestEven);
    /// assert_eq!(f, 48.0);
    /// assert!(x.mul_exp2(i64::MAX).is_infinite());
    /// assert!(x.mul_exp2(i64::MIN).is_zero());
    /// ```
    pub fn mul_exp2(&self, e: i64) -> Self {
        self.mul_exp2_status(e, BigFormat::INTERNAL).0
    }

    /// Multiplies `self` by 2^e, then rounds to `format`.
    ///
    /// Corresponds to `bf_mul_2exp` in libbf: `e` is clamped to +-(2^62 - 1)
    /// before the exponent is adjusted, so that a huge `e` overflows to
    /// infinity or underflows to zero (with the corresponding status flags)
    /// instead of producing a malformed value.
    ///
    /// ```
    /// use libbeef::{BigFloat, BigFormat, Status};
    /// let (r, st) = BigFloat::from_i64(1).mul_exp2_status(i64::MAX, BigFormat::BINARY64);
    /// assert!(r.is_infinite());
    /// assert_eq!(st, Status::OVERFLOW | Status::INEXACT);
    /// ```
    pub fn mul_exp2_status(&self, e: i64, format: BigFormat) -> (Self, Status) {
        let mut r = self.clone();
        let status = r.mul_exp2_round_mut(e, format);
        (r, status)
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
    /// Corresponds to `bf_remquo` in libbf. Returns `(quotient_low_bits, remainder)`,
    /// where `quotient_low_bits` is the integer quotient modulo 2^64 in two's
    /// complement (exact for quotients that fit in `i64`).
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
        (q.int_mod_u64() as i64, r)
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

        if let Some(r) = check_exp_overflow_underflow(self, self, precision, format) {
            return r;
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
            let e = self.exp.wrapping_mul(2).wrapping_sub(1);
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

        let y_emin = rhs.get_exp_min();
        if self.sign.is_negative() && y_emin < 0 {
            return (Self::nan(), Status::INVALID_OP);
        }
        // The result is negative iff x < 0 and y is an odd integer.
        let r_sign = self.sign.is_negative() && y_emin == 0;
        let mut adjusted_format = format;
        if r_sign {
            adjusted_format.rounding = match adjusted_format.rounding {
                Rounding::TowardNegative => Rounding::TowardPositive,
                Rounding::TowardPositive => Rounding::TowardNegative,
                other => other,
            };
        }

        let (mut result, status) = pow_positive(&self.abs(), rhs, precision, adjusted_format);
        if r_sign && !result.is_nan() {
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
        ..BigFormat::INTERNAL
    }
}

fn fmt_dir(prec: u64, rounding: Rounding) -> BigFormat {
    BigFormat {
        precision: Precision::Bits(prec),
        rounding,
        ..BigFormat::INTERNAL
    }
}

fn fmt_faithful(prec: u64) -> BigFormat {
    BigFormat {
        precision: Precision::Bits(prec),
        rounding: Rounding::Faithful,
        ..BigFormat::INTERNAL
    }
}

/// `fmt_bits` with libbf's `BF_FLAG_EXT_EXP`.
fn fmt_bits_ext(prec: u64) -> BigFormat {
    BigFormat {
        precision: Precision::Bits(prec),
        rounding: Rounding::NearestEven,
        ..BigFormat::INTERNAL_EXT
    }
}

/// `fmt_faithful` with libbf's `BF_FLAG_EXT_EXP`.
fn fmt_faithful_ext(prec: u64) -> BigFormat {
    BigFormat {
        precision: Precision::Bits(prec),
        rounding: Rounding::Faithful,
        ..BigFormat::INTERNAL_EXT
    }
}

fn fmt_inf() -> BigFormat {
    BigFormat {
        precision: Precision::Infinite,
        ..BigFormat::INTERNAL
    }
}

fn sat_add(a: i64, b: i64) -> i64 {
    a.saturating_add(b)
}

/// Crude overflow and underflow tests for `exp(a)` with `a_low <= a <= a_high`.
/// Corresponds to `check_exp_underflow_overflow` in libbf.
fn check_exp_overflow_underflow(
    a_low: &BigFloat,
    a_high: &BigFloat,
    precision: u64,
    format: BigFormat,
) -> Option<(BigFloat, Status)> {
    if a_high.exp <= 0 {
        return None;
    }
    let (e_min, e_max) = super::exp_range_from_format(&format);
    let e_min = if format.subnormal {
        e_min.saturating_sub(precision as i64 - 1)
    } else {
        e_min
    };
    // `cached_log2(128)` is accurate enough for directed rounding to 64 bits
    // to give valid bounds.
    let log2 = cached_log2(128);

    // a_low > e_max * log(2) implies exp(a) > 2^e_max.
    let upper = log2.mul_i64(e_max, fmt_dir(64, Rounding::TowardPositive));
    if a_low.cmp_num(&upper) == Some(core::cmp::Ordering::Greater) {
        return Some(BigFloat::from_i64(1).set_overflow(precision, format));
    }

    // a_high < (e_min - 2) * log(2) implies exp(a) < 2^(e_min - 2).
    let lower = log2.mul_i64(
        e_min.saturating_sub(2),
        fmt_dir(64, Rounding::TowardNegative),
    );
    if a_high.cmp_num(&lower) == Some(core::cmp::Ordering::Less) {
        let result = if format.rounding == Rounding::TowardPositive {
            // The smallest positive value.
            let mut r = BigFloat::from_i64(1);
            r.exp = e_min;
            r
        } else {
            BigFloat::zero(Sign::Positive)
        };
        return Some((result, Status::UNDERFLOW | Status::INEXACT));
    }

    None
}

/// Computes `x^y` for finite `x > 0` and finite non-zero `y` at a finite
/// precision. Corresponds to the body of `bf_pow` in libbf after the sign of
/// the result has been determined.
fn pow_positive(
    x: &BigFloat,
    y: &BigFloat,
    precision: u64,
    format: BigFormat,
) -> (BigFloat, Status) {
    let one = BigFloat::from_i64(1);
    if x.cmp_abs(&one) == core::cmp::Ordering::Equal {
        return one.round_status_owned(format);
    }

    // Crude overflow and underflow tests from bounds of y*log(x).
    let log_lo = x.log(fmt_dir(64, Rounding::TowardNegative));
    let log_hi = x.log(fmt_dir(64, Rounding::TowardPositive));
    let (lo, hi) = if y.sign.is_negative() {
        (log_hi, log_lo)
    } else {
        (log_lo, log_hi)
    };
    let lo = lo.mul(y, fmt_dir(64, Rounding::TowardNegative));
    let hi = hi.mul(y, fmt_dir(64, Rounding::TowardPositive));
    if let Some(r) = check_exp_overflow_underflow(&lo, &hi, precision, format) {
        return r;
    }

    let y_emin = y.get_exp_min();
    if y_emin >= 0 {
        return pow_int_exponent(x, y, precision, format);
    }

    // pow(2^b, y) = 2^(b*y), but only when b*y is an integer. libbf reaches
    // this case through `check_exact_power2n` below.
    // (b*y needs at most 63 fractional bits of y, so 128 bits suffice for
    // any b*y whose integer part could possibly be a meaningful exponent.)
    if x.exp - x.get_exp_min() == 1 {
        let (e, mul_status) = y.mul_i64_status(x.exp - 1, fmt_bits(128));
        if mul_status.is_empty() && e.get_exp_min() >= 0 {
            return pow2_int(&e, format);
        }
    }

    // If x = r^(2^n) for the n given by y = k/2^n, then x^y = r^k with an
    // integer k. The result may then be exact, which the Ziv loop below
    // cannot detect: it would never converge under directed rounding.
    if format.rounding != Rounding::Faithful {
        if let Some(root) = check_exact_power2n(x, -y_emin) {
            let mut k = y.clone();
            k.exp -= y_emin;
            return pow_int_exponent(&root, &k, precision, format);
        }
    }

    ziv_rounding(x, precision, format, |a, prec| pow_generic(a, y, prec))
}

/// Returns `2^e` rounded to `format`, for an integer `e`.
fn pow2_int(e: &BigFloat, format: BigFormat) -> (BigFloat, Status) {
    // Out-of-range exponents saturate, and then overflow or underflow.
    let e = e.get_int64().unwrap_or(if e.sign.is_negative() {
        i64::MIN / 2
    } else {
        i64::MAX / 2
    });
    BigFloat::from_i64(1).mul_exp2_status(e, format)
}

/// `x^y` for finite `x > 0`, `x != 1` and an integer `y`, at a finite
/// precision. Corresponds to the `int_pow` part of `bf_pow` in libbf.
fn pow_int_exponent(
    x: &BigFloat,
    y: &BigFloat,
    precision: u64,
    format: BigFormat,
) -> (BigFloat, Status) {
    let x_bits = x.exp - x.get_exp_min();
    if x_bits == 1 {
        // pow(2^b, y) = 2^(b*y)
        let e = y.mul_i64(x.exp - 1, fmt_dir(64, Rounding::TowardZero));
        return pow2_int(&e, format);
    }
    let can_be_exact = if y.exp <= 31 {
        // Small enough power: use exponentiation in all cases.
        true
    } else if y.sign.is_negative() || format.rounding == Rounding::Faithful {
        false
    } else {
        // If x = m*2^b with m odd, x^y needs at least floor_log2(m)*y bits.
        let e = y.mul_i64(x_bits - 1, fmt_dir(64, Rounding::TowardZero));
        e.get_uint64().is_some_and(|e| e <= precision)
    };
    if can_be_exact {
        ziv_rounding(x, precision, format, |a, prec| pow_int(a, y, prec))
    } else {
        ziv_rounding(x, precision, format, |a, prec| pow_generic(a, y, prec))
    }
}

/// `x^y` by repeated squaring for an integer `y` fitting in 64 bits, at a
/// precision suitable for Ziv rounding. Corresponds to `bf_pow_int` in libbf.
fn pow_int(x: &BigFloat, y: &BigFloat, prec: u64) -> (BigFloat, Status) {
    let y1 = y.abs().get_uint64().unwrap_or(u64::MAX);
    // XXX (from libbf): proof for the added precision
    let ceil_log2_y1 = u64::from(64 - (y1 - 1).leading_zeros());
    let fmt1 = fmt_bits_ext(prec + ceil_log2_y1 * 2 + 8);
    let mut status = Status::empty();
    let mut r = x.clone();
    for i in (0..63 - y1.leading_zeros()).rev() {
        let (t, st) = r.mul_status(&r, fmt1);
        r = t;
        status |= st;
        if (y1 >> i) & 1 != 0 {
            let (t, st) = r.mul_status(x, fmt1);
            r = t;
            status |= st;
        }
    }
    if y.sign.is_negative() {
        let (t, st) = BigFloat::from_i64(1).div_status(&r, fmt1);
        r = t;
        status |= st;
    }
    (r, status)
}

/// Returns `r` such that `x = r^(2^n)` exactly, if there is one.
/// `x` must be finite, positive and non-zero.
/// Corresponds to `check_exact_power2n` in libbf.
fn check_exact_power2n(x: &BigFloat, n: i64) -> Option<BigFloat> {
    // x = m*2^e with m an odd integer.
    let e = x.get_exp_min();
    // Fast check on the exponent.
    let er = if n > 63 {
        if e != 0 {
            return None;
        }
        0
    } else {
        if e & ((1_i64 << n) - 1) != 0 {
            return None;
        }
        e >> n
    };
    let mut t = x.clone();
    t.exp -= e;
    // Every odd perfect square is 1 modulo 8.
    if t.int_mod_u64() & 7 != 1 {
        return None;
    }
    let one = BigFloat::from_i64(1);
    for _ in 0..n {
        // Every further root of 1 is 1; this also bounds the loop for tiny y.
        if t.cmp_abs(&one) == core::cmp::Ordering::Equal {
            break;
        }
        let (root, rem, _) = t.sqrtrem();
        if !rem.is_zero() {
            return None;
        }
        t = root;
    }
    t.exp += er;
    Some(t)
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

    // As in libbf, the squarings use the extended exponent range.
    let fmt_sqr = BigFormat {
        exp_bits: crate::format::ExpBits::Extended,
        ..fmt1
    };
    for _ in 0..k {
        r.sqr_assign(fmt_sqr);
    }

    // As in libbf, this uses the extended exponent range.
    r.mul_exp2_round_mut(n, BigFormat::INTERNAL_EXT);
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

    let mut t = x.log(fmt_faithful_ext(prec1));
    t.mul_assign(y, fmt_faithful_ext(prec1));
    if t.is_nan() {
        return (BigFloat::nan(), Status::INEXACT);
    }
    let (r, _) = exp_internal(&t, prec1);
    (r, Status::INEXACT)
}
