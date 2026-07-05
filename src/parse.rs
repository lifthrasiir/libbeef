use core::fmt;

use crate::format::BigFormat;

/// Options controlling how a string is parsed into a [`BigFloat`](crate::BigFloat) or
/// [`BigDecimal`](crate::BigDecimal).
///
/// Corresponds to the `radix` parameter and `BF_ATOF_*` flag bits passed to
/// `bf_atof` / `bf_atof2` in libbf.
///
/// ```
/// use libbeef::parse::ParseOptions;
///
/// let opts = ParseOptions::default();
/// assert!(!opts.allow_hex_prefix);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParseOptions {
    /// The radix (base) to use when no prefix is recognized (2..=36).
    /// Corresponds to the `radix` parameter of `bf_atof` in libbf.
    pub radix: u8,
    /// The rounding mode and precision to apply after parsing.
    /// Corresponds to the `bf_flags_t` parameter of `bf_atof` in libbf.
    pub format: BigFormat,
    /// When `true`, allow a `0x` / `0X` prefix to switch to hexadecimal.
    /// Setting this to `false` corresponds to the `BF_ATOF_NO_HEX` flag in libbf.
    pub allow_hex_prefix: bool,
    /// When `true`, allow a `0b` / `0B` prefix to switch to binary.
    ///
    /// Together with [`allow_octal_prefix`](Self::allow_octal_prefix), this
    /// loosely corresponds to the `BF_ATOF_BIN_OCT` flag in libbf, but each
    /// prefix can be enabled independently here.
    pub allow_binary_prefix: bool,
    /// When `true`, allow a `0o` / `0O` prefix to switch to octal.
    ///
    /// Together with [`allow_binary_prefix`](Self::allow_binary_prefix), this
    /// loosely corresponds to the `BF_ATOF_BIN_OCT` flag in libbf, but each
    /// prefix can be enabled independently here.
    pub allow_octal_prefix: bool,
    /// When `true`, the strings `NaN` and `Inf`/`Infinity` are recognized.
    /// Setting this to `false` corresponds to the `BF_ATOF_NO_NAN_INF` flag in libbf.
    pub allow_nan_inf: bool,
    /// When `true`, the returned exponent is in the input radix rather than base-2.
    /// Corresponds to the `BF_ATOF_EXPONENT` flag in libbf.
    pub return_radix_exponent: bool,
}

impl ParseOptions {
    /// Returns options that mimic C-style auto-detection: radix 10 by default,
    /// with `0x`, `0b`, and `0o` prefixes all recognized.
    pub fn auto_radix() -> Self {
        Self {
            allow_hex_prefix: true,
            allow_binary_prefix: true,
            allow_octal_prefix: true,
            ..Self::default()
        }
    }
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            radix: 10,
            format: BigFormat::default(),
            allow_hex_prefix: false,
            allow_binary_prefix: false,
            allow_octal_prefix: false,
            allow_nan_inf: true,
            return_radix_exponent: false,
        }
    }
}

/// Error returned when parsing a string into a [`BigFloat`](crate::BigFloat) or
/// [`BigDecimal`](crate::BigDecimal) fails.
///
/// Implements [`Display`](fmt::Display) and, with the `std` feature,
/// [`std::error::Error`].
///
/// ```
/// let err = "not_a_number".parse::<f64>().unwrap_err();
/// // ParseFloatError behaves similarly — it is returned from BigFloat::parse
/// // and friends when the input is not a valid numeric literal.
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseFloatError {
    _priv: (),
}

impl ParseFloatError {
    pub(crate) const fn new() -> Self {
        Self { _priv: () }
    }
}

impl fmt::Display for ParseFloatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid floating point literal")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ParseFloatError {}
