use crate::format::Rounding;

/// The layout used when converting a number to a string.
///
/// Corresponds to the `BF_FTOA_FORMAT_*` values of libbf.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FormatStyle {
    /// A given number of significant digits (`BF_FTOA_FORMAT_FIXED`).
    ///
    /// The value is rounded to exactly [`ToStringOptions::precision`]
    /// significant digits (counted from the radix point for power-of-two
    /// radixes). Exponential notation is used when the exponent is out of
    /// range for plain notation.
    Fixed,
    /// A given number of digits after the radix point (`BF_FTOA_FORMAT_FRAC`).
    ///
    /// The value is rounded to [`ToStringOptions::precision`] fractional
    /// digits and is never printed in exponential notation.
    Fractional,
    /// Enough digits to read back the value at [`ToStringOptions::precision`]
    /// bits (`BF_FTOA_FORMAT_FREE`).
    ///
    /// For power-of-two radixes the exact value is printed without trailing
    /// zero digits. Other radixes need a finite precision.
    Free,
    /// Like [`Free`](Self::Free) but with the minimum number of digits that
    /// still read back as the same value (`BF_FTOA_FORMAT_FREE_MIN`). This
    /// takes more computation time, and is identical to `Free` for
    /// power-of-two radixes.
    FreeMin,
}

/// Options controlling how a [`BigFloat`](crate::BigFloat) is converted to a
/// string by [`BigFloat::to_string_with`](crate::BigFloat::to_string_with).
///
/// Corresponds to the `radix`, `prec` and `BF_FTOA_*` flag bits passed to
/// `bf_ftoa` in libbf.
///
/// ```
/// use libbeef::ftoa::{FormatStyle, ToStringOptions};
/// use libbeef::BigFloat;
///
/// let x = BigFloat::from_f64(0.1);
/// let opts = ToStringOptions { precision: 53, ..ToStringOptions::default() };
/// assert_eq!(x.to_string_with(opts).unwrap(), "0.1");
///
/// let opts = ToStringOptions {
///     radix: 16,
///     precision: 3,
///     style: FormatStyle::Fractional,
///     add_prefix: true,
///     ..ToStringOptions::default()
/// };
/// assert_eq!(x.to_string_with(opts).unwrap(), "0x0.19a");
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ToStringOptions {
    /// The radix (2..=36) of the output.
    pub radix: u8,
    /// The number of digits for [`FormatStyle::Fixed`] (significant digits)
    /// and [`FormatStyle::Fractional`] (digits after the radix point), or the
    /// number of bits of precision of the value for the free styles.
    pub precision: u64,
    /// The rounding mode used when digits are dropped. Ignored by the free
    /// styles for power-of-two radixes, which are exact.
    pub rounding: Rounding,
    /// The layout of the result.
    pub style: FormatStyle,
    /// Always use exponential notation (`BF_FTOA_FORCE_EXP`). The exponent
    /// is introduced by `e` for radixes up to 10, by `p` (a binary exponent)
    /// for the other power-of-two radixes up to 16, and by `@` otherwise.
    pub force_exponent: bool,
    /// Prefix the digits of radix 16, 8 and 2 with `0x`, `0o` and `0b`
    /// (`BF_FTOA_ADD_PREFIX`).
    pub add_prefix: bool,
    /// Follow the conventions of JavaScript (`BF_FTOA_JS_QUIRKS`): infinity is
    /// printed as `Infinity` and the exponent always has a sign.
    pub js_quirks: bool,
}

impl Default for ToStringOptions {
    /// Decimal output with the minimum number of digits that round-trips a
    /// 53-bit value, rounded to nearest even.
    fn default() -> Self {
        Self {
            radix: 10,
            precision: 53,
            rounding: Rounding::NearestEven,
            style: FormatStyle::FreeMin,
            force_exponent: false,
            add_prefix: false,
            js_quirks: false,
        }
    }
}
