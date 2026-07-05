use core::marker::PhantomData;

/// A runtime-configurable floating-point format descriptor.
///
/// Encodes precision, rounding mode, exponent size, subnormal support, and
/// radix-point precision flag—the Rust equivalent of the combination of
/// `(prec, bf_flags_t)` in libbf, where `bf_flags_t` packs the rounding mode,
/// `exp_bits`, subnormal flag, and radix-point precision flag.
///
/// ```
/// use libbeef::BigFormat;
///
/// let fmt = BigFormat::BINARY64;
/// assert_eq!(fmt.subnormal, true);
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BigFormat {
    /// The precision to use for the result (in bits or digits).
    pub precision: Precision,
    /// The rounding mode applied after each operation.
    pub rounding: Rounding,
    /// The number of exponent bits (controls the exponent range).
    pub exp_bits: ExpBits,
    /// Whether subnormal (denormalized) numbers are supported.
    pub subnormal: bool,
    /// When `true`, precision is relative to the radix point rather than the
    /// most significant digit.
    pub radix_point_precision: bool,
}

impl BigFormat {
    /// IEEE 754 binary64 (double-precision) format: 53-bit significand,
    /// 11-bit exponent, round-to-nearest-even, subnormals enabled.
    ///
    /// Corresponds to the `bf_flags_t` combination for binary64 in libbf.
    pub const BINARY64: Self = Self {
        precision: Precision::Bits(53),
        rounding: Rounding::NearestEven,
        exp_bits: ExpBits::Bits(11),
        subnormal: true,
        radix_point_precision: false,
    };

    /// IEEE 754 binary128 (quad-precision) format: 113-bit significand,
    /// 15-bit exponent, round-to-nearest-even, subnormals enabled.
    ///
    /// Corresponds to the `bf_flags_t` combination for binary128 in libbf.
    pub const BINARY128: Self = Self {
        precision: Precision::Bits(113),
        rounding: Rounding::NearestEven,
        exp_bits: ExpBits::Bits(15),
        subnormal: true,
        radix_point_precision: false,
    };

    /// IEEE 754 decimal64 format: 16 decimal digits, 11-bit exponent,
    /// round-to-nearest-even, subnormals enabled.
    ///
    /// Corresponds to the `bf_flags_t` combination for decimal64 in libbf.
    pub const DECIMAL64: Self = Self {
        precision: Precision::Digits(16),
        rounding: Rounding::NearestEven,
        exp_bits: ExpBits::Bits(11),
        subnormal: true,
        radix_point_precision: false,
    };
}

impl Default for BigFormat {
    fn default() -> Self {
        Self::BINARY64
    }
}

/// The precision component of a [`BigFormat`].
///
/// Corresponds to the `prec` field in libbf's `bf_t`, plus the
/// `BF_FLAG_RADPNT_PREC` distinction between bit-precision and digit-precision.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Precision {
    /// Fixed precision measured in binary bits.
    Bits(u64),
    /// Fixed precision measured in decimal digits (used with `BigDecimal`).
    Digits(u64),
    /// Unlimited (infinite) precision—no rounding is performed.
    Infinite,
}

/// The exponent-width component of a [`BigFormat`].
///
/// Corresponds to the `BF_EXP_BITS_*` family of constants encoded in
/// `bf_flags_t` in libbf.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ExpBits {
    /// Maximum exponent range supported by the implementation.
    Max,
    /// A specific exponent width in bits (e.g. 11 for binary64).
    Bits(u8),
    /// Extended exponent range (larger than any fixed bit count).
    Extended,
}

/// IEEE 754 rounding mode selection.
///
/// Corresponds to `bf_rnd_t` in libbf.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Rounding {
    /// Round to nearest, ties to even (IEEE default).
    NearestEven,
    /// Round toward zero (truncation).
    TowardZero,
    /// Round toward negative infinity (floor).
    TowardNegative,
    /// Round toward positive infinity (ceiling).
    TowardPositive,
    /// Round to nearest, ties away from zero.
    NearestAway,
    /// Round away from zero.
    AwayFromZero,
    /// Faithful rounding—result is one of the two nearest representable values.
    Faithful,
}

const INV_LOG2_RADIX: [[u32; 3]; 35] = [
    [0x80000000, 0x00000000, 0x00000000],
    [0x50c24e60, 0xd4d4f4a7, 0x021f57bc],
    [0x40000000, 0x00000000, 0x00000000],
    [0x372068d2, 0x0a1ee5ca, 0x19ea911b],
    [0x3184648d, 0xb8153e7a, 0x7fc2d2e1],
    [0x2d983275, 0x9d5369c4, 0x4dec1661],
    [0x2aaaaaaa, 0xaaaaaaaa, 0xaaaaaaab],
    [0x28612730, 0x6a6a7a53, 0x810fabde],
    [0x268826a1, 0x3ef3fde6, 0x23e2566b],
    [0x25001383, 0xbac8a744, 0x385a3349],
    [0x23b46706, 0x82c0c709, 0x3f891718],
    [0x229729f1, 0xb2c83ded, 0x15fba800],
    [0x219e7ffd, 0xa5ad572a, 0xe169744b],
    [0x20c33b88, 0xda7c29aa, 0x9bddee52],
    [0x20000000, 0x00000000, 0x00000000],
    [0x1f50b57e, 0xac5884b3, 0x70e28eee],
    [0x1eb22cc6, 0x8aa6e26f, 0x06d1a2a2],
    [0x1e21e118, 0x0c5daab1, 0x81b4f4bf],
    [0x1d9dcd21, 0x439834e3, 0x81667575],
    [0x1d244c78, 0x367a0d64, 0xc8204d6d],
    [0x1cb40589, 0xac173e0c, 0x3b7b16ba],
    [0x1c4bd95b, 0xa8d72b0d, 0x5879f25a],
    [0x1bead768, 0x98f8ce4c, 0x66cc2858],
    [0x1b903469, 0x050f72e5, 0x0cf5488e],
    [0x1b3b433f, 0x2eb06f14, 0x8c89719c],
    [0x1aeb6f75, 0x9c46fc37, 0xab5fc7e9],
    [0x1aa038eb, 0x0e3bfd17, 0x1bd62080],
    [0x1a593062, 0xb38d8c56, 0x7998ab45],
    [0x1a15f4c3, 0x2b95a2e6, 0x46aed6a0],
    [0x19d630dc, 0xcc7ddef9, 0x5aadd61b],
    [0x19999999, 0x99999999, 0x9999999a],
    [0x195fec80, 0x8a609430, 0xe1106014],
    [0x1928ee7b, 0x0b4f22f9, 0x5f69791d],
    [0x18f46acf, 0x8c06e318, 0x4d2aeb2c],
    [0x18c23246, 0xdc0a9f3d, 0x3fe16970],
];

const LOG2_RADIX: [u64; 35] = [
    0x2000000000000000,
    0x32b803473f7ad0f4,
    0x4000000000000000,
    0x4a4d3c25e68dc57f,
    0x52b803473f7ad0f4,
    0x59d5d9fd5010b366,
    0x6000000000000000,
    0x6570068e7ef5a1e8,
    0x6a4d3c25e68dc57f,
    0x6eb3a9f01975077f,
    0x72b803473f7ad0f4,
    0x766a008e4788cbcd,
    0x79d5d9fd5010b366,
    0x7d053f6d26089673,
    0x8000000000000000,
    0x82cc7edf592262d0,
    0x8570068e7ef5a1e8,
    0x87ef05ae409a0289,
    0x8a4d3c25e68dc57f,
    0x8c8ddd448f8b845a,
    0x8eb3a9f01975077f,
    0x90c10500d63aa659,
    0x92b803473f7ad0f4,
    0x949a784bcd1b8afe,
    0x966a008e4788cbcd,
    0x982809d5be7072dc,
    0x99d5d9fd5010b366,
    0x9b74948f5532da4b,
    0x9d053f6d26089673,
    0x9e88c6b3626a72aa,
    0xa000000000000000,
    0xa16bad3758efd873,
    0xa2cc7edf592262d0,
    0xa4231623369e78e6,
    0xa570068e7ef5a1e8,
];

/// Multiplies or divides `a` by log2(radix), with optional ceiling.
///
/// Corresponds to `bf_mul_log2_radix` in libbf. This is useful for converting
/// between a number of binary bits and a number of digits in a given radix.
/// When `is_inverse` is `false`, computes `a * log2(radix)` (digits to bits).
/// When `is_inverse` is `true`, computes `a / log2(radix)` (bits to digits).
/// Returns `None` if the radix is out of range (2..=36) or on overflow.
///
/// ```
/// use libbeef::mul_log2_radix;
///
/// // 10 decimal digits need about 34 bits
/// let bits = mul_log2_radix(10, 10, false, true).unwrap();
/// assert_eq!(bits, 34);
///
/// // 34 bits hold about 10 decimal digits
/// let digits = mul_log2_radix(34, 10, true, false).unwrap();
/// assert_eq!(digits, 10);
/// ```
pub fn mul_log2_radix(a: i64, radix: u8, is_inverse: bool, is_ceil: bool) -> Option<i64> {
    if !(2..=36).contains(&radix) {
        return None;
    }

    if a == 0 {
        return Some(0);
    }

    let is_negative = a.is_negative();
    let mut magnitude = a.unsigned_abs();
    let is_ceil = is_ceil ^ is_negative;

    if radix.is_power_of_two() {
        let radix_bits = u64::from(radix.trailing_zeros());
        if is_inverse {
            if is_ceil {
                magnitude = magnitude.checked_add(radix_bits - 1)?;
            }
            magnitude /= radix_bits;
        } else {
            magnitude = magnitude.checked_mul(radix_bits)?;
        }
    } else {
        let idx = usize::from(radix - 2);
        if is_inverse {
            let tab = INV_LOG2_RADIX[idx];
            let b1 = (u64::from(tab[0]) << 32) | u64::from(tab[1]);
            let b0 = u64::from(tab[2]) << 32;
            let t = u128::from(b0) * u128::from(magnitude);
            let t = u128::from(b1) * u128::from(magnitude) + (t >> 64);
            magnitude = (t >> 63).try_into().ok()?;
        } else {
            let t = u128::from(LOG2_RADIX[idx]) * u128::from(magnitude);
            magnitude = (t >> 61).try_into().ok()?;
        }
        if is_ceil {
            magnitude = magnitude.checked_add(1)?;
        }
    }

    if is_negative {
        if magnitude == (1_u64 << 63) {
            Some(i64::MIN)
        } else {
            i64::try_from(magnitude).ok().map(|value| -value)
        }
    } else {
        i64::try_from(magnitude).ok()
    }
}

/// A type-level binary floating-point format parameterized by precision,
/// rounding mode, exponent bits, subnormal support, and radix-point precision.
///
/// Implements [`StaticFormat`] so it can be used as a generic parameter to
/// specify the format at compile time.
///
/// ```
/// use libbeef::formats::Binary64;
/// use libbeef::StaticFormat;
///
/// assert_eq!(libbeef::BigFormat::BINARY64, Binary64::FORMAT);
/// ```
pub struct Format<const PREC: u64, R, const EXP_BITS: u8, S, P>(PhantomData<(R, S, P)>);

/// A type-level decimal floating-point format, analogous to [`Format`] but
/// with precision measured in decimal digits rather than binary bits.
///
/// Implements [`StaticFormat`] for compile-time decimal format specification.
pub struct DecimalFormat<const DIGITS: u64, R, const EXP_BITS: u8, S, P>(PhantomData<(R, S, P)>);

/// Trait for types that encode a complete [`BigFormat`] at the type level.
///
/// Implemented by [`Format`] and [`DecimalFormat`] to allow generic code to
/// be parameterized by the floating-point format without runtime overhead.
pub trait StaticFormat {
    /// The [`BigFormat`] descriptor encoded by this type.
    const FORMAT: BigFormat;
}

/// Trait for types that encode a [`Rounding`] at the type level.
///
/// Implemented by the rounding-mode marker types ([`NearestEven`],
/// [`TowardZero`], etc.).
pub trait StaticRounding {
    /// The [`Rounding`] encoded by this type.
    const ROUNDING: Rounding;
}

/// Trait for types that encode a subnormal-support flag at the type level.
///
/// Implemented by [`NoSubnormal`] and [`Subnormal`].
pub trait StaticSubnormal {
    /// Whether subnormal numbers are enabled.
    const ENABLED: bool;
}

/// Trait for types that encode a radix-point-precision flag at the type level.
///
/// Implemented by [`NoRadixPointPrec`] and [`RadixPointPrec`].
pub trait StaticRadixPointPrecision {
    /// Whether precision is relative to the radix point.
    const ENABLED: bool;
}

/// Type-level marker for [`Rounding::NearestEven`] rounding.
pub enum NearestEven {}
/// Type-level marker for [`Rounding::TowardZero`] rounding.
pub enum TowardZero {}
/// Type-level marker for [`Rounding::TowardNegative`] rounding.
pub enum TowardNegative {}
/// Type-level marker for [`Rounding::TowardPositive`] rounding.
pub enum TowardPositive {}
/// Type-level marker for [`Rounding::NearestAway`] rounding.
pub enum NearestAway {}
/// Type-level marker for [`Rounding::AwayFromZero`] rounding.
pub enum AwayFromZero {}
/// Type-level marker for [`Rounding::Faithful`] rounding.
pub enum Faithful {}

impl StaticRounding for NearestEven {
    const ROUNDING: Rounding = Rounding::NearestEven;
}
impl StaticRounding for TowardZero {
    const ROUNDING: Rounding = Rounding::TowardZero;
}
impl StaticRounding for TowardNegative {
    const ROUNDING: Rounding = Rounding::TowardNegative;
}
impl StaticRounding for TowardPositive {
    const ROUNDING: Rounding = Rounding::TowardPositive;
}
impl StaticRounding for NearestAway {
    const ROUNDING: Rounding = Rounding::NearestAway;
}
impl StaticRounding for AwayFromZero {
    const ROUNDING: Rounding = Rounding::AwayFromZero;
}
impl StaticRounding for Faithful {
    const ROUNDING: Rounding = Rounding::Faithful;
}

/// Type-level marker indicating subnormal numbers are **disabled**.
pub enum NoSubnormal {}
/// Type-level marker indicating subnormal numbers are **enabled**.
pub enum Subnormal {}

impl StaticSubnormal for NoSubnormal {
    const ENABLED: bool = false;
}
impl StaticSubnormal for Subnormal {
    const ENABLED: bool = true;
}

/// Type-level marker indicating radix-point precision is **disabled**.
pub enum NoRadixPointPrec {}
/// Type-level marker indicating radix-point precision is **enabled**.
pub enum RadixPointPrec {}

impl StaticRadixPointPrecision for NoRadixPointPrec {
    const ENABLED: bool = false;
}
impl StaticRadixPointPrecision for RadixPointPrec {
    const ENABLED: bool = true;
}

impl<const PREC: u64, R, const EXP_BITS: u8, S, P> StaticFormat for Format<PREC, R, EXP_BITS, S, P>
where
    R: StaticRounding,
    S: StaticSubnormal,
    P: StaticRadixPointPrecision,
{
    const FORMAT: BigFormat = BigFormat {
        precision: Precision::Bits(PREC),
        rounding: R::ROUNDING,
        exp_bits: ExpBits::Bits(EXP_BITS),
        subnormal: S::ENABLED,
        radix_point_precision: P::ENABLED,
    };
}

impl<const DIGITS: u64, R, const EXP_BITS: u8, S, P> StaticFormat
    for DecimalFormat<DIGITS, R, EXP_BITS, S, P>
where
    R: StaticRounding,
    S: StaticSubnormal,
    P: StaticRadixPointPrecision,
{
    const FORMAT: BigFormat = BigFormat {
        precision: Precision::Digits(DIGITS),
        rounding: R::ROUNDING,
        exp_bits: ExpBits::Bits(EXP_BITS),
        subnormal: S::ENABLED,
        radix_point_precision: P::ENABLED,
    };
}

/// Predefined type-level format aliases for common IEEE 754 formats.
///
/// These type aliases can be used as generic parameters wherever a
/// [`StaticFormat`] is required.
///
/// ```
/// use libbeef::formats::Binary64;
/// use libbeef::StaticFormat;
///
/// let fmt = Binary64::FORMAT;
/// assert_eq!(fmt, libbeef::BigFormat::BINARY64);
/// ```
pub mod formats {
    use super::{DecimalFormat, Format, NearestEven, NoRadixPointPrec, Subnormal};

    /// Type-level IEEE 754 binary64 (double-precision) format.
    ///
    /// Equivalent to [`BigFormat::BINARY64`](super::BigFormat::BINARY64) at the type level.
    pub type Binary64 = Format<53, NearestEven, 11, Subnormal, NoRadixPointPrec>;
    /// Type-level IEEE 754 binary128 (quad-precision) format.
    ///
    /// Equivalent to [`BigFormat::BINARY128`](super::BigFormat::BINARY128) at the type level.
    pub type Binary128 = Format<113, NearestEven, 15, Subnormal, NoRadixPointPrec>;
    /// Type-level IEEE 754 decimal64 format with 16-digit precision.
    ///
    /// Equivalent to [`BigFormat::DECIMAL64`](super::BigFormat::DECIMAL64) at the type level.
    pub type Decimal64 = DecimalFormat<16, NearestEven, 11, Subnormal, NoRadixPointPrec>;
}
