use core::fmt;
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Not};

/// A set of status flags returned by arithmetic operations.
///
/// Corresponds to the `BF_ST_*` flags returned by libbf functions via `bf_flags_t`.
/// Flags can be combined with `|` and tested with [`contains`](Status::contains).
///
/// ```
/// use libbeef::Status;
///
/// let s = Status::INEXACT | Status::OVERFLOW;
/// assert!(!s.is_empty());
/// assert!(s.contains(Status::INEXACT));
/// assert!(!s.contains(Status::UNDERFLOW));
/// ```
#[derive(Clone, Copy, Default, Eq, PartialEq, Hash)]
pub struct Status(u32);

impl Status {
    /// The operation was mathematically invalid (e.g. 0/0, sqrt of negative).
    ///
    /// Corresponds to `BF_ST_INVALID_OP` in libbf.
    pub const INVALID_OP: Self = Self(1 << 0);

    /// A finite value was divided by zero.
    ///
    /// Corresponds to `BF_ST_DIVIDE_ZERO` in libbf.
    pub const DIVIDE_ZERO: Self = Self(1 << 1);

    /// The result overflowed the representable range.
    ///
    /// Corresponds to `BF_ST_OVERFLOW` in libbf.
    pub const OVERFLOW: Self = Self(1 << 2);

    /// The result underflowed to zero or a subnormal.
    ///
    /// Corresponds to `BF_ST_UNDERFLOW` in libbf.
    pub const UNDERFLOW: Self = Self(1 << 3);

    /// The result was rounded and is not exact.
    ///
    /// Corresponds to `BF_ST_INEXACT` in libbf.
    pub const INEXACT: Self = Self(1 << 4);

    /// Returns an empty status with no flags set.
    ///
    /// No direct libbf equivalent.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Creates a `Status` from a raw `u32` bitmask, retaining all bits.
    ///
    /// No direct libbf equivalent.
    pub const fn from_bits_retain(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the underlying `u32` bitmask.
    ///
    /// No direct libbf equivalent.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Returns `true` if no status flags are set.
    ///
    /// No direct libbf equivalent.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns `true` if all flags in `other` are set in `self`.
    ///
    /// No direct libbf equivalent.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl fmt::Debug for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        let items = [
            (Self::INVALID_OP, "INVALID_OP"),
            (Self::DIVIDE_ZERO, "DIVIDE_ZERO"),
            (Self::OVERFLOW, "OVERFLOW"),
            (Self::UNDERFLOW, "UNDERFLOW"),
            (Self::INEXACT, "INEXACT"),
        ];

        f.write_str("Status(")?;
        for (flag, name) in items {
            if self.contains(flag) {
                if !first {
                    f.write_str(" | ")?;
                }
                first = false;
                f.write_str(name)?;
            }
        }
        if first {
            f.write_str("empty")?;
        }
        f.write_str(")")
    }
}

impl BitOr for Status {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for Status {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitAnd for Status {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAndAssign for Status {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl Not for Status {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}
