# beef Rust Interface Design

`beef` is a Rust translation of Fabrice Bellard's `libbf` arbitrary precision
number library. The name stands for "Beeg Float".

The implementation goal is to stay faithful to `libbf`'s numeric model and
algorithms while exposing a Rust API that feels familiar to users of
`num-bigint`, `num-traits`, `rug`, and the standard arithmetic traits.

## Source Model To Preserve

`libbf` has a few design choices that should remain visible in Rust:

- Binary floating point (`bf_t`) follows IEEE 754 semantics, including signed
  zero, infinities, NaN, configurable exponent width, subnormals, rounding modes,
  and status flags.
- Decimal floating point (`bfdec_t`) is a separate base-10 type with IEEE 754
  2008 style behavior, with the upstream limitation that the mantissa is always
  normalized.
- Values do not carry their own precision. Each operation receives precision,
  exponent configuration, rounding mode, and subnormal behavior.
- Arbitrarily large integers are represented by the binary float type using
  infinite precision.
- Memory allocation failure is part of the numeric status surface.

The `bfcalc.c` demo follows the same model: `BCValue` stores only `bf_t` or
`bfdec_t`; precision and flags live in `BCContext` as `float_prec`,
`float_flags`, `dec_prec`, and `dec_flags`. `setprec` and `setbprec` change the
calculation environment for subsequent operations rather than mutating stored
values.

Therefore `beef` must not make operation format part of the identity or storage
of the underlying numeric value.

## Two Public Layers

`beef` exposes two layers.

1. `BigFloat` and `BigDecimal` are verbose, faithful value types. They do not
   implement arithmetic operator traits. Arithmetic methods receive a
   `BigFormat` argument.
2. `Float<F>` and `Decimal<F>` are ergonomic typed wrappers. `F` is a static
   type-level format. Arithmetic operators are available only when both operands
   have the same `F`.

Both layers store the same numeric value representation. Converting
`Float<F1>` to `Float<F2>`, or `Float<F>` to `BigFloat`, must be a no-op wrapper
change. It must not round or normalize beyond preserving internal invariants.

## Crate Layout

Public modules:

```rust
pub mod decimal;
pub mod float;
pub mod format;
pub mod parse;
pub mod status;

pub use decimal::{BigDecimal, Decimal};
pub use float::{BigFloat, Float, Integer};
pub use format::{BigFormat, ExpBits, Format, Precision, Radix, Rounding};
pub use status::Status;
```

Internal modules will mirror upstream organization where useful:

```text
src/
  limb.rs       // limb_t/slimb_t equivalents and low-level helpers
  mp.rs         // multi-precision integer kernels
  float/        // bf_t translation
  decimal/      // bfdec_t translation
  conv.rs       // atof/ftoa equivalents
  transc.rs     // exp/log/trig constants and caches
```

## Core Value Types

### `BigFloat`

`BigFloat` is the Rust equivalent of `bf_t`.

```rust
pub struct BigFloat {
    sign: Sign,
    exp: RawExp,
    limbs: Vec<Limb>,
}
```

Notes:

- The physical representation should remain close to upstream: sign, raw
  exponent, normalized limb vector.
- It preserves signed zero, infinities, and NaN.
- It has no attached precision, rounding mode, exponent size, or subnormal
  setting.
- It does not implement `Add`, `Sub`, `Mul`, `Div`, or `Rem`.

### `BigDecimal`

`BigDecimal` is the Rust equivalent of `bfdec_t`.

```rust
pub struct BigDecimal {
    sign: Sign,
    exp: RawExp,
    limbs: Vec<Limb>,
}
```

`BigDecimal` has the same storage philosophy as `BigFloat`, but uses decimal
arithmetic and conversion routines.

### `Float<F>`

`Float<F>` is an ergonomic wrapper over `BigFloat` with a compile-time operation
format.

```rust
pub struct Float<F> {
    value: BigFloat,
    _format: PhantomData<F>,
}
```

`F` does not change the stored value. It only supplies the operation format for
methods and operator traits.

```rust
impl<F> Float<F> {
    pub fn into_big(self) -> BigFloat;
    pub fn as_big(&self) -> &BigFloat;
    pub fn as_big_mut(&mut self) -> &mut BigFloat;
    pub fn from_big(value: BigFloat) -> Self;

    pub fn into_format<G>(self) -> Float<G>;
}
```

`into_format` is intentionally a no-op wrapper conversion. If a user wants to
round to the new format, they must call an explicit rounding method.

### `Decimal<F>`

`Decimal<F>` mirrors `Float<F>` over `BigDecimal`.

```rust
pub struct Decimal<F> {
    value: BigDecimal,
    _format: PhantomData<F>,
}
```

## Formats

### Dynamic Format

`BigFormat` is the runtime form of upstream `prec` plus `bf_flags_t`.

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BigFormat {
    pub precision: Precision,
    pub rounding: Rounding,
    pub exp_bits: ExpBits,
    pub subnormal: bool,
    pub radix_point_precision: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Precision {
    Bits(u64),
    Digits(u64),
    Infinite,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpBits {
    Max,
    Bits(u8),
    Extended,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Rounding {
    NearestEven,
    TowardZero,
    TowardNegative,
    TowardPositive,
    NearestAway,
    AwayFromZero,
    Faithful,
}
```

### Static Format

`Format<...>` is the type-level binary format used by `Float<F>`.
`DecimalFormat<...>` is the type-level decimal format used by `Decimal<F>`.

```rust
pub struct Format<
    const PREC: u64,
    R,
    const EXP_BITS: u8,
    S,
    P,
>(PhantomData<(R, S, P)>);

pub struct DecimalFormat<
    const DIGITS: u64,
    R,
    const EXP_BITS: u8,
    S,
    P,
>(PhantomData<(R, S, P)>);
```

Marker traits convert static policy to `BigFormat`:

```rust
pub trait StaticFormat {
    const FORMAT: BigFormat;
}

pub trait StaticRounding {
    const ROUNDING: Rounding;
}

pub trait StaticSubnormal {
    const ENABLED: bool;
}

pub trait StaticRadixPointPrecision {
    const ENABLED: bool;
}
```

Example marker types:

```rust
pub enum NearestEven {}
pub enum TowardZero {}
pub enum NoSubnormal {}
pub enum Subnormal {}
pub enum NoRadixPointPrec {}
pub enum RadixPointPrec {}
```

Example fully explicit binary64 type:

```rust
type Binary64Format = Format<
    53,
    NearestEven,
    11,
    Subnormal,
    NoRadixPointPrec,
>;

type Binary64 = Float<Binary64Format>;
```

`u8::MAX` may be reserved as a const-generic sentinel for `ExpBits::Max` or
`ExpBits::Extended`; the exact sentinel mapping should be finalized during the
first implementation pass.

## Status And Errors

Upstream operations return bit flags. Rust should preserve that shape for
status-aware methods.

```rust
bitflags::bitflags! {
    pub struct Status: u32 {
        const INVALID_OP  = 1 << 0;
        const DIVIDE_ZERO = 1 << 1;
        const OVERFLOW    = 1 << 2;
        const UNDERFLOW   = 1 << 3;
        const INEXACT     = 1 << 4;
        const MEM_ERROR   = 1 << 5;
    }
}
```

Status policy:

- Plain arithmetic methods and operator traits discard `Status`.
- `*_status` methods return `Status`.
- Numeric exceptional conditions such as invalid operation, overflow,
  underflow, divide-by-zero, and inexact remain status flags, not Rust errors.
- Discarding status is valid as long as every numeric result can be represented
  by the value type. This includes signed zero, infinities, and NaN.
- If an implementation path cannot represent a numeric result, the API must be
  reconsidered before stabilizing.
- Allocation failure should be surfaced explicitly where Rust allocation APIs
  make that practical. It may also appear as `Status::MEM_ERROR` for parity with
  upstream.

## Faithful `BigFloat` API

`BigFloat` uses explicit runtime formats. Method names are not suffixed with
`_with`; the format argument already makes the operation explicit.

```rust
impl BigFloat {
    pub fn add(&self, rhs: &Self, format: BigFormat) -> Self;
    pub fn sub(&self, rhs: &Self, format: BigFormat) -> Self;
    pub fn mul(&self, rhs: &Self, format: BigFormat) -> Self;
    pub fn div(&self, rhs: &Self, format: BigFormat) -> Self;
    pub fn rem(&self, rhs: &Self, format: BigFormat, mode: DivRemMode) -> Self;

    pub fn add_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status);
    pub fn sub_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status);
    pub fn mul_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status);
    pub fn div_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status);
    pub fn rem_status(
        &self,
        rhs: &Self,
        format: BigFormat,
        mode: DivRemMode,
    ) -> (Self, Status);

    pub fn div_rem_status(
        &self,
        rhs: &Self,
        format: BigFormat,
        mode: DivRemMode,
    ) -> (Self, Self, Status);

    pub fn rem_quo_status(
        &self,
        rhs: &Self,
        format: BigFormat,
        mode: DivRemMode,
    ) -> (i64, Self, Status);

    pub fn round(&self, format: BigFormat) -> Self;
    pub fn round_status(&self, format: BigFormat) -> (Self, Status);
    pub fn rint(&self, rounding: Rounding) -> Self;
    pub fn rint_status(&self, rounding: Rounding) -> (Self, Status);

    pub fn sqrt(&self, format: BigFormat) -> Self;
    pub fn sqrt_status(&self, format: BigFormat) -> (Self, Status);
}
```

Transcendentals:

```rust
impl BigFloat {
    pub fn const_pi(format: BigFormat) -> Self;
    pub fn const_pi_status(format: BigFormat) -> (Self, Status);
    pub fn const_log2(format: BigFormat) -> Self;
    pub fn const_log2_status(format: BigFormat) -> (Self, Status);

    pub fn exp(&self, format: BigFormat) -> Self;
    pub fn exp_status(&self, format: BigFormat) -> (Self, Status);
    pub fn ln(&self, format: BigFormat) -> Self;
    pub fn ln_status(&self, format: BigFormat) -> (Self, Status);
    pub fn pow(&self, rhs: &Self, format: BigFormat) -> Self;
    pub fn pow_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status);

    pub fn sin(&self, format: BigFormat) -> Self;
    pub fn sin_status(&self, format: BigFormat) -> (Self, Status);
    pub fn cos(&self, format: BigFormat) -> Self;
    pub fn cos_status(&self, format: BigFormat) -> (Self, Status);
    pub fn tan(&self, format: BigFormat) -> Self;
    pub fn tan_status(&self, format: BigFormat) -> (Self, Status);
    pub fn asin(&self, format: BigFormat) -> Self;
    pub fn asin_status(&self, format: BigFormat) -> (Self, Status);
    pub fn acos(&self, format: BigFormat) -> Self;
    pub fn acos_status(&self, format: BigFormat) -> (Self, Status);
    pub fn atan(&self, format: BigFormat) -> Self;
    pub fn atan_status(&self, format: BigFormat) -> (Self, Status);
    pub fn atan2(&self, x: &Self, format: BigFormat) -> Self;
    pub fn atan2_status(&self, x: &Self, format: BigFormat) -> (Self, Status);
}
```

Format-independent operations:

```rust
impl BigFloat {
    pub fn neg(&self) -> Self;
    pub fn abs(&self) -> Self;

    pub fn is_finite(&self) -> bool;
    pub fn is_nan(&self) -> bool;
    pub fn is_zero(&self) -> bool;
    pub fn is_sign_negative(&self) -> bool;
    pub fn classify(&self) -> FpCategory;

    pub fn cmp_abs(&self, rhs: &Self) -> Ordering;
    pub fn cmp_num(&self, rhs: &Self) -> Option<Ordering>;
    pub fn cmp_full(&self, rhs: &Self) -> Ordering;
}
```

## Ergonomic `Float<F>` API

`Float<F>` uses `F::FORMAT`.

```rust
impl<F: StaticFormat> Float<F> {
    pub fn add(&self, rhs: &Self) -> Self;
    pub fn sub(&self, rhs: &Self) -> Self;
    pub fn mul(&self, rhs: &Self) -> Self;
    pub fn div(&self, rhs: &Self) -> Self;
    pub fn rem(&self, rhs: &Self, mode: DivRemMode) -> Self;

    pub fn add_status(&self, rhs: &Self) -> (Self, Status);
    pub fn sub_status(&self, rhs: &Self) -> (Self, Status);
    pub fn mul_status(&self, rhs: &Self) -> (Self, Status);
    pub fn div_status(&self, rhs: &Self) -> (Self, Status);
    pub fn rem_status(&self, rhs: &Self, mode: DivRemMode) -> (Self, Status);

    pub fn round(&self) -> Self;
    pub fn round_status(&self) -> (Self, Status);
    pub fn sqrt(&self) -> Self;
    pub fn sqrt_status(&self) -> (Self, Status);
}
```

Operators are implemented only for matching formats:

```rust
impl<F: StaticFormat> Add for Float<F>;
impl<F: StaticFormat> Sub for Float<F>;
impl<F: StaticFormat> Mul for Float<F>;
impl<F: StaticFormat> Div for Float<F>;
impl<F: StaticFormat> Rem for Float<F>;
impl<F> Neg for Float<F>;

impl<F: StaticFormat> AddAssign for Float<F>;
impl<F: StaticFormat> SubAssign for Float<F>;
impl<F: StaticFormat> MulAssign for Float<F>;
impl<F: StaticFormat> DivAssign for Float<F>;
impl<F: StaticFormat> RemAssign for Float<F>;
```

Borrowed combinations should also be implemented, following `num-bigint` style:

```rust
impl<'a, F: StaticFormat> Add<&'a Float<F>> for Float<F>;
impl<'a, F: StaticFormat> Add<Float<F>> for &'a Float<F>;
impl<'a, 'b, F: StaticFormat> Add<&'b Float<F>> for &'a Float<F>;
```

No cross-format arithmetic is provided:

```rust
// Not implemented:
// impl<F1, F2> Add<Float<F2>> for Float<F1>
```

Users must pick the result format explicitly:

```rust
let z = x.into_format::<Binary128Format>() + y.into_format::<Binary128Format>();
```

## Decimal APIs

`BigDecimal` and `Decimal<F>` mirror the float structure but expose only the
operations supported by upstream `bfdec_*`.

```rust
impl BigDecimal {
    pub fn add(&self, rhs: &Self, format: BigFormat) -> Self;
    pub fn sub(&self, rhs: &Self, format: BigFormat) -> Self;
    pub fn mul(&self, rhs: &Self, format: BigFormat) -> Self;
    pub fn div(&self, rhs: &Self, format: BigFormat) -> Self;
    pub fn sqrt(&self, format: BigFormat) -> Self;
    pub fn round(&self, format: BigFormat) -> Self;
    pub fn pow_u64(&self, exp: u64) -> Self;

    pub fn add_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status);
    pub fn sub_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status);
    pub fn mul_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status);
    pub fn div_status(&self, rhs: &Self, format: BigFormat) -> (Self, Status);
    pub fn sqrt_status(&self, format: BigFormat) -> (Self, Status);
    pub fn round_status(&self, format: BigFormat) -> (Self, Status);
    pub fn pow_u64_status(&self, exp: u64) -> (Self, Status);
}

impl<F: StaticFormat> Decimal<F> {
    pub fn add(&self, rhs: &Self) -> Self;
    pub fn add_status(&self, rhs: &Self) -> (Self, Status);
    // same pattern for sub/mul/div/sqrt/round
}
```

Decimal operators are implemented only for matching `F`, just like `Float<F>`.

## Integer Wrapper

`Integer` is a convenience wrapper over `BigFloat` constrained to finite
integral values:

```rust
pub struct Integer(BigFloat);
```

The upstream library intentionally avoids a separate integer representation.
`Integer` should therefore be a Rust-facing convenience wrapper, not a separate
arithmetic engine. Its operations use `Precision::Infinite`.

```rust
impl Integer {
    pub fn div_rem(&self, rhs: &Self) -> (Self, Self);
    pub fn gcd(&self, rhs: &Self) -> Self;
    pub fn is_even(&self) -> bool;
    pub fn bits(&self) -> u64;
}
```

Integer operators should follow `num-bigint` conventions where possible:

- `/` returns truncated integer division.
- `div_rem`, `div_floor`, and `mod_floor` can be added via optional
  `num-integer` integration.

## Constructors And Conversion

Basic constructors:

```rust
impl BigFloat {
    pub fn new() -> Self; // +0
    pub fn nan() -> Self;
    pub fn infinity(sign: Sign) -> Self;
    pub fn zero(sign: Sign) -> Self;

    pub fn from_u64(value: u64) -> Self;
    pub fn from_i64(value: i64) -> Self;
    pub fn from_f64(value: f64) -> Self;
    pub fn to_f64(&self, rounding: Rounding) -> (f64, Status);
}

impl<F> Float<F> {
    pub fn new() -> Self;
    pub fn nan() -> Self;
    pub fn infinity(sign: Sign) -> Self;
    pub fn zero(sign: Sign) -> Self;
}
```

Trait conversions:

```rust
impl From<u64> for BigFloat;
impl From<i64> for BigFloat;
impl From<f64> for BigFloat;
impl<F> From<BigFloat> for Float<F>;
impl<F> From<Float<F>> for BigFloat;

impl<F, G> From<Float<F>> for Float<G>; // no-op wrapper conversion
```

Rust coherence may prevent the exact blanket `From<Float<F>> for Float<G>`
implementation because it overlaps with `impl<T> From<T> for T`. If so,
`into_format::<G>()` remains the stable public API for no-op format changes.

## Parsing And Formatting

Parsing should preserve upstream radix and flag behavior while exposing
idiomatic Rust entry points.

```rust
pub struct ParseOptions {
    pub radix: Radix,
    pub format: BigFormat,
    pub allow_hex_prefix: bool,
    pub allow_binary_octal_prefix: bool,
    pub allow_nan_inf: bool,
    pub return_radix_exponent: bool,
}

impl BigFloat {
    pub fn parse(input: &str, options: ParseOptions)
        -> Result<(Self, usize), ParseFloatError>;

    pub fn parse_status(input: &str, options: ParseOptions)
        -> Result<(Self, Status, usize), ParseFloatError>;

    pub fn parse_with_exponent_status(input: &str, options: ParseOptions)
        -> Result<(Self, i64, Status, usize), ParseFloatError>;
}

impl<F: StaticFormat> Float<F> {
    pub fn parse(input: &str) -> Result<Self, ParseFloatError>;
    pub fn parse_status(input: &str) -> Result<(Self, Status), ParseFloatError>;
}
```

Formatting:

```rust
pub enum FormatStyle {
    Fixed,
    Fractional,
    Free,
    FreeMin,
}

pub struct ToStringOptions {
    pub radix: Radix,
    pub precision: Precision,
    pub rounding: Rounding,
    pub style: FormatStyle,
    pub force_exponent: bool,
    pub add_prefix: bool,
    pub js_quirks: bool,
}

impl BigFloat {
    pub fn to_string_with(&self, options: ToStringOptions) -> String;
}

impl<F: StaticFormat> Float<F> {
    pub fn to_string_with_static_format(&self) -> String;
}
```

`Display` for `Float<F>` should use base 10 free-min formatting derived from
`F::FORMAT`. `Display` for `BigFloat` should use a documented default or require
callers to use `to_string_with`; this can be finalized during implementation.

## Comparison Semantics

Rust traits:

- `PartialEq` and `PartialOrd` use numeric IEEE-like comparisons. NaN is
  unordered and not equal to itself.
- `Eq` and `Ord` are not implemented for `BigFloat`, `BigDecimal`, `Float<F>`,
  or `Decimal<F>`.
- `Integer` implements `Eq` and `Ord`.

Explicit comparison methods:

```rust
impl BigFloat {
    pub fn cmp_abs(&self, rhs: &Self) -> Ordering;
    pub fn cmp_num(&self, rhs: &Self) -> Option<Ordering>;
    pub fn cmp_full(&self, rhs: &Self) -> Ordering;
}

impl<F> Float<F> {
    pub fn cmp_abs(&self, rhs: &Self) -> Ordering;
    pub fn cmp_num(&self, rhs: &Self) -> Option<Ordering>;
    pub fn cmp_full(&self, rhs: &Self) -> Ordering;
}
```

`Float<F1>` and `Float<F2>` comparisons are not provided initially. Users can
compare through `as_big()` or convert one wrapper format explicitly if needed.

## `num-*` Compatibility

Initial optional features:

```toml
[features]
default = ["std"]
std = []
num-traits = ["dep:num-traits"]
num-integer = ["dep:num-integer", "num-traits"]
serde = ["dep:serde"]
```

With `num-traits`:

- `Zero`, `One`, `Signed`, `Num`, `ToPrimitive`, `FromPrimitive` for `Integer`.
- `Zero`, `One`, `Signed`, `ToPrimitive`, `FromPrimitive` for `Float<F>` where
  the trait semantics are compatible.
- Avoid implementing traits whose contract conflicts with NaN, infinities, or
  explicit rounding.

With `num-integer`:

- `Integer` implements integer trait methods using upstream-style division and
  remainder.

## Context And Caches

Upstream `bf_context_t` stores allocator hooks and caches for constants/NTT
state. Rust should hide this unless a real need appears.

Proposed internal design:

- A `Context` owns constant caches and temporary work buffers.
- Public APIs can initially use an internal default context.
- Add explicit `Context` methods later if benchmarks show cache reuse matters.

```rust
pub struct Context { /* caches */ }

impl Context {
    pub fn new() -> Self;
    pub fn clear_cache(&mut self);
    pub fn add_status(
        &mut self,
        a: &BigFloat,
        b: &BigFloat,
        format: BigFormat,
    ) -> (BigFloat, Status);
}
```

The implementation must not introduce process-global mutable numeric settings.
Thread-local caches are acceptable if they do not change results.

## Features And Porting Order

Milestone 1:

- Crate skeleton.
- Limb representation.
- `BigFloat` storage, constructors, normalization, comparison.
- `BigFormat`, static `Format`, and `Status`.
- Addition, subtraction, multiplication, integer conversion.
- `Float<F>` wrapper and matching-format operator traits.

Milestone 2:

- Division, remainder, rounding, sqrt.
- Parsing and formatting in bases 2 through 36.
- `Integer` wrapper and `num-traits` integration.

Milestone 3:

- `BigDecimal` and `Decimal<F>`.
- Decimal arithmetic.
- Decimal parsing/formatting.

Milestone 4:

- Constants and transcendental functions.
- Cache management.

Milestone 5:

- NTT/large multiplication optimizations.
- Architecture-specific acceleration behind feature flags.

## Testing Strategy

Tests should be ported from upstream behavior, not invented only from Rust API
expectations.

- Differential tests against `../libbf` C binaries or a small FFI harness during
  development.
- Golden tests for special values: signed zero, infinities, NaN, subnormals, and
  every status flag.
- Tests proving status-returning and status-discarding methods compute the same
  numeric result.
- Cross-check finite ordinary values against MPFR/rug when available.
- Property tests for integer operations and exact add/mul cases.
- Round-trip tests for parse/format across radices 2 through 36.
- Operator tests proving `Float<F>` arithmetic only works for matching `F`.
- No-op conversion tests for `Float<F1> -> Float<F2> -> BigFloat`.

## Non-Goals For The First Implementation

- Replacing `BigFloat` with `num-bigint` or another big number backend.
- Making the underlying numeric values carry operation formats.
- Cross-format `Float<F1> + Float<F2>` arithmetic.
- Guaranteeing that the first port matches upstream AVX2 performance.
- Implementing `Ord` for floating types.
