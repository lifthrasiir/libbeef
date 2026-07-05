# 🥩 libbeef

[![docs.rs](https://img.shields.io/docsrs/libbeef)](https://docs.rs/libbeef)

A Rust translation of Fabrice Bellard's [libbf](https://bellard.org/libbf/) —
a tiny arbitrary-precision floating-point library. The name stands for "Beeg
Float".

- Full IEEE 754 semantics: signed zeros, NaN, infinities, configurable exponent
  width, subnormals, all five rounding modes, all five status flags.
- Transcendental functions: exp, log, pow, sin, cos, tan, atan, atan2, asin, acos.
- Decimal floating point (`BigDecimal`) with independent base-10 arithmetic.
- `no_std` compatible (requires `alloc`).
- Pure Rust with zero dependencies.

## Quick example

```rust
use libbeef::format::formats;
use libbeef::Float;

type Quad = Float<formats::Binary128>;

fn main() {
    let a: Quad = "3.14159265358979323846".parse().unwrap();
    let b: Quad = "2.71828182845904523536".parse().unwrap();
    let result = (a * b).sin();
    println!("{result}");
    // 0.773942685266708278263054855332479932...
}
```

`Float<F>` pairs a value with a compile-time format, so `*` and `.sin()` use
128-bit precision and round-to-nearest-even automatically — no format argument
at every call site.

## Performance

libbeef implements the same algorithms as the C libbf: NTT-based multiplication,
Newton iteration for division/sqrt, and AGM/binary-splitting for
transcendentals. The asymptotic complexity is optimal for each operation class:

| Operation class | Complexity | Algorithm |
|---|---|---|
| add, sub, cmp | O(n) | Linear scan |
| mul | O(n log n) | Number-theoretic transform |
| div, sqrt | O(M(n)) | Newton iteration over NTT mul |
| exp, log, sin, … | O(M(n) · log n) | AGM / binary splitting |

Empirically (full data in [`docs/benchmark-report.md`](docs/benchmark-report.md)),
libbeef tracks the C libbf's throughput with constant-factor overhead from Rust's
bounds checking and allocation model:

| op | 256 bits | 30 000 bits | 300 000 bits | vs libbf | vs rug (GMP/MPFR) |
|---|---:|---:|---:|---|---|
| mul | 6.7 | 84 | 100 ns/limb | 1.3–2.0× | 0.9–1.3× |
| div | 17.3 | 515 | 659 ns/limb | 1.1–1.4× | 2.6–3.5× |
| sqrt | 38.3 | 384 | 597 ns/limb | 1.2–1.3× | 3.2–4.3× |
| sin | 1051 | — | — ns/limb | 0.7× (faster) | 3.6× |

The **mul** row is the most informative: a quadratic algorithm would show
~10× growth per decade of operand size (47 → 469 → 4688 limbs), but libbeef grows
4.2× then 1.2× — the O(n log n) NTT envelope, the same shape as the C
original. At 300k bits libbeef is ~2× libbf and 1.3× GMP, while being 4× faster
than num-bigint's schoolbook/Toom multiplication.

For **transcendentals**, libbeef matches or beats the C libbf on sin/cos/tan/pow
and is within 15% on log/atan. The uniform 3–5× gap to MPFR is algorithmic
(MPFR uses different, better algorithms for these functions; the C libbf shows
the same gap).

Division and sqrt show a larger constant-factor gap to GMP/MPFR (~3×). This is
an inherent property of libbf's Newton-reciprocal approach vs. GMP's tuned
divide-and-conquer — the same ratio appears in the C original.

## Why libbeef?

**1. Pure Rust, no system dependencies.** rug/GMP requires a C compiler, system
GMP/MPFR libraries, and a build script that probes the host. libbeef is a single
`cargo add` with no `build.rs`, `-lm`, nor `pkg-config`. It builds on any
target `rustc` supports — including WASM, embedded, and cross-compilation — with
zero configuration.

**2. Small binary footprint.** With GMP/MPFR statically linked, a trivial
program that multiplies two numbers and computes sin produces:

| Library | Binary size (stripped) |
|---|---:|
| **libbeef** | 482 KiB |
| num-bigint (integers only, no trig) | 448 KiB |
| malachite (integers only, no trig) | 658 KiB |
| rug (GMP + MPFR statically linked) | 680 KiB |

libbeef delivers full floating-point arithmetic *and* transcendentals in less space
than malachite or rug need for integers alone. The num-bigint binary is smaller
only because it cannot compute sin at all — it has no floating-point layer.

**3. Correct and complete.** libbeef passes libbf's own verification suite across
every operation, precision, and rounding mode. It is not a "good enough"
approximation library — it implements IEEE 754 correctly-rounded arithmetic with
configurable exponent width and subnormals.

**4. `no_std` ready.** Only requires `alloc`. No file I/O, no threads, no
system calls beyond allocation.

**5. More permissive license.** libbeef is MIT-licensed, while GMP/MPFR are LGPL.
This makes libbeef a better choice for license-sensitive projects where additional
legal review for LGPL is undesirable.

### When to use something else

- **Maximum throughput at large precisions (>10k bits):** GMP/MPFR (via rug) has
  a heavily tuned FFT and Toom-Cook stack with ~2–3× better constants for
  multiplication and ~3× for division. If raw ns/op at million-bit precision is
  the bottleneck, use rug.
- **Integer-only workloads:** If you never need floating point, rounding, or
  transcendentals, num-bigint or malachite give you a simpler API focused
  purely on integers.
- **Decimal arithmetic at scale:** libbeef's decimal path is functional but not yet
  performance-tuned (it routes through binary conversion rather than native
  base-10⁹ kernels).

## Build & test

```bash
cargo build                    # build (default features: std)
cargo test                     # run all tests
cargo test --test bftest       # libbf verification suite (quick, single-seed)
cargo doc --open               # generate and view API docs
```

## Features

| Feature | Description |
|---|---|
| `std` (default) | Enables `std` support |
| `num-traits` | Trait impls for the `num` ecosystem |
| `num-integer` | Additional `num` integer traits |
| `serde` | Serialization support |

## License

MIT (same as in libbf).
