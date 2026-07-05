# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`beef` is a Rust translation of Fabrice Bellard's `libbf` — a tiny arbitrary-precision floating-point library. The name stands for "Beeg Float". It is `no_std` compatible (requires `alloc`).

## Build & test commands

```bash
cargo build                    # build (default features: std)
cargo test                     # run all tests
cargo test --test api_surface  # run only the API-surface tests
cargo test --test bftest       # run the libbf-ported test suite (quick, single-seed)

# cross-library comparison tests (each needs its optional dep)
cargo test --features vs_num_bigint
cargo test --features vs_rug
cargo test --features vs_malachite
cargo test --features vs              # all three at once

cargo doc --open               # generate and view rustdoc
```

No linter or formatter config is checked in; use `cargo fmt` and `cargo clippy` as usual.

### bftest in detail

`cargo test --test bftest` runs the quick, single-pass verification suite. Two additional `#[ignore]` test functions provide long-running modes:

```bash
# Continuous verification — loops over all ops indefinitely, printing stats
cargo test --test bftest --release bftest_continuous -- --ignored --nocapture

# Pure benchmark — measures only operation time, no assertion overhead
cargo test --test bftest --release bftest_bench -- --ignored --nocapture

# Benchmark with other libraries
cargo test --test bftest --release bftest_bench --features vs -- --ignored --nocapture
```

Both modes respect these environment variables:

| Variable | Default | Description |
|---|---|---|
| `BFTEST_SEED` | `1234` | Starting RNG seed (u64). |
| `BFTEST_DURATION` | `100` | Milliseconds to spend per (op, precision) combination. |
| `BFTEST_SINGLE` | *(unset)* | If set (any value), run one seed only then stop. Continuous mode only (like `-S` in C bftest). |
| `BFTEST_OP` | *(unset)* | Substring filter on operation name — only ops whose name contains this string will run. |

Example: run only multiplication-related ops for 500ms each, starting at seed 42:

```bash
BFTEST_SEED=42 BFTEST_DURATION=500 BFTEST_OP=mul \
  cargo test --test bftest --release bftest_continuous -- --ignored --nocapture
```

Operation names used by `BFTEST_OP` include: `mul`, `add`, `sub`, `div`, `sqrt`, `exp`, `log`, `cos`, `sin`, `tan`, `atan`, `atan2`, `pow`, `rint`, `fmod`, `rem`, `round`, `or`, `xor`, `and`, `atof`, `ftoa`, `mp_sqrtrem`, `mp_recip`, `can_round`, `mul_l2radix`, `div_l2radix`, and their `_dec` decimal variants.

## Architecture

The crate exposes two layers (see `docs/interface-design.md` for the full rationale):

1. **Value types** — `BigFloat` (binary) and `BigDecimal` (decimal). These are verbose, faithful translations of libbf's `bf_t` / `bfdec_t`. Arithmetic methods take an explicit `BigFormat` argument; they do not implement operator traits.
2. **Typed wrappers** — `Float<F>` and `Decimal<F>` pair a value with a compile-time `StaticFormat` so that `+`, `-`, `*`, `/` use the correct precision and rounding automatically. Converting between wrappers or back to the underlying value is a zero-cost wrapper change (no rounding).

Values do not carry their own precision — this mirrors libbf's design. Each operation receives format parameters (precision, rounding mode, exponent width, subnormal flag).

### Source layout

| Path | Contents |
|---|---|
| `src/float/mod.rs` | `BigFloat`, `Float<F>`, `Integer`, `Sign`, core arithmetic (~6k lines) |
| `src/float/conv.rs` | `bf_atof` / `bf_ftoa` — string ↔ number conversion |
| `src/float/mp.rs` | Multi-precision integer kernels (`mp_recip`, `mp_sqrtrem`) |
| `src/float/ntt.rs` | Number-theoretic transform for large multiplication |
| `src/float/transc.rs` | Transcendental functions (exp, log, pow, trig) |
| `src/decimal.rs` | `BigDecimal`, `Decimal<F>` — base-10 counterpart (~2.5k lines) |
| `src/format.rs` | `BigFormat`, `StaticFormat`, `Precision`, `Rounding`, `ExpBits` |
| `src/parse.rs` | `ParseOptions`, `ParseFloatError` |
| `src/status.rs` | `Status` flags (inexact, overflow, underflow, etc.) |

### Tests

- `tests/api_surface.rs` — high-level API smoke tests (constructors, classification, conversions).
- `tests/bftest.rs` + `tests/bftest/ops.rs` — ported from libbf's `bftest` verification suite; exercises every operation with known-good inputs/outputs.
- `tests/bftest/{num_bigint,rug,malachite}.rs` — feature-gated comparison tests against other bignum crates.

## Key design constraints

- **Aliasing-safe in-place ops**: libbf allows aliased pointers (`bf_mul(r, a, a, …)`). Rust forbids simultaneous `&mut` and `&`. Dedicated methods (`sqr`, `rsub_assign`, `rdiv_assign`) fill this gap — see `src/lib.rs` doc comment.
- **IEEE 754 fidelity**: signed zeros, NaN, infinities, configurable exponent width, subnormals, all five rounding modes, and all five status flags must be preserved.
- **Limb type**: `u64` for both binary and decimal representations.

## Features

- `std` (default) — enables `std` support.
- `num-traits` / `num-integer` — trait impls for the `num` ecosystem.
- `serde` — serialization support.
- `vs_num_bigint`, `vs_rug`, `vs_malachite` — enable comparison tests against other bignum crates. `vs` enables all three.
