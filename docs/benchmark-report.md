# Benchmark report: beef vs libbf, MPFR (rug), num-bigint, malachite

*Measured 2026-07-05.*

## Environment

| | |
|---|---|
| CPU / RAM | Apple M4, 32 GiB |
| OS | macOS 26.5 |
| Rust | rustc 1.96.1, `--release` |
| beef | this repository (commit `a15c3743f262931a8a1f87fc1260529815afabe5`) |
| libbf (C) | version 2025-06-03, prebuilt `bftest` at `../libbf`, clang `-O2` |
| rug | 1.30.0 (gmp-mpfr-sys 1.7.1 → GMP 6.3 / MPFR 4.2) |
| num-bigint | 0.4.7 |
| malachite-nz | 0.4.22 |

## Methodology

All numbers are **ns per 64-bit limb per operation** (`total_ns / iterations /
ceil(prec/64)`), lower is better. Every harness follows the C `bftest`
methodology: inputs are pre-generated outside the timed region, each call is
timed individually with a monotonic clock (`std::time::Instant` in Rust,
`clock_gettime(CLOCK_MONOTONIC)` in C — both report nanoseconds), and the
accumulated time is divided by the iteration count. The result variable is
reused across iterations wherever the library allows it (`set_*`/`*_assign` in
beef, `assign_round` in rug, `bf_t` reuse in C), so allocator behaviour under
reuse is part of what is measured — as it is in real workloads.

- **beef, rug, num-bigint, malachite** — `cargo test --test bftest --release
  --features vs bftest_bench -- --ignored --nocapture` with
  `BFTEST_SEED=1234 BFTEST_DURATION=200`. All four libraries run inside the
  same process with the same seed; per-(op, precision) input counts are 1024,
  reduced to 64 at ≥ 20 000 bits and 32 at ≥ 100 000 bits so that one pass over
  the inputs (the granularity of the deadline check) stays short.
- **libbf (C)** — `./bftest -S -p <prec> <op> <op>` per combination, which
  prints libbf's own ns/limb (and an MPFR reference measured through libbf's
  harness; we quote rug's numbers instead since they come from the same
  process and generator as beef's).

### Comparability caveats

- **rug (MPFR)** computes the same thing as beef/libbf: correctly rounded
  floating point at the given precision. This is the like-for-like comparison.
- **num-bigint and malachite are integer libraries.** Their columns measure
  `prec`-bit integer operations (mul/add/sub/cmp/logic; div is a `prec`-bit ÷
  `prec/2`-bit floor division). No rounding or exponent handling is performed,
  so treat these as a raw limb-throughput reference, not an equivalent
  operation.
- beef and libbf draw operands from the same `bf_rrandom`-style generator
  (run-length random bits, sizes uniform in `[1, 2·prec]` bits); the other
  libraries use full-width `prec`-bit operands. This matches how the original
  C `bftest` benchmarks itself.
- Decimal rows are **not** cross-comparable with the C decimal rows (different
  operand sizes and different limb normalization); see the decimal section.

## Core arithmetic

ns/limb; `beef` and `rug` are correctly rounded floats, `libbf` is the C
original, `nbi`/`mal` are integer-library references.

### mul

| prec (bits) | beef | libbf (C) | rug | nbi | mal |
|---:|---:|---:|---:|---:|---:|
| 53 | 19.4 | 43.4 | 15.6 | 19.8 | 18.6 |
| 112 | 10.7 | 24.9 | 8.3 | 11.0 | 11.3 |
| 256 | 6.7 | 17.4 | 7.3 | 7.0 | 6.4 |
| 3 000 | 20.0 | 13.0 | 9.4 | 27.0 | 20.9 |
| 30 000 | 84.0 | 43.0 | 35.3 | 141.6 | 73.0 |
| 300 000 | 100.4 | 51.2 | 78.9 | 416.1 | 130.5 |

### add / sub

| prec | beef add | libbf add | rug add | beef sub | libbf sub | rug sub |
|---:|---:|---:|---:|---:|---:|---:|
| 53 | 21.6 | 35.4 | 16.7 | 21.4 | 35.6 | 16.6 |
| 112 | 10.8 | 26.8 | 8.5 | 11.0 | 28.6 | 8.5 |
| 256 | 6.2 | 20.7 | 4.5 | 6.3 | 19.5 | 4.5 |
| 3 000 | 1.9 | 3.2 | 0.7 | 1.9 | 3.2 | 0.7 |
| 30 000 | 1.5 | 1.3 | 0.4 | 1.4 | 1.3 | 0.4 |
| 300 000 | 1.4 | 1.1 | 0.4 | 1.4 | 1.1 | 0.4 |

### div

| prec | beef | libbf (C) | rug | nbi | mal |
|---:|---:|---:|---:|---:|---:|
| 53 | 31.8 | 99.6 | 18.1 | 18.2 | 15.2 |
| 112 | 20.9 | 50.0 | 9.1 | 10.8 | 11.5 |
| 256 | 17.3 | 32.3 | 16.0 | 36.4 | 16.4 |
| 3 000 | 52.3 | 59.3 | 18.4 | 25.8 | 10.6 |
| 30 000 | 515.2 | 370.9 | 64.9 | 96.1 | 49.1 |
| 300 000 | 659.2 | 485.2 | 187.0 | 390.9 | 160.2 |

### sqrt

| prec | beef | libbf (C) | rug |
|---:|---:|---:|---:|
| 53 | 32.2 | 55.1 | 17.7 |
| 112 | 31.7 | 47.9 | 8.9 |
| 256 | 38.3 | 50.4 | 28.9 |
| 3 000 | 31.3 | 38.1 | 14.9 |
| 30 000 | 383.5 | 303.1 | 46.0 |
| 300 000 | 597.3 | 464.1 | 139.2 |

## Rounding, compare, logic, remainder

| op / prec | 53 | 112 | 256 | 3 000 |
|---|---:|---:|---:|---:|
| rint — beef | 20.7 | 10.5 | 5.5 | 0.6 |
| rint — libbf | 34.0 | 26.2 | 16.4 | 2.4 |
| round — beef | 23.0 | 11.7 | 6.1 | 0.8 |
| round — libbf | 24.2 | 18.6 | 13.4 | 1.6 |
| cmp — beef | 16.4 | 8.2 | 4.1 | 0.4 |
| cmp — nbi / mal | 15.8 / 15.6 | 8.0 / 8.4 | 4.0 / 4.2 | 0.3 / 0.4 |
| logic (or+xor+and) — beef | 47.4 | 28.8 | 20.3 | 5.0 |
| or — libbf | 127.3 | 84.0 | 47.0 | 7.4 |
| logic — nbi | 21.4 | 21.4 | 12.8 | 2.3 |
| fmod — beef | 96.6 | 57.6 | 36.0 | — |
| fmod — libbf | 110.7 | 68.9 | 44.5 | — |
| rem — beef | 135.2 | 76.7 | 46.4 | — |
| rem — libbf | 138.9 | 85.6 | 54.2 | — |

Note: the beef `logic` row times **three** operations (or, xor, and) per
iteration, the libbf `or` row times one; per single operation beef is roughly
3× faster than the numbers suggest relative to libbf's column, and comparable
per-op with num-bigint once the float→integer view reconstruction is accounted
for.

## Transcendental functions

ns/limb at 53 / 112 / 256 bits.

| op | beef | libbf (C) | rug (MPFR) |
|---|---|---|---|
| exp | 2758 / 1792 / 1565 | 2985 / 1371 / 655 | 518 / 384 / 304 |
| log | 3319 / 2360 / 1991 | 3125 / 2256 / 2099 | 472 / 634 / 535 |
| sin | 2070 / 1378 / 1051 | 2933 / 2026 / 1486 | 489 / 328 / 287 |
| cos | 2049 / 1349 / 1056 | 2804 / 2012 / 1472 | 406 / 272 / 231 |
| tan | 2278 / 1482 / 1128 | 3345 / 2181 / 1643 | 604 / 443 / 342 |
| atan | 3644 / 2502 / 2162 | 3452 / 2499 / 2243 | 1072 / 757 / 1250 |
| atan2 | 4487 / 2936 / 2537 | 4767 / 4588 / 2428 | 1236 / 1057 / 1380 |
| asin | 3979 / 2743 / 2283 | 3527 / 2384 / 2157 | 1286 / 1091 / 1399 |
| acos | 4059 / 2850 / 2338 | 3971 / 2712 / 2439 | 1298 / 1147 / 1428 |
| pow | 7114 / 4809 / 3965 | 8535 / 5056 / 3351 | 1175 / 1071 / 901 |

beef is at or below the C original for sin/cos/tan/atan2/pow at most
precisions, and within ~15 % for log/atan/asin/acos. The visible outlier is
**exp at 112–256 bits**, where the C original pulls ahead (1371/655 vs
1792/1565); this is the one transcendental where a real gap to the C port
remains. The uniform ~3–5× gap to MPFR is algorithmic — MPFR uses different
(better) algorithms for these functions, and the C libbf shows the same gap.

## Decimal (BigDecimal)

ns per **operation** (beef bench uses ≤ 2-decimal-limb operands at 16-digit
precision and normalizes by 1 limb; the C bench uses ~100-digit random
precision and normalizes by 6 limbs — the two columns are *not* directly
comparable and are listed for absolute orientation only).

| op | beef (ns/op, 16 digits) | libbf (ns/limb, ~100 digits) |
|---|---:|---:|
| add_dec | 190.6 | 58.5 |
| mul_dec | 149.3 | 64.1 |
| div_dec | 348.1 | 180.0 |
| sqrt_dec | 2535.9 | 179.5 |
| rint_dec | 51.0 | 40.6 |

`src/decimal.rs` is not a port of libbf's base-10⁹/10¹⁹ kernels: it routes
through `ScaledDecimalLimbs` with per-step allocation and decimal↔binary limb
conversion. Reaching libbf-level decimal performance (most visibly for
`sqrt_dec`) requires porting the native decimal kernels (`mp_sqrtrem_dec`,
`mp_divnorm_dec`, …) — tracked as future work.

## Asymptotic behaviour

The 53 → 300 000-bit ladder confirms beef sits in the same asymptotic class as
libbf and GMP/MPFR; constant factors differ but growth does not:

- **mul** — ns/limb *falls* from 19.4 (53 b) to 6.7 (256 b) as per-call
  overhead amortizes, then grows slowly through the NTT range: 20 → 84 → 100
  for 3 k → 30 k → 300 k bits. A quadratic algorithm would grow ~10× per
  decade of size (47 → 469 → 4 688 limbs); beef grows 4.2× then 1.2×,
  i.e. the O(n log n) FFT envelope, the same shape as libbf (13 → 43 → 51)
  and rug/GMP (9.4 → 35.3 → 78.9). At 300 k bits beef is ~2× libbf's cost and
  1.3× GMP's, and 4× faster than num-bigint's.
- **div** — beef 52 → 515 → 659 vs libbf 59 → 371 → 485 vs rug 18 → 65 → 187:
  the same Newton-reciprocal-over-FFT shape as the C original (beef ≤ 1.4×
  libbf everywhere), while GMP's divide-and-conquer division keeps a ~3×
  better constant.
- **sqrt** — beef 31 → 384 → 597 tracks libbf's 38 → 303 → 464 (≤ 1.3×);
  rug/MPFR again wins on constants (15 → 46 → 139).
- **add/sub/rint/round/cmp** — ns/limb decreases monotonically toward the
  memory-bandwidth floor (≈ 1.4 ns/limb for add at 300 k bits, vs 1.1 libbf /
  0.4 rug / 0.9 num-bigint): O(n), as expected.

In short: linear ops are linear, multiplication-class ops follow the FFT
envelope, and division/sqrt cost a small constant multiple of multiplication —
beef never falls out of libbf's asymptotic class; remaining differences at
large sizes are constant-factor (NTT vs GMP's tuned FFT/Toom stack).

## Reproducing

```bash
# beef + rug + num-bigint + malachite (one process, same seed)
BFTEST_DURATION=200 cargo test --test bftest --release --features vs \
    bftest_bench -- --ignored --nocapture

# libbf C reference (built in ../libbf)
cd ../libbf && ./bftest -S -p 300000 mul mul   # last two columns: libbf, MPFR (ns/limb)
```

`BFTEST_SEED`, `BFTEST_DURATION` and `BFTEST_OP` (substring filter) control
the Rust harness; see `CLAUDE.md`.
