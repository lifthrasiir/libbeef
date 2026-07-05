use super::Mt19937_64;
use rug::float::Round;
use rug::ops::{AssignRound, Pow};
use rug::Float;

const BENCH_N: usize = 1024;

/// Number of pre-generated inputs. Reduced at very large precisions so that a
/// single pass over the inputs (the granularity of the deadline check) stays
/// short and input generation does not dominate wall-clock time.
fn bench_n(prec: u64) -> usize {
    if prec >= 100_000 {
        32
    } else if prec >= 20_000 {
        64
    } else {
        BENCH_N
    }
}

fn gen_random_float(rng: &mut Mt19937_64, prec: u64) -> Float {
    let p = prec.max(53) as u32;
    let n_limbs = (p as usize + 63) / 64;
    let mut int = rug::Integer::from(rng.next_u64());
    for _ in 1..n_limbs {
        int <<= 64u32;
        int += rng.next_u64();
    }
    int.set_bit(p - 1, true);
    let mut f = Float::with_val(p, &int);
    if let Some(exp) = f.get_exp() {
        let shift = exp - 1;
        if shift > 0 {
            f >>= shift as u32;
        } else if shift < 0 {
            f <<= (-shift) as u32;
        }
    }
    if rng.next_u64() & 1 != 0 {
        f = -f;
    }
    f
}

fn gen_positive_float(rng: &mut Mt19937_64, prec: u64) -> Float {
    gen_random_float(rng, prec).abs()
}

fn gen_unit_float(rng: &mut Mt19937_64, prec: u64) -> Float {
    let mut f = gen_random_float(rng, prec).abs();
    f >>= 1u32;
    if rng.next_u64() & 1 != 0 {
        f = -f;
    }
    f
}

macro_rules! rug_bench_binary {
    ($name:ident, $gen_a:ident, $gen_b:ident, $op:tt) => {
        fn $name(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
            let p = prec.max(53) as u32;
            let inputs: Vec<(Float, Float)> = (0..bench_n(prec))
                .map(|_| ($gen_a(rng, prec), $gen_b(rng, prec)))
                .collect();
            let deadline = std::time::Instant::now()
                + std::time::Duration::from_millis(duration_ms);
            let mut total = 0usize;
            let mut accum_ns = 0u128;
            let mut r = Float::with_val(p, 1);
            loop {
                for (a, b) in &inputs {
                    let t0 = std::time::Instant::now();
                    r.assign_round(a $op b, Round::Nearest);
                    accum_ns += t0.elapsed().as_nanos();
                    total += 1;
                }
                if std::time::Instant::now() >= deadline { break; }
            }
            std::hint::black_box(r);
            (total, accum_ns)
        }
    };
}

macro_rules! rug_bench_unary_ref {
    ($name:ident, $gen:ident, $method:ident) => {
        fn $name(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
            let p = prec.max(53) as u32;
            let inputs: Vec<Float> = (0..bench_n(prec)).map(|_| $gen(rng, prec)).collect();
            let deadline =
                std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
            let mut total = 0usize;
            let mut accum_ns = 0u128;
            let mut r = Float::with_val(p, 1);
            loop {
                for a in &inputs {
                    let t0 = std::time::Instant::now();
                    r.assign_round(a.$method(), Round::Nearest);
                    accum_ns += t0.elapsed().as_nanos();
                    total += 1;
                }
                if std::time::Instant::now() >= deadline {
                    break;
                }
            }
            std::hint::black_box(r);
            (total, accum_ns)
        }
    };
}

rug_bench_binary!(bench_add, gen_random_float, gen_random_float, +);
rug_bench_binary!(bench_sub, gen_random_float, gen_random_float, -);
rug_bench_binary!(bench_mul, gen_random_float, gen_random_float, *);
rug_bench_binary!(bench_div, gen_random_float, gen_random_float, /);

rug_bench_unary_ref!(bench_sqrt, gen_positive_float, sqrt_ref);
rug_bench_unary_ref!(bench_exp, gen_random_float, exp_ref);
rug_bench_unary_ref!(bench_log, gen_positive_float, ln_ref);
rug_bench_unary_ref!(bench_sin, gen_random_float, sin_ref);
rug_bench_unary_ref!(bench_cos, gen_random_float, cos_ref);
rug_bench_unary_ref!(bench_tan, gen_random_float, tan_ref);
rug_bench_unary_ref!(bench_atan, gen_random_float, atan_ref);
rug_bench_unary_ref!(bench_asin, gen_unit_float, asin_ref);
rug_bench_unary_ref!(bench_acos, gen_unit_float, acos_ref);

fn bench_atan2(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let p = prec.max(53) as u32;
    let inputs: Vec<(Float, Float)> = (0..bench_n(prec))
        .map(|_| (gen_random_float(rng, prec), gen_random_float(rng, prec)))
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0usize;
    let mut accum_ns = 0u128;
    let mut r = Float::with_val(p, 1);
    loop {
        for (a, b) in &inputs {
            let t0 = std::time::Instant::now();
            r.assign_round(a.atan2_ref(b), Round::Nearest);
            accum_ns += t0.elapsed().as_nanos();
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    std::hint::black_box(r);
    (total, accum_ns)
}

fn bench_pow(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let p = prec.max(53) as u32;
    let inputs: Vec<(Float, Float)> = (0..bench_n(prec))
        .map(|_| (gen_positive_float(rng, prec), gen_random_float(rng, prec)))
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0usize;
    let mut accum_ns = 0u128;
    let mut r = Float::with_val(p, 1);
    loop {
        for (a, b) in &inputs {
            let t0 = std::time::Instant::now();
            r.assign_round(Pow::pow(a, b), Round::Nearest);
            accum_ns += t0.elapsed().as_nanos();
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    std::hint::black_box(r);
    (total, accum_ns)
}

fn bench_cmp(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let inputs: Vec<(Float, Float)> = (0..bench_n(prec))
        .map(|_| (gen_random_float(rng, prec), gen_random_float(rng, prec)))
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0usize;
    let mut accum_ns = 0u128;
    loop {
        for (a, b) in &inputs {
            let t0 = std::time::Instant::now();
            let eq = a == b;
            let lt = a < b;
            let le = a <= b;
            accum_ns += t0.elapsed().as_nanos();
            std::hint::black_box((eq, lt, le));
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    (total, accum_ns)
}

pub fn bench_lookup(name: &str) -> Option<fn(&mut Mt19937_64, u64, u64) -> (usize, u128)> {
    match name {
        "add" => Some(bench_add),
        "sub" => Some(bench_sub),
        "mul" => Some(bench_mul),
        "div" => Some(bench_div),
        "sqrt" => Some(bench_sqrt),
        "exp" => Some(bench_exp),
        "log" => Some(bench_log),
        "sin" => Some(bench_sin),
        "cos" => Some(bench_cos),
        "tan" => Some(bench_tan),
        "atan" => Some(bench_atan),
        "atan2" => Some(bench_atan2),
        "asin" => Some(bench_asin),
        "acos" => Some(bench_acos),
        "pow" => Some(bench_pow),
        "cmp" => Some(bench_cmp),
        _ => None,
    }
}
