use super::Mt19937_64;
use num_bigint::BigInt;

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

fn gen_random_bigint(rng: &mut Mt19937_64, bits: u64) -> BigInt {
    let bits = bits.max(64);
    let n_hex = ((bits + 3) / 4) as usize;
    let mut hex = String::with_capacity(n_hex);
    hex.push(super::digit_char(1 + (rng.next_u64() % 15) as u8));
    for _ in 1..n_hex {
        hex.push(super::digit_char((rng.next_u64() % 16) as u8));
    }
    let mag = BigInt::parse_bytes(hex.as_bytes(), 16).expect("valid hex");
    if rng.next_u64() & 1 != 0 {
        -mag
    } else {
        mag
    }
}

macro_rules! bench_binary {
    ($name:ident, $op:tt) => {
        fn $name(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
            let inputs: Vec<(BigInt, BigInt)> = (0..bench_n(prec))
                .map(|_| (gen_random_bigint(rng, prec), gen_random_bigint(rng, prec)))
                .collect();
            let deadline = std::time::Instant::now()
                + std::time::Duration::from_millis(duration_ms);
            let mut total = 0usize;
            let mut accum_ns = 0u128;
            let mut r = BigInt::from(0);
            loop {
                for (a, b) in &inputs {
                    let t0 = std::time::Instant::now();
                    r = a $op b;
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

bench_binary!(bench_mul, *);
bench_binary!(bench_add, +);
bench_binary!(bench_sub, -);

fn bench_div(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let inputs: Vec<(BigInt, BigInt)> = (0..bench_n(prec))
        .map(|_| {
            let a = gen_random_bigint(rng, prec);
            let b = gen_random_bigint(rng, prec / 2 + 1);
            (a, b)
        })
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0usize;
    let mut accum_ns = 0u128;
    loop {
        for (a, b) in &inputs {
            let t0 = std::time::Instant::now();
            let r = a / b;
            accum_ns += t0.elapsed().as_nanos();
            std::hint::black_box(r);
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    (total, accum_ns)
}

fn bench_cmp(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let inputs: Vec<(BigInt, BigInt)> = (0..bench_n(prec))
        .map(|_| (gen_random_bigint(rng, prec), gen_random_bigint(rng, prec)))
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

fn bench_logic(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let inputs: Vec<(BigInt, BigInt)> = (0..bench_n(prec))
        .map(|_| (gen_random_bigint(rng, prec), gen_random_bigint(rng, prec)))
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0usize;
    let mut accum_ns = 0u128;
    loop {
        for (a, b) in &inputs {
            let t0 = std::time::Instant::now();
            let r_or = a | b;
            let r_xor = a ^ b;
            let r_and = a & b;
            accum_ns += t0.elapsed().as_nanos();
            std::hint::black_box((r_or, r_xor, r_and));
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
        "mul" => Some(bench_mul),
        "add" => Some(bench_add),
        "sub" => Some(bench_sub),
        "div" => Some(bench_div),
        "cmp" => Some(bench_cmp),
        "logic" => Some(bench_logic),
        _ => None,
    }
}
