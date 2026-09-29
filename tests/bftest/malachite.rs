use super::Mt19937_64;
use malachite_nz::integer::Integer;
use malachite_nz::natural::Natural;

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

fn gen_random_natural(rng: &mut Mt19937_64, bits: u64) -> Natural {
    let bits = bits.max(64);
    let n_limbs = bits.div_ceil(64) as usize;
    let mut limbs: Vec<u64> = (0..n_limbs).map(|_| rng.next_u64()).collect();
    let bits_in_last = ((bits - 1) % 64) + 1;
    if let Some(last) = limbs.last_mut() {
        *last |= 1u64 << (bits_in_last - 1);
        if bits_in_last < 64 {
            *last &= (1u64 << bits_in_last) - 1;
        }
    }
    Natural::from_owned_limbs_asc(limbs)
}

fn gen_random_integer(rng: &mut Mt19937_64, bits: u64) -> Integer {
    let nat = gen_random_natural(rng, bits);
    if rng.next_u64() & 1 != 0 {
        -Integer::from(nat)
    } else {
        Integer::from(nat)
    }
}

macro_rules! bench_binary_nat {
    ($name:ident, $op:tt) => {
        fn $name(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
            let inputs: Vec<(Natural, Natural)> = (0..bench_n(prec))
                .map(|_| (gen_random_natural(rng, prec), gen_random_natural(rng, prec)))
                .collect();
            let deadline = std::time::Instant::now()
                + std::time::Duration::from_millis(duration_ms);
            let mut total = 0usize;
            let mut accum_ns = 0u128;
            let mut r = Natural::from(0u32);
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

bench_binary_nat!(bench_mul, *);
bench_binary_nat!(bench_add, +);

fn bench_sub(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let inputs: Vec<(Integer, Integer)> = (0..bench_n(prec))
        .map(|_| (gen_random_integer(rng, prec), gen_random_integer(rng, prec)))
        .collect();
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
    let mut total = 0usize;
    let mut accum_ns = 0u128;
    loop {
        for (a, b) in &inputs {
            let t0 = std::time::Instant::now();
            let r = a - b;
            accum_ns += t0.elapsed().as_nanos();
            std::hint::black_box(&r);
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    (total, accum_ns)
}

fn bench_div(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let inputs: Vec<(Natural, Natural)> = (0..bench_n(prec))
        .map(|_| {
            let a = gen_random_natural(rng, prec);
            let b = gen_random_natural(rng, prec / 2 + 1);
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
            std::hint::black_box(&r);
            total += 1;
        }
        if std::time::Instant::now() >= deadline {
            break;
        }
    }
    (total, accum_ns)
}

fn bench_cmp(rng: &mut Mt19937_64, prec: u64, duration_ms: u64) -> (usize, u128) {
    let inputs: Vec<(Natural, Natural)> = (0..bench_n(prec))
        .map(|_| (gen_random_natural(rng, prec), gen_random_natural(rng, prec)))
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

pub fn bench_lookup(name: &str) -> Option<super::BenchFn> {
    match name {
        "mul" => Some(bench_mul),
        "add" => Some(bench_add),
        "sub" => Some(bench_sub),
        "div" => Some(bench_div),
        "cmp" => Some(bench_cmp),
        _ => None,
    }
}
