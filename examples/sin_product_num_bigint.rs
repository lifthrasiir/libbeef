use num_bigint::BigUint;

fn main() {
    // num-bigint is an integer library, so we demonstrate the equivalent
    // integer multiply that the benchmark measures (no trig available).
    let a = "314159265358979323846".parse::<BigUint>().unwrap();
    let b = "271828182845904523536".parse::<BigUint>().unwrap();
    let product = a * b;
    println!("{product}");
}
