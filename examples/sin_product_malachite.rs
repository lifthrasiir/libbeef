use malachite_nz::natural::Natural;

fn main() {
    // malachite is an integer library, so we demonstrate the equivalent
    // integer multiply that the benchmark measures (no trig available).
    let a = Natural::from(314159265358979323846_u128);
    let b = Natural::from(271828182845904523536_u128);
    let product = a * b;
    println!("{product}");
}
