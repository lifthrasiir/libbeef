use rug::ops::CompleteRound;
use rug::Float;

fn main() {
    let a = Float::parse("3.14159265358979323846")
        .unwrap()
        .complete(113);
    let b = Float::parse("2.71828182845904523536")
        .unwrap()
        .complete(113);
    let product = Float::with_val(113, &a * &b);
    let result = product.sin();
    println!("{result}");
}
