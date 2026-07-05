use beef::format::formats;
use beef::Float;

type Quad = Float<formats::Binary128>;

fn main() {
    let a: Quad = "3.14159265358979323846".parse().unwrap();
    let b: Quad = "2.71828182845904523536".parse().unwrap();
    let result = (a * b).sin();
    println!("{result}");
}
