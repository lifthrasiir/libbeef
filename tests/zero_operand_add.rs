use libbeef::{BigFloat, BigFormat, Rounding, Sign};

fn fmt(r: Rounding) -> BigFormat {
    BigFormat {
        rounding: r,
        ..BigFormat::BINARY64
    }
}

fn is_neg_zero(x: &BigFloat) -> bool {
    x.is_zero() && x.is_sign_negative()
}

#[test]
fn add_assign_with_zero_operands_matches_libbf() {
    let pz = || BigFloat::zero(Sign::Positive);
    let nz = || BigFloat::zero(Sign::Negative);

    let mut x = pz();
    x.add_assign(&BigFloat::infinity(Sign::Positive), BigFormat::BINARY64);
    assert!(x.is_infinite() && !x.is_sign_negative());

    let mut x = nz();
    x.add_assign(&pz(), fmt(Rounding::NearestEven));
    assert!(x.is_zero() && !x.is_sign_negative());
    let mut x = nz();
    x.add_assign(&pz(), fmt(Rounding::NearestAway));
    assert!(!x.is_sign_negative());

    let mut x = pz();
    x.add_assign(&nz(), fmt(Rounding::TowardNegative));
    assert!(is_neg_zero(&x));

    let mut x = pz();
    x.sub_assign(&pz(), fmt(Rounding::TowardNegative));
    assert!(is_neg_zero(&x));

    let mut x = nz();
    x.sub_assign(&nz(), fmt(Rounding::NearestEven));
    assert!(x.is_zero() && !x.is_sign_negative());

    let mut x = nz();
    x.add_i64_assign(0, fmt(Rounding::NearestEven));
    assert!(x.is_zero() && !x.is_sign_negative());

    let mut x = BigFloat::from_i64(5);
    let st = x.add_assign(&pz(), BigFormat::BINARY64);
    assert!(st.is_empty());
    assert_eq!(x.to_f64(Rounding::NearestEven), 5.0);
}
