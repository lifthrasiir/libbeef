#![allow(dead_code)]

use alloc::vec;

const UDIV1NORM_THRESHOLD: usize = 3;
const DIVNORM_LARGE_THRESHOLD: usize = 50;

fn mp_scan_nz(tab: &[u64], n: usize) -> bool {
    tab[..n].iter().any(|&x| x != 0)
}

pub(crate) fn mp_add_ui(tab: &mut [u64], mut b: u64, n: usize) -> u64 {
    for item in tab.iter_mut().take(n) {
        let v = *item;
        let a = v.wrapping_add(b);
        b = u64::from(a < v);
        *item = a;
        if b == 0 {
            break;
        }
    }
    b
}

pub(crate) fn mp_sub_ui(tab: &mut [u64], mut b: u64, n: usize) -> u64 {
    for item in tab.iter_mut().take(n) {
        let v = *item;
        let a = v.wrapping_sub(b);
        b = u64::from(a > v);
        *item = a;
        if b == 0 {
            break;
        }
    }
    b
}

fn mp_shr(tab_r: &mut [u64], tab: &[u64], n: usize, shift: u32, high: u64) -> u64 {
    assert!((1..64).contains(&shift));
    let mut l = high;
    for i in (0..n).rev() {
        let a = tab[i];
        tab_r[i] = (a >> shift) | (l << (64 - shift));
        l = a;
    }
    l & ((1_u64 << shift) - 1)
}

fn mp_mul1(tabr: &mut [u64], taba: &[u64], n: usize, b: u64, mut l: u64) -> u64 {
    for i in 0..n {
        let t = u128::from(taba[i]) * u128::from(b) + u128::from(l);
        tabr[i] = t as u64;
        l = (t >> 64) as u64;
    }
    l
}

fn mp_add_mul1(tabr: &mut [u64], taba: &[u64], n: usize, b: u64) -> u64 {
    let mut l = 0_u64;
    for i in 0..n {
        let t = u128::from(taba[i]) * u128::from(b) + u128::from(l) + u128::from(tabr[i]);
        tabr[i] = t as u64;
        l = (t >> 64) as u64;
    }
    l
}

fn mp_sub_mul1(tabr: &mut [u64], taba: &[u64], n: usize, b: u64) -> u64 {
    let mut borrow = 0_u64;
    for i in 0..n {
        let product = u128::from(taba[i]) * u128::from(b);
        let prod_lo = product as u64;
        let prod_hi = (product >> 64) as u64;
        let (sub1, b1) = tabr[i].overflowing_sub(prod_lo);
        let (sub2, b2) = sub1.overflowing_sub(borrow);
        tabr[i] = sub2;
        borrow = prod_hi + u64::from(b1) + u64::from(b2);
    }
    borrow
}

/// In-place variant of `mp_sub`: `res[..n] -= op2[..n]` (mirrors C's aliased
/// `mp_sub(tab, tab, op2, …)` calls, which Rust's borrow rules cannot express
/// through `mp_sub`).
fn mp_sub_inplace(res: &mut [u64], op2: &[u64], n: usize, mut borrow: u64) -> u64 {
    for i in 0..n {
        let a = res[i];
        let diff = a.wrapping_sub(op2[i]);
        let b1 = u64::from(diff > a);
        let diff2 = diff.wrapping_sub(borrow);
        let b2 = u64::from(diff2 > diff);
        res[i] = diff2;
        borrow = b1 | b2;
    }
    borrow
}

/// In-place variant of `mp_add`: `res[..n] += op2[..n]`.
fn mp_add_inplace(res: &mut [u64], op2: &[u64], n: usize, mut carry: u64) -> u64 {
    for i in 0..n {
        let v = res[i];
        let s = v.wrapping_add(op2[i]);
        let c1 = u64::from(s < v);
        let s2 = s.wrapping_add(carry);
        carry = u64::from(s2 < carry) | c1;
        res[i] = s2;
    }
    carry
}

fn mp_sub(res: &mut [u64], op1: &[u64], op2: &[u64], n: usize, mut borrow: u64) -> u64 {
    for i in 0..n {
        let a = op1[i];
        let diff = a.wrapping_sub(op2[i]);
        let b1 = u64::from(diff > a);
        let diff2 = diff.wrapping_sub(borrow);
        let b2 = u64::from(diff2 > diff);
        res[i] = diff2;
        borrow = b1 | b2;
    }
    borrow
}

fn mp_add(res: &mut [u64], op1: &[u64], op2: &[u64], n: usize, mut carry: u64) -> u64 {
    for i in 0..n {
        let v = op1[i];
        let s = v.wrapping_add(op2[i]);
        let c1 = u64::from(s < v);
        let s2 = s.wrapping_add(carry);
        carry = u64::from(s2 < carry) | c1;
        res[i] = s2;
    }
    carry
}

fn mp_neg(tab: &mut [u64], n: usize) {
    let mut carry = 1_u64;
    for item in tab.iter_mut().take(n) {
        let v = (!*item).wrapping_add(carry);
        carry = if *item == 0 && carry == 1 { 1 } else { 0 };
        *item = v;
    }
}

fn mp_cmp(a: &[u64], b: &[u64], n: usize) -> i32 {
    for i in (0..n).rev() {
        if a[i] != b[i] {
            return if a[i] < b[i] { -1 } else { 1 };
        }
    }
    0
}

fn udiv1norm_init(d: u64) -> u64 {
    let a1 = (!d).wrapping_sub(0); // -d - 1 in unsigned
    let a0 = u64::MAX;
    let a = (u128::from(a1) << 64) | u128::from(a0);
    (a / u128::from(d)) as u64
}

fn udiv1norm(pr: &mut u64, a1: u64, a0: u64, d: u64, d_inv: u64) -> u64 {
    let n1m = (a0 as i64 >> 63) as u64;
    let n_adj = a0.wrapping_add(n1m & d);
    let a_approx = u128::from(d_inv) * u128::from(a1.wrapping_sub(n1m)) + u128::from(n_adj);
    let q = (a_approx >> 64) as u64 + a1;
    let a_full = (u128::from(a1) << 64) | u128::from(a0);
    let a_rem = a_full
        .wrapping_sub(u128::from(q) * u128::from(d))
        .wrapping_sub(u128::from(d));
    let ah = (a_rem >> 64) as u64;
    let q = q.wrapping_add(1).wrapping_add(ah);
    let r = (a_rem as u64).wrapping_add(ah & d);
    *pr = r;
    q
}

fn mp_div1norm(tabr: &mut [u64], taba: &[u64], n: usize, b: u64, mut r: u64) -> u64 {
    if n >= UDIV1NORM_THRESHOLD {
        let b_inv = udiv1norm_init(b);
        for i in (0..n).rev() {
            let mut new_r = 0_u64;
            tabr[i] = udiv1norm(&mut new_r, r, taba[i], b, b_inv);
            r = new_r;
        }
    } else {
        for i in (0..n).rev() {
            let al = (u128::from(r) << 64) | u128::from(taba[i]);
            tabr[i] = (al / u128::from(b)) as u64;
            r = (al % u128::from(b)) as u64;
        }
    }
    r
}

pub(crate) fn mp_divnorm(tabq: &mut [u64], taba: &mut [u64], na: usize, tabb: &[u64], nb: usize) {
    let b1 = tabb[nb - 1];
    if nb == 1 {
        taba[0] = mp_div1norm(tabq, taba, na, b1, 0);
        return;
    }
    let n = na - nb;

    if n >= DIVNORM_LARGE_THRESHOLD && nb >= DIVNORM_LARGE_THRESHOLD {
        mp_divnorm_large(tabq, taba, na, tabb, nb);
        return;
    }

    let b1_inv = if n >= UDIV1NORM_THRESHOLD {
        udiv1norm_init(b1)
    } else {
        0
    };

    let mut q: u64 = 1;
    for j in (0..nb).rev() {
        if taba[n + j] != tabb[j] {
            if taba[n + j] < tabb[j] {
                q = 0;
            }
            break;
        }
    }
    tabq[n] = q;
    if q != 0 {
        let mut borrow = 0_u64;
        for j in 0..nb {
            let diff = taba[n + j].wrapping_sub(tabb[j]).wrapping_sub(borrow);
            borrow = u64::from(
                taba[n + j] < tabb[j].wrapping_add(borrow) || (borrow != 0 && tabb[j] == u64::MAX),
            );
            taba[n + j] = diff;
        }
    }

    for i in (0..n).rev() {
        let q_val;
        if taba[i + nb] >= b1 {
            q_val = u64::MAX;
        } else if b1_inv != 0 {
            let mut dummy_r = 0_u64;
            q_val = udiv1norm(&mut dummy_r, taba[i + nb], taba[i + nb - 1], b1, b1_inv);
        } else {
            let al = (u128::from(taba[i + nb]) << 64) | u128::from(taba[i + nb - 1]);
            q_val = (al / u128::from(b1)) as u64;
        }

        let r = mp_sub_mul1(&mut taba[i..], tabb, nb, q_val);
        let v = taba[i + nb];
        let a = v.wrapping_sub(r);
        let c = u64::from(a > v);
        taba[i + nb] = a;

        let mut q_val = q_val;
        if c != 0 {
            loop {
                q_val = q_val.wrapping_sub(1);
                let carry = mp_add_inplace(&mut taba[i..], tabb, nb, 0);
                if carry != 0 {
                    taba[i + nb] = taba[i + nb].wrapping_add(1);
                    if taba[i + nb] == 0 {
                        break;
                    }
                }
            }
        }
        tabq[i] = q_val;
    }
}

fn mp_divnorm_large(tabq: &mut [u64], taba: &mut [u64], na: usize, tabb: &[u64], nb: usize) {
    let nq = na - nb;
    assert!(nq >= 1);
    let n = if nq < nb { nq + 1 } else { nq };

    let mut tabt = vec![0_u64; n];
    let mut tabb_inv = vec![0_u64; n + 1];
    let mut recip_done = false;
    if n >= nb {
        for i in 0..nb {
            tabt[i + n - nb] = tabb[i];
        }
    } else {
        // Truncate B: increment it so that the approximate inverse is
        // smaller than the exact inverse.
        for i in 0..n {
            tabt[i] = tabb[i + nb - n];
        }
        if mp_add_ui(&mut tabt, 1, n) != 0 {
            // tabt == B^n, so the reciprocal is exactly B^n
            // (mirrors C mp_divnorm_large's `recip_done` path).
            tabb_inv[n] = 1;
            recip_done = true;
        }
    }
    if !recip_done {
        mp_recip_internal(&mut tabb_inv, &tabt, n);
    }

    // Q = A * B^-1, using the top n+1 limbs of A (na >= n+1 since nb >= 2)
    let mut prod = vec![0_u64; 2 * (n + 1)];
    let a_start = na - (n + 1);
    super::ntt::mp_mul_into(
        &mut prod,
        &tabb_inv[..n + 1],
        &taba[a_start..a_start + n + 1],
    );

    let q_start = 2 * (n + 1) - (nq + 1);
    tabq[..=nq].copy_from_slice(&prod[q_start..q_start + nq + 1]);

    let mut bq = vec![0_u64; na + 1];
    super::ntt::mp_mul_into(&mut bq, &tabq[..nq + 1], &tabb[..nb]);
    mp_sub_inplace(taba, &bq, nb + 1, 0);

    loop {
        if taba[nb] == 0 && mp_cmp(taba, tabb, nb) < 0 {
            break;
        }
        let borrow = mp_sub_inplace(taba, tabb, nb, 0);
        taba[nb] = taba[nb].wrapping_sub(borrow);
        mp_add_ui(tabq, 1, nq + 1);
    }
}

fn mp_recip_internal(tabr: &mut [u64], taba: &[u64], n: usize) {
    if n <= 2 {
        // Stack buffers: 2n+1 <= 5 and n+2 <= 4
        let mut tabu = [0_u64; 5];
        let mut tabt = [0_u64; 4];
        tabu[2 * n] = 1;
        mp_divnorm(
            &mut tabt[..n + 2],
            &mut tabu[..2 * n + 1],
            2 * n + 1,
            taba,
            n,
        );
        tabr[..=n].copy_from_slice(&tabt[..=n]);
        if !mp_scan_nz(&tabu, n) {
            mp_sub_ui(tabr, 1, n + 1);
        }
        return;
    }

    let l = (n - 1) / 2;
    let h = n - l;

    mp_recip_internal(&mut tabr[l..], &taba[l..], h);

    let mut tabt = vec![0_u64; n + h + 1];
    super::ntt::mp_mul_into(&mut tabt, &taba[..n], &tabr[l..l + h + 1]);

    while tabt[n + h] != 0 {
        mp_sub_ui(&mut tabr[l..], 1, h + 1);
        let c = mp_sub_inplace(&mut tabt, &taba[..n], n, 0);
        mp_sub_ui(&mut tabt[n..], c, h + 1);
    }

    mp_neg(&mut tabt, n + h + 1);
    tabt[n + h] = tabt[n + h].wrapping_add(1);

    let mut tabu = vec![0_u64; n + 2 * h - l + 2];
    super::ntt::mp_mul_into(&mut tabu, &tabt[l..n + h + 1], &tabr[l..l + h + 1]);

    let k = 2 * h - l;
    tabr[..l].copy_from_slice(&tabu[k..k + l]);
    let carry = mp_add_inplace(&mut tabr[l..], &tabu[2 * h..2 * h + h], h, 0);
    let _ = carry;
}

const SQRT_TABLE: [u16; 192] = [
    128, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 144,
    145, 146, 147, 148, 149, 150, 150, 151, 152, 153, 154, 155, 155, 156, 157, 158, 159, 160, 160,
    161, 162, 163, 163, 164, 165, 166, 167, 167, 168, 169, 170, 170, 171, 172, 173, 173, 174, 175,
    176, 176, 177, 178, 178, 179, 180, 181, 181, 182, 183, 183, 184, 185, 185, 186, 187, 187, 188,
    189, 189, 190, 191, 192, 192, 193, 193, 194, 195, 195, 196, 197, 197, 198, 199, 199, 200, 201,
    201, 202, 203, 203, 204, 204, 205, 206, 206, 207, 208, 208, 209, 209, 210, 211, 211, 212, 212,
    213, 214, 214, 215, 215, 216, 217, 217, 218, 218, 219, 219, 220, 221, 221, 222, 222, 223, 224,
    224, 225, 225, 226, 226, 227, 227, 228, 229, 229, 230, 230, 231, 231, 232, 232, 233, 234, 234,
    235, 235, 236, 236, 237, 237, 238, 238, 239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244,
    245, 245, 246, 246, 247, 247, 248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254,
    254, 255,
];

pub(crate) fn mp_sqrtrem1(pr: &mut u64, a: u64) -> u64 {
    let mut s1 = SQRT_TABLE[((a >> 56) - 64) as usize] as u64;
    let mut r1 = (a >> 48) - s1 * s1;
    if r1 > 2 * s1 {
        r1 -= 2 * s1 + 1;
        s1 += 1;
    }

    // 16→32 bit sqrt
    let num = (r1 << 8) | ((a >> (64 - 32 + 8)) & 0xff);
    let q = num / (2 * s1);
    let u = num % (2 * s1);
    let mut s: u64 = (s1 << 8) + q;
    let mut r: u64 = ((u << 8) | ((a >> (64 - 32)) & 0xff)).wrapping_sub(q * q);
    if (r as i64) < 0 {
        s -= 1;
        r = r.wrapping_add(2 * s + 1);
    }

    // 32→64 bit sqrt
    s1 = s;
    r1 = r;
    let num = (r1 << 16) | ((a >> 16) & 0xffff);
    let q = num / (2 * s1);
    let u = num % (2 * s1);
    s = (s1 << 16) + q;
    r = ((u << 16) | (a & 0xffff)).wrapping_sub(q * q);
    if (r as i64) < 0 {
        s -= 1;
        r = r.wrapping_add(2 * s + 1);
    }

    *pr = r;
    s
}

fn mp_sqrtrem2(tabs: &mut [u64], taba: &mut [u64]) -> u64 {
    let a0 = taba[0];
    let a1 = taba[1];
    let mut r1 = 0_u64;
    let s1 = mp_sqrtrem1(&mut r1, a1);
    let l = 32_u32;
    let num = (u128::from(r1) << l) | u128::from(a0 >> l);
    let q = (num / (2 * u128::from(s1))) as u64;
    let u = (num % (2 * u128::from(s1))) as u64;
    let s = (s1 << l).wrapping_add(q);
    let mut r = (u128::from(u) << l) | u128::from(a0 & ((1_u64 << l) - 1));
    if (q >> l) != 0 {
        r = r.wrapping_sub(1_u128 << 64);
    } else {
        r = r.wrapping_sub(u128::from(q) * u128::from(q));
    }
    if ((r >> 64) as i64) < 0 {
        let s2 = s.wrapping_sub(1);
        r = r.wrapping_add(2 * u128::from(s2) + 1);
        tabs[0] = s2;
        taba[0] = r as u64;
        return (r >> 64) as u64;
    }
    tabs[0] = s;
    taba[0] = r as u64;
    (r >> 64) as u64
}

/// Recursive Karatsuba square root. Mirrors C `mp_sqrtrem_rec` allocation-free:
/// `tmp_buf` (>= n/2+1 limbs) holds the quotient across all recursion levels and
/// `taba[n..n+2*(n/2)]` is used as scratch for the q^2 product, exactly like C.
/// `taba` must have at least `n + 2*(n/2)` limbs.
pub(crate) fn mp_sqrtrem_rec(
    tabs: &mut [u64],
    taba: &mut [u64],
    n: usize,
    tmp_buf: &mut [u64],
    prh: &mut u64,
) {
    if n == 1 {
        *prh = mp_sqrtrem2(tabs, taba);
        return;
    }

    let l = n / 2;
    let h = n - l;

    let mut qh = 0_u64;
    mp_sqrtrem_rec(&mut tabs[l..], &mut taba[2 * l..], h, tmp_buf, &mut qh);

    // The remainder is in taba[2l..]. Its high bit is in qh.
    if qh != 0 {
        mp_sub_inplace(&mut taba[2 * l..], &tabs[l..l + h], h, 0);
    }

    // Instead of dividing by 2*s, divide by s (which is normalized)
    // and update q and r.
    mp_divnorm(
        &mut tmp_buf[..l + 1],
        &mut taba[l..l + n],
        n,
        &tabs[l..l + h],
        h,
    );
    qh += tmp_buf[l];
    let ql = mp_shr(&mut tabs[..l], &tmp_buf[..l], l, 1, qh & 1);
    qh >>= 1; // 0 or 1

    let mut rh: u64 = if ql != 0 {
        mp_add_inplace(&mut taba[l..], &tabs[l..l + h], h, 0)
    } else {
        0
    };

    mp_add_ui(&mut tabs[l..], qh, h);

    // q = qh, tabs[l-1..0], r = taba[n-1..l]
    // Subtract q^2. If qh = 1 then q = B^l, so we can take a shortcut.
    let c: u64 = if qh != 0 {
        qh
    } else {
        let (taba_lo, taba_hi) = taba.split_at_mut(n);
        super::ntt::mp_mul_into(&mut taba_hi[..2 * l], &tabs[..l], &tabs[..l]);
        mp_sub_inplace(&mut taba_lo[..2 * l], &taba_hi[..2 * l], 2 * l, 0)
    };
    rh = rh.wrapping_sub(mp_sub_ui(&mut taba[2 * l..], c, n - 2 * l));
    if (rh as i64) < 0 {
        mp_sub_ui(tabs, 1, n);
        rh = rh.wrapping_add(mp_add_mul1(taba, tabs, n, 2));
        rh = rh.wrapping_add(mp_add_ui(taba, 1, n));
    }
    *prh = rh;
}

pub(crate) fn mp_sqrtrem_full(tabs: &mut [u64], taba: &mut [u64], n: usize) {
    let n2 = n / 2 + 1;
    // Stack buffer for small sizes, mirroring C mp_sqrtrem's tmp_buf1[8]
    let mut tmp_buf1 = [0_u64; 8];
    let mut tmp_vec;
    let tmp_buf: &mut [u64] = if n2 <= 8 {
        &mut tmp_buf1
    } else {
        tmp_vec = vec![0_u64; n2];
        &mut tmp_vec
    };
    let mut rh = 0_u64;
    mp_sqrtrem_rec(tabs, taba, n, tmp_buf, &mut rh);
    // Store the high remainder limb at taba[n], matching libbf's convention
    if taba.len() > n {
        taba[n] = rh;
    }
}

pub(crate) fn mp_recip_full(tabr: &mut [u64], taba: &[u64], n: usize) {
    mp_recip_internal(tabr, taba, n);
}
