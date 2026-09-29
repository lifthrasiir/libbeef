use alloc::vec;
use alloc::vec::Vec;

use super::{mp_mul_basecase_abs, trim_limbs};

const LIMB_BITS: u32 = 64;
const LIMB_LOG2_BITS: u32 = 6;
const FFT_MUL_THRESHOLD: usize = 100;

const NTT_MOD_LOG2_MIN: u32 = 61;
const NTT_MOD_LOG2_MAX: u32 = 62;
const NB_MODS: usize = 5;
const NTT_PROOT_2EXP: usize = 51;
const NTT_TRIG_K_MAX: usize = 19;
const STRIP_LEN: usize = 16;

static NTT_INT_BITS: [i32; NB_MODS] = [307, 246, 185, 123, 61];

static NTT_MODS: [u64; NB_MODS] = [
    0x28d8000000000001,
    0x2a88000000000001,
    0x2ed8000000000001,
    0x3508000000000001,
    0x3aa8000000000001,
];

static NTT_PROOT: [[u64; NB_MODS]; 2] = [
    [
        0x1b8ea61034a2bea7,
        0x21a9762de58206fb,
        0x02ca782f0756a8ea,
        0x278384537a3e50a1,
        0x106e13fee74ce0ab,
    ],
    [
        0x233513af133e13b8,
        0x1d13140d1c6f75f1,
        0x12cde57f97e3eeda,
        0x0d6149e23cbe654f,
        0x36cd204f522a1379,
    ],
];

static NTT_MODS_CR: [u64; NB_MODS * (NB_MODS - 1) / 2] = [
    0x08a9ed097b425eea,
    0x18a44aaaaaaaaab3,
    0x2493f57f57f57f5d,
    0x126b8d0649a7f8d4,
    0x09d80ed7303b5ccc,
    0x25b8bcf3cf3cf3d5,
    0x2ce6ce63398ce638,
    0x0e31fad40a57eb59,
    0x02a3529fd4a7f52f,
    0x3a5493e93e93e94a,
];

type NTTLimb = u64;

struct BFNTTState {
    ntt_mods_div: [u64; NB_MODS],
    ntt_proot_pow: [[[u64; NTT_PROOT_2EXP + 1]; 2]; NB_MODS],
    ntt_proot_pow_inv: [[[u64; NTT_PROOT_2EXP + 1]; 2]; NB_MODS],
    ntt_trig: Vec<Vec<Vec<Option<Vec<NTTLimb>>>>>,
    ntt_len_inv: [[[u64; 2]; NTT_PROOT_2EXP + 1]; NB_MODS],
    ntt_mods_cr_inv: [u64; NB_MODS * (NB_MODS - 1) / 2],
}

#[inline(always)]
fn add_mod(a: u64, b: u64, m: u64) -> u64 {
    let r = a.wrapping_add(b);
    if r >= m {
        r - m
    } else {
        r
    }
}

#[inline(always)]
fn sub_mod(a: u64, b: u64, m: u64) -> u64 {
    let r = a.wrapping_sub(b);
    if r > a {
        r.wrapping_add(m)
    } else {
        r
    }
}

#[inline(always)]
fn mod_fast(r: u128, m: u64, m_inv: u64) -> u64 {
    let a1 = (r >> NTT_MOD_LOG2_MIN) as u64;
    let q = ((a1 as u128).wrapping_mul(m_inv as u128) >> LIMB_BITS) as u64;
    let mut r = r
        .wrapping_sub((q as u128).wrapping_mul(m as u128))
        .wrapping_sub(m as u128 * 2);
    let r1 = (r >> LIMB_BITS) as u64;
    let t0 = (r1 as i64 >> 1) as u64;
    r = r.wrapping_add((m as u128) & (t0 as u128));
    let r0 = r as u64;
    let r1 = (r >> LIMB_BITS) as u64;
    r0.wrapping_add(m & r1)
}

#[inline(always)]
fn mul_mod_fast(a: u64, b: u64, m: u64, m_inv: u64) -> u64 {
    mod_fast(a as u128 * b as u128, m, m_inv)
}

#[inline(always)]
fn init_mul_mod_fast(m: u64) -> u64 {
    let t: u128 = 1_u128 << (LIMB_BITS + NTT_MOD_LOG2_MIN);
    (t / m as u128) as u64
}

#[inline(always)]
fn mul_mod_fast2(a: u64, b: u64, m: u64, b_inv: u64) -> u64 {
    let q = ((a as u128).wrapping_mul(b_inv as u128) >> LIMB_BITS) as u64;
    let mut r = a.wrapping_mul(b).wrapping_sub(q.wrapping_mul(m));
    if r >= m {
        r -= m;
    }
    r
}

#[inline(always)]
fn mul_mod_fast3(a: u64, b: u64, m: u64, b_inv: u64) -> u64 {
    let q = ((a as u128).wrapping_mul(b_inv as u128) >> LIMB_BITS) as u64;
    a.wrapping_mul(b).wrapping_sub(q.wrapping_mul(m))
}

#[inline(always)]
fn init_mul_mod_fast2(b: u64, m: u64) -> u64 {
    ((b as u128) << LIMB_BITS).wrapping_div(m as u128) as u64
}

#[inline(always)]
fn ntt_limb_to_int(a: NTTLimb, m: u64) -> u64 {
    if a >= m {
        a - m
    } else {
        a
    }
}

#[inline(always)]
fn int_to_ntt_limb(a: u64, _m: u64) -> NTTLimb {
    a
}

fn get_bits(tab: &[u64], len: usize, pos: i64) -> u64 {
    if pos < 0 {
        return 0;
    }
    let i = (pos >> LIMB_LOG2_BITS) as usize;
    let p = (pos & (LIMB_BITS as i64 - 1)) as u32;
    let a0 = if i < len { tab[i] } else { 0 };
    if p == 0 {
        a0
    } else {
        let a1 = if i + 1 < len { tab[i + 1] } else { 0 };
        (a0 >> p) | (a1 << (LIMB_BITS - p))
    }
}

fn put_bits(tab: &mut [u64], len: usize, pos: i64, val: u64) {
    let i = (pos >> LIMB_LOG2_BITS) as usize;
    let p = (pos & (LIMB_BITS as i64 - 1)) as u32;
    if i < len {
        tab[i] |= val << p;
    }
    if p != 0 {
        let i1 = i + 1;
        if i1 < len {
            tab[i1] |= val >> (LIMB_BITS - p);
        }
    }
}

fn ceil_log2(a: u64) -> i32 {
    if a <= 1 {
        0
    } else {
        (LIMB_BITS - (a - 1).leading_zeros()) as i32
    }
}

impl BFNTTState {
    #[allow(clippy::needless_range_loop)]
    fn new() -> Self {
        let mut s = BFNTTState {
            ntt_mods_div: [0; NB_MODS],
            ntt_proot_pow: [[[0; NTT_PROOT_2EXP + 1]; 2]; NB_MODS],
            ntt_proot_pow_inv: [[[0; NTT_PROOT_2EXP + 1]; 2]; NB_MODS],
            ntt_trig: (0..NB_MODS)
                .map(|_| {
                    (0..2)
                        .map(|_| (0..NTT_TRIG_K_MAX + 1).map(|_| None).collect())
                        .collect()
                })
                .collect(),
            ntt_len_inv: [[[0; 2]; NTT_PROOT_2EXP + 1]; NB_MODS],
            ntt_mods_cr_inv: [0; NB_MODS * (NB_MODS - 1) / 2],
        };

        for j in 0..NB_MODS {
            let m = NTT_MODS[j];
            let m_inv = init_mul_mod_fast(m);
            s.ntt_mods_div[j] = m_inv;
            let c_inv2 = m.div_ceil(2);
            let mut c_inv = 1_u64;
            for i in 0..=NTT_PROOT_2EXP {
                s.ntt_len_inv[j][i][0] = c_inv;
                s.ntt_len_inv[j][i][1] = init_mul_mod_fast2(c_inv, m);
                c_inv = mul_mod_fast(c_inv, c_inv2, m, m_inv);
            }
            for inverse in 0..2 {
                let mut c = NTT_PROOT[inverse][j];
                for i in 0..NTT_PROOT_2EXP {
                    s.ntt_proot_pow[j][inverse][NTT_PROOT_2EXP - i] = c;
                    s.ntt_proot_pow_inv[j][inverse][NTT_PROOT_2EXP - i] = init_mul_mod_fast2(c, m);
                    c = mul_mod_fast(c, c, m, m_inv);
                }
            }
        }
        let mut l = 0;
        for j in 0..NB_MODS - 1 {
            for k in j + 1..NB_MODS {
                s.ntt_mods_cr_inv[l] = init_mul_mod_fast2(NTT_MODS_CR[l], NTT_MODS[k]);
                l += 1;
            }
        }
        s
    }

    fn get_trig(&mut self, k: usize, inverse: usize, m_idx: usize) -> Option<&[NTTLimb]> {
        if k > NTT_TRIG_K_MAX {
            return None;
        }
        if self.ntt_trig[m_idx][inverse][k].is_some() {
            return self.ntt_trig[m_idx][inverse][k].as_deref();
        }
        let n2 = 1_usize << (k - 1);
        let m = NTT_MODS[m_idx];
        let mut tab = vec![0_u64; n2 * 2];
        let mut c = 1_u64;
        let c_mul = self.ntt_proot_pow[m_idx][inverse][k];
        let c_mul_inv = self.ntt_proot_pow_inv[m_idx][inverse][k];
        for i in 0..n2 {
            tab[2 * i] = int_to_ntt_limb(c, m);
            tab[2 * i + 1] = init_mul_mod_fast2(c, m);
            c = mul_mod_fast2(c, c_mul, m, c_mul_inv);
        }
        self.ntt_trig[m_idx][inverse][k] = Some(tab);
        self.ntt_trig[m_idx][inverse][k].as_deref()
    }

    fn ntt_fft(
        &mut self,
        out_buf: &mut [NTTLimb],
        in_buf: &[NTTLimb],
        fft_len_log2: usize,
        inverse: usize,
        m_idx: usize,
    ) -> bool {
        let m = NTT_MODS[m_idx];
        let m2 = 2_u64.wrapping_mul(m);
        let n = 1_usize << fft_len_log2;
        let stride_in = n / 2;

        let mut buf_a = in_buf.to_vec();
        let mut buf_b = vec![0_u64; n];
        let mut a_is_current = true;
        let mut l = fft_len_log2;
        let mut nb_blocks = n;
        let mut fft_per_block = 1_usize;

        while nb_blocks != 2 {
            nb_blocks >>= 1;
            let trig = match self.get_trig(l, inverse, m_idx) {
                Some(t) => t.to_vec(),
                None => return false,
            };
            let mut p = 0_usize;
            let mut k = 0_usize;
            let mut ti = 0_usize;
            for _i in 0..nb_blocks {
                let c = trig[ti];
                let c_inv = trig[ti + 1];
                ti += 2;
                for j in 0..fft_per_block {
                    let (a0, a1) = if a_is_current {
                        (buf_a[k + j], buf_a[k + j + stride_in])
                    } else {
                        (buf_b[k + j], buf_b[k + j + stride_in])
                    };
                    let b0 = add_mod(a0, a1, m2);
                    let b1 = mul_mod_fast3(a0.wrapping_sub(a1).wrapping_add(m2), c, m, c_inv);
                    if a_is_current {
                        buf_b[p + j] = b0;
                        buf_b[p + j + fft_per_block] = b1;
                    } else {
                        buf_a[p + j] = b0;
                        buf_a[p + j + fft_per_block] = b1;
                    }
                }
                k += fft_per_block;
                p += 2 * fft_per_block;
            }
            fft_per_block <<= 1;
            l -= 1;
            a_is_current = !a_is_current;
        }

        let src = if a_is_current { &buf_a } else { &buf_b };
        for k in 0..stride_in {
            let a0 = src[k];
            let a1 = src[k + stride_in];
            out_buf[k] = add_mod(a0, a1, m2);
            out_buf[k + stride_in] = sub_mod(a0, a1, m2);
        }
        true
    }

    fn ntt_vec_mul(
        &self,
        tab1: &mut [NTTLimb],
        tab2: &[NTTLimb],
        fft_len_log2: usize,
        k_tot: usize,
        m_idx: usize,
    ) {
        let m = NTT_MODS[m_idx];
        let m_inv = self.ntt_mods_div[m_idx];
        let norm = self.ntt_len_inv[m_idx][k_tot][0];
        let norm_inv = self.ntt_len_inv[m_idx][k_tot][1];
        let n = 1_usize << fft_len_log2;
        for i in 0..n {
            let mut a = tab1[i];
            if a >= m {
                a -= m;
            }
            a = mul_mod_fast(a, tab2[i], m, m_inv);
            a = mul_mod_fast3(a, norm, m, norm_inv);
            tab1[i] = a;
        }
    }
}

fn mul_trig(buf: &mut [NTTLimb], n: usize, c_mul: u64, m: u64, m_inv: u64) {
    let mut c0 = 1_u64;
    let c_mul_inv = init_mul_mod_fast2(c_mul, m);
    for item in buf.iter_mut().take(n) {
        *item = mul_mod_fast(*item, c0, m, m_inv);
        c0 = mul_mod_fast2(c0, c_mul, m, c_mul_inv);
    }
}

#[allow(clippy::too_many_arguments)]
fn ntt_fft_partial(
    s: &mut BFNTTState,
    buf1: &mut [NTTLimb],
    k1: usize,
    k2: usize,
    n1: usize,
    n2: usize,
    inverse: usize,
    m_idx: usize,
) -> bool {
    if k2 == 0 {
        let in_copy = buf1[..n1].to_vec();
        return s.ntt_fft(buf1, &in_copy, k1, inverse, m_idx);
    }
    let strip_len = STRIP_LEN;
    let mut buf2 = vec![0_u64; n1 * strip_len];
    let m = NTT_MODS[m_idx];
    let m_inv = s.ntt_mods_div[m_idx];
    let c0 = s.ntt_proot_pow[m_idx][inverse][k1 + k2];
    let mut c_mul = 1_u64;

    let mut j = 0_usize;
    while j < n2 {
        for i in 0..n1 {
            for l in 0..strip_len {
                buf2[i + l * n1] = buf1[i * n2 + (j + l)];
            }
        }
        for l in 0..strip_len {
            if inverse != 0 {
                mul_trig(&mut buf2[l * n1..], n1, c_mul, m, m_inv);
            }
            let in_copy = buf2[l * n1..l * n1 + n1].to_vec();
            if !s.ntt_fft(&mut buf2[l * n1..l * n1 + n1], &in_copy, k1, inverse, m_idx) {
                return false;
            }
            if inverse == 0 {
                mul_trig(&mut buf2[l * n1..], n1, c_mul, m, m_inv);
            }
            c_mul = mul_mod_fast(c_mul, c0, m, m_inv);
        }
        for i in 0..n1 {
            for l in 0..strip_len {
                buf1[i * n2 + (j + l)] = buf2[i + l * n1];
            }
        }
        j += strip_len;
    }
    true
}

fn ntt_conv(
    s: &mut BFNTTState,
    buf1: &mut [NTTLimb],
    buf2: &mut [NTTLimb],
    k: usize,
    k_tot: usize,
    m_idx: usize,
) -> bool {
    let k1 = if k <= NTT_TRIG_K_MAX {
        k
    } else {
        (k / 2).min(NTT_TRIG_K_MAX)
    };
    let k2 = k - k1;
    let n1 = 1_usize << k1;
    let n2 = 1_usize << k2;

    if !ntt_fft_partial(s, buf1, k1, k2, n1, n2, 0, m_idx) {
        return false;
    }
    if !ntt_fft_partial(s, buf2, k1, k2, n1, n2, 0, m_idx) {
        return false;
    }
    if k2 == 0 {
        let b2c = buf2.to_vec();
        s.ntt_vec_mul(buf1, &b2c, k, k_tot, m_idx);
    } else {
        for i in 0..n1 {
            let off = i * n2;
            let mut b2s = buf2[off..off + n2].to_vec();
            if !ntt_conv(s, &mut buf1[off..off + n2], &mut b2s, k2, k_tot, m_idx) {
                return false;
            }
        }
    }
    if !ntt_fft_partial(s, buf1, k1, k2, n1, n2, 1, m_idx) {
        return false;
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn limb_to_ntt(
    s: &BFNTTState,
    tabr: &mut [NTTLimb],
    fft_len: usize,
    taba: &[u64],
    a_len: usize,
    dpl: i32,
    first_m_idx: usize,
    nb_mods: usize,
) {
    for v in tabr.iter_mut() {
        *v = 0;
    }
    let shift = dpl & (LIMB_BITS as i32 - 1);
    let base_mask1: u64 = if shift == 0 {
        u64::MAX
    } else {
        (1_u64 << shift) - 1
    };
    let n = (fft_len as i64).min((a_len as i64 * LIMB_BITS as i64 + dpl as i64 - 1) / dpl as i64);
    for i in 0..n as usize {
        let a0 = get_bits(taba, a_len, i as i64 * dpl as i64);
        let (a, a0_low) = if dpl <= LIMB_BITS as i32 {
            ((a0 & base_mask1) as u128, 0_u64)
        } else {
            let a1 = get_bits(taba, a_len, i as i64 * dpl as i64 + LIMB_BITS as i64);
            if dpl <= (LIMB_BITS as i32 + NTT_MOD_LOG2_MIN as i32) {
                (a0 as u128 | ((a1 & base_mask1) as u128) << LIMB_BITS, 0_u64)
            } else {
                let (a1v, a2) = if dpl > 2 * LIMB_BITS as i32 {
                    (
                        a1,
                        get_bits(taba, a_len, i as i64 * dpl as i64 + LIMB_BITS as i64 * 2)
                            & base_mask1,
                    )
                } else {
                    (a1 & base_mask1, 0_u64)
                };
                let sa = LIMB_BITS - NTT_MOD_LOG2_MAX + NTT_MOD_LOG2_MIN;
                let av = (a0 >> sa) as u128
                    | ((a1v as u128) << (NTT_MOD_LOG2_MAX - NTT_MOD_LOG2_MIN))
                    | ((a2 as u128) << (LIMB_BITS + NTT_MOD_LOG2_MAX - NTT_MOD_LOG2_MIN));
                (av, a0 & ((1_u64 << sa) - 1))
            }
        };
        for j in 0..nb_mods {
            let m = NTT_MODS[first_m_idx + j];
            let m_inv = s.ntt_mods_div[first_m_idx + j];
            let mut r = mod_fast(a, m, m_inv);
            if dpl > (LIMB_BITS as i32 + NTT_MOD_LOG2_MIN as i32) {
                let sa = LIMB_BITS - NTT_MOD_LOG2_MAX + NTT_MOD_LOG2_MIN;
                let b = ((r as u128) << sa) | a0_low as u128;
                r = mod_fast(b, m, m_inv);
            }
            tabr[i + j * fft_len] = int_to_ntt_limb(r, m);
        }
    }
}

fn ntt_to_limb(
    s: &BFNTTState,
    tabr: &mut [u64],
    r_len: usize,
    buf: &[NTTLimb],
    fft_len_log2: usize,
    dpl: i32,
    nb_mods: usize,
) {
    let mo = NB_MODS - nb_mods;
    let mods = &NTT_MODS[mo..];
    let cr_off = NB_MODS * (NB_MODS - 1) / 2 - nb_mods * (nb_mods - 1) / 2;
    let mods_cr = &NTT_MODS_CR[cr_off..];
    let mods_cr_inv = &s.ntt_mods_cr_inv[cr_off..];
    let shift = dpl & (LIMB_BITS as i32 - 1);
    let base_mask1: u64 = if shift == 0 {
        u64::MAX
    } else {
        (1_u64 << shift) - 1
    };
    let n_limb1 = ((dpl as u32 - 1) / LIMB_BITS) as usize;
    let mut carry = [0_u64; NB_MODS];
    let mut u = [0_u64; NB_MODS];
    for v in tabr[..r_len].iter_mut() {
        *v = 0;
    }
    let fft_len = 1_usize << fft_len_log2;
    let len = (fft_len as i64).min((r_len as i64 * LIMB_BITS as i64 + dpl as i64 - 1) / dpl as i64)
        as usize;

    for i in 0..len {
        let mut y = [0_u64; NB_MODS];
        for j in 0..nb_mods {
            y[j] = ntt_limb_to_int(buf[i + fft_len * j], mods[j]);
        }
        let mut l = 0_usize;
        for j in 0..nb_mods - 1 {
            for k in j + 1..nb_mods {
                let m = mods[k];
                y[k] = mul_mod_fast2(
                    y[k].wrapping_sub(y[j]).wrapping_add(m),
                    mods_cr[l],
                    m,
                    mods_cr_inv[l],
                );
                l += 1;
            }
        }
        u[0] = y[nb_mods - 1];
        let mut lc = 1_usize;
        for j in (1..nb_mods - 1).rev() {
            let mut r = y[j];
            for item in u.iter_mut().take(lc) {
                let t = *item as u128 * mods[j] as u128 + r as u128;
                r = (t >> LIMB_BITS) as u64;
                *item = t as u64;
            }
            u[lc] = r;
            lc += 1;
        }
        let mut r = y[0];
        for k in 0..lc {
            let t = u[k] as u128 * mods[0] as u128 + r as u128 + carry[k] as u128;
            r = (t >> LIMB_BITS) as u64;
            u[k] = t as u64;
        }
        u[lc] = r.wrapping_add(carry[lc]);

        let pos = i as i64 * dpl as i64;
        for (j, &item) in u.iter().enumerate().take(n_limb1) {
            put_bits(tabr, r_len, pos + j as i64 * LIMB_BITS as i64, item);
        }
        put_bits(
            tabr,
            r_len,
            pos + n_limb1 as i64 * LIMB_BITS as i64,
            u[n_limb1] & base_mask1,
        );

        if shift == 0 {
            for j in n_limb1 + 1..nb_mods {
                carry[j - (n_limb1 + 1)] = u[j];
            }
        } else {
            for j in n_limb1..nb_mods - 1 {
                carry[j - n_limb1] =
                    (u[j] >> shift as u32) | (u[j + 1] << (LIMB_BITS - shift as u32));
            }
            carry[nb_mods - 1 - n_limb1] = u[nb_mods - 1] >> shift as u32;
        }
    }
}

pub(crate) fn bf_get_fft_size(len: u64) -> (i32, i32, i32) {
    let mut min_cost = u64::MAX;
    let mut dpl_found = 0_i32;
    let mut nb_mods_found = 4_i32;
    let mut fft_len_log2_found = 0_i32;
    for nb_mods in 3..=NB_MODS as i32 {
        let int_bits = NTT_INT_BITS[(NB_MODS as i32 - nb_mods) as usize];
        let mut dpl = ((int_bits - 4) / 2)
            .min(2 * LIMB_BITS as i32 + 2 * NTT_MOD_LOG2_MIN as i32 - NTT_MOD_LOG2_MAX as i32);
        loop {
            let fft_len_log2 = ceil_log2((len * LIMB_BITS as u64).div_ceil(dpl as u64));
            if fft_len_log2 > NTT_PROOT_2EXP as i32 {
                break;
            }
            let n_bits = fft_len_log2 + 2 * dpl;
            if n_bits <= int_bits {
                let cost = ((fft_len_log2 as u64 + 1) << fft_len_log2 as u64) * nb_mods as u64;
                if cost < min_cost {
                    min_cost = cost;
                    dpl_found = dpl;
                    nb_mods_found = nb_mods;
                    fft_len_log2_found = fft_len_log2;
                }
                break;
            }
            dpl -= 1;
            if dpl == 0 {
                break;
            }
        }
    }
    if dpl_found != 0
        && dpl_found > (LIMB_BITS as i32 + NTT_MOD_LOG2_MIN as i32)
        && (((LIMB_BITS as i32 + NTT_MOD_LOG2_MIN as i32) as u64) << fft_len_log2_found as u64)
            >= len * LIMB_BITS as u64
    {
        dpl_found = LIMB_BITS as i32 + NTT_MOD_LOG2_MIN as i32;
    }
    (dpl_found, nb_mods_found, fft_len_log2_found)
}

fn fft_mul(
    s: &mut BFNTTState,
    a_tab: &[u64],
    a_len: usize,
    b_tab: &[u64],
    b_len: usize,
) -> Vec<u64> {
    let len = a_len + b_len;
    let (dpl, nb_mods, fft_len_log2) = bf_get_fft_size(len as u64);
    let fft_len = 1_usize << fft_len_log2;
    let nb_mods = nb_mods as usize;
    let mut buf1 = vec![0_u64; fft_len * nb_mods];
    limb_to_ntt(
        s,
        &mut buf1,
        fft_len,
        a_tab,
        a_len,
        dpl,
        NB_MODS - nb_mods,
        nb_mods,
    );
    let reduced_mem = fft_len_log2 >= 14;
    if !reduced_mem {
        let mut buf2 = vec![0_u64; fft_len * nb_mods];
        limb_to_ntt(
            s,
            &mut buf2,
            fft_len,
            b_tab,
            b_len,
            dpl,
            NB_MODS - nb_mods,
            nb_mods,
        );
        for j in 0..nb_mods {
            let off = fft_len * j;
            let mut b2s = buf2[off..off + fft_len].to_vec();
            ntt_conv(
                s,
                &mut buf1[off..off + fft_len],
                &mut b2s,
                fft_len_log2 as usize,
                fft_len_log2 as usize,
                j + NB_MODS - nb_mods,
            );
        }
    } else {
        let mut buf2 = vec![0_u64; fft_len];
        for j in 0..nb_mods {
            limb_to_ntt(
                s,
                &mut buf2,
                fft_len,
                b_tab,
                b_len,
                dpl,
                NB_MODS - nb_mods + j,
                1,
            );
            let off = fft_len * j;
            let mut b2c = buf2.clone();
            ntt_conv(
                s,
                &mut buf1[off..off + fft_len],
                &mut b2c,
                fft_len_log2 as usize,
                fft_len_log2 as usize,
                j + NB_MODS - nb_mods,
            );
        }
    }
    let mut result = vec![0_u64; len];
    ntt_to_limb(
        s,
        &mut result,
        len,
        &buf1,
        fft_len_log2 as usize,
        dpl,
        nb_mods,
    );
    result
}

#[cfg(feature = "std")]
fn with_ntt_state<F, R>(f: F) -> R
where
    F: FnOnce(&mut BFNTTState) -> R,
{
    use std::cell::RefCell;
    thread_local! {
        static NTT_STATE: RefCell<Option<BFNTTState>> = const { RefCell::new(None) };
    }
    NTT_STATE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        if borrow.is_none() {
            *borrow = Some(BFNTTState::new());
        }
        f(borrow.as_mut().unwrap())
    })
}

#[cfg(not(feature = "std"))]
fn with_ntt_state<F, R>(f: F) -> R
where
    F: FnOnce(&mut BFNTTState) -> R,
{
    f(&mut BFNTTState::new())
}

pub(crate) fn mp_mul(op1: &[u64], op1_size: usize, op2: &[u64], op2_size: usize) -> Vec<u64> {
    if op1_size == 0 || op2_size == 0 {
        return Vec::new();
    }
    let (a, b) = if op1_size >= op2_size {
        (&op1[..op1_size], &op2[..op2_size])
    } else {
        (&op2[..op2_size], &op1[..op1_size])
    };
    if b.len() == 1 {
        let multiplier = b[0];
        let mut result = vec![0_u64; a.len() + 1];
        let mut carry = 0_u128;
        for (i, &v) in a.iter().enumerate() {
            let product = v as u128 * multiplier as u128 + carry;
            result[i] = product as u64;
            carry = product >> 64;
        }
        result[a.len()] = carry as u64;
        trim_limbs(&mut result);
        return result;
    }
    if b.len() >= FFT_MUL_THRESHOLD {
        let mut result = with_ntt_state(|s| fft_mul(s, a, a.len(), b, b.len()));
        trim_limbs(&mut result);
        result
    } else {
        let mut result = vec![0_u64; a.len() + b.len()];
        mp_mul_basecase_abs(&mut result, a, b);
        trim_limbs(&mut result);
        result
    }
}

/// Multiply `op1 * op2` into `out`, reusing `out`'s allocation when possible.
/// `out` must not alias the inputs. The result is not trimmed; the caller is
/// expected to normalize (mirrors C `bf_mul` writing into `r->tab`).
pub(crate) fn mp_mul_into_vec(out: &mut Vec<u64>, op1: &[u64], op2: &[u64]) {
    debug_assert!(!op1.is_empty() && !op2.is_empty());
    let (a, b) = if op1.len() >= op2.len() {
        (op1, op2)
    } else {
        (op2, op1)
    };
    if b.len() == 1 {
        let multiplier = b[0];
        out.clear();
        out.resize(a.len() + 1, 0);
        let mut carry = 0_u128;
        for (i, &v) in a.iter().enumerate() {
            let product = v as u128 * multiplier as u128 + carry;
            out[i] = product as u64;
            carry = product >> 64;
        }
        out[a.len()] = carry as u64;
    } else if b.len() >= FFT_MUL_THRESHOLD {
        *out = with_ntt_state(|s| fft_mul(s, a, a.len(), b, b.len()));
    } else {
        out.clear();
        out.resize(a.len() + b.len(), 0);
        mp_mul_basecase_abs(out, a, b);
    }
}

pub(crate) fn mp_mul_into(result: &mut [u64], op1: &[u64], op2: &[u64]) {
    if op1.is_empty() || op2.is_empty() {
        for v in result.iter_mut() {
            *v = 0;
        }
        return;
    }
    let (a, b) = if op1.len() >= op2.len() {
        (op1, op2)
    } else {
        (op2, op1)
    };
    // Fast path: multiply directly into the destination without a temporary
    // (mirrors C mp_mul writing into the caller's buffer).
    if result.len() >= a.len() + b.len() && b.len() < FFT_MUL_THRESHOLD {
        let (dst, rest) = result.split_at_mut(a.len() + b.len());
        for v in rest.iter_mut() {
            *v = 0;
        }
        if b.len() == 1 {
            let multiplier = b[0];
            let mut carry = 0_u128;
            for (i, &v) in a.iter().enumerate() {
                let product = v as u128 * multiplier as u128 + carry;
                dst[i] = product as u64;
                carry = product >> 64;
            }
            dst[a.len()] = carry as u64;
        } else {
            for v in dst.iter_mut() {
                *v = 0;
            }
            mp_mul_basecase_abs(dst, a, b);
        }
        return;
    }
    let product = mp_mul(a, a.len(), b, b.len());
    let copy_len = product.len().min(result.len());
    result[..copy_len].copy_from_slice(&product[..copy_len]);
    for v in &mut result[copy_len..] {
        *v = 0;
    }
}
