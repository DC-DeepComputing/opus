/* Copyright (c) 2026 RISC-V RVV optimization work.
 *
 * RVV butterfly kernels for kiss_fft (float build). Lanes run across
 * independent butterflies; twiddles are scalar within a lane group, so
 * arithmetic per element follows the C reference in the same order.
 * The compiler's FMA contraction of the scalar code may differ from the
 * explicit vector mul/add pairs by <=1 ulp (accepted float policy).
 */
#if defined(OPUS_RISCV_MAY_HAVE_RVV) && !defined(FIXED_POINT) && !defined(ENABLE_QEXT)

int opus_rvv_fft_off = 0; /* debug toggle for A/B harness */
int opus_rvv_verify_run = 0; /* re-entry flag for FFT_VERIFY shadow check */

#include <riscv_vector.h>

#include "kiss_fft.h"
#include "_kiss_fft_guts.h"

/* four cpx at Fout[4i..4i+3], i = lane, stride 32 bytes */
void kf_bfly4_m1_rvv(kiss_fft_cpx *Fout, int N) {
  float *p = (float *)Fout;
  const size_t vlmax = __riscv_vsetvlmax_e32m2();
  int i = 0;
  while (i < N) {
    const size_t vl = __riscv_vsetvl_e32m2((size_t)(N - i));
    const size_t byte_off = (size_t)i * 32;
    float *q = p + (byte_off >> 2);
    vfloat32m2_t x0r = __riscv_vlse32_v_f32m2(q + 0, 32, vl);
    vfloat32m2_t x0i = __riscv_vlse32_v_f32m2(q + 1, 32, vl);
    vfloat32m2_t x1r = __riscv_vlse32_v_f32m2(q + 2, 32, vl);
    vfloat32m2_t x1i = __riscv_vlse32_v_f32m2(q + 3, 32, vl);
    vfloat32m2_t x2r = __riscv_vlse32_v_f32m2(q + 4, 32, vl);
    vfloat32m2_t x2i = __riscv_vlse32_v_f32m2(q + 5, 32, vl);
    vfloat32m2_t x3r = __riscv_vlse32_v_f32m2(q + 6, 32, vl);
    vfloat32m2_t x3i = __riscv_vlse32_v_f32m2(q + 7, 32, vl);

    vfloat32m2_t s0r = __riscv_vfsub_vv_f32m2(x0r, x2r, vl);
    vfloat32m2_t s0i = __riscv_vfsub_vv_f32m2(x0i, x2i, vl);
    x0r = __riscv_vfadd_vv_f32m2(x0r, x2r, vl);
    x0i = __riscv_vfadd_vv_f32m2(x0i, x2i, vl);
    vfloat32m2_t s1r = __riscv_vfadd_vv_f32m2(x1r, x3r, vl);
    vfloat32m2_t s1i = __riscv_vfadd_vv_f32m2(x1i, x3i, vl);
    x2r = __riscv_vfsub_vv_f32m2(x0r, s1r, vl);
    x2i = __riscv_vfsub_vv_f32m2(x0i, s1i, vl);
    x0r = __riscv_vfadd_vv_f32m2(x0r, s1r, vl);
    x0i = __riscv_vfadd_vv_f32m2(x0i, s1i, vl);
    vfloat32m2_t d1r = __riscv_vfsub_vv_f32m2(x1r, x3r, vl);
    vfloat32m2_t d1i = __riscv_vfsub_vv_f32m2(x1i, x3i, vl);

    vfloat32m2_t y1r = __riscv_vfadd_vv_f32m2(s0r, d1i, vl);
    vfloat32m2_t y1i = __riscv_vfsub_vv_f32m2(s0i, d1r, vl);
    vfloat32m2_t y3r = __riscv_vfsub_vv_f32m2(s0r, d1i, vl);
    vfloat32m2_t y3i = __riscv_vfadd_vv_f32m2(s0i, d1r, vl);

    __riscv_vsse32_v_f32m2(q + 0, 32, x0r, vl);
    __riscv_vsse32_v_f32m2(q + 1, 32, x0i, vl);
    __riscv_vsse32_v_f32m2(q + 2, 32, y1r, vl);
    __riscv_vsse32_v_f32m2(q + 3, 32, y1i, vl);
    __riscv_vsse32_v_f32m2(q + 4, 32, x2r, vl);
    __riscv_vsse32_v_f32m2(q + 5, 32, x2i, vl);
    __riscv_vsse32_v_f32m2(q + 6, 32, y3r, vl);
    __riscv_vsse32_v_f32m2(q + 7, 32, y3i, vl);
    i += (int)vl;
  }
}

/* general radix-4: butterflies over Fout[i*mm + j], j fixed per pass.
 * lane = i, stride mm cpx; twiddle is constant per j. */
void kf_bfly4_gen_rvv(kiss_fft_cpx *Fout, const size_t fstride,
                      const kiss_fft_state *st, int m, int N, int mm) {
  const kiss_twiddle_cpx *tw = st->twiddles;
  int j;
  const size_t vlmax = __riscv_vsetvlmax_e32m2();
  for (j = 0; j < m; j++) {
    const float tr1 = tw[(size_t)j * fstride].r;
    const float ti1 = tw[(size_t)j * fstride].i;
    const float tr2 = tw[(size_t)j * fstride * 2].r;
    const float ti2 = tw[(size_t)j * fstride * 2].i;
    const float tr3 = tw[(size_t)j * fstride * 3].r;
    const float ti3 = tw[(size_t)j * fstride * 3].i;
    float *base = (float *)Fout + (size_t)j * 2;
    int i = 0;
    while (i < N) {
      const size_t vl = __riscv_vsetvl_e32m2((size_t)(N - i));
      float *q = base + (size_t)i * mm * 2;
      const long sbytes = (long)mm * 8;
      vfloat32m2_t x0r = __riscv_vlse32_v_f32m2(q, sbytes, vl);
      vfloat32m2_t x0i = __riscv_vlse32_v_f32m2(q + 1, sbytes, vl);
      vfloat32m2_t x1r = __riscv_vlse32_v_f32m2(q + (size_t)m * 2, sbytes, vl);
      vfloat32m2_t x1i =
          __riscv_vlse32_v_f32m2(q + (size_t)m * 2 + 1, sbytes, vl);
      vfloat32m2_t x2r =
          __riscv_vlse32_v_f32m2(q + (size_t)m * 4, sbytes, vl);
      vfloat32m2_t x2i =
          __riscv_vlse32_v_f32m2(q + (size_t)m * 4 + 1, sbytes, vl);
      vfloat32m2_t x3r =
          __riscv_vlse32_v_f32m2(q + (size_t)m * 6, sbytes, vl);
      vfloat32m2_t x3i =
          __riscv_vlse32_v_f32m2(q + (size_t)m * 6 + 1, sbytes, vl);

      /* C_MUL with scalar twiddle */
      vfloat32m2_t s0r =
          __riscv_vfsub_vv_f32m2(__riscv_vfmul_vf_f32m2(x1r, tr1, vl),
                                 __riscv_vfmul_vf_f32m2(x1i, ti1, vl), vl);
      vfloat32m2_t s0i =
          __riscv_vfadd_vv_f32m2(__riscv_vfmul_vf_f32m2(x1r, ti1, vl),
                                 __riscv_vfmul_vf_f32m2(x1i, tr1, vl), vl);
      vfloat32m2_t s1r =
          __riscv_vfsub_vv_f32m2(__riscv_vfmul_vf_f32m2(x2r, tr2, vl),
                                 __riscv_vfmul_vf_f32m2(x2i, ti2, vl), vl);
      vfloat32m2_t s1i =
          __riscv_vfadd_vv_f32m2(__riscv_vfmul_vf_f32m2(x2r, ti2, vl),
                                 __riscv_vfmul_vf_f32m2(x2i, tr2, vl), vl);
      vfloat32m2_t s2r =
          __riscv_vfsub_vv_f32m2(__riscv_vfmul_vf_f32m2(x3r, tr3, vl),
                                 __riscv_vfmul_vf_f32m2(x3i, ti3, vl), vl);
      vfloat32m2_t s2i =
          __riscv_vfadd_vv_f32m2(__riscv_vfmul_vf_f32m2(x3r, ti3, vl),
                                 __riscv_vfmul_vf_f32m2(x3i, tr3, vl), vl);

      vfloat32m2_t s5r = __riscv_vfsub_vv_f32m2(x0r, s1r, vl);
      vfloat32m2_t s5i = __riscv_vfsub_vv_f32m2(x0i, s1i, vl);
      x0r = __riscv_vfadd_vv_f32m2(x0r, s1r, vl);
      x0i = __riscv_vfadd_vv_f32m2(x0i, s1i, vl);
      vfloat32m2_t s3r = __riscv_vfadd_vv_f32m2(s0r, s2r, vl);
      vfloat32m2_t s3i = __riscv_vfadd_vv_f32m2(s0i, s2i, vl);
      vfloat32m2_t s4r = __riscv_vfsub_vv_f32m2(s0r, s2r, vl);
      vfloat32m2_t s4i = __riscv_vfsub_vv_f32m2(s0i, s2i, vl);
      vfloat32m2_t y2r = __riscv_vfsub_vv_f32m2(x0r, s3r, vl);
      vfloat32m2_t y2i = __riscv_vfsub_vv_f32m2(x0i, s3i, vl);
      x0r = __riscv_vfadd_vv_f32m2(x0r, s3r, vl);
      x0i = __riscv_vfadd_vv_f32m2(x0i, s3i, vl);
      vfloat32m2_t y1r = __riscv_vfadd_vv_f32m2(s5r, s4i, vl);
      vfloat32m2_t y1i = __riscv_vfsub_vv_f32m2(s5i, s4r, vl);
      vfloat32m2_t y3r = __riscv_vfsub_vv_f32m2(s5r, s4i, vl);
      vfloat32m2_t y3i = __riscv_vfadd_vv_f32m2(s5i, s4r, vl);

      __riscv_vsse32_v_f32m2(q, sbytes, x0r, vl);
      __riscv_vsse32_v_f32m2(q + 1, sbytes, x0i, vl);
      __riscv_vsse32_v_f32m2(q + (size_t)m * 2, sbytes, y1r, vl);
      __riscv_vsse32_v_f32m2(q + (size_t)m * 2 + 1, sbytes, y1i, vl);
      __riscv_vsse32_v_f32m2(q + (size_t)m * 4, sbytes, y2r, vl);
      __riscv_vsse32_v_f32m2(q + (size_t)m * 4 + 1, sbytes, y2i, vl);
      __riscv_vsse32_v_f32m2(q + (size_t)m * 6, sbytes, y3r, vl);
      __riscv_vsse32_v_f32m2(q + (size_t)m * 6 + 1, sbytes, y3i, vl);
      i += (int)vl;
    }
  }
}

/* radix-2 with m==4 (the only variant used after a radix-4 stage).
 * lane = i, Fout advances 8 cpx per group. */
void kf_bfly2_rvv(kiss_fft_cpx *Fout, int N) {
  float *p = (float *)Fout;
  const float tw = 0.7071067812f;
  const size_t vlmax = __riscv_vsetvlmax_e32m2();
  int i = 0;
  while (i < N) {
    const size_t vl = __riscv_vsetvl_e32m2((size_t)(N - i));
    float *q = p + (size_t)i * 16;
    int k;
    for (k = 0; k < 4; k++) {
      vfloat32m2_t x0r = __riscv_vlse32_v_f32m2(q + 2 * k, 64, vl);
      vfloat32m2_t x0i = __riscv_vlse32_v_f32m2(q + 2 * k + 1, 64, vl);
      vfloat32m2_t x1r = __riscv_vlse32_v_f32m2(q + 8 + 2 * k, 64, vl);
      vfloat32m2_t x1i = __riscv_vlse32_v_f32m2(q + 9 + 2 * k, 64, vl);
      vfloat32m2_t tr, ti;
      switch (k) {
        case 0:
          tr = x1r;
          ti = x1i;
          break;
        case 1:
          tr = __riscv_vfmul_vf_f32m2(
              __riscv_vfadd_vv_f32m2(x1r, x1i, vl), tw, vl);
          ti = __riscv_vfmul_vf_f32m2(
              __riscv_vfsub_vv_f32m2(x1i, x1r, vl), tw, vl);
          break;
        case 2:
          tr = x1i;
          ti = __riscv_vfneg_v_f32m2(x1r, vl);
          break;
        default:
          tr = __riscv_vfmul_vf_f32m2(
              __riscv_vfsub_vv_f32m2(x1i, x1r, vl), tw, vl);
          ti = __riscv_vfneg_v_f32m2(
              __riscv_vfmul_vf_f32m2(
                  __riscv_vfadd_vv_f32m2(x1i, x1r, vl), tw, vl),
              vl);
          break;
      }
      {
        vfloat32m2_t tr_, ti_;
        if (k == 1 || k == 3) {
          tr_ = tr;
          ti_ = ti;
        } else {
          tr_ = x1r;
          ti_ = x1i;
          if (k == 2) {
            tr_ = x1i;
            ti_ = __riscv_vfneg_v_f32m2(x1r, vl);
          }
        }
        vfloat32m2_t y1r = __riscv_vfsub_vv_f32m2(x0r, tr_, vl);
        vfloat32m2_t y1i = __riscv_vfsub_vv_f32m2(x0i, ti_, vl);
        vfloat32m2_t y0r = __riscv_vfadd_vv_f32m2(x0r, tr_, vl);
        vfloat32m2_t y0i = __riscv_vfadd_vv_f32m2(x0i, ti_, vl);
        (void)tr;
        (void)ti;
        __riscv_vsse32_v_f32m2(q + 8 + 2 * k, 64, y1r, vl);
        __riscv_vsse32_v_f32m2(q + 9 + 2 * k, 64, y1i, vl);
        __riscv_vsse32_v_f32m2(q + 2 * k, 64, y0r, vl);
        __riscv_vsse32_v_f32m2(q + 2 * k + 1, 64, y0i, vl);
      }
    }
    i += (int)vl;
  }
}

#endif /* OPUS_RISCV_MAY_HAVE_RVV */
