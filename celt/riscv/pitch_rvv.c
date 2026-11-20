/* Copyright (c) 2026 RISC-V RVV optimization work.
 *
 * RVV float-domain reductions for CELT pitch/inner-product family.
 * Compiled with -march=rv64gcv; runtime-gated by opus_riscv_has_rvv().
 *
 * NOTE: unlike the integer kernels in other projects, float reductions
 * change summation order vs the scalar FMA chain, so results differ in
 * the low-order bits (same as the official NEON float ports). Callers
 * are all perceptual-codec paths; validation is done end-to-end.
 */
#if defined(OPUS_RISCV_MAY_HAVE_RVV)

#include <riscv_vector.h>
#include <stdio.h>
#include <string.h>

#include "arch.h"
#include "stack_alloc.h"
#include "mathops.h"
#include "pitch.h"

int opus_riscv_has_rvv(void) {
  static int cached = -1;
  if (cached < 0) {
    int ok = 0;
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
      char line[512];
      while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "isa", 3) == 0) {
          const char *p = strstr(line, "rv64");
          if (p) {
            p += 4; /* skip rv64 */
            while (*p >= 'a' && *p <= 'z') {
              if (*p == 'v') { ok = 1; break; }
              p++;
            }
          }
          break;
        }
      }
      fclose(f);
    }
    cached = ok;
  }
  return cached;
}

opus_val32 celt_inner_prod_rvv(const opus_val16 *x, const opus_val16 *y,
                               int N) {
  const size_t vlmax = __riscv_vsetvlmax_e32m2();
  vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
  int i = 0;
  while (i < N) {
    const size_t vl = __riscv_vsetvl_e32m2((size_t)(N - i));
    const vfloat32m2_t a = __riscv_vle32_v_f32m2(x + i, vl);
    const vfloat32m2_t b = __riscv_vle32_v_f32m2(y + i, vl);
    acc = __riscv_vfmacc_vv_f32m2(acc, a, b, vl);
    i += (int)vl;
  }
  const vfloat32m1_t s = __riscv_vfredusum_vs_f32m2_f32m1(
      acc, __riscv_vfmv_v_f_f32m1(0.f, vlmax), vlmax);
  return __riscv_vfmv_f_s_f32m1_f32(s);
}

void dual_inner_prod_rvv(const opus_val16 *x, const opus_val16 *y01,
                         const opus_val16 *y02, int N, opus_val32 *xy1,
                         opus_val32 *xy2) {
  const size_t vlmax = __riscv_vsetvlmax_e32m2();
  vfloat32m2_t a1 = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
  vfloat32m2_t a2 = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
  int i = 0;
  while (i < N) {
    const size_t vl = __riscv_vsetvl_e32m2((size_t)(N - i));
    const vfloat32m2_t xv = __riscv_vle32_v_f32m2(x + i, vl);
    a1 = __riscv_vfmacc_vv_f32m2(
        a1, xv, __riscv_vle32_v_f32m2(y01 + i, vl), vl);
    a2 = __riscv_vfmacc_vv_f32m2(
        a2, xv, __riscv_vle32_v_f32m2(y02 + i, vl), vl);
    i += (int)vl;
  }
  const vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.f, vlmax);
  *xy1 = __riscv_vfmv_f_s_f32m1_f32(
      __riscv_vfredusum_vs_f32m2_f32m1(a1, z, vlmax));
  *xy2 = __riscv_vfmv_f_s_f32m1_f32(
      __riscv_vfredusum_vs_f32m2_f32m1(a2, z, vlmax));
}

void xcorr_kernel_rvv(const opus_val16 *x, const opus_val16 *y,
                      opus_val32 sum[4], int len) {
  const size_t vlmax = __riscv_vsetvlmax_e32m2();
  vfloat32m2_t s0 = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
  vfloat32m2_t s1 = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
  vfloat32m2_t s2 = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
  vfloat32m2_t s3 = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
  int i = 0;
  while (i < len) {
    const size_t vl = __riscv_vsetvl_e32m2((size_t)(len - i));
    const vfloat32m2_t xv = __riscv_vle32_v_f32m2(x + i, vl);
    const vfloat32m2_t y0 = __riscv_vle32_v_f32m2(y + i, vl);
    const vfloat32m2_t y1 = __riscv_vle32_v_f32m2(y + i + 1, vl);
    const vfloat32m2_t y2 = __riscv_vle32_v_f32m2(y + i + 2, vl);
    const vfloat32m2_t y3 = __riscv_vle32_v_f32m2(y + i + 3, vl);
    s0 = __riscv_vfmacc_vv_f32m2(s0, xv, y0, vl);
    s1 = __riscv_vfmacc_vv_f32m2(s1, xv, y1, vl);
    s2 = __riscv_vfmacc_vv_f32m2(s2, xv, y2, vl);
    s3 = __riscv_vfmacc_vv_f32m2(s3, xv, y3, vl);
    i += (int)vl;
  }
  {
    const vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.f, vlmax);
    sum[0] = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(s0, z, vlmax));
    sum[1] = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(s1, z, vlmax));
    sum[2] = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(s2, z, vlmax));
    sum[3] = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(s3, z, vlmax));
  }
}

void celt_pitch_xcorr_rvv(const opus_val16 *_x, const opus_val16 *_y,
                          opus_val32 *xcorr, int len, int max_pitch,
                          int arch) {
  (void)arch;
  int i;
  for (i = 0; i < max_pitch - 3; i += 4) {
    opus_val32 sum[4] = {0.f, 0.f, 0.f, 0.f};
    xcorr_kernel_rvv(_x, _y + i, sum, len);
    xcorr[i] = sum[0];
    xcorr[i + 1] = sum[1];
    xcorr[i + 2] = sum[2];
    xcorr[i + 3] = sum[3];
  }
  for (; i < max_pitch; i++)
    xcorr[i] = celt_inner_prod_rvv(_x, _y + i, len);
}

opus_int32 stereo_itheta_rvv(const opus_val16 *X, const opus_val16 *Y,
                             int stereo, int N, int arch) {
  opus_val32 mid, side;
  opus_val32 Emid, Eside;
  Emid = Eside = 0;
  if (stereo) {
    const size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t am = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
    vfloat32m2_t as = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
    int i = 0;
    while (i < N) {
      const size_t vl = __riscv_vsetvl_e32m2((size_t)(N - i));
      const vfloat32m2_t xv = __riscv_vle32_v_f32m2(X + i, vl);
      const vfloat32m2_t yv = __riscv_vle32_v_f32m2(Y + i, vl);
      const vfloat32m2_t m = __riscv_vfadd_vv_f32m2(xv, yv, vl);
      const vfloat32m2_t s = __riscv_vfsub_vv_f32m2(xv, yv, vl);
      am = __riscv_vfmacc_vv_f32m2(am, m, m, vl);
      as = __riscv_vfmacc_vv_f32m2(as, s, s, vl);
      i += (int)vl;
    }
    const vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(0.f, vlmax);
    Emid = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(am, z, vlmax));
    Eside = __riscv_vfmv_f_s_f32m1_f32(
        __riscv_vfredusum_vs_f32m2_f32m1(as, z, vlmax));
  } else {
    Emid += celt_inner_prod_rvv(X, X, N);
    Eside += celt_inner_prod_rvv(Y, Y, N);
  }
  mid = celt_sqrt(Emid);
  side = celt_sqrt(Eside);
  return (opus_int32)floor(.5f + 65536.f * 16384.f *
                                       celt_atan2p_norm(side, mid));
}

#endif /* OPUS_RISCV_MAY_HAVE_RVV */

/* spreading_decision CDF counters: exact float comparisons, so results
 * are bit-identical to the scalar loop. x2N = x*x*N; count x2N below
 * 0.25 / 0.0625 / 0.015625. */
void celt_spread_counts_rvv(const float *x, int N, int tcount[3]) {
  const size_t vlmax = __riscv_vsetvlmax_e32m2();
  const vfloat32m2_t vn = __riscv_vfmv_v_f_f32m2((float)N, vlmax);
  const vfloat32m2_t th0 = __riscv_vfmv_v_f_f32m2(0.25f, vlmax);
  const vfloat32m2_t th1 = __riscv_vfmv_v_f_f32m2(0.0625f, vlmax);
  const vfloat32m2_t th2 = __riscv_vfmv_v_f_f32m2(0.015625f, vlmax);
  const vuint32m2_t ones = __riscv_vmv_v_x_u32m2(1u, vlmax);
  const vuint32m2_t zeros = __riscv_vmv_v_x_u32m2(0u, vlmax);
  vuint32m2_t c0 = __riscv_vmv_v_x_u32m2(0u, vlmax);
  vuint32m2_t c1 = __riscv_vmv_v_x_u32m2(0u, vlmax);
  vuint32m2_t c2 = __riscv_vmv_v_x_u32m2(0u, vlmax);
  int i = 0;
  while (i < N) {
    const size_t vl = __riscv_vsetvl_e32m2((size_t)(N - i));
    const vfloat32m2_t v = __riscv_vle32_v_f32m2(x + i, vl);
    const vfloat32m2_t t = __riscv_vfmul_vv_f32m2(
        __riscv_vfmul_vv_f32m2(v, v, vl), vn, vl);
    const vbool16_t m0 = __riscv_vmflt_vv_f32m2_b16(t, th0, vl);
    const vbool16_t m1 = __riscv_vmflt_vv_f32m2_b16(t, th1, vl);
    const vbool16_t m2 = __riscv_vmflt_vv_f32m2_b16(t, th2, vl);
    c0 = __riscv_vadd_vv_u32m2(
        c0, __riscv_vmerge_vvm_u32m2(zeros, ones, m0, vl), vl);
    c1 = __riscv_vadd_vv_u32m2(
        c1, __riscv_vmerge_vvm_u32m2(zeros, ones, m1, vl), vl);
    c2 = __riscv_vadd_vv_u32m2(
        c2, __riscv_vmerge_vvm_u32m2(zeros, ones, m2, vl), vl);
    i += (int)vl;
  }
  {
    const vuint32m1_t z = __riscv_vmv_v_x_u32m1(0u, vlmax);
    tcount[0] = (int)__riscv_vmv_x_s_u32m1_u32(
        __riscv_vredsum_vs_u32m2_u32m1(c0, z, vlmax));
    tcount[1] = (int)__riscv_vmv_x_s_u32m1_u32(
        __riscv_vredsum_vs_u32m2_u32m1(c1, z, vlmax));
    tcount[2] = (int)__riscv_vmv_x_s_u32m1_u32(
        __riscv_vredsum_vs_u32m2_u32m1(c2, z, vlmax));
  }
}

/* op_pvq_search: float build. The per-j score computation is vectorized;
 * the argmax is a scalar left-fold over the stored (num, den) arrays so
 * the winner selection is exactly the C predicate in C order. The
 * pre-search reductions use vector sums (float-order difference, same
 * tolerance policy as the rest of this file). */
opus_val16 op_pvq_search_rvv(celt_norm *X_, int *iy, int K, int N, int arch) {
  (void)arch;
  VARDECL(celt_norm, y);
  VARDECL(int, signx);
  VARDECL(celt_norm, vnum);
  VARDECL(celt_norm, vden);
  SAVE_STACK;
  celt_norm *X = X_;
  int i, j;
  int pulsesLeft;
  float sum = 0.f;
  float xy = 0.f;
  float yy = 0.f;
  ALLOC(y, N, celt_norm);
  ALLOC(signx, N, int);
  ALLOC(vnum, N, celt_norm);
  ALLOC(vden, N, celt_norm);

  /* strip signs, zero iy/y */
  {
    const size_t vlmax = __riscv_vsetvlmax_e32m2();
    const vfloat32m2_t zero = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
    const vint32m2_t izero = __riscv_vmv_v_x_i32m2(0, vlmax);
    i = 0;
    while (i < N) {
      const size_t vl = __riscv_vsetvl_e32m2((size_t)(N - i));
      const vfloat32m2_t v = __riscv_vle32_v_f32m2(X + i, vl);
      const vbool16_t neg = __riscv_vmflt_vv_f32m2_b16(v, zero, vl);
      const vint32m2_t s = __riscv_vmerge_vvm_i32m2(
          izero, __riscv_vmv_v_x_i32m2(1, vl), neg, vl);
      const vfloat32m2_t a =
          __riscv_vmerge_vvm_f32m2(v, __riscv_vfneg_v_f32m2(v, vl), neg, vl);
      __riscv_vse32_v_i32m2(signx + i, s, vl);
      __riscv_vse32_v_f32m2(X + i, a, vl);
      __riscv_vse32_v_i32m2(iy + i, izero, vl);
      __riscv_vse32_v_f32m2(y + i, zero, vl);
      i += (int)vl;
    }
  }

  pulsesLeft = K;

  if (K > (N >> 1)) {
    float rcp;
    const size_t vlmax = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
    i = 0;
    while (i < N) {
      const size_t vl = __riscv_vsetvl_e32m2((size_t)(N - i));
      acc = __riscv_vfadd_vv_f32m2(
          acc, __riscv_vle32_v_f32m2(X + i, vl), vl);
      i += (int)vl;
    }
    sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m2_f32m1(
        acc, __riscv_vfmv_v_f_f32m1(0.f, vlmax), vlmax));

    if (!(sum > EPSILON && sum < 64)) {
      X[0] = 1.f;
      for (j = 1; j < N; j++) X[j] = 0.f;
      sum = 1.f;
    }
    rcp = (K + 0.8f) * (1.f / sum);

    {
      vfloat32m2_t accy = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
      vfloat32m2_t accxy = __riscv_vfmv_v_f_f32m2(0.f, vlmax);
      vint32m2_t accp = __riscv_vmv_v_x_i32m2(0, vlmax);
      i = 0;
      while (i < N) {
        const size_t vl = __riscv_vsetvl_e32m2((size_t)(N - i));
        const vfloat32m2_t xv = __riscv_vle32_v_f32m2(X + i, vl);
        const vfloat32m2_t t = __riscv_vfmul_vf_f32m2(xv, rcp, vl);
        /* X[j] >= 0 here, so round-toward-zero == floor() */
        const vint32m2_t q = __riscv_vfcvt_x_f_v_i32m2_rm(
            t, __RISCV_FRM_RTZ, vl);
        const vfloat32m2_t yv = __riscv_vfcvt_f_x_v_f32m2(q, vl);
        accy = __riscv_vfmacc_vv_f32m2(accy, yv, yv, vl);
        accxy = __riscv_vfmacc_vv_f32m2(accxy, xv, yv, vl);
        accp = __riscv_vadd_vv_i32m2(accp, q, vl);
        __riscv_vse32_v_i32m2(iy + i, q, vl);
        __riscv_vse32_v_f32m2(y + i,
                              __riscv_vfmul_vf_f32m2(yv, 2.f, vl), vl);
        i += (int)vl;
      }
      const vfloat32m1_t zf = __riscv_vfmv_v_f_f32m1(0.f, vlmax);
      yy = __riscv_vfmv_f_s_f32m1_f32(
          __riscv_vfredusum_vs_f32m2_f32m1(accy, zf, vlmax));
      xy = __riscv_vfmv_f_s_f32m1_f32(
          __riscv_vfredusum_vs_f32m2_f32m1(accxy, zf, vlmax));
      pulsesLeft -= __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m2_i32m1(
          accp, __riscv_vmv_v_x_i32m1(0, vlmax), vlmax));
    }
  }

  if (pulsesLeft > N + 3) {
    float tmp = (float)pulsesLeft;
    yy = yy + tmp * tmp + tmp * y[0];
    iy[0] += pulsesLeft;
    pulsesLeft = 0;
  }

  for (i = 0; i < pulsesLeft; i++) {
    int best_id;
    float best_num, best_den;
    yy = yy + 1.f;

    /* vector: num_j = (xy + X[j])^2, den_j = yy + y[j] */
    {
      const size_t vlmax = __riscv_vsetvlmax_e32m2();
      i = 0;
      while (i < N) {
        const size_t vl = __riscv_vsetvl_e32m2((size_t)(N - i));
        const vfloat32m2_t t = __riscv_vfadd_vf_f32m2(
            __riscv_vle32_v_f32m2(X + i, vl), xy, vl);
        __riscv_vse32_v_f32m2(
            vnum + i, __riscv_vfmul_vv_f32m2(t, t, vl), vl);
        __riscv_vse32_v_f32m2(
            vden + i,
            __riscv_vfadd_vv_f32m2(__riscv_vle32_v_f32m2(y + i, vl),
                                   __riscv_vfmv_v_f_f32m2(yy, vl), vl),
            vl);
        i += (int)vl;
      }
    }

    /* scalar left-fold argmax: identical predicate and order to C */
    best_num = vnum[0];
    best_den = vden[0];
    best_id = 0;
    for (j = 1; j < N; j++) {
      if (best_den * vnum[j] > vden[j] * best_num) {
        best_den = vden[j];
        best_num = vnum[j];
        best_id = j;
      }
    }

    xy = xy + X[best_id];
    yy = yy + y[best_id];
    y[best_id] += 2.f;
    iy[best_id]++;
  }

  /* restore signs */
  {
    const size_t vlmax = __riscv_vsetvlmax_e32m2();
    const vint32m2_t izero = __riscv_vmv_v_x_i32m2(0, vlmax);
    i = 0;
    while (i < N) {
      const size_t vl = __riscv_vsetvl_e32m2((size_t)(N - i));
      const vint32m2_t v = __riscv_vle32_v_i32m2(iy + i, vl);
      const vbool16_t neg =
          __riscv_vmsne_vv_i32m2_b16(__riscv_vle32_v_i32m2(signx + i, vl),
                                     izero, vl);
      __riscv_vse32_v_i32m2(
          iy + i,
          __riscv_vmerge_vvm_i32m2(v, __riscv_vneg_v_i32m2(v, vl), neg, vl),
          vl);
      i += (int)vl;
    }
  }

  RESTORE_STACK;
  return yy;
}
