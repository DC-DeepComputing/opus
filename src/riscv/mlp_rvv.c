/* Copyright (c) 2026 RISC-V RVV optimization work.
 *
 * RVV int8-weight matvec for the analysis MLP/GRU (gemm_accum).
 * Lane = output row; the j loop keeps the C order, so accumulation is
 * bit-identical to the scalar reference.
 */
#if defined(OPUS_RISCV_MAY_HAVE_RVV)

#include <riscv_vector.h>

#include "opus_types.h"

void gemm_accum_rvv(float *out, const opus_int8 *weights, int rows, int cols,
                    int col_stride, const float *x) {
  const size_t vlmax = __riscv_vsetvlmax_e32m4();
  int i0;
  for (i0 = 0; i0 < rows; i0 += (int)vlmax) {
    const size_t vl =
        (size_t)(rows - i0) < vlmax ? (size_t)(rows - i0) : vlmax;
    vfloat32m4_t acc = __riscv_vle32_v_f32m4(out + i0, vl);
    int j;
    for (j = 0; j < cols; j++) {
      const vint8m1_t w8 =
          __riscv_vle8_v_i8m1(weights + (size_t)j * col_stride + i0, vl);
      const vint16m2_t w16 = __riscv_vsext_vf2_i16m2(w8, vl);
      const vint32m4_t w32 = __riscv_vsext_vf2_i32m4(w16, vl);
      const vfloat32m4_t wf = __riscv_vfcvt_f_x_v_f32m4(w32, vl);
      acc = __riscv_vfmacc_vf_f32m4(acc, x[j], wf, vl);
    }
    __riscv_vse32_v_f32m4(out + i0, acc, vl);
  }
}

#endif /* OPUS_RISCV_MAY_HAVE_RVV */
