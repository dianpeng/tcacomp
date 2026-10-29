> **SUPERSEDED** — 2026-10-28
>
> Superseded by doc/tile_scheduling.md (canonical tile scheduling design doc).
> The worked-codegen content of this file corresponds to its §9 (Worked Example:
> Matmul [128] × [128] × [16]) and §8 (Reduce Scheduling). Retained for
> history. Do not modify — all changes should go to the canonical document.

---

---

uI = [128][128][16]

--------------------------------------------------------
            SHAPE                         STRIDE
--------------------------------------------------------
(0/CTA):   [4][4][1]   | [32][32][1]  | [32][32][16]      <- [32][32]
(1/WARP):  [2][2][2]   | [16][16][8]  | [16][16][16]      <- [16][16]
(2/THR):   [4][4][2]   | [4][4][4]    | [4][4][4]         <- [4][2]
(3/SEQ)    [4][4][4]   | [1][1][1]    | [1][1][1]
--------------------------------------------------------

(0/CTA):   [4][4][1] |       combine project
(1/WARP):  [2][2][2] | cache combine   -     cache
(2/THR):   [4][4][2] | cache combine   -     cache
(3/SEQ)    [4][4][4] | map   reduce    -     map

tile(4, 4, 1) -> {
  smem reduce;
  reduce.init();
  tile(2, 2, 2) -> {
    reduce.init()
    tile(4, 4, 2) -> {
      reduce.init();
      tile(4, 4, 4) -> {
        auto lhs = mem_load(g_lhs[i4][j4])
        auto rhs = mem_load(g_rhs[k4][j4])
        const auto value = lhs * rhs;
        reduce.reduce(value);
      }
      reduce.combine();
    }
    reduce.combine()
  }

  reduce.combine(reduce);
  mem_write(gxx, reduce);
}

// CTA level
for (auto cta_0 = 0; cta_0 < 4; cta_0++) {
  auto cta_0_offset = cta_0 * 32 + blockIdx.x;
  for (auto cta_1 = 0; cta_1 < 4; cta_1++) {
    auto cta_2_offset = cta_1 * 32 + blockIdx.y;
    for (auto cta_2 = 0; cta_2 < 1; cta_2++) {
      auto cta_3_offset = cta_2 * 1 + blockIdx.z;

      reducer<cta, shape<2, 2>, stride<32, 32>> reducer;

      smem lhs_smem[32][32];
      smem rhs_smem[32][32];
      smem acc_cta[32][32];

      load_frag<4, 2>(
        lhs,
        lhs_smem,
        cta_0_offset + warp_0_offset,
        cta_1_offset + warp_1_offset,
      );

      load_frag<4, 2>(
        rhs,
        rhs_smem,
        cta_1_offset + warp_1_offset,
        cta_2_offset + warp_2_offset,
      );

      reducer.init(cc);


      // WARP level
      for (auto warp_0 = 0; warp_0 < 2; warp_0++) {
        for (auto warp_1 = 0; warp_1 < 2; warp_1++) {
          for (auto warp_2 = 0; warp_2 < 2; warp_2++) {
            wmem lhs_smem[16][16];
            wmem rhs_smem[16][16];
            wmem acc_warp[16][16]; // <- assume this exists

            reducer<warp, shape<4, 4>, stride<16, 16>> reducer.init(acc_warp);

            load_frag<4, 2>(
              lhs,
              lhs_smem,
              cta_0_offset + warp_0_offset,
              cta_1_offset + warp_1_offset,
            );

            load_frag<4, 2>(
              rhs,
              rhs_smem,
              cta_1_offset + warp_1_offset,
              cta_2_offset + warp_2_offset,
            );

            // THR level
            for (auto thr_0 = 0; thr_0 < 4; thr_0++) {
              for (auto thr_1 = 0; thr_1 < 4; thr_1++) {
                for (auto thr_2 = 0; thr_2 < 2; thr_2++) {

                  // SEQ level
                  fragment lhs_frag[4][4];
                  fragment rhs_frag[4][4];
                  fragment acc_thr[4][4];

                  reducer<thread, shape<4, 4>, stride<1, 1>> reducer.init(acc_frag);

                  load_frag<4, 2>(
                    lhs_smem,
                    lhs_frag,
                    thr0_offset,
                    thr1_offset,
                  );

                  load_frag<4, 2>(
                    rhs_smem,
                    rhs_frag,
                    thr1_offset,
                    thr2_offset,
                  );

                  for (auto seq_0 = 0; seq_0 < 4; seq_0++) {
                    for (auto seq_1 = 0; seq_1 < 4; seq_1++) {
                      for (auto seq_2 = 0; seq_2 < 4; seq_2++) {

                        auto global_0 = cta_0_offset +
                                        warp_0_offset +
                                        thr_0_offset +
                                        seq_0_offset;

                        auto global_1 = cta_1_offset +
                                        warp_1_offset +
                                        thr_1_offset +
                                        seq_1_offset;

                        auto global_2 = cta_2_offset +
                                        warp_2_offset +
                                        thr_2_offset +
                                        seq_2_offset;

                        // (global_0/1/2)

                        if (...) {
                          auto lhs_value = lhs_frag[global_0][global_1];
                          auto rhs_value = rhs_frag[global_1][global_2];
                          auto value = lhs_value * rhs_value;
                          reducer.reduce((global_0, global_2), value);
                        }
                      }
                    }
                  }

                  // for each thread, it writes all its acc_frag value into
                  // acc_warp level position. This will use warp level
                  // specialization instruction __shfl_down_sync with specific
                  // mask. The mask is derived as following,
                  reducer.combine<thread>(thr_0, thr_1, thr_2);
                }
              }
            }

            // combine warp into smem[32][32]
            reducer.combine<warp>(warp_0, warp_1, warp_2);
          }
        }
      }

      // finally not combine, but written out
      mem_write(gout, ...)
    }
  }
}
