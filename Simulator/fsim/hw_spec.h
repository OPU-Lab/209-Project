#include <stdint.h>
#include <vector>
#include <algorithm>
#include <iterator>

#ifndef FSIM_HW_SPEC_H_
#define FSIM_HW_SPEC_H_

/*! Mem ID constant: input feature map memory */
#define OPU_MEM_ID_FM 0
/*! Mem ID constant: kernel memory */
#define OPU_MEM_ID_WGT 1
/*! Mem ID constant: bias memory */
#define OPU_MEM_ID_BIAS 2
/*! Mem ID constant: residual memory */
#define OPU_MEM_ID_RESIDUAL 3
/*! Mem ID constant: instruction memory */
#define OPU_MEM_ID_INS 4

/*! DDR load trigger constant */
#define OPU_DDRLD_TRIG_DDRLD 0
// dma done here means the completion of multiple consecutive dmas
#define OPU_DDRLD_TRIG_DDRLD_DMA 1
#define OPU_DDRLD_TRIG_LAYER_START 2
#define OPU_DDRLD_TRIG_DDRST 3
#define OPU_DDRLD_TRIG_BRAMST 5
#define OPU_DDRLD_TRIG_BRAMWB_DDRLD 6
#define OPU_DDRLD_TRIG_DDRLD_DDRST 7

/*! COMPUTE trigger constant */
#define OPU_DMA_TRIG_DMA 0
#define OPU_DMA_TRIG_DDRLD 1
#define OPU_DMA_TRIG_DMA_DDRLD 2
#define OPU_DMA_TRIG_DDRST_NOT_1_6 3
#define OPU_DMA_TRIG_DDRST 5

/*! DDR store trigger constant */
#define OPU_DDRST_TRIG_BRAMST_DDRLD 0
#define OPU_DDRST_TRIG_DDRST 1
#define OPU_DDRST_TRIG_BRAMWB_DDRLD 3
#define OPU_DDRST_TRIG_DDRLD 4
#define OPU_DDRST_TRIG_DDRLD_DDRST 5
#define OPU_DDRST_TRIG_SINGLE_DDRLD 6

/*! Layer type constant */
#define OPU_LAYER_FC 0
#define OPU_LAYER_CONV2D 1
#define OPU_LAYER_SINGLE_POOL 2
#define OPU_LAYER_SE 3
#define OPU_LAYER_ZTCONV 4
#define OPU_LAYER_NTCONV 5
#define OPU_LAYER_CONV3D 6

/*! Copy mode for local input buffer of compute engine */
#define OPU_COPY_MODE_REP_8 0
#define OPU_COPY_MODE_REP_16 1
#define OPU_COPY_MODE_REP_32 2

/*! Actiovation type */
#define OPU_ACTIVATION_NONE 0
#define OPU_ACTIVATION_RELU 1
#define OPU_ACTIVATION_PRELU 2
#define OPU_ACTIVATION_HSIGMOID 3
#define OPU_ACTIVATION_HSWISH 4
#define OPU_ACTIVATION_LUT 5

/*! Pooling type */
#define OPU_POOL_NONE 0
#define OPU_POOL_MAX 1
#define OPU_POOL_AVG 2

/*! IPA output number */
#define OPU_IPA_OUT_NUM_2 0
#define OPU_IPA_OUT_NUM_4 1
#define OPU_IPA_OUT_NUM_8 2
#define OPU_IPA_OUT_NUM_16 3
#define OPU_IPA_OUT_NUM_32 4
#define OPU_IPA_OUT_NUM_64 5

/*! DDR write destination */
#define OPU_DDRST_TO_DDR 0
#define OPU_DDRST_TO_BRAM 1

#endif  // FSIM_HW_SPEC_H_

