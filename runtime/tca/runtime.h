// tca/runtime.h — THE umbrella header: the only include generated code uses
// (doc/rt.md §1). Brings in the CUDA headers plus every tca/*.h.
#ifndef TCA_RT_RUNTIME_H_
#define TCA_RT_RUNTIME_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Host surface.
#include "tca/check.h"
#include "tca/tensor.h"
#include "tca/launch.h"

// Device surface.
#include "tca/mapping.h"
#include "tca/iterator.h"
#include "tca/view.h"
#include "tca/loader.h"
#include "tca/reducer.h"

#endif  // TCA_RT_RUNTIME_H_
