#pragma once

#include "../tensor.hpp"

namespace infinicore::op {

// Converts a contiguous two-dimensional Ascend tensor from ACL_FORMAT_ND to
// ACL_FORMAT_FRACTAL_NZ. The returned tensor retains the source logical shape;
// only operators that explicitly describe it as FRACTAL_NZ may consume it.
Tensor ascend_format_cast_nz(const Tensor &src);

} // namespace infinicore::op
