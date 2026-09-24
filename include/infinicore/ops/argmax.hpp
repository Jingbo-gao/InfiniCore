#pragma once

#include "../tensor.hpp"

namespace infinicore::op {

// Return the index of the maximum element along `dim`.
Tensor argmax(const Tensor &input, int64_t dim);
void argmax_(Tensor output, const Tensor &input, int64_t dim);

} // namespace infinicore::op
