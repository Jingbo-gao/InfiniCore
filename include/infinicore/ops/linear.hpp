#pragma once

#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

Tensor linear(Tensor input, Tensor weight, std::optional<Tensor> bias, float alpha = 1.0f);

void linear_(Tensor out, Tensor input, Tensor weight, std::optional<Tensor> bias, float alpha = 1.0f);

// Pre-packed weight variants: packed_weight is already [IC, OC] contiguous.
// These skip the runtime weight->permute({1, 0}) call inside linear_.
Tensor linear_packed(Tensor input, Tensor packed_weight,
                     std::optional<Tensor> bias, float alpha = 1.0f);
void linear_packed_(Tensor out, Tensor input, Tensor packed_weight,
                    std::optional<Tensor> bias, float alpha = 1.0f);

// Ascend-only variant. nz_weight has logical shape [IC, OC] and physical
// ACL_FORMAT_FRACTAL_NZ storage produced by ascend_format_cast_nz().
Tensor linear_packed_nz(Tensor input, Tensor nz_weight,
                        std::optional<Tensor> bias, float alpha = 1.0f);
void linear_packed_nz_(Tensor out, Tensor input, Tensor nz_weight,
                       std::optional<Tensor> bias, float alpha = 1.0f);

} // namespace infinicore::op
