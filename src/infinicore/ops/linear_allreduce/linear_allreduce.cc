#include "infinicore/ops/linear_allreduce.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/linear.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(LinearAllReduce);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(LinearAllReduceNz);

LinearAllReduce::LinearAllReduce(
    Tensor output,
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    infinicclComm_t communicator) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, weight);
    if (bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, *bias);
    }
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(), output, input, weight, bias, communicator);
}

void LinearAllReduce::execute(
    Tensor output,
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        LinearAllReduce, output, input, weight, bias, communicator);
}

LinearAllReduceNz::LinearAllReduceNz(
    Tensor output,
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    infinicclComm_t communicator) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, weight);
    if (bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, *bias);
    }
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(), output, input, weight, bias, communicator);
}

void LinearAllReduceNz::execute(
    Tensor output,
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        LinearAllReduceNz, output, input, weight, bias, communicator);
}

enum class PackedLayout {
    NONE,
    ND,
    ASCEND_NZ,
};

static Tensor linear_allreduce_impl(
    Tensor input,
    Tensor weight,
    std::optional<Tensor> bias,
    infinicclComm_t communicator,
    PackedLayout packed_layout) {
    const bool weight_is_packed = packed_layout != PackedLayout::NONE;
    Size in_features = weight->shape()[weight_is_packed ? 0 : 1];
    Size out_features = weight->shape()[weight_is_packed ? 1 : 0];
    auto output_shape = input->shape();
    output_shape.back() = out_features;

    const bool aclnn_supported = input->dtype() == DataType::F16 || input->dtype() == DataType::BF16;
    if (input->device().getType() == Device::Type::ASCEND && aclnn_supported) {
        auto output = Tensor::empty(output_shape, input->dtype(), input->device());
        Size rows = 1;
        for (Size i = 0; i + 1 < input->ndim(); ++i) {
            rows *= input->size(i);
        }
        auto input_matrix = input->view({rows, in_features});
        auto output_matrix = output->view({rows, out_features});
        auto weight_matrix = weight_is_packed
                               ? weight
                               : weight->permute({1, 0});
        if (packed_layout == PackedLayout::ASCEND_NZ) {
            LinearAllReduceNz::execute(
                output_matrix, input_matrix, weight_matrix, bias, communicator);
        } else {
            LinearAllReduce::execute(
                output_matrix, input_matrix, weight_matrix, bias, communicator);
        }
        return output;
    }

    if (packed_layout == PackedLayout::ASCEND_NZ) {
        throw std::runtime_error(
            "FRACTAL_NZ fused linear/all-reduce is only supported on Ascend");
    }

    auto output = weight_is_packed
                    ? linear_packed(input, weight, bias)
                    : linear(input, weight, bias);
    distributed::allreduce_(
        output, output, INFINICCL_SUM, communicator);
    return output;
}

Tensor linear_allreduce(
    Tensor input,
    Tensor weight,
    std::optional<Tensor> bias,
    infinicclComm_t communicator) {
    return linear_allreduce_impl(
        input, weight, bias, communicator, PackedLayout::NONE);
}

Tensor linear_allreduce_packed(
    Tensor input,
    Tensor packed_weight,
    std::optional<Tensor> bias,
    infinicclComm_t communicator) {
    return linear_allreduce_impl(
        input, packed_weight, bias, communicator, PackedLayout::ND);
}

Tensor linear_allreduce_packed_nz(
    Tensor input,
    Tensor nz_weight,
    std::optional<Tensor> bias,
    infinicclComm_t communicator) {
    return linear_allreduce_impl(
        input, nz_weight, bias, communicator, PackedLayout::ASCEND_NZ);
}

} // namespace infinicore::op
