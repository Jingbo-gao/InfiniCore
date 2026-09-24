#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/gemm.hpp"
#include "infinicore/ops/rearrange.hpp"

#include <stdexcept>

#ifdef ENABLE_ASCEND_API
#include "infinicore/context/context.hpp"
#include "../../../infiniop/devices/ascend/common_ascend.h"
#include <aclnnop/aclnn_add.h>
#include <memory>
#include <string>
#endif

namespace infinicore::op {

#ifdef ENABLE_ASCEND_API
namespace {

void check_acl_linear_bias_add(aclnnStatus status, const char *api) {
    if (status == ACL_SUCCESS) {
        return;
    }
    const char *detail = aclGetRecentErrMsg();
    throw std::runtime_error(
        std::string(api) + " failed with status " + std::to_string(status)
        + (detail == nullptr ? "" : std::string(": ") + detail));
}

void add_bias_inplace_ascend(Tensor out_matrix, Tensor bias_matrix) {
    aclnnTensorDescriptor self_desc(out_matrix->desc(), out_matrix->data());
    aclnnTensorDescriptor other_desc(
        bias_matrix->desc(), const_cast<std::byte *>(bias_matrix->data()));

    const float alpha_value = 1.0f;
    aclnnScalarDescriptor alpha_desc(
        ACL_FLOAT, &alpha_value, sizeof(alpha_value));

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    check_acl_linear_bias_add(
        aclnnInplaceAddGetWorkspaceSize(
            self_desc.tensor, other_desc.tensor, alpha_desc.scalar,
            &workspace_size, &executor),
        "aclnnInplaceAddGetWorkspaceSize");

    std::shared_ptr<Memory> workspace;
    void *workspace_ptr = nullptr;
    if (workspace_size != 0) {
        workspace = context::allocateMemory(workspace_size);
        workspace_ptr = workspace->data();
    }

    check_acl_linear_bias_add(
        aclnnInplaceAdd(
            workspace_ptr, workspace_size, executor,
            reinterpret_cast<aclrtStream>(context::getStream())),
        "aclnnInplaceAdd");
}
class AscendBiasAddInplaceOp : public graph::GraphOperator {
public:
    AscendBiasAddInplaceOp(Tensor out_matrix, Tensor bias_matrix)
        : out_matrix_(out_matrix), bias_matrix_(bias_matrix) {}

    void run() const override {
        add_bias_inplace_ascend(out_matrix_, bias_matrix_);
    }

    bool is_device_graph_capture_safe() const override {
        return false;
    }

private:
    graph::GraphTensor out_matrix_;
    graph::GraphTensor bias_matrix_;
};

void execute_bias_add_inplace_ascend(Tensor out_matrix, Tensor bias_matrix) {
    auto op = std::make_shared<AscendBiasAddInplaceOp>(out_matrix, bias_matrix);
    if (context::isGraphRecording()) {
        context::addGraphOperator(op);
    } else {
        op->run();
    }
}

} // namespace
#endif

Tensor linear(Tensor input,
              Tensor weight,
              std::optional<Tensor> bias,
              float alpha) {

    Size ndim = input->ndim();
    Size out_features = weight->shape()[0];

    // Assign memory to out variables
    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());

    // Inplace Calculate
    linear_(out, input, weight, bias, alpha);
    return out;
}

void linear_(Tensor out,
             Tensor input,
             Tensor weight,
             std::optional<Tensor> bias,
             float alpha) {

    auto weight_shape = weight->shape();
    Size out_features = weight_shape[0];
    Size in_features = weight_shape[1];

    Size ndim = input->ndim();
    assert(out->ndim() == ndim);

    // Calculate the number of features
    Size N = 1;
    auto input_shape = input->shape();
    for (size_t i = 0; i < ndim - 1; ++i) {
        N *= input_shape[i];
    }

    // linear transformation
    Tensor out_view = out->view({N, out_features});
    // Add bias
    float beta = 0.0f;
    if (bias.has_value()) {
        rearrange_(out_view,
                   bias.value()->as_strided({N, out_features}, {0, 1}));
        beta = 1.0f;
    }

    gemm_(out_view,
          input->view({N, in_features}),
          weight->permute({1, 0}), alpha, beta);
}

// --- Pre-packed weight variants ---
// packed_weight layout: [in_features, out_features] (already transposed from [OC,IC])

Tensor linear_packed(Tensor input,
                     Tensor packed_weight,
                     std::optional<Tensor> bias,
                     float alpha) {

    Size ndim = input->ndim();
    // packed_weight shape is [IC, OC], so OC is dim 1
    Size out_features = packed_weight->shape()[1];

    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());

    linear_packed_(out, input, packed_weight, bias, alpha);
    return out;
}

void linear_packed_(Tensor out,
                    Tensor input,
                    Tensor packed_weight,
                    std::optional<Tensor> bias,
                    float alpha) {

    auto weight_shape = packed_weight->shape();
    // packed_weight is [IC, OC] — already the layout GEMM expects
    Size in_features = weight_shape[0];
    Size out_features = weight_shape[1];

    Size ndim = input->ndim();
    assert(out->ndim() == ndim);

    Size N = 1;
    auto input_shape = input->shape();
    for (size_t i = 0; i < ndim - 1; ++i) {
        N *= input_shape[i];
    }

    Tensor out_view = out->view({N, out_features});

    // Add bias (same logic as linear_)
    float beta = 0.0f;
    if (bias.has_value()) {
        rearrange_(out_view,
                   bias.value()->as_strided({N, out_features}, {0, 1}));
        beta = 1.0f;
    }

    // KEY: no weight->permute({1, 0}) here!
    // packed_weight is already [IC, OC] contiguous.
    gemm_(out_view,
          input->view({N, in_features}),
          packed_weight, alpha, beta);
}

Tensor linear_packed_nz(Tensor input,
                        Tensor nz_weight,
                        std::optional<Tensor> bias,
                        float alpha) {
    Size ndim = input->ndim();
    Size out_features = nz_weight->shape()[1];
    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());
    linear_packed_nz_(out, input, nz_weight, bias, alpha);
    return out;
}

void linear_packed_nz_(Tensor out,
                       Tensor input,
                       Tensor nz_weight,
                       std::optional<Tensor> bias,
                       float alpha) {
    Size in_features = nz_weight->shape()[0];
    Size out_features = nz_weight->shape()[1];
    Size ndim = input->ndim();
    assert(out->ndim() == ndim);

    Size rows = 1;
    auto input_shape = input->shape();
    for (size_t i = 0; i < ndim - 1; ++i) {
        rows *= input_shape[i];
    }

    Tensor out_view = out->view({rows, out_features});

    // aclnnMatmulWeightNz has pure MatMul semantics and does not accept the
    // beta=1 accumulation pattern used by the ND GEMM bias path. Keep the NZ
    // GEMM itself bias-free, then add the broadcast bias in-place on Ascend.
    gemm_nz_(out_view,
             input->view({rows, in_features}),
             nz_weight, alpha, 0.0f);

    if (bias.has_value()) {
#ifdef ENABLE_ASCEND_API
        if (out->device().getType() == Device::Type::ASCEND) {
            execute_bias_add_inplace_ascend(
                out_view,
                bias.value()->as_strided({rows, out_features}, {0, 1}));
        } else
#endif
        {
            throw std::runtime_error(
                "linear_packed_nz with bias is only implemented on Ascend");
        }
    }
}

} // namespace infinicore::op
