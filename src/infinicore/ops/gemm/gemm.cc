#include "infinicore/ops/gemm.hpp"

#include "../../utils.hpp"

namespace infinicore::op {
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Gemm);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(GemmNz);

Gemm::Gemm(Tensor c, const Tensor &a, const Tensor &b, float alpha, float beta) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c, a, b, alpha, beta);
}

void Gemm::execute(Tensor c, const Tensor &a, const Tensor &b, float alpha, float beta) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Gemm, c, a, b, alpha, beta);
}

GemmNz::GemmNz(Tensor c, const Tensor &a, const Tensor &b, float alpha, float beta) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(c, a, b);
    INFINICORE_GRAPH_OP_DISPATCH(c->device().getType(), c, a, b, alpha, beta);
}

void GemmNz::execute(Tensor c, const Tensor &a, const Tensor &b, float alpha, float beta) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(GemmNz, c, a, b, alpha, beta);
}

Tensor gemm(const Tensor &a, const Tensor &b, float alpha, float beta) {
    Shape shape = a->shape();
    Size size = a->ndim();
    shape[size - 1] = b->size(size - 1);
    auto c = Tensor::empty(shape, a->dtype(), a->device());
    gemm_(c, a, b, alpha, beta);
    return c;
}

void gemm_(Tensor c, const Tensor &a, const Tensor &b, float alpha, float beta) {
    Gemm::execute(c, a, b, alpha, beta);
}

Tensor gemm_nz(const Tensor &a, const Tensor &b_nz, float alpha, float beta) {
    Shape shape = a->shape();
    Size size = a->ndim();
    shape[size - 1] = b_nz->size(size - 1);
    auto c = Tensor::empty(shape, a->dtype(), a->device());
    gemm_nz_(c, a, b_nz, alpha, beta);
    return c;
}

void gemm_nz_(Tensor c, const Tensor &a, const Tensor &b_nz, float alpha, float beta) {
    GemmNz::execute(c, a, b_nz, alpha, beta);
}

} // namespace infinicore::op
