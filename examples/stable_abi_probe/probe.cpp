#include <filesystem>
#include <string>
#include <string_view>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <deep_jit/backend/cuda/backend.hpp>

namespace {

using Runtime = deep_jit::Runtime<deep_jit::CUDA>;

constexpr std::string_view add_one_source = R"cuda(
extern "C" __global__ void add_one_kernel(
        const float* input, float* output, long long numel) {
    const long long index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < numel)
        output[index] = input[index] + 1.0f;
}
)cuda";

Runtime& get_runtime() {
    static Runtime runtime(deep_jit::Config(
        std::filesystem::path(__FILE__).parent_path(),
        "DEEP_JIT_STABLE_ABI_PROBE",
        "add-one-v1"));
    return runtime;
}

torch::stable::Tensor add_one(const torch::stable::Tensor& input) {
    STD_TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    STD_TORCH_CHECK(
        input.scalar_type() == torch::headeronly::ScalarType::Float,
        "input must have dtype torch.float32");

    torch::stable::accelerator::DeviceGuard guard(input.get_device_index());
    auto output = torch::stable::empty_like(input);
    const int64_t numel = input.numel();
    if (numel == 0)
        return output;

    auto& runtime = get_runtime();
    const auto kernel = runtime.compile("add_one", std::string(add_one_source));
    constexpr unsigned int block_size = 256;
    const auto grid_size = static_cast<unsigned int>(
        (numel + block_size - 1) / block_size);
    runtime.launch(
        kernel,
        {
            .grid_dim = dim3(grid_size),
            .block_dim = dim3(block_size),
        },
        input.const_data_ptr<float>(),
        output.mutable_data_ptr<float>(),
        numel);
    return output;
}

}  // namespace

STABLE_TORCH_LIBRARY(deep_jit_stable_abi_probe, m) {
    m.def("add_one(Tensor input) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(deep_jit_stable_abi_probe, CUDA, m) {
    m.impl("add_one", TORCH_BOX(&add_one));
}
