#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAMathCompat.h>

static inline __device__ int8_t sign_extend_4bit(uint8_t v) {
    return (int8_t)((int8_t)(v << 4) >> 4);
}

template <typename scalar_t>
__global__ void quantize_pack_int4_kernel_3d(
    const scalar_t* __restrict__ x,
    const float* __restrict__ scale,
    uint8_t* __restrict__ packed,
    int64_t rows_total,
    int64_t layers,
    int64_t layer_row_stride,
    int64_t orig_D,
    int64_t packed_D) {

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows_total * packed_D;
    if (idx >= n) return;

    int64_t row = idx / packed_D;
    int64_t p = idx - row * packed_D;

    int64_t layer = (row / layer_row_stride) % layers;
    float s = scale[layer];

    int64_t d0 = p * 2;
    int64_t d1 = d0 + 1;

    const int64_t row_offset = row * orig_D;

    float qf0 = nearbyintf(static_cast<float>(x[row_offset + d0]) / s);
    qf0 = qf0 < -8.0f ? -8.0f : (qf0 > 7.0f ? 7.0f : qf0);
    int8_t v0 = static_cast<int8_t>(qf0);
    uint8_t low = static_cast<uint8_t>(v0) & 0x0F;

    uint8_t high = 0;
    if (d1 < orig_D) {
        float qf1 = nearbyintf(static_cast<float>(x[row_offset + d1]) / s);
        qf1 = qf1 < -8.0f ? -8.0f : (qf1 > 7.0f ? 7.0f : qf1);
        int8_t v1 = static_cast<int8_t>(qf1);
        high = (static_cast<uint8_t>(v1) & 0x0F) << 4;
    }

    packed[row * packed_D + p] = static_cast<uint8_t>(low | high);
}

__global__ void dequant_unpack_int4_kernel_2d(
    const uint8_t* __restrict__ packed,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int64_t rows_total,
    int64_t layers,
    int64_t layer_row_stride,
    int64_t orig_D,
    int64_t packed_D) {

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_elems = rows_total * orig_D;
    if (idx >= n_elems) return;

    int64_t row = idx / orig_D;
    int64_t d = idx - row * orig_D;

    int64_t layer = (row / layer_row_stride) % layers;
    float s = scale[layer];

    int64_t p = d >> 1;
    uint8_t byte = packed[row * packed_D + p];
    uint8_t nibble = (d & 1) ? (byte >> 4) : (byte & 0x0F);
    int8_t q = sign_extend_4bit(nibble);

    out[row * orig_D + d] = static_cast<float>(q) * s;
}

std::vector<at::Tensor> quantize_pack_int4_cuda(at::Tensor x) {
    c10::cuda::CUDAGuard device_guard(x.device());

    auto sizes = x.sizes();
    const int64_t layer_dim = x.dim() >= 4 ? 2 : 0;
    TORCH_CHECK(
        layer_dim < x.dim() - 1,
        "layer_dim must be a valid non-last dimension"
    );
    const int64_t layers = sizes[layer_dim];
    const int64_t orig_D = sizes[sizes.size() - 1];
    const int64_t packed_D = (orig_D + 1) / 2;

    int64_t rows_total = 1;
    for (int i = 0; i < x.dim() - 1; ++i) rows_total *= sizes[i];

    int64_t layer_row_stride = 1;
    for (int i = layer_dim + 1; i < x.dim() - 1; ++i) {
        layer_row_stride *= sizes[i];
    }

    std::vector<int64_t> packed_sizes(sizes.begin(), sizes.end());
    packed_sizes.back() = packed_D;
    auto packed = torch::empty(packed_sizes, x.options().dtype(torch::kUInt8));

    std::vector<int64_t> scale_sizes(sizes.begin(), sizes.end());
    for (int i = 0; i < scale_sizes.size(); ++i) {
        if (i != layer_dim) {
            scale_sizes[i] = 1;
        }
    }
    auto scale = torch::empty(scale_sizes, x.options().dtype(torch::kFloat32));
    std::vector<int64_t> reduce_dims;
    reduce_dims.reserve(x.dim() - 1);
    for (int i = 0; i < x.dim(); ++i) {
        if (i != layer_dim) {
            reduce_dims.push_back(i);
        }
    }
    auto scale_reduced = x.abs().amax(reduce_dims, /*keepdim=*/true);
    scale.copy_(torch::clamp(scale_reduced / 7.0, 1e-6));

    auto x_2d = x.view({rows_total, orig_D});
    auto packed_2d = packed.view({rows_total, packed_D});
    auto scale_1d = scale.view({layers});

    auto stream = c10::cuda::getCurrentCUDAStream(x.device().index());

    int threads = 256;
    int64_t n = rows_total * packed_D;
    int blocks = static_cast<int>((n + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "quantize_pack_int4_kernel_3d", [&] {
        quantize_pack_int4_kernel_3d<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            x_2d.data_ptr<scalar_t>(),
            scale_1d.data_ptr<float>(),
            packed_2d.data_ptr<uint8_t>(),
            rows_total,
            layers,
            layer_row_stride,
            orig_D,
            packed_D
        );
    });

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "quantize_pack_int4 kernel launch failed: ", cudaGetErrorString(err));
    }

    return {packed, scale};
}

at::Tensor dequant_unpack_int4_cuda(at::Tensor packed, at::Tensor scale, int64_t orig_D) {
    c10::cuda::CUDAGuard device_guard(packed.device());

    auto sizes = packed.sizes();
    int64_t packed_D = sizes.back();
    const int64_t layer_dim = packed.dim() >= 4 ? 2 : 0;
    TORCH_CHECK(
        layer_dim < packed.dim() - 1,
        "layer_dim must be a valid non-last dimension"
    );
    int64_t layers = sizes[layer_dim];

    int64_t rows_total = 1;
    for (int i = 0; i < packed.dim() - 1; ++i) rows_total *= sizes[i];

    int64_t layer_row_stride = 1;
    for (int i = layer_dim + 1; i < packed.dim() - 1; ++i) {
        layer_row_stride *= sizes[i];
    }

    TORCH_CHECK(scale.numel() == layers, "scale.numel() must equal number of layers");

    std::vector<int64_t> out_sizes(sizes.begin(), sizes.end());
    out_sizes.back() = orig_D;

    auto out = torch::empty(out_sizes, packed.options().dtype(torch::kFloat32));

    auto packed_2d = packed.view({rows_total, packed_D});
    auto out_2d = out.view({rows_total, orig_D});
    auto scale_1d = scale.view({layers});

    int threads = 256;
    int64_t n = rows_total * orig_D;
    int blocks = static_cast<int>((n + threads - 1) / threads);

    auto stream = c10::cuda::getCurrentCUDAStream(packed.device().index());

    dequant_unpack_int4_kernel_2d<<<blocks, threads, 0, stream.stream()>>>(
        packed_2d.data_ptr<uint8_t>(),
        scale_1d.data_ptr<float>(),
        out_2d.data_ptr<float>(),
        rows_total,
        layers,
        layer_row_stride,
        orig_D,
        packed_D
    );

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "dequant_unpack_int4 kernel launch failed: ", cudaGetErrorString(err));
    }
    return out;
}
