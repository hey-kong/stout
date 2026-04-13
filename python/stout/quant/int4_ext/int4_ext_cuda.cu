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
__global__ void layer_absmax_kernel(
    const scalar_t* __restrict__ x,
    float* __restrict__ scale,
    int64_t layers,
    int64_t inner_elems) {

    int layer = blockIdx.x;
    if (layer >= layers) return;

    float local_max = 0.0f;
    const int64_t offset = (int64_t)layer * inner_elems;

    for (int64_t i = threadIdx.x; i < inner_elems; i += blockDim.x) {
        float v = static_cast<float>(x[offset + i]);
        float a = fabsf(v);
        if (a > local_max) local_max = a;
    }

    __shared__ float sdata[1024];
    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other = sdata[threadIdx.x + stride];
            if (other > sdata[threadIdx.x]) sdata[threadIdx.x] = other;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float s = sdata[0] / 7.0f;
        if (s < 1e-6f) s = 1e-6f;
        scale[layer] = s;
    }
}

template <typename scalar_t>
__global__ void quantize_pack_int4_kernel_3d(
    const scalar_t* __restrict__ x,
    const float* __restrict__ scale,
    uint8_t* __restrict__ packed,
    int64_t rows_total,
    int64_t rows_per_layer,
    int64_t orig_D,
    int64_t packed_D) {

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = rows_total * packed_D;
    if (idx >= n) return;

    int64_t row = idx / packed_D;
    int64_t p = idx - row * packed_D;

    int64_t layer = row / rows_per_layer;
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
    int64_t rows_per_layer,
    int64_t orig_D,
    int64_t packed_D) {

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n_elems = rows_total * orig_D;
    if (idx >= n_elems) return;

    int64_t row = idx / orig_D;
    int64_t d = idx - row * orig_D;

    int64_t layer = row / rows_per_layer;
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
    const int64_t layers = sizes[0];
    const int64_t orig_D = sizes[sizes.size() - 1];
    const int64_t packed_D = (orig_D + 1) / 2;

    int64_t rows_total = 1;
    for (int i = 0; i < x.dim() - 1; ++i) rows_total *= sizes[i];

    int64_t rows_per_layer = 1;
    for (int i = 1; i < x.dim() - 1; ++i) rows_per_layer *= sizes[i];

    const int64_t inner_elems = rows_per_layer * orig_D;

    std::vector<int64_t> packed_sizes(sizes.begin(), sizes.end());
    packed_sizes.back() = packed_D;
    auto packed = torch::empty(packed_sizes, x.options().dtype(torch::kUInt8));

    std::vector<int64_t> scale_sizes(sizes.begin(), sizes.end());
    for (size_t i = 1; i < scale_sizes.size(); ++i) scale_sizes[i] = 1;
    auto scale = torch::empty(scale_sizes, x.options().dtype(torch::kFloat32));

    auto x_2d = x.view({rows_total, orig_D});
    auto packed_2d = packed.view({rows_total, packed_D});
    auto scale_1d = scale.view({layers});

    auto stream = c10::cuda::getCurrentCUDAStream(x.device().index());

    int threads = 256;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "layer_absmax_kernel", [&] {
        layer_absmax_kernel<scalar_t><<<layers, threads, 0, stream.stream()>>>(
            x.data_ptr<scalar_t>(),
            scale_1d.data_ptr<float>(),
            layers,
            inner_elems
        );
    });

    int64_t n = rows_total * packed_D;
    int blocks = static_cast<int>((n + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "quantize_pack_int4_kernel_3d", [&] {
        quantize_pack_int4_kernel_3d<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
            x_2d.data_ptr<scalar_t>(),
            scale_1d.data_ptr<float>(),
            packed_2d.data_ptr<uint8_t>(),
            rows_total,
            rows_per_layer,
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
    int64_t layers = sizes[0];

    int64_t rows_total = 1;
    for (int i = 0; i < packed.dim() - 1; ++i) rows_total *= sizes[i];

    int64_t rows_per_layer = 1;
    for (int i = 1; i < packed.dim() - 1; ++i) rows_per_layer *= sizes[i];

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
        rows_per_layer,
        orig_D,
        packed_D
    );

    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "dequant_unpack_int4 kernel launch failed: ", cudaGetErrorString(err));
    }
    return out;
}
