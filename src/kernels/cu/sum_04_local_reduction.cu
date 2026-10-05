#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#define WARP_SIZE 32

__global__ void sum_04_local_reduction(
    const unsigned int* a,
    unsigned int* b,
    unsigned int n)
{
    const uint index = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ uint local[GROUP_SIZE];
    if (index >= n) {
        local[threadIdx.x] = 0;
    } else {
        local[threadIdx.x] = a[index];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint tmp = 0;
        for (int i = 0; i < GROUP_SIZE; ++i) {
            tmp += local[i];
        }

        b[blockIdx.x] = tmp;
    }
}

namespace cuda {
void sum_04_local_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, gpu::gpu_mem_32u& sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_04_local_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
