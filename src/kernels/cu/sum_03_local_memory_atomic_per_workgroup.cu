#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/work_size.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

__global__ void sum_03_local_memory_atomic_per_workgroup(
    const unsigned int* a,
    unsigned int* sum,
    unsigned int n)
{
    __shared__ uint local[GROUP_SIZE];
    const uint index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= n) {
        local[threadIdx.x] = 0;
    } else {
        local[threadIdx.x] = a[index];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint suma = 0;

        for (int i = 0; i < GROUP_SIZE; ++i) {
            suma += local[i];
        }

        atomicAdd(sum, suma);
    }
}

namespace cuda {
void sum_03_local_memory_atomic_per_workgroup(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, gpu::gpu_mem_32u& sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_03_local_memory_atomic_per_workgroup<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
