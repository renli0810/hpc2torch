#include <cuda.h>
#include <cub/cub.cuh>

constexpr long long BLOCKdim = 128;
// constexpr long long GRIDdim_x =  ;
// constexpr long long GRIDdim_y = 1;

template <typename T, typename Tind>
__global__ void gatherkernel(T const *input, Tind const *indices, T *output, long long stride, long long allSize)
{
    long long overcount = (stride + BLOCKdim - 1) / BLOCKdim;
    long long outputindex = blockIdx.x * gridDim.y * stride + blockIdx.y * stride + threadIdx.x;
    long long index = indices[blockIdx.y];
    long long inputindex = blockIdx.x * gridDim.y * stride + index * stride + threadIdx.x;
    if (threadIdx.x >= stride)
        return;
    for (int i = 0; i < overcount; i++)
    {
        if (outputindex >= allSize)
        {
            return;
        }
        output[outputindex] = input[inputindex];
        // printf("%d %d %d %d stride:%d\n",gridDim.x,gridDim.y,blockDim.x, blockDim.y, stride);
        // printf("outputindex=%d inputindex=%d input[inputindex]=%lf index=%d allSize=%d blockx=%d blocky=%d threadx=%d\n", outputindex, inputindex, input[inputindex], index, allSize, blockIdx.x, blockIdx.y, threadIdx.x);
        inputindex += BLOCKdim;
        outputindex += BLOCKdim;
    }
}

template <typename T, typename Tind>
void gatherLaunch(void const *input, void const *indices, void *output, long long stride, long long axisSize, long long inputSize, long long indexSize)
{
    long long griddim_x = inputSize / axisSize / stride;
    long long griddim_y = indexSize;
    long long allSize = indexSize * inputSize / axisSize;
    // printf("x:%d y:%d\n", griddim_x, griddim_y);
    dim3 GRIDdim(griddim_x, griddim_y);
    gatherkernel<T, Tind>
        <<<GRIDdim, BLOCKdim>>>((T *)input, (Tind *)indices, (T *)output, stride, allSize);
}
extern "C" void gather_nv_f32(void const *input, void const *indices, void *output, long long stride, long long axisSize, long long inputSize, long long indexSize)
{
    gatherLaunch<float, uint64_t>(input, indices, output, stride, axisSize, inputSize, indexSize);
}
extern "C" void gather_nv_f16(void const *input, void const *indices, void *output, long long stride, long long axisSize, long long inputSize, long long indexSize)
{
    gatherLaunch<half, uint64_t>(input, indices, output, stride, axisSize, inputSize, indexSize);
}