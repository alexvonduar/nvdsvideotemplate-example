
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "nv12_to_rgb_kernel.hpp"

#define checkRuntime(call)  check_runtime(call, #call, __LINE__, __FILE__)
#define half2short(h)   (*(unsigned short*)&h)

typedef unsigned char uint8_t;

template<typename _T>struct AsUnion4{};
template<>struct AsUnion4<uint8_t>{typedef uchar4  type;};
template<>struct AsUnion4<float>  {typedef float4  type;};
template<>struct AsUnion4<__half> {typedef ushort4 type;};

template<typename _T>struct AsUnion3{};
template<>struct AsUnion3<uint8_t>{typedef uchar3  type;};
template<>struct AsUnion3<float>  {typedef float3  type;};
template<>struct AsUnion3<__half> {typedef ushort3 type;};

template<DataType _T>struct AsPODType{};
template<>struct AsPODType<DataType::Uint8>   {typedef uint8_t type;};
template<>struct AsPODType<DataType::Float32> {typedef float   type;};
template<>struct AsPODType<DataType::Float16> {typedef __half  type;};

enum class Parallel : unsigned int{
    None        = 0,
    SinglePixel = 1,
    FourPixel   = 2
};

static __device__ __forceinline__ uchar4 make4(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3){return make_uchar4(v0, v1, v2, v3);}
static __device__ __forceinline__ float4 make4(float v0, float v1, float v2, float v3){return make_float4(v0, v1, v2, v3);}
static __device__ __forceinline__ ushort4 make4(__half v0, __half v1, __half v2, __half v3){return make_ushort4(half2short(v0), half2short(v1), half2short(v2), half2short(v3)); }

static __device__ __forceinline__ uchar3 make3(uint8_t v0, uint8_t v1, uint8_t v2){return make_uchar3(v0, v1, v2);}
static __device__ __forceinline__ float3 make3(float v0, float v1, float v2){return make_float3(v0, v1, v2);}
static __device__ __forceinline__ ushort3 make3(__half v0, __half v1, __half v2){return make_ushort3(half2short(v0), half2short(v1), half2short(v2)); }

#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)

template<typename _T>
static __forceinline__ __device__ _T limit(_T value, _T low, _T high){
    return value < low ? low : (value > high ? high : value);
}

template<typename _T>
static __host__ __device__ __forceinline__ uint8_t u8cast(_T value){
    return value < 0 ? 0 : (value >= 255 ? 255 : value);
}

template<typename _T>
static __host__ __device__ __forceinline__ _T fpcast(_T value){
    return value < 0 ? 0 : (value >= 255 ? 255 : value);
}

static bool __inline__ check_runtime(cudaError_t e, const char* call, int line, const char *file){
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d\n", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
        return false;
    }
    return true;
}

template<typename _DataType, PixelLayout _Layout, Parallel parallel>
struct DataLayoutInvoker{};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NHWC RGB
template<typename _DataType>
struct DataLayoutInvoker<_DataType, PixelLayout::NHWC_RGB, Parallel::SinglePixel>{
    static __device__ __forceinline__ void call(_DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height){

        _DataType* p = pdst + (ib * height + y) * stride + x * 3;
        p[0] = r; p[1] = g; p[2] = b;
    }
};

template<typename _DataType>
struct DataLayoutInvoker<_DataType, PixelLayout::NHWC_RGB, Parallel::FourPixel>{
    static __device__ __forceinline__ void call(_DataType* pdst, _DataType r[4], _DataType g[4], _DataType b[4], int ib, int x, int y, int width, int stride, int height){
        
        _DataType* p0 = pdst + (ib * height + y) * stride + (x+0) * 3;
        _DataType* p1 = pdst + (ib * height + y) * stride + (x+1) * 3;
        _DataType* p2 = pdst + (ib * height + y) * stride + (x+2) * 3;
        _DataType* p3 = pdst + (ib * height + y) * stride + (x+3) * 3;
        *(typename AsUnion3<_DataType>::type*)p0 = make3(r[0], g[0], b[0]);
        *(typename AsUnion3<_DataType>::type*)p1 = make3(r[1], g[1], b[1]);
        *(typename AsUnion3<_DataType>::type*)p2 = make3(r[2], g[2], b[2]);
        *(typename AsUnion3<_DataType>::type*)p3 = make3(r[3], g[3], b[3]);
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NHWC BGR
template<typename _DataType>
struct DataLayoutInvoker<_DataType, PixelLayout::NHWC_BGR, Parallel::SinglePixel>{
    static __device__ __forceinline__ void call(_DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height){

        _DataType* p = pdst + (ib * height + y) * stride + x * 3;
        p[0] = b; p[1] = g; p[2] = r;
    }
};

template<typename _DataType>
struct DataLayoutInvoker<_DataType, PixelLayout::NHWC_BGR, Parallel::FourPixel>{
    static __device__ __forceinline__ void call(_DataType* pdst, _DataType r[4], _DataType g[4], _DataType b[4], int ib, int x, int y, int width, int stride, int height){

        _DataType* p0 = pdst + (ib * height + y) * stride + (x+0) * 3;
        _DataType* p1 = pdst + (ib * height + y) * stride + (x+1) * 3;
        _DataType* p2 = pdst + (ib * height + y) * stride + (x+2) * 3;
        _DataType* p3 = pdst + (ib * height + y) * stride + (x+3) * 3;
        *(typename AsUnion3<_DataType>::type*)p0 = make3(b[0], g[0], r[0]);
        *(typename AsUnion3<_DataType>::type*)p1 = make3(b[1], g[1], r[1]);
        *(typename AsUnion3<_DataType>::type*)p2 = make3(b[2], g[2], r[2]);
        *(typename AsUnion3<_DataType>::type*)p3 = make3(b[3], g[3], r[3]);
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NCHW RGB
template<typename _DataType>
struct DataLayoutInvoker<_DataType, PixelLayout::NCHW_RGB, Parallel::SinglePixel>{
    static __device__ __forceinline__ void call(_DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height){

        *(pdst + (((ib * 3 + 0) * height + y) * width + x)) = r;
        *(pdst + (((ib * 3 + 1) * height + y) * width + x)) = g;
        *(pdst + (((ib * 3 + 2) * height + y) * width + x)) = b;
    }
};

template<typename _DataType>
struct DataLayoutInvoker<_DataType, PixelLayout::NCHW_RGB, Parallel::FourPixel>{
    static __device__ __forceinline__ void call(_DataType* pdst, _DataType r[4], _DataType g[4], _DataType b[4], int ib, int x, int y, int width, int stride, int height){

        *(typename AsUnion4<_DataType>::type*)(pdst + (((ib * 3 + 0) * height + y) * width + x)) = make4(r[0], r[1], r[2], r[3]);
        *(typename AsUnion4<_DataType>::type*)(pdst + (((ib * 3 + 1) * height + y) * width + x)) = make4(g[0], g[1], g[2], g[3]);
        *(typename AsUnion4<_DataType>::type*)(pdst + (((ib * 3 + 2) * height + y) * width + x)) = make4(b[0], b[1], b[2], b[3]);
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NCHW BGR
template<typename _DataType>
struct DataLayoutInvoker<_DataType, PixelLayout::NCHW_BGR, Parallel::SinglePixel>{
    static __device__ __forceinline__ void call(_DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height){

        *(pdst + (((ib * 3 + 0) * height + y) * width + x)) = b;
        *(pdst + (((ib * 3 + 1) * height + y) * width + x)) = g;
        *(pdst + (((ib * 3 + 2) * height + y) * width + x)) = r;
    }
};

template<typename _DataType>
struct DataLayoutInvoker<_DataType, PixelLayout::NCHW_BGR, Parallel::FourPixel>{
    static __device__ __forceinline__ void call(_DataType* pdst, _DataType r[4], _DataType g[4], _DataType b[4], int ib, int x, int y, int width, int stride, int height){

        *(typename AsUnion4<_DataType>::type*)(pdst + (((ib * 3 + 0) * height + y) * width + x)) = make4(b[0], b[1], b[2], b[3]);
        *(typename AsUnion4<_DataType>::type*)(pdst + (((ib * 3 + 1) * height + y) * width + x)) = make4(g[0], g[1], g[2], g[3]);
        *(typename AsUnion4<_DataType>::type*)(pdst + (((ib * 3 + 2) * height + y) * width + x)) = make4(r[0], r[1], r[2], r[3]);
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NCHW16 RGB
template<typename _DataType>
struct DataLayoutInvoker<_DataType, PixelLayout::NCHW16_RGB, Parallel::SinglePixel>{
    static __device__ __forceinline__ void call(_DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height){

        _DataType* p = pdst + ((ib * height + y) * width + x) * 16;
        p[0] = r; p[1] = g; p[2] = b;
    }
};

template<typename _DataType>
struct DataLayoutInvoker<_DataType, PixelLayout::NCHW16_RGB, Parallel::FourPixel>{
    static __device__ __forceinline__ void call(_DataType* pdst, _DataType r[4], _DataType g[4], _DataType b[4], int ib, int x, int y, int width, int stride, int height){

        _DataType* p0 = pdst + ((ib * height + y) * width + x+0) * 16;
        _DataType* p1 = pdst + ((ib * height + y) * width + x+1) * 16;
        _DataType* p2 = pdst + ((ib * height + y) * width + x+2) * 16;
        _DataType* p3 = pdst + ((ib * height + y) * width + x+3) * 16;
        *(typename AsUnion3<_DataType>::type*)p0 = make3(r[0], g[0], b[0]);
        *(typename AsUnion3<_DataType>::type*)p1 = make3(r[1], g[1], b[1]);
        *(typename AsUnion3<_DataType>::type*)p2 = make3(r[2], g[2], b[2]);
        *(typename AsUnion3<_DataType>::type*)p3 = make3(r[3], g[3], b[3]);
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// NCHW16 BGR
template<typename _DataType>
struct DataLayoutInvoker<_DataType, PixelLayout::NCHW16_BGR, Parallel::SinglePixel>{
    static __device__ __forceinline__ void call(_DataType* pdst, _DataType r, _DataType g, _DataType b, int ib, int x, int y, int width, int stride, int height){

        _DataType* p = pdst + ((ib * height + y) * width + x) * 16;
        p[0] = b; p[1] = g; p[2] = r;
    }
};

template<typename _DataType>
struct DataLayoutInvoker<_DataType, PixelLayout::NCHW16_BGR, Parallel::FourPixel>{
    static __device__ __forceinline__ void call(_DataType* pdst, _DataType r[4], _DataType g[4], _DataType b[4], int ib, int x, int y, int width, int stride, int height){

        _DataType* p0 = pdst + ((ib * height + y) * width + x+0) * 16;
        _DataType* p1 = pdst + ((ib * height + y) * width + x+1) * 16;
        _DataType* p2 = pdst + ((ib * height + y) * width + x+2) * 16;
        _DataType* p3 = pdst + ((ib * height + y) * width + x+3) * 16;
        *(typename AsUnion3<_DataType>::type*)p0 = make3(b[0], g[0], r[0]);
        *(typename AsUnion3<_DataType>::type*)p1 = make3(b[1], g[1], r[1]);
        *(typename AsUnion3<_DataType>::type*)p2 = make3(b[2], g[2], r[2]);
        *(typename AsUnion3<_DataType>::type*)p3 = make3(b[3], g[3], r[3]);
    }
};
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static __device__ unsigned int __forceinline__ round_down2(unsigned int num){
    return num & (~1);
}

template<typename _T>
static __device__ void __forceinline__ scale_rgb(
    uint8_t r0, uint8_t g0, uint8_t b0, _T& r, _T& g, _T& b,
    float mean0, float mean1, float mean2, float scale0, float scale1, float scale2
){
    r = (r0 - mean0) * scale0;
    g = (g0 - mean1) * scale1;
    b = (b0 - mean2) * scale2;
}

static __device__ void __forceinline__ yuv2rgb(
    uint8_t y, uint8_t u, uint8_t v, uint8_t& r, uint8_t& g, uint8_t& b
){
    int iyval = 1220542*(y - 16);
    r = u8cast((iyval + 1673527*(v - 128)                      + (1 << 19)) >> 20);
    g = u8cast((iyval - 852492*(v - 128) - 409993*(u - 128)    + (1 << 19)) >> 20);
    b = u8cast((iyval                      + 2116026*(u - 128) + (1 << 19)) >> 20);
}

template<NV12Format nv12_format>
static __device__ void __forceinline__ load_nv12_pixel(
    const void* luma, const void* chroma,
    int x, int y, int down_x, int width, int stride, uint8_t& r, uint8_t& g, uint8_t& b
);

// BL sample pixel implmentation
template<>
__device__ void __forceinline__ load_nv12_pixel<NV12Format::BlockLinear>(
    const void* luma, const void* chroma,
    int x, int y, int down_x, int width, int stride, uint8_t& r, uint8_t& g, uint8_t& b
){
    uint8_t yv = tex2D<uint8_t>((cudaTextureObject_t)luma,   x,          y    );
    uchar2 uv = tex2D<uchar2>((cudaTextureObject_t)chroma, x / 2, y / 2);
    yuv2rgb(yv, uv.x, uv.y, r, g, b);
}

// PL sample pixel implmentation
template<>
__device__ void __forceinline__ load_nv12_pixel<NV12Format::PitchLinear>(
    const void* luma, const void* chroma,
    int x, int y, int down_x, int width, int stride, uint8_t& r, uint8_t& g, uint8_t& b
){
    uint8_t yv = *((const unsigned char*)luma + y * stride + x);
    uint8_t uv = *((const unsigned char*)chroma + (y / 2) * stride + down_x + 0);
    uint8_t vv = *((const unsigned char*)chroma + (y / 2) * stride + down_x + 1);
    yuv2rgb(yv, uv, vv, r, g, b);
}

template<typename DType, NV12Format nv12_format, Interpolation interp>
struct SamplePixel{};

// BL sample pixel implmentation
template<typename DType, NV12Format format>
struct SamplePixel<DType, format, Interpolation::Nearest>{
    static __device__ void __forceinline__ call(
        const void* luma, const void* chroma, 
        int x, int y, float sx, float sy, int ybatch_offset, int width, int stride, int height,
        uint8_t& r, uint8_t& g, uint8_t& b
    ){
        // In some cases, the floating point precision will lead to miscalculation of the value, 
        // making the result not exactly match with opencv, 
        // so here you need to add eps as precision compensation
        //
        // A special case is when the input is 3840 and the output is 446, x = 223:
        // const int src_x_double = 223.0  * (3840.0  / 446.0);            // -> 1920
        // const int src_x_float  = 223.0f * (3840.0f / 446.0f);           // -> 1919
        // const int src_x_float  = 223.0f * (3840.0f / 446.0f) + 1e-5;    // -> 1920
        //
        // !!! If you want to use the double for sx/sy, you'll get a 2x speed drop
        const float eps = 1e-5;
        int ix = x * sx + eps;
        int iy = y * sy + eps + ybatch_offset;
        load_nv12_pixel<format>(luma, chroma, ix, iy, round_down2(ix), width, stride, r, g, b);
    }
};

template<typename DType, NV12Format format>
struct SamplePixel<DType, format, Interpolation::Bilinear>{
    static __device__ void __forceinline__ call(
        const void* luma, const void* chroma, 
        int x, int y, float sx, float sy, int ybatch_offset, int width, int stride, int height,
        uint8_t& r, uint8_t& g, uint8_t& b
    ){
        uint8_t r0[4], g0[4], b0[4];
        float src_x = (x + 0.5f) * sx - 0.5f;
        float src_y = (y + 0.5f) * sy - 0.5f;
        int y_low  = floorf(src_y);
        int x_low  = floorf(src_x);
        int y_high = limit(y_low + 1, 0, height - 1);
        int x_high = limit(x_low + 1, 0, width - 1);
        y_low = limit(y_low, 0, height - 1);
        x_low = limit(x_low, 0, width - 1);

        int ly = rint((src_y - y_low) * INTER_RESIZE_COEF_SCALE);
        int lx = rint((src_x - x_low) * INTER_RESIZE_COEF_SCALE);
        int hy = INTER_RESIZE_COEF_SCALE - ly;
        int hx = INTER_RESIZE_COEF_SCALE - lx;

        load_nv12_pixel<format>(luma, chroma, x_low,  y_low + ybatch_offset,  round_down2(x_low),  width, stride, r0[0], g0[0], b0[0]);
        load_nv12_pixel<format>(luma, chroma, x_high, y_low + ybatch_offset,  round_down2(x_high), width, stride, r0[1], g0[1], b0[1]);
        load_nv12_pixel<format>(luma, chroma, x_low,  y_high + ybatch_offset, round_down2(x_low),  width, stride, r0[2], g0[2], b0[2]);
        load_nv12_pixel<format>(luma, chroma, x_high, y_high + ybatch_offset, round_down2(x_high), width, stride, r0[3], g0[3], b0[3]);

        r = ( ((hy * ((hx * r0[0] + lx * r0[1]) >> 4)) >> 16) + ((ly * ((hx * r0[2] + lx * r0[3]) >> 4)) >> 16) + 2 )>>2;
        g = ( ((hy * ((hx * g0[0] + lx * g0[1]) >> 4)) >> 16) + ((ly * ((hx * g0[2] + lx * g0[3]) >> 4)) >> 16) + 2 )>>2;
        b = ( ((hy * ((hx * b0[0] + lx * b0[1]) >> 4)) >> 16) + ((ly * ((hx * b0[2] + lx * b0[3]) >> 4)) >> 16) + 2 )>>2;
    }
};

template<NV12Format nv12_format, typename OutDType, PixelLayout layout, Interpolation interp>
static __global__ void convert_nv12_to_rgb_kernel_4x(
    const void* luma, const void* chroma, 
    OutDType* pdst, float sx, float sy,
    int src_height, int src_width, int src_stride, float mean0, float mean1, float mean2, float scale0, float scale1, float scale2,
    int dst_width, int dst_stride, int dst_height, int nbatch
){
    int x = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= dst_width-3 || y >= dst_height) return;

    OutDType r[4], g[4], b[4];
    uint8_t r0, g0, b0;
    for(int ib = blockIdx.z; ib < nbatch; ib += gridDim.z){
        int ybatch_offset = ib * src_height;
        for(int ip = 0; ip < 4; ++ip){
            SamplePixel<OutDType, nv12_format, interp>::call(
                luma, chroma, x+ip, y, sx, sy, ybatch_offset, src_width, src_stride, src_height,
                r0, g0, b0
            );
            scale_rgb(r0, g0, b0, r[ip], g[ip], b[ip], mean0, mean1, mean2, scale0, scale1, scale2);
        }

        DataLayoutInvoker<OutDType, layout, Parallel::FourPixel>::call(
            pdst, r, g, b, ib, x, y, dst_width, dst_stride, dst_height
        );
    }
}

template<NV12Format nv12_format, typename OutDType, PixelLayout layout, Interpolation interp>
static __global__ void convert_nv12_to_rgb_kernel_1x(
    const void* luma, const void* chroma, 
    OutDType* pdst, float sx, float sy,
    int src_height, int src_width, int src_stride, float mean0, float mean1, float mean2, float scale0, float scale1, float scale2,
    int dst_width, int dst_stride, int dst_height, int nbatch
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= dst_width || y >= dst_height) return;

    OutDType r, g, b;
    uint8_t r0, g0, b0;
    for(int ib = blockIdx.z; ib < nbatch; ib += gridDim.z){
        int ybatch_offset = ib * src_height;
        SamplePixel<OutDType, nv12_format, interp>::call(
            luma, chroma, x, y, sx, sy, ybatch_offset, src_width, src_stride, src_height,
            r0, g0, b0
        );

        scale_rgb(r0, g0, b0, r, g, b, mean0, mean1, mean2, scale0, scale1, scale2);
        DataLayoutInvoker<OutDType, layout, Parallel::SinglePixel>::call(
            pdst, r, g, b, ib, x, y, dst_width, dst_stride, dst_height
        );
    }
}

template<NV12Format nv12_format, DataType out_dtype, PixelLayout layout, Interpolation interp>
void batched_convert_nv12_to_rgb_impl(
    const void* luma, const void* chroma, int input_width, int input_stride, int input_height, int input_batch,
    void* out_ptr, int out_width, int out_stride, int out_height, 
    float mean0, float mean1, float mean2, float scale0, float scale1, float scale2,
    cudaStream_t stream
){
    float sx = input_width  / (float)out_width;
    float sy = input_height / (float)out_height;
    using OutDType = typename AsPODType<out_dtype>::type;

    if(
        layout == PixelLayout::NHWC_BGR   || 
        layout == PixelLayout::NHWC_RGB   || 
        layout == PixelLayout::NCHW16_RGB || 
        layout == PixelLayout::NCHW16_BGR ||
        input_stride % 4 != 0
    ){
        int grid_z = input_batch >= 32 ? 32 : input_batch;
        dim3 dim_block(32, 32);
        dim3 dim_grid((out_width  + dim_block.x - 1) / dim_block.x,
                    (out_height + dim_block.y - 1) / dim_block.y, grid_z);
        convert_nv12_to_rgb_kernel_1x<nv12_format, OutDType, layout, interp> <<<dim_grid, dim_block, 0, stream>>>(
            luma, chroma, 
            (OutDType*)out_ptr, sx, sy, 
            input_height, input_width, input_stride,
            mean0, mean1, mean2, scale0, scale1, scale2, 
            out_width, out_stride, out_height, input_batch
        );

    }else{
        int grid_z = input_batch >= 32 ? 32 : input_batch;
        dim3 dim_block(16, 32);
        dim3 dim_grid(((out_width + 3) / 4  + dim_block.x - 1) / dim_block.x,
                       (out_height + dim_block.y - 1) / dim_block.y, grid_z);
        convert_nv12_to_rgb_kernel_4x<nv12_format, OutDType, layout, interp> <<<dim_grid, dim_block, 0, stream>>>(
            luma, chroma, 
            (OutDType*)out_ptr, sx, sy, 
            input_height, input_width, input_stride,
            mean0, mean1, mean2, scale0, scale1, scale2, 
            out_width, out_stride, out_height, input_batch
        );
    }
    checkRuntime(cudaPeekAtLastError());
}

typedef void(*batched_convert_nv12_to_rgb_impl_function)(
    const void* luma, const void* chroma, int input_width, int input_stride, int input_height, int input_batch,
    void* out_ptr, int out_width, int out_stride, int out_height, 
    float mean0, float mean1, float mean2, float scale0, float scale1, float scale2,
    cudaStream_t stream
);


// If you want to modify this part of the code, 
// please note that the order of the enumerated types must match the integer values of this type 
// (note: that the order starts from 1)

#define DefineNV12Format(...)                                               \
    batched_convert_nv12_to_rgb_impl<NV12Format::BlockLinear, __VA_ARGS__>,  \
    batched_convert_nv12_to_rgb_impl<NV12Format::PitchLinear, __VA_ARGS__>,

#define DefineDType(...)                                              \
    DefineNV12Format(DataType::Uint8, __VA_ARGS__)                    \
    DefineNV12Format(DataType::Float32, __VA_ARGS__)                  \
    DefineNV12Format(DataType::Float16, __VA_ARGS__)                  

#define DefineLayout(...)                                            \
    DefineDType(PixelLayout::NCHW_RGB, __VA_ARGS__)                  \
    DefineDType(PixelLayout::NCHW_BGR, __VA_ARGS__)                  \
    DefineDType(PixelLayout::NHWC_RGB, __VA_ARGS__)                  \
    DefineDType(PixelLayout::NHWC_BGR, __VA_ARGS__)                  \
    DefineDType(PixelLayout::NCHW16_RGB, __VA_ARGS__)                \
    DefineDType(PixelLayout::NCHW16_BGR, __VA_ARGS__)                  

#define DefineInterp                                          \
    DefineLayout(Interpolation::Nearest)                      \
    DefineLayout(Interpolation::Bilinear)              

#define DefineAllFunction   DefineInterp


template<typename T>struct EnumCount{};
template<> struct EnumCount<NV12Format>{static const unsigned int value = 2;};
template<> struct EnumCount<DataType>{static const unsigned int value = 3;};
template<> struct EnumCount<PixelLayout>{static const unsigned int value = 6;};
template<> struct EnumCount<Interpolation>{static const unsigned int value = 2;};

static const batched_convert_nv12_to_rgb_impl_function func_list[] = {
    DefineAllFunction
    nullptr
};  

void batched_convert_nv12_to_rgb(
    const void* luma, const void* chroma, int input_width, int input_stride, int input_height, int input_batch, NV12Format nv12_format, 
    void* out_ptr, int out_width, int out_stride, int out_height, DataType out_dtype, PixelLayout out_layout, Interpolation interp,
    float mean0, float mean1, float mean2, float scale0, float scale1, float scale2,
    void* stream
){
    int iformat = (int)nv12_format - 1;
    int odtype  = (int)out_dtype - 1;
    int olayout = (int)out_layout - 1;
    int iinterp  = (int)interp - 1;
    int index = ((iinterp * EnumCount<PixelLayout>::value + olayout) * EnumCount<DataType>::value + odtype) * EnumCount<NV12Format>::value + iformat;
    if(
        iformat < 0 || iformat >= EnumCount<NV12Format>::value ||
        odtype < 0  || odtype >= EnumCount<DataType>::value ||
        olayout < 0 || olayout >= EnumCount<PixelLayout>::value ||
        iinterp < 0 || iinterp >= EnumCount<Interpolation>::value ||
        index < 0 || index >= sizeof(func_list) / sizeof(func_list[0]) - 1
    ){
        fprintf(stderr, "Unsupported configure %d.\n", index);
        return;
    }

    batched_convert_nv12_to_rgb_impl_function func = func_list[index];
    func(
        luma, chroma, input_width, input_stride, input_height, input_batch,
        out_ptr, out_width, out_stride, out_height,
        mean0, mean1, mean2, scale0, scale1, scale2,
        (cudaStream_t)stream
    );
}
