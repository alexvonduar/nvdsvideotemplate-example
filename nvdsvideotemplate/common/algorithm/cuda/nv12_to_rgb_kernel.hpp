#ifndef NV12_TO_RGB_KERNEL_HPP
#define NV12_TO_RGB_KERNEL_HPP

enum class PixelLayout : unsigned int{
    NONE       = 0,
    NCHW_RGB   = 1,
    NCHW_BGR   = 2,
    NHWC_RGB   = 3,
    NHWC_BGR   = 4,
    NCHW16_RGB = 5,  // c = (c + 15) / 16 * 16 if c % 16 != 0 else c
    NCHW16_BGR = 6   // NHW16 if c == 3
};

enum class DataType : unsigned int{
    NONE             = 0,
    Uint8            = 1,
    Float32          = 2,
    Float16          = 3
};

enum class Interpolation : unsigned int{
    NONE     = 0,
    Nearest  = 1,
    Bilinear = 2
};

enum class NV12Format : unsigned int{
    NONE         = 0,
    BlockLinear  = 1,
    PitchLinear  = 2
};

// If nv12_format == BlockLinear, luma must be of type cudaTexture_t, otherwise luma must be ydata of type unsigned char*.
// If nv12_format == BlockLinear, chroma must be of type cudaTexture_t, otherwise chroma must be uvdata of type unsigned char*.
// if out_layout  == NHWC_RGB or NHWC_BGR, out_stride are used, otherwise ignore out_stride
void batched_convert_nv12_to_rgb(
    const void* luma, const void* chroma, int input_width, int input_stride, int input_height, int input_batch, NV12Format nv12_format, 
    void* out_ptr, int out_width, int out_stride, int out_height, DataType out_dtype, PixelLayout out_layout, Interpolation interp,
    float mean0, float mean1, float mean2, float scale0, float scale1, float scale2,
    void* stream
);

#endif // NV12_TO_RGB_KERNEL_HPP