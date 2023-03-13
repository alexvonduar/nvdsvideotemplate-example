#ifndef _CUSTOMLIB_TRT_HPP_
#define _CUSTOMLIB_TRT_HPP_


#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
#include <cudaEGL.h>
#endif
#include <nppdefs.h>

#include <opencv2/opencv.hpp>

#include "NvInfer.h"

#include <nvbufsurface.h>
#include <nvdsmeta.h>

typedef struct _TRTBuffer
{
    nvinfer1::Dims dims;
    nvinfer1::DataType dataType;
    NvBufSurfaceMemType memType;
    int batchSize;
    void *buffer;
    size_t size;
} TRTBuffer;

typedef struct _TRTJob
{
    int networkInputBatchSize;
    cudaStream_t stream;
    std::shared_ptr<nvinfer1::IExecutionContext> trtcontext;
    NppStreamContext nppcontext;
    int inputWidth;
    int inputHeight;
    std::vector<TRTBuffer> inputbuffers;
    std::vector<TRTBuffer> outputbuffers;
    int numBindings;
    void **bindings;
    std::vector<std::vector<cv::Rect>> bboxes;
} TRTJob;

class TRTInfer
{
public:
    void initialize(const std::string &trtModelFile, const int &inputBatch, const std::vector<cv::Point2i> roi, const std::vector<std::string> &inputNames, const std::vector<std::string> &outputNames);

    void preprocessingImage(NvBufSurface *surf, const uint64_t& frame_count, const std::vector<int>& buffer_remap);


    void postprocessingImage(NvBufSurface *surf, const uint64_t& frame_count, const std::vector<int>& buffer_remap);

    void trtInference(NvBufSurface *input, NvBufSurface *output, const uint64_t& frame_count, const std::vector<int>& buffer_remap);

    ~TRTInfer();

    TRTInfer() : m_scratchSurface(nullptr)
#if defined(PLATFORM_TEGRA) and PLATFORM_TEGRA
    , m_eglFramePtr(nullptr), m_pResource(nullptr)
#endif
    , m_firstFrame(true)
#if !defined(NDEBUG) or NDEBUG == 0
    , dump_max_frames(1), dump_max_nchw(3)
#endif
    {
    }

    void fillBatchMetaData (NvDsBatchMeta *batch_meta, const std::unordered_map<int,int>& source_id);

private:
    void updateReference(const int& jobIndex, const int& batchIndex, const cv::Mat& mask);
    std::unique_ptr<nvinfer1::IRuntime> m_trtruntime;
    std::shared_ptr<nvinfer1::ICudaEngine> m_trtengine;
    std::string m_trtModelFile;
    std::vector<TRTJob> m_trtJobs;
    NvBufSurface *m_scratchSurface;
#if defined(PLATFORM_TEGRA) and PLATFORM_TEGRA
    CUeglFrame *m_eglFramePtr;
    CUgraphicsResource *m_pResource;
#endif
    std::vector<cv::Point2i> mRoi;
    cv::Rect mRoiRect;
    cv::Rect mSrcRect;
    double mScale;
    bool m_firstFrame;
#if !defined(NDEBUG) or NDEBUG == 0
    int dump_max_frames;
    int dump_max_nchw;
#endif
    std::vector<std::vector<cv::Rect>> m_bboxes;

    // Helper function to dump the nvbufsurface, used for debugging purpose
    void DumpNvBufSurface(NvBufSurface *in_surface, NvDsBatchMeta *batch_meta, const std::string& prefix = "");
    // Helper function to dump the nchw buffer, used for debugging purpose
    void DumpNCHW(TRTBuffer *input, const cudaStream_t& stream, const std::string& prefix = "");
};

#endif //_CUSTOMLIB_TRT_HPP_
