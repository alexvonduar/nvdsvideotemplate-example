#include <string>

#include <opencv2/core/cuda.hpp>

#include <nppcore.h>
#include <nppi_arithmetic_and_logical_operations.h>

#include <nvbufsurftransform.h>

#include "common.h"
#include "logger.h"

#include "nv12_to_rgb_kernel.hpp"
#include "customlib_trt.hpp"

static std::string dataTypeName(const nvinfer1::DataType &t)
{
    switch (t)
    {
    case nvinfer1::DataType::kFLOAT:
        return "FLOAT";
    case nvinfer1::DataType::kHALF:
        return "HALF";
    case nvinfer1::DataType::kINT8:
        return "INT8";
    case nvinfer1::DataType::kINT32:
        return "INT32";
    case nvinfer1::DataType::kBOOL:
        return "BOOL";
    }
    return "UNKNOWN";
}

static std::string dimsToString(const nvinfer1::Dims &dims)
{
    std::string ret = "(";
    for (int i = 0; i < dims.nbDims; i++)
    {
        ret += std::to_string(dims.d[i]);
        if (i != dims.nbDims - 1)
        {
            ret += ", ";
        }
    }
    ret += ")";
    return ret;
}

static inline bool _CHECK_CUDA(int e, int iLine, const char *szFile)
{
    if (e != cudaSuccess)
    {
        std::cout << "CUDA runtime error " << e << " at line " << iLine << " in file " << szFile;
        exit(-1);
        return false;
    }
    return true;
}
#define CHECK_CUDA(call) _CHECK_CUDA(call, __LINE__, __FILE__)

static inline cv::Rect findBestRoi(const cv::Rect &srcRoi, const int &srcWidth, const int &srcHeight, const int &dstWidth, const int &dstHeight, const bool &doScale = false)
{
    cv::Rect result;
    assert(srcRoi.width <= srcWidth);
    assert(srcRoi.height <= srcHeight);
    if (doScale)
    {
        if (srcRoi.width * dstHeight > srcRoi.height * dstWidth)
        {
            const int newHeight = srcRoi.width * dstHeight / dstWidth;
            const auto newTop = srcRoi.y + newHeight < srcHeight ? srcRoi.y : srcHeight - newHeight - 1;
            const auto newLeft = srcRoi.x + srcRoi.width < srcWidth ? srcRoi.x : srcWidth - srcRoi.width - 1;
            result = cv::Rect(newLeft, newTop, srcRoi.width, newHeight);
        }
        else
        {
            const auto newWidth = srcRoi.height * dstWidth / dstHeight;
            const auto newLeft = srcRoi.x + newWidth < srcWidth ? srcRoi.x : srcWidth - newWidth - 1;
            const auto newTop = srcRoi.y + srcRoi.height < srcHeight ? srcRoi.y : srcHeight - srcRoi.height - 1;
            result = cv::Rect(newLeft, newTop, newWidth, srcRoi.height);
        }
    }
    else
    {
        const auto newLeft = srcRoi.x + dstWidth < srcWidth ? srcRoi.x : srcWidth - dstWidth - 1;
        const auto newTop = srcRoi.y + dstHeight < srcHeight ? srcRoi.y : srcHeight - dstHeight - 1;
        result = cv::Rect(newLeft, newTop, dstWidth, dstHeight);
    }
    std::cout << "input roi: " << srcRoi << " adjusted roi " << result << std::endl;
    return result;
}

void TRTInfer::initialize(const std::string &trtModelFile, const int &inputBatch, const std::vector<cv::Point2i> roi, const std::vector<std::string> &inputNames, const std::vector<std::string> &outputNames)
{
#if !defined(NDEBUG)
    sample::gLogger.setReportableSeverity(nvinfer1::ILogger::Severity::kVERBOSE);
#endif
    sample::gLogInfo << "Initializing TensorRT" << std::endl;
    // cudaStreamCreate(&m_stream);
    // nppSetStream(m_stream);
    // nppGetStreamContext(&m_nppcontext);

    mRoi = roi;
    // mRoiRect = cv::boundingRect(roi);
    m_firstFrame = true;

    if (!m_trtruntime)
    {
        m_trtruntime = std::unique_ptr<nvinfer1::IRuntime>{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};
    }

    if (!m_trtruntime)
    {
        sample::gLogError << "Failed to create InferRuntime" << std::endl;
        return;
    }

    if (!m_trtengine)
    {
        std::vector<char> engineData;
        std::ifstream engineFile(trtModelFile, std::ios::in | std::ios::binary);
        if (engineFile.is_open())
        {
            engineFile.seekg(0, std::ios::end);
            const auto engineSize = engineFile.tellg();
            sample::gLogInfo << "Engine size: " << engineSize << std::endl;
            engineData.resize(engineSize);
            engineFile.seekg(0, std::ios::beg);
            engineFile.read(engineData.data(), engineSize);
        }
        else
        {
            // std::stringstream msg;
            sample::gLogError << "Cannot open file " << m_trtModelFile << "!";
            // throw std::invalid_argument(msg.str());
        }

        // std::unique_ptr<nvinfer1::ICudaEngine> mEngine(runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr));
        m_trtengine = std::shared_ptr<nvinfer1::ICudaEngine>(
            m_trtruntime->deserializeCudaEngine(engineData.data(), engineData.size()), samplesCommon::InferDeleter());
    }

    if (!m_trtengine)
    {
        sample::gLogError << "Failed to create ICudaEngine" << std::endl;
        return;
    }

    // if (!m_trtcontext)
    //{
    //     m_trtcontext = std::unique_ptr<nvinfer1::IExecutionContext>{(m_trtengine->createExecutionContext())};
    // }

    // if (!m_trtcontext)
    //{
    //     sample::gLogError << "Failed to create IExecutionContext" << std::endl;
    //     return;
    // }

    int networkInputBatchSize = 0;

    for (const auto &inputName : inputNames)
    {
        const auto inputIndex = m_trtengine->getBindingIndex(inputName.c_str());
        const auto dims = m_trtengine->getBindingDimensions(inputIndex);
        if (networkInputBatchSize == 0)
        {
            networkInputBatchSize = dims.nbDims < 3 ? 1 : dims.d[0];
        }
        else
        {
            assert(networkInputBatchSize == (dims.nbDims < 3 ? 1 : dims.d[0]));
        }
    }

    int numJobs = (inputBatch + networkInputBatchSize - 1) / networkInputBatchSize;

    sample::gLogInfo << "input buffer batch size: " << inputBatch << " network input batch size: " << networkInputBatchSize << " numJobs: " << numJobs << std::endl;
    for (int i = 0; i < numJobs; i++)
    {
        TRTJob job;
        memset(&job, 0, sizeof(TRTJob));
        job.networkInputBatchSize = networkInputBatchSize;
        job.bboxes.resize(networkInputBatchSize);
        cudaStreamCreate(&job.stream);
        job.trtcontext = std::shared_ptr<nvinfer1::IExecutionContext>{(m_trtengine->createExecutionContext())};

        if (job.trtcontext == nullptr)
        {
            sample::gLogError << "Failed to create IExecutionContext" << std::endl;
            return;
        }
        // job.nppcontext = m_nppcontext;
        // job.numBindings = m_trtengine->getNbBindings();
        // job.bindings = new void*[job.numBindings];
        // for (int j = 0; j < job.numBindings; j++) {
        //     job.bindings[j] = nullptr;
        // }
        for (const auto &inputName : inputNames)
        {
            TRTBuffer buffer;
            const auto inputIndex = m_trtengine->getBindingIndex(inputName.c_str());
            buffer.dataType = m_trtengine->getBindingDataType(inputIndex);
            buffer.dims = m_trtengine->getBindingDimensions(inputIndex);
            buffer.batchSize = buffer.dims.nbDims < 3 ? 1 : buffer.dims.d[0];
            if (job.inputWidth == 0 or job.inputHeight == 0)
            {
                if (buffer.dims.nbDims == 3)
                {
                    job.inputWidth = buffer.dims.d[2];
                    job.inputHeight = buffer.dims.d[1];
                }
                else
                {
                    job.inputWidth = buffer.dims.d[3];
                    job.inputHeight = buffer.dims.d[2];
                }
            }
            sample::gLogInfo << "Job: " << i << " input: " << inputIndex << " type: " << dataTypeName(buffer.dataType) << " dims: " << dimsToString(buffer.dims) << std::endl;
            buffer.size = samplesCommon::volume(buffer.dims) * samplesCommon::getElementSize(buffer.dataType);
            // sample::gLogInfo << "buffer size: " << buffer.size << std::endl;
            buffer.buffer = nullptr;
#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
            buffer.memType = NVBUF_MEM_CUDA_PINNED;
            cudaMallocManaged(&buffer.buffer, buffer.size, cudaMemAttachGlobal);
#else
            buffer.memType = NVBUF_MEM_CUDA_DEVICE;
            cudaMallocManaged(&buffer.buffer, buffer.size);
#endif
            if (buffer.buffer == nullptr)
            {
                sample::gLogError << "Failed to allocate input buffer" << std::endl;
                // return;
            }
            sample::gLogInfo << "Job: " << i << " input: " << inputIndex << " buffer: " << buffer.buffer << " size " << buffer.size << std::endl;
            job.inputbuffers.emplace_back(buffer);
        }

        for (const auto &outputName : outputNames)
        {
            TRTBuffer buffer;
            const auto outputIndex = m_trtengine->getBindingIndex(outputName.c_str());
            buffer.dataType = m_trtengine->getBindingDataType(outputIndex);
            buffer.dims = m_trtengine->getBindingDimensions(outputIndex);
            buffer.batchSize = buffer.dims.nbDims < 3 ? 1 : buffer.dims.d[0];
            // assert(buffer.batchSize == batchSize);
            sample::gLogInfo << "Job: " << i << " output: " << outputIndex << " type: " << dataTypeName(buffer.dataType) << " dims: " << dimsToString(buffer.dims) << std::endl;
            buffer.size = samplesCommon::volume(buffer.dims) * samplesCommon::getElementSize(buffer.dataType);
            // sample::gLogInfo << "buffer size: " << buffer.size << std::endl;
            buffer.buffer = nullptr;
#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
            buffer.memType = NVBUF_MEM_CUDA_PINNED;
            //cudaMalloc(&buffer.buffer, buffer.size);
            cudaMallocManaged(&buffer.buffer, buffer.size, cudaMemAttachGlobal);
#else
            buffer.memType = NVBUF_MEM_CUDA_UNIFIED;
            cudaMallocManaged(&buffer.buffer, buffer.size);
#endif
            // std::cout <<"m_trtOutputBuffer: " << m_trtOutputBuffer << std::endl;
            if (buffer.buffer == nullptr)
            {
                sample::gLogError << "Failed to allocate output buffer" << std::endl;
                // return;
            }
            sample::gLogInfo << "Job: " << i << " output: " << outputIndex << " buffer: " << buffer.buffer << " size " << buffer.size << std::endl;
            job.outputbuffers.emplace_back(buffer);
        }

        if (job.bindings != nullptr)
        {
            delete[] job.bindings;
        }
        job.numBindings = job.inputbuffers.size() + job.outputbuffers.size();
        job.bindings = new void *[job.numBindings];
        memset(job.bindings, 0, sizeof(void *) * job.numBindings);
        for (size_t j = 0; j < job.inputbuffers.size(); j++)
        {
            job.bindings[j] = job.inputbuffers[j].buffer;
        }
        for (size_t j = 0; j < job.outputbuffers.size(); j++)
        {
            job.bindings[j + job.inputbuffers.size()] = job.outputbuffers[j].buffer;
        }
        m_trtJobs.emplace_back(job);
    }

    sample::gLogInfo << "try create scratch surface" << std::endl;
    if (m_scratchSurface != nullptr)
    {
        NvBufSurfaceDestroy(m_scratchSurface);
    }
    // else
    //{
    //     m_scratchSurface = new NvBufSurface();
    // }

    // FIXME: create temp surface
    NvBufSurfaceCreateParams surfaceCreateParams;
    surfaceCreateParams.layout = NVBUF_LAYOUT_PITCH;
    surfaceCreateParams.colorFormat = NVBUF_COLOR_FORMAT_NV12;
    assert(m_trtJobs[0].inputbuffers[0].dims.nbDims == 3 || m_trtJobs[0].inputbuffers[0].dims.nbDims == 4);
    surfaceCreateParams.width = m_trtJobs[0].inputWidth;
    surfaceCreateParams.height = m_trtJobs[0].inputHeight;
    surfaceCreateParams.isContiguous = true;
    surfaceCreateParams.size = 0;
    surfaceCreateParams.gpuId = 0;
#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
    surfaceCreateParams.memType = NVBUF_MEM_SURFACE_ARRAY;
#else
    surfaceCreateParams.memType = NVBUF_MEM_CUDA_DEVICE;
#endif

    if (NvBufSurfaceCreate(&m_scratchSurface, inputBatch, &surfaceCreateParams) != 0)
    {
        sample::gLogError << "create surface failed!" << std::endl;
    }
    NvBufSurfaceMemSet(m_scratchSurface, 0, 1, 128);
#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
    if (NvBufSurfaceMapEglImage(m_scratchSurface, -1) != 0)
    {
        sample::gLogError << "Unable to map EGL Image" << std::endl;
        return;
    }
    m_pResource = new CUgraphicsResource[m_scratchSurface->batchSize];
    m_eglFramePtr = new CUeglFrame[m_scratchSurface->batchSize];
    CHECK_CUDA(cudaFree(0));
    for (int i = 0; i < m_scratchSurface->batchSize; ++i) {
        auto status = cuGraphicsEGLRegisterImage(&m_pResource[i], m_scratchSurface->surfaceList[i].mappedAddr.eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
        if (status != CUDA_SUCCESS)
        {
            sample::gLogError << "cuGraphicsEGLRegisterImage failed: " << status << ", cuda process stop" << std::endl;
            return;
        }

        status = cuGraphicsResourceGetMappedEglFrame(&m_eglFramePtr[i], m_pResource[i], 0, 0);
        if (status != CUDA_SUCCESS)
        {
            sample::gLogError << "cuGraphicsSubResourceGetMappedArray failed" << std::endl;
        }
    }
#endif
}

void TRTInfer::preprocessingImage(NvBufSurface *surf, const uint64_t& frame_count, const std::vector<int>& buffer_remap)
{
    //assert(surf->numFilled == 2);
    //  FIXME: check gpuid to support multi-gpu
    // assert(surf->memType == NVBUF_MEM_CUDA_UNIFIED);
    //assert(surf->surfaceList[0].layout == NVBUF_LAYOUT_PITCH);
    assert(m_scratchSurface->surfaceList[0].layout == NVBUF_LAYOUT_PITCH);
    //auto unifiedMem = surf->memType == NVBUF_MEM_CUDA_UNIFIED;
    sample::gLogInfo << "preprocess: input image memory type " << surf->memType << std::endl;
    switch(surf->memType) {
#if !defined(PLATFORM_TEGRA) or PLATFORM_TEGRA == 0
    case NVBUF_MEM_DEFAULT:
#endif
    case NVBUF_MEM_CUDA_PINNED:
    case NVBUF_MEM_CUDA_DEVICE:
    case NVBUF_MEM_CUDA_UNIFIED:
    {
#if 0
        int status = NvBufSurfaceMap(surf, -1, -1, NVBUF_MAP_READ);
        if (status != 0)
        {
            sample::gLogError << "Failed to map surface" << std::endl;
            return;
        }
#endif
        break;
    }
#if defined(PLATFORM_TEGRA) and PLATFORM_TEGRA
    case NVBUF_MEM_DEFAULT:
#endif
    case NVBUF_MEM_SURFACE_ARRAY:
    {
        break;
    }
    case NVBUF_MEM_HANDLE:
    {
        break;
    }
    case NVBUF_MEM_SYSTEM:
    {
        break;
    }
    default:
    {
        sample::gLogError << "Unsupported memory type" << std::endl;
        return;
    }
    };

    // crop and scale input surface
    const auto inputW = surf->surfaceList[0].width;
    const auto inputH = surf->surfaceList[0].height;
    if (mRoiRect.empty())
    {
        // sample::gLogInfo << "input roi: " << mRoi << std::endl;
        std::vector<cv::Point2i> inputRect{{0, 0}, {inputW, 0}, {inputW, inputH}, {0, inputH}};
        std::vector<cv::Point2f> intersect;
        cv::intersectConvexConvex(inputRect, mRoi, intersect);
        mRoiRect = cv::boundingRect(intersect);
        sample::gLogInfo << "input roi: " << mRoi << " intersect roi: " << mRoiRect << std::endl;
    }

    // do actual transform
    // if (mRoiRect.width != inputW or mRoiRect.height != inputH) {
    sample::gLogInfo << "do scale and crop from input source to scratch buffer" << std::endl;
    mSrcRect = findBestRoi(mRoiRect, inputW, inputH, m_scratchSurface->surfaceList[0].width, m_scratchSurface->surfaceList[0].height, true);
    mScale = float(m_scratchSurface->surfaceList[0].width) / mSrcRect.width;
    sample::gLogInfo << "best src rect: " << mSrcRect  << " scale factor " << mScale << std::endl;
    NvBufSurfTransformParams transform_params;
    memset(&transform_params, 0, sizeof(NvBufSurfTransformParams));
    transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC;
    transform_params.transform_flip = NvBufSurfTransform_None;
#if defined(PLATFORM_TEGRA) and PLATFORM_TEGRA
    transform_params.transform_filter = NvBufSurfTransformInter_Default;
#else
    transform_params.transform_filter = NvBufSurfTransformInter_Algo4;
#endif
    NvBufSurfTransformRect src_rect;
    src_rect.left = mSrcRect.x, src_rect.top = mSrcRect.y, src_rect.width = mSrcRect.width, src_rect.height = mSrcRect.height;
    std::vector<NvBufSurfTransformRect> src_rects(surf->numFilled, src_rect);
    NvBufSurfTransformRect dst_rect;
    dst_rect.left = 0, dst_rect.top = 0, dst_rect.width = m_scratchSurface->surfaceList[0].width, dst_rect.height = m_scratchSurface->surfaceList[0].height;
    transform_params.src_rect = &(src_rects[0]);
    // transform_params.dst_rect = &dst_rect;
    const auto tstatus = NvBufSurfTransform(surf, m_scratchSurface, &transform_params);
    if (tstatus != NvBufSurfTransformError_Success) {
        sample::gLogError << "failed transform: " << tstatus << std::endl;
        return;
    }
    m_scratchSurface->numFilled = surf->numFilled;
    DumpNvBufSurface(m_scratchSurface, nullptr, "NVTransform_frame_" + std::to_string(frame_count) + "_");

    //}

    // assert(surf->batchSize == m_trtInputBuffers[0].batchSize);

    // iterate at batch size
    const auto numJobs = m_trtJobs.size();
    const auto networkInputBatchSize = m_trtJobs[0].networkInputBatchSize;
    NppStatus npp_status;
    for (int i = 0; i < m_scratchSurface->numFilled; i++)
    {
        const auto id = buffer_remap[i];
        const auto jobIndex = id / networkInputBatchSize;
        const auto batchIndex = id % networkInputBatchSize;
        const auto networkInputBatchSize = m_trtJobs[jobIndex].networkInputBatchSize;
        const auto inputSizePerBatch = m_trtJobs[jobIndex].inputbuffers[0].size / networkInputBatchSize;
        const auto offset = batchIndex * inputSizePerBatch;
        auto inputBuffer = reinterpret_cast<unsigned char *>(m_trtJobs[jobIndex].inputbuffers[0].buffer) + offset;
        sample::gLogInfo << "preprocessing: " << (void *)inputBuffer << std::endl;
        const auto w = m_scratchSurface->surfaceList[i].width;
        const auto h = m_scratchSurface->surfaceList[i].height;
        const auto p = m_scratchSurface->surfaceList[i].pitch;
        assert(m_scratchSurface->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_NV12);
        assert(m_scratchSurface->surfaceList[i].layout == NVBUF_LAYOUT_PITCH);
        assert(m_scratchSurface->surfaceList[i].planeParams.num_planes == 2);
        const auto plane_w = m_scratchSurface->surfaceList[i].planeParams.width[0];
        const auto plane_h = m_scratchSurface->surfaceList[i].planeParams.height[0];
        const auto plane_p = m_scratchSurface->surfaceList[i].planeParams.pitch[0];
        assert(w == plane_w);
        assert(h == plane_h);
        assert(p == plane_p);
        // assert(w == p);
#if defined(PLATFORM_TEGRA) and PLATFORM_TEGRA
        const auto src_y = m_eglFramePtr[i].frame.pPitch[0];
        const auto src_uv = m_eglFramePtr[i].frame.pPitch[1];
#else
        const auto src_y = ((unsigned char *)m_scratchSurface->surfaceList[i].dataPtr) + m_scratchSurface->surfaceList[i].planeParams.offset[0];
        const auto src_uv = ((unsigned char *)m_scratchSurface->surfaceList[i].dataPtr) + m_scratchSurface->surfaceList[i].planeParams.offset[1];
#endif
        auto dst_y = reinterpret_cast<Npp32f *>(inputBuffer);

        // normalize parameters
        // mean [0.485*255,0.456*255,0.406*255]
        // sigma [0.229*255,0.224*255,0.225*255]
        batched_convert_nv12_to_rgb(
            src_y,
            src_uv,
            w,
            p,
            h,
            1,
            NV12Format::PitchLinear,
            dst_y,
            w,
            w,
            h,
            DataType::Float32,
            PixelLayout::NCHW_BGR,
            Interpolation::Bilinear,
            //0.485f * 255, 0.456 * 255, 0.406 * 255, 255.f / 0.229, 255.f / 0.224, 255.f / 0.225,
            128,128,128,1.0/128,1.0/128,1.0/128,
            m_trtJobs[jobIndex].stream);
        if (m_firstFrame) {
        }
    }
    if (m_firstFrame) {
        for (auto &j : m_trtJobs) {
            assert(j.inputbuffers[1].size == j.inputbuffers[0].size);
            cudaMemcpyAsync(j.inputbuffers[1].buffer, j.inputbuffers[0].buffer, j.inputbuffers[0].size, cudaMemcpyDeviceToDevice, j.stream);
        }
        m_firstFrame = false;
    }
    DumpNCHW(&(m_trtJobs[0].inputbuffers[0]), m_trtJobs[0].stream, "input_" + std::to_string(frame_count) + "_job[0]_0_");
    DumpNCHW(&(m_trtJobs[0].inputbuffers[1]), m_trtJobs[0].stream, "input_" + std::to_string(frame_count) + "_job[0]_1_");

    switch (surf->memType)
    {
#if !defined(PLATFORM_TEGRA) or PLATFORM_TEGRA == 0
    case NVBUF_MEM_DEFAULT:
#endif
    case NVBUF_MEM_CUDA_PINNED:
    case NVBUF_MEM_CUDA_DEVICE:
    case NVBUF_MEM_CUDA_UNIFIED:
    {
#if 0
        int status = NvBufSurfaceUnMap(surf, -1, -1);
        if (status != 0)
        {
            sample::gLogError << "Failed to map surface" << std::endl;
            return;
        }
#endif
        break;
    }
#if defined(PLATFORM_TEGRA) and PLATFORM_TEGRA
    case NVBUF_MEM_DEFAULT:
#endif
    case NVBUF_MEM_SURFACE_ARRAY:
    {
        break;
    }
    case NVBUF_MEM_HANDLE:
    {
        break;
    }
    case NVBUF_MEM_SYSTEM:
    {
        break;
    }
    default:
    {
        sample::gLogError << "Unsupported memory type" << std::endl;
        return;
    }
    };
}

static inline int trtDType2cvDType(const nvinfer1::DataType& type, const int& channels)
{
    switch (type)
    {
    case nvinfer1::DataType::kFLOAT:
        return CV_MAKE_TYPE(CV_32F, channels);
    case nvinfer1::DataType::kHALF:
        return CV_MAKE_TYPE(CV_16S, channels);
    case nvinfer1::DataType::kINT8:
        return CV_MAKE_TYPE(CV_8S, channels);
    case nvinfer1::DataType::kINT32:
        return CV_MAKE_TYPE(CV_32S, channels);
    case nvinfer1::DataType::kBOOL:
        return CV_MAKE_TYPE(CV_8U, channels);
#if ((NV_TENSORRT_MAJOR * 1000) + (NV_TENSORRT_MINOR * 100) + (NV_TENSORRT_PATCH * 10)) >= 8510
    case nvinfer1::DataType::kUINT8:
        return CV_MAKE_TYPE(CV_8U, channels);
#endif
    default:
        return 0;
    }
}

void TRTInfer::updateReference(const int& jobIndex, const int& batchIndex, const cv::Mat& mask)
{
    cv::Mat mask8u, invMask8u;
    if (mask.type() == CV_8UC1) {
        mask8u = mask;
    } else {
        mask.convertTo(mask8u, CV_8UC1);
    }
    invMask8u = cv::Mat(mask8u.size(), mask8u.type(), 255);
    invMask8u = invMask8u - mask8u;
    cv::cuda::GpuMat maskGpu(mask8u);
    cv::cuda::GpuMat invMaskGpu(invMask8u);

    //const auto numJobs = m_trtJobs.size();
    //const auto networkInputBatchSize = m_trtJobs[0].networkInputBatchSize;
    //const auto jobIndex = frameIndex / networkInputBatchSize;
    //const auto batchIndex = frameIndex % networkInputBatchSize;
    auto& j = m_trtJobs[jobIndex];
    //for (const auto& j: m_trtJobs) {
        if (j.nppcontext.hStream == nullptr) {
            nppSetStream(j.stream);
            //NppStreamContext nppStreamCtx;
            nppGetStreamContext(&(j.nppcontext));
        }
        const auto batchSize = j.networkInputBatchSize;
        const auto outputSizePerBatch = j.outputbuffers[0].size / batchSize;
        const auto dims = j.inputbuffers[0].dims;
        assert(dims.nbDims == 4);
        const auto N = dims.d[0];
        assert(N == batchSize);
        const auto C = dims.d[1];
        const auto H = dims.d[2];
        const auto W = dims.d[3];
        const auto dtype = j.inputbuffers[0].dataType;
        assert(H == mask8u.rows);
        assert(W == mask8u.cols);
        const auto bufferSizePerChannelInBytes = H * W * samplesCommon::getElementSize(dtype);
        auto referenceBuffer = reinterpret_cast<unsigned char *>(j.inputbuffers[1].buffer);
        auto currentBuffer = reinterpret_cast<unsigned char *>(j.inputbuffers[0].buffer);
        NppiSize roi{W, H};
        const auto n = batchIndex;
        //for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                sample::gLogInfo << "update reference " << n << " " << c << " " << H << " " << W << std::endl;
                const auto reference = reinterpret_cast<float *>(referenceBuffer + (n * C + c) * bufferSizePerChannelInBytes);
                const auto current = reinterpret_cast<float *>(currentBuffer + (n * C + c) * bufferSizePerChannelInBytes);
                NppStatus npp_status = nppiAddWeighted_32f_C1IMR_Ctx(current, W * sizeof(float), maskGpu.data, W, reference, W * sizeof(float), roi, 0.3, j.nppcontext);
                if (npp_status != NPP_NO_ERROR)
                {
                    sample::gLogError << "update reference change part npp error " << npp_status << std::endl;
                }
                npp_status = nppiAddWeighted_32f_C1IMR_Ctx(current, W * sizeof(float), invMaskGpu.data, W, reference, W * sizeof(float), roi, 0.5, j.nppcontext);
                if (npp_status != NPP_NO_ERROR) {
                    sample::gLogError << "update reference steady pard npp error " << npp_status << std::endl;
                }
            }
        //}
    //}
}

void TRTInfer::postprocessingImage(NvBufSurface *surf, const uint64_t& frame_count, const std::vector<int>& buffer_remap)
{
    const auto numJobs = m_trtJobs.size();
    const auto networkInputBatchSize = m_trtJobs[0].networkInputBatchSize;
    // const auto batchSize = surf->batchSize;
    // const auto outputSizePerBatch = m_trtOutputBuffers[0].size / batchSize;
    // auto outputBuffer = reinterpret_cast<unsigned char *>(m_trtOutputBuffers[0].buffer);
    NppStatus npp_status = NPP_NO_ERROR;
    m_bboxes.clear();
    for (int id = 0; id < surf->numFilled; id++)
    {
        const auto i = buffer_remap[id];
        const auto jobIndex = i / networkInputBatchSize;
        const auto batchIndex = i % networkInputBatchSize;
        const auto networkInputBatchSize = m_trtJobs[jobIndex].networkInputBatchSize;
        const auto outputSizePerBatch = m_trtJobs[jobIndex].outputbuffers[0].size / networkInputBatchSize;
        DumpNCHW(&(m_trtJobs[jobIndex].outputbuffers[0]), m_trtJobs[jobIndex].stream, "output_"+ std::to_string(frame_count) + "_job_" + std::to_string(jobIndex) + "_0_");
        auto outputBuffer = reinterpret_cast<unsigned char *>(m_trtJobs[jobIndex].outputbuffers[0].buffer) + batchIndex * outputSizePerBatch;
        sample::gLogInfo << "output job " << jobIndex << " batch " \
            << batchIndex << " buffer " << reinterpret_cast<void *>(outputBuffer) \
            << " size " << outputSizePerBatch << std::endl;
        const auto dims = m_trtJobs[jobIndex].outputbuffers[0].dims;
        const auto dataType = m_trtJobs[jobIndex].outputbuffers[0].dataType;
        assert(dims.nbDims == 4 /*and dims.d[0] == surf->batchSize*/ and dims.d[1] == 2 and dims.d[2] == 800 and dims.d[3] == 1408);
        const auto bufferSize = outputSizePerBatch / dims.d[1];
        const auto output0 = outputBuffer;
        const auto output1 = outputBuffer + bufferSize;
        if (m_trtJobs[jobIndex].nppcontext.hStream == nullptr) {
            nppSetStream(m_trtJobs[jobIndex].stream);
            nppGetStreamContext(&m_trtJobs[jobIndex].nppcontext);
        }
        NppiSize npp_size;
        npp_size.width = dims.d[3];
        npp_size.height = dims.d[2];
        const auto step = npp_size.width * sizeof(float);
        //NppStatus npp_status = nppiSub_32f_C1R_Ctx((Npp32f *)output0, step, (Npp32f *)output1, step, (Npp32f *)m_postprocessScratch, step, npp_size, m_trtJobs[jobIndex].nppcontext);
        NppStatus npp_status = nppiSub_32f_C1IR_Ctx((Npp32f *)output0, step, (Npp32f *)output1, step, npp_size, m_trtJobs[jobIndex].nppcontext);
        if (npp_status != NPP_NO_ERROR) {
            sample::gLogError << "npp output1 - output0 error: " << npp_status << std::endl;
        }
        cv::cuda::GpuMat gpuMask(dims.d[2], dims.d[3], trtDType2cvDType(dataType, 1), output1);
        cv::Mat cpuMask(dims.d[2], dims.d[3], trtDType2cvDType(dataType, 1));
        CHECK_CUDA(cudaStreamSynchronize(m_trtJobs[jobIndex].stream));
        gpuMask.download(cpuMask);
        //CHECK_CUDA(cudaMemcpyAsync(cpuMask.data, output1, bufferSize, cudaMemcpyDeviceToHost, m_trtJobs[jobIndex].stream));
        //cv::Mat cpuMask(dims.d[2], dims.d[3], CV_8UC1);
        cv::threshold(cpuMask, cpuMask, 0, 255, cv::THRESH_BINARY);
        cpuMask.convertTo(cpuMask, CV_8UC1);
        cv::dilate(cpuMask, cpuMask, cv::Mat());
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(cpuMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        //m_trtJobs[jobIndex].bboxes[batchIndex].clear();
        std::vector<cv::Rect> bboxes;
        if (contours.size()) {
            std::vector<cv::Point> hull;
            //std::vector<std::vector<cv::Point> >hulls;
            //hulls.reserve(contours.size());
            for (const auto& c: contours) {
                //std::vector<cv::Point> hull;
                cv::convexHull(c, hull);
                //hulls.emplace_back(hull);
                auto rect = cv::boundingRect(hull);
                rect.x = (rect.x / mScale) + mSrcRect.x;
                rect.y = (rect.y / mScale) + mSrcRect.y;
                rect.width /= mScale;
                rect.height /= mScale;
                //m_trtJobs[jobIndex].bboxes[batchIndex].emplace_back(rect);
                bboxes.emplace_back(rect);
            }
        }
        m_bboxes.emplace_back(bboxes);
        updateReference(jobIndex, batchIndex, cpuMask);
    }
}

void TRTInfer::trtInference(NvBufSurface *input, NvBufSurface *output, const uint64_t& frame_count, const std::vector<int>& buffer_remap)
{
    preprocessingImage(input, frame_count, buffer_remap);
    // void *bindings[] = {m_trtInputBuffer, m_trtOutputBuffer};
    //  auto context = m_trtengine->getExecutionContext();
    //  cudaStream_t stream;
    //  cudaStreamCreate(&stream);
    for (int i = 0; i < m_trtJobs.size(); ++i)
    {
        sample::gLogInfo << "infer job: " << i << std::endl;
        bool status = m_trtJobs[i].trtcontext->enqueueV2(m_trtJobs[i].bindings, m_trtJobs[i].stream, nullptr);
        // cudaStreamSynchronize(m_stream);
        //  cudaStreamDestroy(stream);
        if (!status)
        {
            sample::gLogError << "Failed to do " << i << " infer job" << std::endl;
        }
    }
    postprocessingImage(output, frame_count, buffer_remap);

    for (const auto &j : m_trtJobs)
    {
        CHECK_CUDA(cudaStreamSynchronize(j.stream));
    }
}

TRTInfer::~TRTInfer()
{
    for (auto &j : m_trtJobs)
    {
        if (j.bindings)
        {
            delete[] j.bindings;
            j.bindings = nullptr;
        }
        // delete j.trtcontext;
        cudaStreamDestroy(j.stream);
        j.stream = nullptr;
        for (auto &b : j.inputbuffers)
        {
            cudaFree(b.buffer);
            b.buffer = nullptr;
        }
        for (auto &b : j.outputbuffers)
        {
            cudaFree(b.buffer);
            b.buffer = nullptr;
        }
    }

    if (m_scratchSurface)
    {
#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
        /* Destroy EGLImage */
        if (NvBufSurfaceUnMapEglImage(m_scratchSurface, -1) != 0)
        {
            sample::gLogError << "Unable to unmap EGL Image" << std::endl;
        }
        for (int i = 0; i < m_scratchSurface->batchSize; ++i) {

            //m_eglFramePtr[i] = nullptr;

            auto status = cuGraphicsUnregisterResource(m_pResource[i]);
            if (status != CUDA_SUCCESS)
            {
                sample::gLogError << "cuGraphicsEGLUnRegisterResource failed: " << status << std::endl;
            }
        }
        delete []m_eglFramePtr;
        delete []m_pResource;
#endif
        NvBufSurfaceDestroy(m_scratchSurface);
        // delete m_scratchSurface;
        m_scratchSurface = nullptr;
    }
}

void TRTInfer::fillBatchMetaData (NvDsBatchMeta *batch_meta, const std::unordered_map<int,int>& source_id_to_stream_id)
{
    if (batch_meta == nullptr)
    {
        return;
    }
    const auto numFilled = source_id_to_stream_id.size();
    //assert(numFilled == batch_meta->frame_meta_list->num_elements);
    NvDsMetaList * l_frame = NULL;
    //for (int i = 0; i < numFilled; i ++)
    int frame_index = 0;
    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        //const auto id = source_id[i];
        //NvDsFrameMetaList * l_frame = nullptr;
        //for (l_frame = batch_meta->frame_meta_list; l_frame != nullptr; l_frame = l_frame->next)
        //{
        //    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        //    if (frame_meta->source_id == id)
        //    {
        //        break;
        //    }
        //}
        //if (l_frame == nullptr)
        //{
        //    sample::gLogError << "source id: " << id << " not found in meta data" << std::endl;
        //    continue;
        //}
        //NvDsFrameMeta *frame_meta = nvds_get_nth_frame_meta(batch_meta->frame_meta_list, i);
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        const auto i = source_id_to_stream_id.find(frame_meta->source_id)->second;
        sample::gLogInfo << "frame index: " << i << " pad idx: " << frame_meta->pad_index << " source id: " << frame_meta->source_id << std::endl;
        /*printf ("pad_index = %d frame_width = %d frame_height = %d\n",
                frame_meta->pad_index, frame_meta->source_frame_width, frame_meta->source_frame_height);*/
        const auto inputBatchSize = m_trtJobs[0].networkInputBatchSize;
        const auto jobIndex = i / inputBatchSize;
        const auto batchIndex = i % inputBatchSize;
        //for (const auto& b : m_trtJobs[jobIndex].bboxes[batchIndex])
        for (const auto& b : m_bboxes[frame_index])
        {
            NvDsObjectMeta *obj_meta = nvds_acquire_obj_meta_from_pool (batch_meta);
            obj_meta->unique_component_id = 0xAB;
            obj_meta->confidence = 0.0;

            /* This is an untracked object. Set tracking_id to -1. */
            obj_meta->object_id = UNTRACKED_OBJECT_ID;
            obj_meta->class_id = 2;

            NvOSD_RectParams & rect_params = obj_meta->rect_params;
            /* Assign bounding box coordinates. */
            rect_params.left = b.x;
            rect_params.top = b.y;
            rect_params.width = b.width;
            rect_params.height = b.height;
            rect_params.border_color.red = 1.0;
            rect_params.border_color.green = 0.5;
            rect_params.border_color.blue = 0.0;
            rect_params.border_width = 2;

            nvds_add_obj_meta_to_frame (frame_meta, obj_meta, NULL);
        }
        frame_index++;
    }
}

void TRTInfer::DumpNvBufSurface(NvBufSurface *in_surface, NvDsBatchMeta *batch_meta, const std::string& prefix)
{
#if !defined(NDEBUG) or NDEBUG == 0
    void *tmpBuffer = nullptr;
    const auto numFilled = in_surface->numFilled;
    int source_id = 0;
    int i = 0;
    std::ofstream outfile;
    void *input_data = NULL;
    int size = 0;

    if (dump_max_frames)
    {
        dump_max_frames--;
    }
    else
    {
        return;
    }

    if (in_surface->memType == NVBUF_MEM_SURFACE_ARRAY) {
        const auto status = NvBufSurfaceMap(in_surface, -1, -1, NVBUF_MAP_READ);
        if (status != 0)
        {
            sample::gLogError << "Failed to map surface" << std::endl;
            return;
        }
        NvBufSurfaceSyncForCpu(in_surface, -1, -1);
    }

    for (i = 0; i < numFilled; i++)
    {
        if (batch_meta)
        {
            NvDsFrameMeta *frame_meta = nvds_get_nth_frame_meta(batch_meta->frame_meta_list, i);
            source_id = frame_meta->pad_index;
        }

        std::string tmp = prefix.length() ? prefix : "NvSurface_";
        tmp += std::to_string(in_surface->surfaceList[i].pitch) + "x" + std::to_string(in_surface->surfaceList[i].height) + "_" +
                            "BS-" + std::to_string(source_id);

        input_data = in_surface->surfaceList[i].dataPtr;

        // input_size = in_surface->surfaceList[i].dataSize;

        switch (in_surface->surfaceList[i].colorFormat)
        {
        case NVBUF_COLOR_FORMAT_NV12:
        case NVBUF_COLOR_FORMAT_NV12_2020:
        case NVBUF_COLOR_FORMAT_NV12_709:
        case NVBUF_COLOR_FORMAT_NV12_709_ER:
        case NVBUF_COLOR_FORMAT_NV12_ER:
        {
            std::string fname = tmp + ".nv12";

            size = (in_surface->surfaceList[i].pitch * in_surface->surfaceList[i].height * 3) / 2;
            assert(in_surface->surfaceList[i].layout == NVBUF_LAYOUT_PITCH);

            if (tmpBuffer == nullptr)
            {
#if defined(PLATFORM_TEGRA) and PLATFORM_TEGRA
                tmpBuffer = malloc(size);
#else
                CHECK_CUDA(cudaMallocHost(&tmpBuffer, size));
#endif
                outfile.open(fname, std::ofstream::out);
            }
            else
            {
                outfile.open(fname, std::ofstream::out | std::ofstream::app);
            }

            if (in_surface->memType == NVBUF_MEM_SURFACE_ARRAY) {
                const auto y = (char *)(in_surface->surfaceList[i].mappedAddr.addr[0]);
                const auto uv = (char *)(in_surface->surfaceList[i].mappedAddr.addr[1]);
                outfile.write(y, in_surface->surfaceList[i].pitch * in_surface->surfaceList[i].height);
                outfile.write(uv, in_surface->surfaceList[i].pitch * in_surface->surfaceList[i].height / 2);
            } else {
                cudaMemcpy2D(tmpBuffer,
                                in_surface->surfaceList[i].pitch, input_data, in_surface->surfaceList[i].pitch,
                                in_surface->surfaceList[i].pitch,
                                (in_surface->surfaceList[i].height * 3) / 2, cudaMemcpyDeviceToHost);
                outfile.write(reinterpret_cast<char *>(tmpBuffer), size);
            }

            outfile.close();
        }
        break;

        case NVBUF_COLOR_FORMAT_RGBA:
        case NVBUF_COLOR_FORMAT_BGRx:
        {
            std::string fname = tmp;

            if (in_surface->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_RGBA)
                fname.append(".rgba");
            else if (in_surface->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_BGRx)
                fname.append(".bgrx");

            size = in_surface->surfaceList[i].width * in_surface->surfaceList[i].height * 4;

            if (tmpBuffer == NULL)
            {
                CHECK_CUDA(cudaMallocHost(&tmpBuffer, size));
                outfile.open(fname, std::ofstream::out);
            }
            else
            {
                outfile.open(fname, std::ofstream::out | std::ofstream::app);
            }

            cudaMemcpy2D(tmpBuffer,
                            in_surface->surfaceList[i].width * 4, input_data,
                            in_surface->surfaceList[i].width * 4, in_surface->surfaceList[i].width * 4,
                            in_surface->surfaceList[i].height, cudaMemcpyDeviceToHost);

            outfile.write(reinterpret_cast<char *>(tmpBuffer), size);
            outfile.close();
        }
        break;

        default:
            sample::gLogError << "dump surface : NOT SUPPORTED FORMAT " << in_surface->surfaceList[i].colorFormat << std::endl;
            break;
        }
    }
    if (in_surface->memType == NVBUF_MEM_SURFACE_ARRAY) {
        const int status = NvBufSurfaceUnMap(in_surface, -1, -1);
        if (status != 0)
        {
            sample::gLogError << "Failed to map surface" << std::endl;
            return;
        }
    }

    if (tmpBuffer != nullptr)
    {
#if defined(PLATFORM_TEGRA) and PLATFORM_TEGRA
        free(tmpBuffer);
        tmpBuffer = nullptr;
#else
        CHECK_CUDA(cudaFreeHost(tmpBuffer));
#endif
    }
#endif
}

void TRTInfer::DumpNCHW(TRTBuffer *input, const cudaStream_t& stream, const std::string& prefix)
{
#if !defined(NDEBUG) or NDEBUG == 0
    std::ofstream outfile;
    void *input_data = nullptr;
    int size = 0;

    if (dump_max_nchw)
    {
        dump_max_nchw--;
    }
    else
    {
        return;
    }

    std::string tmp = prefix.length() ? prefix : "NCHW_";
    tmp += std::to_string(input->dims.d[0]);
    for (int j = 1; j < input->dims.nbDims; ++j)
    {
        tmp += "x" + std::to_string(input->dims.d[j]);
    }

    // tmp += "_";
    input_data = input->buffer;

    int cv_input_type;
    switch (input->dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        tmp += "_float";
        cv_input_type = CV_32FC1;
        size = input->size / sizeof(float);
        break;
    case nvinfer1::DataType::kHALF:
        tmp += "_half";
        cv_input_type = CV_16FC1;
        size = input->size / sizeof(short);
        break;
    case nvinfer1::DataType::kINT8:
        tmp += "_int8";
        cv_input_type = CV_8SC1;
        size = input->size;
        break;
    default:
        tmp += "_unknown";
        break;
    }
    std::string fname = tmp + ".nchw";
    cv::Mat normMat;
#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
    CHECK_CUDA(cudaStreamAttachMemAsync(stream, input_data, 0, cudaMemAttachHost));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    cv::Mat gpuMat{1, size, cv_input_type, input_data};
    cv::normalize(gpuMat, normMat, 0, 255, cv::NORM_MINMAX, cv_input_type);
    CHECK_CUDA(cudaStreamAttachMemAsync(stream, input_data, 0, cudaMemAttachGlobal));
#else
    cv::cuda::GpuMat gpuMat{1, size, cv_input_type, input_data};
    //void * tmpBuff = malloc(input->size);
    //CHECK_CUDA(cudaMemcpy(tmpBuff, input_data, input->size, cudaMemcpyDeviceToHost));
    normMat = cv::Mat(1, size, cv_input_type);
    CHECK_CUDA(cudaStreamSynchronize(stream));
    gpuMat.download(normMat);
    cv::normalize(normMat, normMat, 0, 255, cv::NORM_MINMAX, cv_input_type);
#endif
    normMat.convertTo(normMat, CV_8UC1);
#if 0
    //cv::namedWindow(fname);
    const int width = 1408;
    const int height = 800*3;
    assert(size == width*height);
    cv::Mat img(height, width, CV_8UC1, normMat.ptr());
    cv::imshow(fname, img);
    //cv::waitKey(0);
#endif
    outfile.open(fname, std::ofstream::out);
    outfile.write((char *)(normMat.ptr()), size);
    outfile.close();
    //free(tmpBuff);
#endif
}
