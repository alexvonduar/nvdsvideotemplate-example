#ifndef _CUSTOMLIB_TRT_HPP_
#define _CUSTOMLIB_TRT_HPP_

#include <string>

#include "NvInfer.h"
#include <nppcore.h>
#include <nppi_data_exchange_and_initialization.h>

#include <nvbufsurface.h>

#include "common.h"
#include "logger.h"
#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
#include <cudaEGL.h>
#endif

typedef struct _TRTBuffer
{
    nvinfer1::Dims dims;
    nvinfer1::DataType dataType;
    NvBufSurfaceMemType memType;
    int batchSize;
    void *buffer;
    size_t size;
} TRTBuffer;

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

class TRTInfer
{
public:
    void initialize(const std::string &trtModelFile, const std::vector<std::string> &inputNames, const std::vector<std::string> &outputNames)
    {
        sample::gLogInfo << "Initializing TensorRT" << std::endl;
        cudaStreamCreate(&m_stream);
        nppSetStream(m_stream);
        nppGetStreamContext(&m_nppcontext);

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

        if (!m_trtcontext)
        {
            m_trtcontext = std::unique_ptr<nvinfer1::IExecutionContext>{(m_trtengine->createExecutionContext())};
        }

        if (!m_trtcontext)
        {
            sample::gLogError << "Failed to create IExecutionContext" << std::endl;
            return;
        }

        int batchSize = 0;

        for (const auto &inputName : inputNames)
        {
            TRTBuffer buffer;
            const auto inputIndex = m_trtengine->getBindingIndex(inputName.c_str());
            buffer.dataType = m_trtengine->getBindingDataType(inputIndex);
            buffer.dims = m_trtengine->getBindingDimensions(inputIndex);
            buffer.batchSize = buffer.dims.nbDims < 3 ? 1 : buffer.dims.d[0];
            if (batchSize)
            {
                assert(buffer.batchSize == batchSize);
            }
            else
            {
                batchSize = buffer.batchSize;
            }
            sample::gLogInfo << "inputIndex: " << inputIndex << " type: " << dataTypeName(buffer.dataType) << " dims: " << dimsToString(buffer.dims) << std::endl;
            buffer.size = samplesCommon::volume(buffer.dims) * samplesCommon::getElementSize(buffer.dataType);
            sample::gLogInfo << "size: " << buffer.size << std::endl;
            buffer.buffer = nullptr;
#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
            buffer.memType = NVBUF_MEM_CUDA_DEVICE;
            cudaMalloc(&buffer.buffer, buffer.size);
#else
            buffer.memType = NVBUF_MEM_CUDA_UNIFIED;
            cudaMallocManaged(&buffer.buffer, buffer.size);
#endif
            if (buffer.buffer == nullptr)
            {
                sample::gLogError << "Failed to allocate input buffer" << std::endl;
                // return;
            }
            m_trtInputBuffers.emplace_back(buffer);
        }

        for (const auto &outputName : outputNames)
        {
            TRTBuffer buffer;
            const auto outputIndex = m_trtengine->getBindingIndex(outputName.c_str());
            buffer.dataType = m_trtengine->getBindingDataType(outputIndex);
            buffer.dims = m_trtengine->getBindingDimensions(outputIndex);
            buffer.batchSize = buffer.dims.nbDims < 3 ? 1 : buffer.dims.d[0];
            assert(buffer.batchSize == batchSize);
            sample::gLogInfo << "outputIndex: " << outputIndex << " type: " << dataTypeName(buffer.dataType) << " dims: " << dimsToString(buffer.dims) << std::endl;
            buffer.size = samplesCommon::volume(buffer.dims) * samplesCommon::getElementSize(buffer.dataType);
            sample::gLogInfo << "size: " << buffer.size << std::endl;
            buffer.buffer = nullptr;
#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
            buffer.memType = NVBUF_MEM_CUDA_DEVICE;
            cudaMalloc(&buffer.buffer, buffer.size);
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
            m_trtOutputBuffers.emplace_back(buffer);
        }

        if (m_bindings)
        {
            delete[] m_bindings;
        }
        m_numBindings = m_trtInputBuffers.size() + m_trtOutputBuffers.size();
        m_bindings = new void *[m_numBindings];
        for (size_t i = 0; i < m_trtInputBuffers.size(); i++)
        {
            m_bindings[i] = m_trtInputBuffers[i].buffer;
        }
        for (size_t i = 0; i < m_trtOutputBuffers.size(); i++)
        {
            m_bindings[i + m_trtInputBuffers.size()] = m_trtOutputBuffers[i].buffer;
        }
    }

#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
    void preprocessingImage(NvBufSurface *_surf)
    {
        // assert(_surf->numFilled == 1);
        const auto numFilled = _surf->numFilled;
        const auto batchSize = _surf->batchSize;
        assert(_surf->memType == NVBUF_MEM_SURFACE_ARRAY);
        NvBufSurface *surf = nullptr;
        bool input_pitch = _surf->surfaceList[0].layout == NVBUF_LAYOUT_PITCH;
        if (input_pitch)
        {
            surf = _surf;
        }
        else
        {
            if (m_tmpsurf == nullptr)
            {
                sample::gLogError << "try creat scratch buffer" << std::endl;
                // FIXME: create temp surface
                NvBufSurfaceCreateParams surfaceCreateParams;
                surfaceCreateParams.layout = NVBUF_LAYOUT_PITCH;
                surfaceCreateParams.colorFormat = NVBUF_COLOR_FORMAT_NV12;
                surfaceCreateParams.width = _surf->surfaceList[0].width;
                surfaceCreateParams.height = _surf->surfaceList[0].height;
                surfaceCreateParams.isContiguous = true;
                surfaceCreateParams.size = 0;
                surfaceCreateParams.gpuId = 0;
                surfaceCreateParams.memType = _surf->memType;

                m_tmpsurf = new NvBufSurface();
                NvBufSurfaceCreate(&m_tmpsurf, batchSize, &surfaceCreateParams);
            }
            surf = m_tmpsurf;
            NvBufSurfTransformParams transform_params;
            transform_params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
            transform_params.transform_flip = NvBufSurfTransform_None;
            transform_params.transform_filter = NvBufSurfTransformInter_Default;

            NvBufSurfTransform(_surf, surf, &transform_params);
            // NvBufSurfaceMemSet(surf, 0, 0, 128);
        }

        // map all surfaces
        int status = NvBufSurfaceMapEglImage(surf, -1);
        if (status != 0)
        {
            sample::gLogError << "Failed to map surface" << std::endl;
            return;
        }

        for (int i = 0; i < surf->numFilled; i++)
        {
            CUresult egl_status;
            CUeglFrame eglFrame;
            CUgraphicsResource pResource = NULL;

            EGLImageKHR eglImage = surf->surfaceList[i].mappedAddr.eglImage;

            cudaFree(0);
            egl_status = cuGraphicsEGLRegisterImage(&pResource, eglImage,
                                                    CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
            if (egl_status != CUDA_SUCCESS)
            {
                printf("cuGraphicsEGLRegisterImage failed: %d, cuda process stop\n",
                       egl_status);
                return;
            }

            egl_status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
            if (egl_status != CUDA_SUCCESS)
            {
                printf("cuGraphicsSubResourceGetMappedArray failed\n");
            }

            egl_status = cuCtxSynchronize();
            if (egl_status != CUDA_SUCCESS)
            {
                printf("cuCtxSynchronize failed\n");
            }

            /// sample::gLogError << "egl frame type: " << eglFrame.frameType << std::endl;
            if (eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH)
            {
                const auto w = surf->surfaceList[i].width;
                const auto h = surf->surfaceList[i].height;
                const auto p = surf->surfaceList[i].pitch;
                assert(surf->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_NV12);
                // assert(surf->surfaceList[i].layout == NVBUF_LAYOUT_PITCH);
                assert(surf->surfaceList[i].planeParams.num_planes == 2);
                const auto plane_w = surf->surfaceList[i].planeParams.width[0];
                const auto plane_h = surf->surfaceList[i].planeParams.height[0];
                const auto plane_p = surf->surfaceList[i].planeParams.pitch[0];
                assert(w == plane_w);
                assert(h == plane_h);
                assert(p == plane_p);
                // assert(w == p);
                const auto plane_offset = surf->surfaceList[i].planeParams.offset[0];
                NppiSize oSizeROI{plane_w, plane_h};
                // nppSetStream(m_stream);
                // NppStreamContext nppStreamCtx;
                // nppGetStreamContext(&nppStreamCtx);
                auto pSrc = reinterpret_cast<Npp8u *>(eglFrame.frame.pPitch[0]);
                const auto numInputBuffer = m_trtInputBuffers[0].batchSize;
                const auto bufferSizePerBatch = m_trtInputBuffers[0].size / numInputBuffer;
                auto _pDst = reinterpret_cast<Npp8u *>(m_trtInputBuffers[0].buffer) + i * bufferSizePerBatch;
                auto pDst = reinterpret_cast<Npp32f *>(_pDst);
                auto npp_status = nppiScale_8u32f_C1R_Ctx(pSrc, plane_p, pDst, plane_p * sizeof(float), oSizeROI, 0.0, 1.0, m_nppcontext);
                // cudaStreamSynchronize(m_nppStream);
            }

            status = cuCtxSynchronize();
            if (status != CUDA_SUCCESS)
            {
                printf("cuCtxSynchronize failed after memcpy\n");
            }

            status = cuGraphicsUnregisterResource(pResource);
            if (status != CUDA_SUCCESS)
            {
                printf("cuGraphicsEGLUnRegisterResource failed: %d\n", status);
            }
        }

        NvBufSurfaceUnMapEglImage(surf, -1);

        ////if (!input_pitch)
        ////{
        ////  NvBufSurfaceDestroy(surf);
        ////}
    }
#else
    void preprocessingImage(NvBufSurface *surf)
    {
        // assert(surf->numFilled == 1);
        //  FIXME: check gpuid to support multi-gpu
        assert(surf->memType == NVBUF_MEM_CUDA_UNIFIED);
        auto unifiedMem = surf->memType == NVBUF_MEM_CUDA_UNIFIED;
        if (unifiedMem)
        {
            int status = NvBufSurfaceMap(surf, -1, -1, NVBUF_MAP_READ_WRITE);
            if (status != 0)
            {
                sample::gLogError << "Failed to map surface" << std::endl;
                return;
            }
        }

        assert(surf->batchSize == m_trtInputBuffers[0].batchSize);

        // iterate at batch size
        const auto batchSize = surf->batchSize;
        const auto inputSizePerBatch = m_trtInputBuffers[0].size / batchSize;
        auto inputBuffer = reinterpret_cast<unsigned char *>(m_trtInputBuffers[0].buffer);
        for (int i = 0; i < surf->numFilled; i++)
        {
            const auto w = surf->surfaceList[i].width;
            const auto h = surf->surfaceList[i].height;
            const auto p = surf->surfaceList[i].pitch;
            assert(surf->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_NV12);
            assert(surf->surfaceList[i].layout == NVBUF_LAYOUT_PITCH);
            assert(surf->surfaceList[i].planeParams.num_planes == 2);
            const auto plane_w = surf->surfaceList[i].planeParams.width[0];
            const auto plane_h = surf->surfaceList[i].planeParams.height[0];
            const auto plane_p = surf->surfaceList[i].planeParams.pitch[0];
            assert(w == plane_w);
            assert(h == plane_h);
            assert(p == plane_p);
            // assert(w == p);
            const auto plane_offset = surf->surfaceList[i].planeParams.offset[0];
            NppiSize oSizeROI{plane_w, plane_h};
            // nppSetStream(m_stream);
            // NppStreamContext nppStreamCtx;
            // nppGetStreamContext(&nppStreamCtx);
            Npp8u *pSrc = nullptr;
            if (unifiedMem)
            {
                pSrc = reinterpret_cast<Npp8u *>(surf->surfaceList[i].mappedAddr.addr[0]) + plane_offset;
            }
            else
            {
                pSrc = reinterpret_cast<Npp8u *>(surf->surfaceList[i].dataPtr) + plane_offset;
            }
            auto pDst = reinterpret_cast<Npp32f *>(inputBuffer + i * inputSizePerBatch);
            auto npp_status = nppiScale_8u32f_C1R_Ctx(pSrc, plane_p, pDst, plane_p * sizeof(float), oSizeROI, 0.0, 255.0, m_nppcontext);
            // cudaStreamSynchronize(m_nppStream);
        }

        if (unifiedMem)
        {
            NvBufSurfaceUnMap(surf, -1, -1);
        }
    }
#endif

#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
    void postprocessingImage(NvBufSurface *surf)
    {
        assert(surf->memType == NVBUF_MEM_SURFACE_ARRAY);
        // assert(surf->numFilled == 1);
        const auto batchSize = surf->batchSize;
        const auto numFilled = surf->numFilled;
        // map all surfaces
        int status = NvBufSurfaceMapEglImage(surf, -1);
        if (status != 0)
        {
            sample::gLogError << "Failed to map surface" << std::endl;
            return;
        }

        for (int i = 0; i < surf->numFilled; i++)
        {
            CUresult egl_status;
            CUeglFrame eglFrame;
            CUgraphicsResource pResource = NULL;

            EGLImageKHR eglImage = surf->surfaceList[i].mappedAddr.eglImage;

            cudaFree(0);
            egl_status = cuGraphicsEGLRegisterImage(&pResource, eglImage,
                                                    CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
            if (egl_status != CUDA_SUCCESS)
            {
                printf("cuGraphicsEGLRegisterImage failed: %d, cuda process stop\n",
                       egl_status);
                return;
            }

            egl_status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pResource, 0, 0);
            if (egl_status != CUDA_SUCCESS)
            {
                printf("cuGraphicsSubResourceGetMappedArray failed\n");
            }

            egl_status = cuCtxSynchronize();
            if (egl_status != CUDA_SUCCESS)
            {
                printf("cuCtxSynchronize failed\n");
            }

            if (eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH)
            {
                const auto w = surf->surfaceList[i].width;
                const auto h = surf->surfaceList[i].height;
                const auto p = surf->surfaceList[i].pitch;
                assert(surf->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_NV12 ||
                       surf->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_NV12_ER ||
                       surf->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_NV12_709 ||
                       surf->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_NV12_709_ER ||
                       surf->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_NV12_2020);
                assert(surf->surfaceList[i].layout == NVBUF_LAYOUT_PITCH);
                assert(surf->surfaceList[i].planeParams.num_planes == 2);
                const auto plane_w = surf->surfaceList[i].planeParams.width[0];
                const auto plane_h = surf->surfaceList[i].planeParams.height[0];
                const auto plane_p = surf->surfaceList[i].planeParams.pitch[0];
                assert(w == plane_w);
                assert(h == plane_h);
                assert(p == plane_p);
                assert(w == p);
                const auto plane_offset = surf->surfaceList[i].planeParams.offset[0];
                NppiSize oSizeROI{plane_w, plane_h};
                // cudaStream_t stream;
                // cudaStreamCreate(&stream);
                // nppSetStream(stream);
                // NppStreamContext nppStreamCtx;
                // nppGetStreamContext(&nppStreamCtx);

                auto pDst = reinterpret_cast<Npp8u *>(eglFrame.frame.pPitch[0]);
                const auto outputBufferSizePerBatch = m_trtOutputBuffers[0].size / batchSize;
                auto pSrc = reinterpret_cast<Npp32f *>(m_trtOutputBuffers[0].buffer) + i * outputBufferSizePerBatch;
                //auto pSrc = reinterpret_cast<Npp32f *>(m_trtOutputBuffer);
                // std::cout << "output plane pitch " << plane_p << std::endl;
                auto npp_status = nppiScale_32f8u_C1R_Ctx(pSrc, plane_p * sizeof(float), pDst, plane_p, oSizeROI, 0.0, 1.0, m_nppcontext);
                if (npp_status != NPP_NO_ERROR)
                {
                    std::cout << "post processing npp error " << npp_status << std::endl;
                }
                // cudaStreamSynchronize(m_nppStream);
                //  cudaStreamDestroy(stream);
            }

            status = cuCtxSynchronize();
            if (status != CUDA_SUCCESS)
            {
                printf("cuCtxSynchronize failed after memcpy\n");
            }

            status = cuGraphicsUnregisterResource(pResource);
            if (status != CUDA_SUCCESS)
            {
                printf("cuGraphicsEGLUnRegisterResource failed: %d\n", status);
            }
            /// NvBufSurfaceMemSet(surf, 0, 0, 128);
        }

        NvBufSurfaceUnMapEglImage(surf, -1);
    }
#else
    void postprocessingImage(NvBufSurface *surf)
    {
        assert(surf->batchSize == m_trtOutputBuffers[0].batchSize);

        // FIXME: check gpuid to support multi-gpu
        assert(surf->memType == NVBUF_MEM_CUDA_UNIFIED);
        auto unifiedMem = surf->memType == NVBUF_MEM_CUDA_UNIFIED;
        if (unifiedMem)
        {
            int status = NvBufSurfaceMap(surf, -1, -1, NVBUF_MAP_READ_WRITE);
            if (status != 0)
            {
                sample::gLogError << "Failed to map surface" << std::endl;
                return;
            }
        }

        const auto batchSize = surf->batchSize;
        const auto outputSizePerBatch = m_trtOutputBuffers[0].size / batchSize;
        auto outputBuffer = reinterpret_cast<unsigned char *>(m_trtOutputBuffers[0].buffer);
        for (int i = 0; i < surf->numFilled; i++)
        {
            const auto w = surf->surfaceList[i].width;
            const auto h = surf->surfaceList[i].height;
            const auto p = surf->surfaceList[i].pitch;
            assert(surf->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_NV12 ||
                   surf->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_NV12_ER ||
                   surf->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_NV12_709 ||
                   surf->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_NV12_709_ER ||
                   surf->surfaceList[i].colorFormat == NVBUF_COLOR_FORMAT_NV12_2020);
            assert(surf->surfaceList[i].layout == NVBUF_LAYOUT_PITCH);
            assert(surf->surfaceList[i].planeParams.num_planes == 2);
            const auto plane_w = surf->surfaceList[i].planeParams.width[0];
            const auto plane_h = surf->surfaceList[i].planeParams.height[0];
            const auto plane_p = surf->surfaceList[i].planeParams.pitch[0];
            assert(w == plane_w);
            assert(h == plane_h);
            assert(p == plane_p);
            assert(w == p);
            const auto plane_offset = surf->surfaceList[i].planeParams.offset[0];
            NppiSize oSizeROI{plane_w, plane_h};
            // cudaStream_t stream;
            // cudaStreamCreate(&stream);
            // nppSetStream(stream);
            // NppStreamContext nppStreamCtx;
            // nppGetStreamContext(&nppStreamCtx);
            Npp8u *pDst = nullptr;
            if (unifiedMem)
            {
                pDst = reinterpret_cast<Npp8u *>(surf->surfaceList[i].mappedAddr.addr[0]) + plane_offset;
            }
            else
            {
                pDst = reinterpret_cast<Npp8u *>(surf->surfaceList[i].dataPtr) + plane_offset;
            }
            auto pSrc = reinterpret_cast<Npp32f *>(outputBuffer + outputSizePerBatch * i);
            // std::cout << "output buffer: " << m_trtOutputBuffer << std::endl;
            std::cout << "output plane pitch " << plane_p << std::endl;
            auto npp_status = nppiScale_32f8u_C1R_Ctx(pSrc, plane_p * sizeof(float), pDst, plane_p, oSizeROI, 0.0, 255.0, m_nppcontext);
            if (npp_status != NPP_NO_ERROR)
            {
                std::cout << "post processing npp error " << npp_status << std::endl;
            }
            // cudaStreamSynchronize(m_nppStream);
            // cudaStreamDestroy(stream);
        }

        if (unifiedMem)
        {
            NvBufSurfaceUnMap(surf, -1, -1);
        }
    }
#endif

    void trtInference(NvBufSurface *input, NvBufSurface *output)
    {
        preprocessingImage(input);
        // void *bindings[] = {m_trtInputBuffer, m_trtOutputBuffer};
        //  auto context = m_trtengine->getExecutionContext();
        //  cudaStream_t stream;
        //  cudaStreamCreate(&stream);
        bool status = m_trtcontext->enqueueV2(m_bindings, m_stream, nullptr);
        cudaStreamSynchronize(m_stream);
        // cudaStreamDestroy(stream);
        if (!status)
        {
            sample::gLogError << "Failed to enqueue infer" << std::endl;
            //    //assert(0);
            ///    //return;
        }
        postprocessingImage(output);
    }

    ~TRTInfer()
    {
        if (m_bindings)
        {
            delete[] m_bindings;
            m_bindings = nullptr;
        }

        for (auto &b : m_trtInputBuffers)
        {
            cudaFree(b.buffer);
            b.buffer = nullptr;
        }

        for (auto &b : m_trtOutputBuffers)
        {
            cudaFree(b.buffer);
            b.buffer = nullptr;
        }

        if (m_stream)
        {
            cudaStreamDestroy(m_stream);
            m_stream = NULL;
        }
        ////if (m_nppStream) {
        ////  cudaStreamDestroy(m_nppStream);
        ////  m_nppStream = NULL;
        ////}
#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
        if (m_tmpsurf)
        {
            NvBufSurfaceDestroy(m_tmpsurf);
            m_tmpsurf = NULL;
        }
#endif
    }

    TRTInfer() : m_numBindings(0), m_bindings(nullptr)
#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
                 ,
                 m_tmpsurf(nullptr)
#endif

    {
    }

private:
    cudaStream_t m_stream;
    std::unique_ptr<nvinfer1::IRuntime> m_trtruntime;
    std::shared_ptr<nvinfer1::ICudaEngine> m_trtengine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_trtcontext;
    std::string m_trtModelFile;
    std::vector<TRTBuffer> m_trtInputBuffers;
    // char *m_trtInputBuffer;
    std::vector<TRTBuffer> m_trtOutputBuffers;
    // char *m_trtOutputBuffer;
    //  cudaStream_t m_nppStream;
    NppStreamContext m_nppcontext;
#if defined(PLATFORM_TEGRA) && PLATFORM_TEGRA
    NvBufSurface *m_tmpsurf;
#endif
    int m_numBindings;
    void **m_bindings;
    static std::string const DEF_ENGINE_NAME;
    static std::string const INPUT_LAYER_NAME;
    static std::string const OUTPUT_LAYER_NAME;
};

const std::string TRTInfer::DEF_ENGINE_NAME = "best_1x1x720x1280.engine"; //"float_int8.engine";
const std::string TRTInfer::INPUT_LAYER_NAME = "Placeholder:0";
const std::string TRTInfer::OUTPUT_LAYER_NAME = "transpose_1:0";
#endif //_CUSTOMLIB_TRT_HPP_
