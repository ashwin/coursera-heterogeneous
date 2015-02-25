////
// wb.h: Header file for Heterogeneous Parallel Programming course (Coursera)
////

#pragma once

////
// Headers
////

// C++
#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <list>
#include <sstream>
#include <string>
#include <vector>

// CUDA
#if defined(__CUDACC__)
    #include <cuda.h>
    #include <cuda_runtime.h>
#else
// OpenCL
    #if defined(__APPLE__)
        #include <OpenCL/cl.h>
    #else
        #include <CL/cl.h>
    #endif
#endif

////
// Macros
////

#ifndef NDEBUG
    #define wbAssert(condition, message)                                                                  \
        do                                                                                                \
        {                                                                                                 \
            if (!(condition))                                                                             \
            {                                                                                             \
                std::cerr << "Assertion failed: (" #condition "), function " << __FUNCTION__ << ", file " \
                          << __FILE__  << ", line " << __LINE__ << ": " << message << std::endl;          \
                std::exit(EXIT_FAILURE);                                                                  \
            }                                                                                             \
        } while (0)
#else
    #define wbAssert(condition, message)
#endif

////
// Constants
////

namespace wbInternal
{
    // Maximum number of errors to display in wbSolution()
    const int kErrorReportLimit = 10;

    // For further information, see the PPM image format documentation at http://netpbm.sourceforge.net
    const int kImageChannels = 3;
    const int kImageColorLimit = 255;
} // namespace wbInternal

////
// Supporting functions
////

namespace wbInternal
{
#if defined(_WIN32)
    std::string wbStrerror(int errnum)
    {
        std::string str;
        char buffer[1024];

        if (errnum)
        {
            (void) strerror_s(buffer, sizeof(buffer), errnum);
            str = buffer;
        }

        return str;
    }
#elif (_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600 || __APPLE__) && ! _GNU_SOURCE
    std::string wbStrerror(int errnum)
    {
        std::string str;
        char buffer[1024];

        if (errnum)
        {
            (void) strerror_r(errnum, buffer, sizeof(buffer));
            str = buffer;
        }

        return str;
    }
#elif defined(_GNU_SOURCE)
    std::string wbStrerror(int errnum)
    {
        std::string str;
        char buffer[1024];

        if (errnum)
        {
            str = strerror_r(errnum, buffer, sizeof(buffer));
        }

        return str;
    }
#else
    std::string wbStrerror(int errnum)
    {
        std::string str;

        if (errnum)
        {
            str = strerror(errnum);
        }

        return str;
    }
#endif
} // namespace wbInternal

////
// OpenCL
////

#if !defined(__CUDACC__)
#define wbOpenCL_ERROR(error) \
    case error:               \
        return #error

const char* wbOpenCLGetErrorString(cl_int error)
{
    switch (error)
    {
/*
 * OpenCL runtime and JIT compile errors
 */
#if defined(CL_VERSION_1_0)
        wbOpenCL_ERROR(CL_SUCCESS);
        wbOpenCL_ERROR(CL_DEVICE_NOT_FOUND);
        wbOpenCL_ERROR(CL_DEVICE_NOT_AVAILABLE);
        wbOpenCL_ERROR(CL_COMPILER_NOT_AVAILABLE);
        wbOpenCL_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        wbOpenCL_ERROR(CL_OUT_OF_RESOURCES);
        wbOpenCL_ERROR(CL_OUT_OF_HOST_MEMORY);
        wbOpenCL_ERROR(CL_PROFILING_INFO_NOT_AVAILABLE);
        wbOpenCL_ERROR(CL_MEM_COPY_OVERLAP);
        wbOpenCL_ERROR(CL_IMAGE_FORMAT_MISMATCH);
        wbOpenCL_ERROR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        wbOpenCL_ERROR(CL_BUILD_PROGRAM_FAILURE);
        wbOpenCL_ERROR(CL_MAP_FAILURE);
#endif
#if defined(CL_VERSION_1_1)
        wbOpenCL_ERROR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
        wbOpenCL_ERROR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
#endif
#if defined(CL_VERSION_1_2)
        wbOpenCL_ERROR(CL_COMPILE_PROGRAM_FAILURE);
        wbOpenCL_ERROR(CL_LINKER_NOT_AVAILABLE);
        wbOpenCL_ERROR(CL_LINK_PROGRAM_FAILURE);
        wbOpenCL_ERROR(CL_DEVICE_PARTITION_FAILED);
        wbOpenCL_ERROR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
#endif
/*
 * OpenCL compile-time errors
 */
#if defined(CL_VERSION_1_0)
        wbOpenCL_ERROR(CL_INVALID_VALUE);
        wbOpenCL_ERROR(CL_INVALID_DEVICE_TYPE);
        wbOpenCL_ERROR(CL_INVALID_PLATFORM);
        wbOpenCL_ERROR(CL_INVALID_DEVICE);
        wbOpenCL_ERROR(CL_INVALID_CONTEXT);
        wbOpenCL_ERROR(CL_INVALID_QUEUE_PROPERTIES);
        wbOpenCL_ERROR(CL_INVALID_COMMAND_QUEUE);
        wbOpenCL_ERROR(CL_INVALID_HOST_PTR);
        wbOpenCL_ERROR(CL_INVALID_MEM_OBJECT);
        wbOpenCL_ERROR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        wbOpenCL_ERROR(CL_INVALID_IMAGE_SIZE);
        wbOpenCL_ERROR(CL_INVALID_SAMPLER);
        wbOpenCL_ERROR(CL_INVALID_BINARY);
        wbOpenCL_ERROR(CL_INVALID_BUILD_OPTIONS);
        wbOpenCL_ERROR(CL_INVALID_PROGRAM);
        wbOpenCL_ERROR(CL_INVALID_PROGRAM_EXECUTABLE);
        wbOpenCL_ERROR(CL_INVALID_KERNEL_NAME);
        wbOpenCL_ERROR(CL_INVALID_KERNEL_DEFINITION);
        wbOpenCL_ERROR(CL_INVALID_KERNEL);
        wbOpenCL_ERROR(CL_INVALID_ARG_INDEX);
        wbOpenCL_ERROR(CL_INVALID_ARG_VALUE);
        wbOpenCL_ERROR(CL_INVALID_ARG_SIZE);
        wbOpenCL_ERROR(CL_INVALID_KERNEL_ARGS);
        wbOpenCL_ERROR(CL_INVALID_WORK_DIMENSION);
        wbOpenCL_ERROR(CL_INVALID_WORK_GROUP_SIZE);
        wbOpenCL_ERROR(CL_INVALID_WORK_ITEM_SIZE);
        wbOpenCL_ERROR(CL_INVALID_GLOBAL_OFFSET);
        wbOpenCL_ERROR(CL_INVALID_EVENT_WAIT_LIST);
        wbOpenCL_ERROR(CL_INVALID_EVENT);
        wbOpenCL_ERROR(CL_INVALID_OPERATION);
        wbOpenCL_ERROR(CL_INVALID_GL_OBJECT);
        wbOpenCL_ERROR(CL_INVALID_BUFFER_SIZE);
        wbOpenCL_ERROR(CL_INVALID_MIP_LEVEL);
        wbOpenCL_ERROR(CL_INVALID_GLOBAL_WORK_SIZE);
#endif
#if defined(CL_VERSION_1_1)
        wbOpenCL_ERROR(CL_INVALID_PROPERTY);
#endif
#if defined(CL_VERSION_1_2)
        wbOpenCL_ERROR(CL_INVALID_IMAGE_DESCRIPTOR);
        wbOpenCL_ERROR(CL_INVALID_COMPILER_OPTIONS);
        wbOpenCL_ERROR(CL_INVALID_LINKER_OPTIONS);
        wbOpenCL_ERROR(CL_INVALID_DEVICE_PARTITION_COUNT);
#endif
#if defined(CL_VERSION_2_0)
        wbOpenCL_ERROR(CL_INVALID_PIPE_SIZE);
        wbOpenCL_ERROR(CL_INVALID_DEVICE_QUEUE);
#endif
/*
 * OpenCL extension error values
 */
#if defined(__OPENCL_CL_GL_H) && defined(cl_khr_gl_sharing)
        wbOpenCL_ERROR(CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR);
#endif
#if defined(__CL_EXT_H) && defined(cl_khr_icd)
        wbOpenCL_ERROR(CL_PLATFORM_NOT_FOUND_KHR);
#endif
#if defined(__OPENCL_CL_D3D10_H)
        wbOpenCL_ERROR(CL_INVALID_D3D10_DEVICE_KHR);
        wbOpenCL_ERROR(CL_INVALID_D3D10_RESOURCE_KHR);
        wbOpenCL_ERROR(CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR);
        wbOpenCL_ERROR(CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR);
#endif
#if defined(__OPENCL_CL_D3D11_H)
        wbOpenCL_ERROR(CL_INVALID_D3D11_DEVICE_KHR);
        wbOpenCL_ERROR(CL_INVALID_D3D11_RESOURCE_KHR);
        wbOpenCL_ERROR(CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR);
        wbOpenCL_ERROR(CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR);
#endif
#if defined(__OPENCL_CL_DX9_MEDIA_SHARING_H)
        wbOpenCL_ERROR(CL_INVALID_DX9_MEDIA_ADAPTER_KHR);
        wbOpenCL_ERROR(CL_INVALID_DX9_MEDIA_SURFACE_KHR);
        wbOpenCL_ERROR(CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR);
        wbOpenCL_ERROR(CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR);
#endif
#if defined(__CL_EXT_H) && defined(cl_ext_device_fission)
        wbOpenCL_ERROR(CL_DEVICE_PARTITION_FAILED_EXT);
        wbOpenCL_ERROR(CL_INVALID_PARTITION_COUNT_EXT);
        wbOpenCL_ERROR(CL_INVALID_PARTITION_NAME_EXT);
#endif
#if defined(__OPENCL_CL_EGL_H)
        wbOpenCL_ERROR(CL_EGL_RESOURCE_NOT_ACQUIRED_KHR);
        wbOpenCL_ERROR(CL_INVALID_EGL_OBJECT_KHR);
#endif
#if defined(__CL_EXT_H) && defined(cl_intel_accelerator)
        wbOpenCL_ERROR(CL_INVALID_ACCELERATOR_INTEL);
        wbOpenCL_ERROR(CL_INVALID_ACCELERATOR_TYPE_INTEL);
        wbOpenCL_ERROR(CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL);
        wbOpenCL_ERROR(CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL);
#endif
    // Undeclared OpenCL error (possibly implementation specific?)
    default:
        return "[Undeclared OpenCL error]";
    }
}
#undef wbOpenCL_ERROR
#endif

////
// Logging
////

namespace wbInternal
{
    enum wbLogLevel
    {
        wbLogLevel_OFF,
        wbLogLevel_FATAL,
        wbLogLevel_ERROR,
        wbLogLevel_WARN,
        wbLogLevel_INFO,
        wbLogLevel_DEBUG,
        wbLogLevel_TRACE,
        wbLogLevel_INVALID // Keep this at the end
    };

    const char* wbLogLevelStr[] =
    {
        "Off",
        "Fatal",
        "Error",
        "Warn",
        "Info",
        "Debug",
        "Trace",
        "***InvalidLogLevel***" // Keep this at the end
    };

    const char* wbLogLevelToStr(const wbLogLevel level)
    {
        wbAssert(level >= wbLogLevel_OFF && level < wbLogLevel_INVALID, "Unrecognized wbLogLevel value");
        return wbLogLevelStr[level];
    }

//-----------------------------------------------------------------------------
// Begin: Ugly C++03 hack
// NVCC 5.0 does not support C++11 variadic templates

    template<typename T1>
    inline void wbLog(T1 const& p1)
    {
        std::cout << p1;
    }

    template<typename T1, typename T2>
    inline void wbLog(T1 const& p1, T2 const& p2)
    {
        std::cout << p1 << p2;
    }

    template<typename T1, typename T2, typename T3>
    inline void wbLog(T1 const& p1, T2 const& p2, T3 const& p3)
    {
        std::cout << p1 << p2 << p3;
    }

    template<typename T1, typename T2, typename T3, typename T4>
    inline void wbLog(T1 const& p1, T2 const& p2, T3 const& p3, T4 const& p4)
    {
        std::cout << p1 << p2 << p3 << p4;
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5>
    inline void wbLog(T1 const& p1, T2 const& p2, T3 const& p3, T4 const& p4, T5 const& p5)
    {
        std::cout << p1 << p2 << p3 << p4 << p5;
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
    inline void wbLog(T1 const& p1, T2 const& p2, T3 const& p3, T4 const& p4, T5 const& p5, T6 const& p6)
    {
        std::cout << p1 << p2 << p3 << p4 << p5 << p6;
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
    inline void wbLog(T1 const& p1, T2 const& p2, T3 const& p3, T4 const& p4, T5 const& p5, T6 const& p6, T7 const& p7)
    {
        std::cout << p1 << p2 << p3 << p4 << p5 << p6 << p7;
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
    inline void wbLog(T1 const& p1, T2 const& p2, T3 const& p3, T4 const& p4, T5 const& p5, T6 const& p6, T7 const& p7, T8 const& p8)
    {
        std::cout << p1 << p2 << p3 << p4 << p5 << p6 << p7 << p8;
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
    inline void wbLog(T1 const& p1, T2 const& p2, T3 const& p3, T4 const& p4, T5 const& p5, T6 const& p6, T7 const& p7, T8 const& p8, T9 const& p9)
    {
        std::cout << p1 << p2 << p3 << p4 << p5 << p6 << p7 << p8 << p9;
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
    inline void wbLog(T1 const& p1, T2 const& p2, T3 const& p3, T4 const& p4, T5 const& p5, T6 const& p6, T7 const& p7, T8 const& p8, T9 const& p9, T10 const& p10)
    {
        std::cout << p1 << p2 << p3 << p4 << p5 << p6 << p7 << p8 << p9 << p10;
    }
} // namespace wbInternal

// End: Ugly C++03 hack
//-----------------------------------------------------------------------------

#define wbLog(level, ...)                                                                \
    do                                                                                   \
    {                                                                                    \
        std::cout << wbInternal::wbLogLevelToStr(wbInternal::wbLogLevel_##level) << " "; \
        std::cout << __FUNCTION__ << "::" << __LINE__ << " ";                            \
        wbInternal::wbLog(__VA_ARGS__);                                                  \
        std::cout << std::endl;                                                          \
    } while (0)

////
// Input arguments
////

struct wbArg_t
{
    int    argc;
    char** argv;
};

wbArg_t wbArg_read(const int argc, char** argv)
{
    wbArg_t argInfo = { argc, argv };
    return argInfo;
}

char* wbArg_getInputFile(const wbArg_t argInfo, const int argNum)
{
    wbAssert(argNum >= 0 && argNum < (argInfo.argc - 1), "Unrecognized command line argument requested");
    return argInfo.argv[argNum + 1];
}

// For assignments MP1, MP4, MP5 & MP12
float* wbImport(const char* fName, int* numElements)
{
    std::ifstream inFile(fName);

    if (!inFile.is_open())
    {
        std::cerr << "Error opening input file " << fName << ". " << wbInternal::wbStrerror(errno) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    inFile >> *numElements;

    std::string sVal;
    std::vector<float> fVec;

    fVec.reserve(*numElements);

    while (inFile >> sVal)
    {
        std::istringstream iss(sVal);
        float fVal;
        iss >> fVal;
        fVec.push_back(fVal);
    }

    inFile.close();

    if (*numElements != static_cast<int>(fVec.size()))
    {
        std::cerr << "Error reading contents of file " << fName << ". Expecting " << *numElements << " elements but got " << fVec.size() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    float* fBuf = (float*) malloc(*numElements * sizeof(float));

    if (!fBuf)
    {
        std::cerr << "Unable to allocate memory for an array of size " << *numElements * sizeof(float) << " bytes" << std::endl;
        inFile.close();
        std::exit(EXIT_FAILURE);
    }

    std::copy(fVec.begin(), fVec.end(), fBuf);

    return fBuf;
}

namespace wbInternal
{
    // For assignment MP6
    float* wbParseCSV(const char* fName, int* numRows, int* numCols)
    {
        std::ifstream inFile(fName);

        if (!inFile.is_open())
        {
            std::cerr << "Error opening input file " << fName << ". " << wbInternal::wbStrerror(errno) << std::endl;
            std::exit(EXIT_FAILURE);
        }

        std::vector<float> fVec;
        std::string rowStr;
        *numRows = *numCols = 0;

        while (std::getline(inFile, rowStr))
        {
            std::istringstream rowStream(rowStr);
            std::string cellStr;
            ++(*numRows);
            *numCols = 0;

            while (std::getline(rowStream, cellStr, ','))
            {
                float fVal;
                ++(*numCols);

                if (!(std::istringstream(cellStr) >> fVal))
                {
                    std::cerr << "Error reading element (" << *numRows << ", " << *numCols << ") in file " << fName << std::endl;
                    inFile.close();
                    std::exit(EXIT_FAILURE);
                }

                fVec.push_back(fVal);
            }
        }

        inFile.close();

        const int numElements = *numRows * *numCols;

        if ((*numRows != *numCols) || (0 == numElements))
        {
            std::cerr << "Error reading contents of file " << fName << ". Last element read (" << *numRows << ", " << *numCols << ")" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        float* fBuf = (float*) malloc(numElements * sizeof(float));

        if (!fBuf)
        {
            std::cerr << "Unable to allocate memory for an array of size " << numElements * sizeof(float) << " bytes" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        std::copy(fVec.begin(), fVec.end(), fBuf);

        return fBuf;
    }
} // namespace wbInternal

// For assignments MP2, MP3 & MP6
float* wbImport(const char* fName, int* numRows, int* numCols)
{
    std::string fNameStr(fName);

    if (fNameStr.substr(fNameStr.find_last_of(".") + 1) == "csv")
    {
        return wbInternal::wbParseCSV(fName, numRows, numCols);
    }

    std::ifstream inFile(fName);

    if (!inFile.is_open())
    {
        std::cerr << "Error opening input file " << fName << ". " << wbInternal::wbStrerror(errno) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    inFile >> *numRows;
    inFile >> *numCols;

    const int numElements = *numRows * *numCols;

    std::string sVal;
    std::vector<float> fVec;

    fVec.reserve(numElements);

    while (inFile >> sVal)
    {
        std::istringstream iss(sVal);
        float fVal;
        iss >> fVal;
        fVec.push_back(fVal);
    }

    inFile.close();

    if (numElements != static_cast<int>(fVec.size()))
    {
        std::cerr << "Error reading contents of file " << fName << ". Expecting " << numElements << " elements but got " << fVec.size() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    float* fBuf = (float*) malloc(numElements * sizeof(float));

    if (!fBuf)
    {
        std::cerr << "Unable to allocate memory for an array of size " << numElements * sizeof(float) << " bytes" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::copy(fVec.begin(), fVec.end(), fBuf);

    return fBuf;
}

struct wbImage_t
{
    int width;
    int height;
    int channels;
    int colors;
    float* data;

    wbImage_t(int imageWidth = 0, int imageHeight = 0, int imageChannels = wbInternal::kImageChannels, int imageColors = wbInternal::kImageColorLimit) : width(imageWidth), height(imageHeight), channels(imageChannels), colors(imageColors), data(NULL)
    {
        const int numElements = width * height * channels;

        // Prevent zero-length memory allocation
        if (numElements > 0)
            data = new float[numElements];
    }
};

// For assignments MP6 & MP11
wbImage_t wbImport(const char* fName)
{
    std::ifstream inFile(fName, std::ios::binary);

    if (!inFile.is_open())
    {
        std::cerr << "Error opening image file " << fName << ". " << wbInternal::wbStrerror(errno) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Read PPM image header
    std::string magic;
    std::getline(inFile, magic);

    if (magic != "P6")
    {
        std::cerr << "Error reading image file " << fName << ". " << "Expecting 'P6' image format but got '" << magic << "'" << std::endl;
        inFile.close();
        std::exit(EXIT_FAILURE);
    }

    // Filter image comments
    if (inFile.peek() == '#')
    {
        std::string commentStr;
        std::getline(inFile, commentStr);
    }

    wbImage_t image;

    inFile >> image.width;

    if (inFile.fail() || 0 >= image.width)
    {
        std::cerr << "Error reading width of image in file " << fName << std::endl;
        inFile.close();
        std::exit(EXIT_FAILURE);
    }

    inFile >> image.height;

    if (inFile.fail() || 0 >= image.height)
    {
        std::cerr << "Error reading height of image in file " << fName << std::endl;
        inFile.close();
        std::exit(EXIT_FAILURE);
    }

    inFile >> image.colors;

    if (inFile.fail() || image.colors != wbInternal::kImageColorLimit)
    {
        std::cerr << "Error reading colors value of image in file " << fName << std::endl;
        inFile.close();
        std::exit(EXIT_FAILURE);
    }

    while (isspace(inFile.peek()))
    {
        inFile.get();
    }

    const int numElements = image.width * image.height * image.channels;

    unsigned char* rawData = new unsigned char[numElements];

    inFile.read(reinterpret_cast<char*>(rawData), numElements);

    const int elementsRead = static_cast<int>(inFile.gcount());

    inFile.close();

    if (elementsRead != numElements)
    {
        std::cerr << "Size of image in file " << fName << " does not match its header. Expecting " << numElements << " bytes, but got " << elementsRead << std::endl;
        delete [] rawData;
        std::exit(EXIT_FAILURE);
    }

    float* data = new float[numElements];

    for (int i = 0; i < numElements; ++i)
    {
        data[i] = rawData[i] * (1.0f / wbInternal::kImageColorLimit);
    }

    image.data = data;
    delete [] rawData;

    return image;
}

int wbImage_getWidth(const wbImage_t& image)
{
    return image.width;
}

int wbImage_getHeight(const wbImage_t& image)
{
    return image.height;
}

int wbImage_getChannels(const wbImage_t& image)
{
    return image.channels;
}

float* wbImage_getData(const wbImage_t& image)
{
     return image.data;
}

wbImage_t wbImage_new(const int imageWidth, const int imageHeight, const int imageChannels)
{
    wbImage_t image(imageWidth, imageHeight, imageChannels);
    return image;
}

void wbImage_delete(wbImage_t& image)
{
    delete [] image.data;
}

////
// Timer
////

#if defined(__CUDACC__)
    #define wbTimerDeviceSynchronize() cudaDeviceSynchronize()
#else
    #define wbTimerDeviceSynchronize()
#endif

// Namespace because Windows.h causes errors
namespace wbInternal
{
#if defined(_WIN32)
    #include <Windows.h>

    // wbTimer class adapted from: https://bitbucket.org/ashwin/cudatimer
    class wbTimer
    {
    private:
        double        timerResolution;
        LARGE_INTEGER startTime;
        LARGE_INTEGER endTime;

    public:
        wbTimer::wbTimer()
        {
            LARGE_INTEGER freq;
            QueryPerformanceFrequency(&freq);
            timerResolution = 1.0 / freq.QuadPart;
        }

        void start()
        {
            wbTimerDeviceSynchronize();
            QueryPerformanceCounter(&startTime);
        }

        void stop()
        {
            wbTimerDeviceSynchronize();
            QueryPerformanceCounter(&endTime);
        }

        double value()
        {
            return (endTime.QuadPart - startTime.QuadPart) * timerResolution;
        }
    };
#elif defined(__APPLE__)
    #include <mach/mach_time.h>

    class wbTimer
    {
    private:
        uint64_t startTime;
        uint64_t endTime;

    public:
        void start()
        {
            wbTimerDeviceSynchronize();
            startTime = mach_absolute_time();
        }

        void stop()
        {
            wbTimerDeviceSynchronize();
            endTime = mach_absolute_time();
        }

        double value()
        {
            static mach_timebase_info_data_t tb;

            if (0 == tb.denom)
                (void) mach_timebase_info(&tb); // Calculate ratio of mach_absolute_time ticks to nanoseconds

            return ((double) endTime - startTime) * (tb.numer / tb.denom) / NSEC_PER_SEC;
        }
    };
#else
    #if defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0
        #include <time.h>
    #else
        #include <sys/time.h>
    #endif

    #if !defined(NSEC_PER_SEC)
        #define NSEC_PER_SEC 1e9L
    #endif
    #if !defined(MSEC_PER_NSEC)
        #define MSEC_PER_NSEC (NSEC_PER_SEC / CLOCKS_PER_SEC)
    #endif

    class wbTimer
    {
    private:
        long startTime;
        long endTime;

        long getTime()
        {
            long time;
        #if defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0
            struct timespec ts;

            if (0 == clock_gettime(CLOCK_MONOTONIC, &ts))
            {
                time  = NSEC_PER_SEC;
                time *= ts.tv_sec;
                time += ts.tv_nsec;
            }
        #else
            struct timeval tv;

            if (0 == gettimeofday(&tv, NULL))
            {
                time  = NSEC_PER_SEC;
                time *= tv.tv_sec;
                time += tv.tv_usec * MSEC_PER_NSEC;
            }
        #endif
            return time;
        }

    public:
        void start()
        {
            wbTimerDeviceSynchronize();
            startTime = getTime();
        }

        void stop()
        {
            wbTimerDeviceSynchronize();
            endTime = getTime();
        }

        double value()
        {
            return ((double) endTime - startTime) / NSEC_PER_SEC;
        }
    };
#endif
} // namespace wbInternal

enum wbTimeType
{
    Generic,
    GPU,
    Compute,
    Copy,
    wbTimeTypeINVALID // Keep this at the end
};

namespace wbInternal
{
    const char* wbTimeTypeStr[] =
    {
        "Generic",
        "GPU    ",
        "Compute",
        "Copy   ",
        "***InvalidTimeType***", // Keep this at the end
    };

    const char* wbTimeTypeToStr(const wbTimeType timeType)
    {
        return wbTimeTypeStr[timeType];
    }

    struct wbTimerInfo
    {
        wbTimeType  type;
        std::string message;
        wbTimer     timer;

        bool operator==(const wbTimerInfo& t2) const
        {
            return (type == t2.type && (0 == message.compare(t2.message)));
        }
    };

    typedef std::list<wbTimerInfo> wbTimerInfoList;

    wbTimerInfoList timerInfoList;
} // namespace wbInternal

void wbTime_start(const wbTimeType timeType, const std::string timeMessage)
{
    wbAssert(timeType >= Generic && timeType < wbTimeTypeINVALID, "Unrecognized wbTimeType value");

    wbInternal::wbTimer timer;
    timer.start();

    wbInternal::wbTimerInfo timerInfo = { timeType, timeMessage, timer };

    wbInternal::timerInfoList.push_front(timerInfo);
}

void wbTime_stop(const wbTimeType timeType, const std::string timeMessage)
{
    wbAssert(timeType >= Generic && timeType < wbTimeTypeINVALID, "Unrecognized wbTimeType value");

    const wbInternal::wbTimerInfo searchInfo = { timeType, timeMessage, wbInternal::wbTimer() };
    const wbInternal::wbTimerInfoList::iterator iter = std::find(wbInternal::timerInfoList.begin(), wbInternal::timerInfoList.end(), searchInfo);

    wbInternal::wbTimerInfo& timerInfo = *iter;

    wbAssert(searchInfo == timerInfo, "Could not find a corresponding wbTimerInfo struct registered by wbTime_start()");

    timerInfo.timer.stop();

    std::cout << "[" << wbInternal::wbTimeTypeToStr( timerInfo.type ) << "] ";
    std::cout << std::fixed << std::setprecision(9) << timerInfo.timer.value() << " ";
    std::cout << timerInfo.message << std::endl;

    wbInternal::timerInfoList.erase(iter);
}

////
// Solutions
////

namespace wbInternal
{
    bool wbFPCloseEnough(const float u, const float v)
    {
        // Note that the tolerance level, e, is still an arbitrarily chosen value. Ideally, this value should scale
        // std::numeric_limits<float>::epsilon() by the number of rounding operations
        const float e = 0.0005f;

        // For floating point values u and v with tolerance e:
        //   |u - v| / |u| <= e || |u - v| / |v| <= e
        // defines a 'close enough' relationship between u and v that scales for magnitude
        // See Knuth, Seminumerical Algorithms 3e, s. 4.2.4, pp. 213-225
        return ((fabs(u - v) / fabs(u == 0.0f ? 1.0f : u) <= e) || (fabs(u - v) / fabs(v == 0.0f ? 1.0f : v) <= e));
    }
} // namespace wbInternal

// For assignments MP1, MP4, MP5 & MP12
template < typename T, typename S >
void wbSolution(const wbArg_t args, const T& t, const S& s)
{
    int solnItems;
    float* soln = wbImport(wbArg_getInputFile(args, args.argc - 2), &solnItems);

    if (solnItems != s)
    {
        std::cout << "Number of elements in solution file " << wbArg_getInputFile(args, args.argc - 2) << " does not match. ";
        std::cout << "Expecting " << s << " but got " << solnItems << ".\n";
    }
    else // Check solution
    {
        int errCnt = 0;

        for (int item = 0; item < solnItems; ++item)
        {
            if (!wbInternal::wbFPCloseEnough(soln[item], t[item]))
            {
                if (errCnt < wbInternal::kErrorReportLimit)
                {
                    std::cout << "The solution did not match the expected result at element " << item << ". ";
                    std::cout << "Expecting " << soln[item] << " but got " << t[item] << ".\n";
                }

                ++errCnt;
            }
        }

        if (!errCnt)
            std::cout << "Solution is correct.\n";
        else
            std::cout << errCnt << " tests failed!\n";
    }

    free(soln);
}

// For assignments MP2 & MP3
template < typename T, typename S, typename U >
void wbSolution(const wbArg_t& args, const T& t, const S& s, const U& u)
{
    int solnRows, solnColumns;
    float* soln = wbImport(wbArg_getInputFile(args, args.argc - 2), &solnRows, &solnColumns);

    if (solnRows != s || solnColumns != u)
    {
        std::cout << "Size of the matrix in solution file " << wbArg_getInputFile(args, args.argc - 2) << " does not match. ";
        std::cout << "Expecting " << solnRows << " x " << solnColumns << " but got " << s << " x " << u << ".\n";
    }
    else // Check solution
    {
        int errCnt = 0;

        for (int row = 0; row < solnRows; ++row)
        {
            for (int col = 0; col < solnColumns; ++col)
            {
                const float expected = row * solnColumns + col + *soln;
                const float result   = row * solnColumns + col + *t;

                if (!wbInternal::wbFPCloseEnough(expected, result))
                {
                    if (errCnt < wbInternal::kErrorReportLimit)
                    {
                        std::cout << "The solution did not match the expected results at column " << col << " and row " << row << "). ";
                        std::cout << "Expecting " << expected << " but got " << result << ".\n";
                    }

                    ++errCnt;
                }
            }
        }

        if (!errCnt)
            std::cout << "Solution is correct.\n";
        else
            std::cout << errCnt << " tests failed!\n";
    }

    free(soln);
}

namespace wbInternal
{
    // For assignments MP6 & MP11
    void wbImage_save(const wbImage_t& image, const wbArg_t& args, const char* fName)
    {
        std::ostringstream oss;
        oss << "P6\n" << "# Created by applying convolution " << wbArg_getInputFile(args, args.argc - 3) << "\n" << image.width << " " << image.height << "\n" << image.colors << "\n";
        std::string headerStr(oss.str());

        std::ofstream outFile(fName, std::ios::binary);
        outFile.write(headerStr.c_str(), headerStr.size());

        const int numElements = image.width * image.height * image.channels;

        unsigned char* rawData = new unsigned char[numElements];

        for (int i = 0; i < numElements; ++i)
        {
            rawData[i] = static_cast<unsigned char>(image.data[i] * wbInternal::kImageColorLimit + 0.5f);
        }

        outFile.write(reinterpret_cast<char*>(rawData), numElements);
        outFile.close();

        delete [] rawData;
    }
} // namespace wbInternal

// For assignment MP6 & MP11
void wbSolution(const wbArg_t& args, const wbImage_t& image)
{
    wbImage_t solnImage = wbImport(wbArg_getInputFile(args, args.argc - 2));

    if (solnImage.width != image.width || solnImage.height != image.height)
    {
        std::cout << "Size of the image in file " << wbArg_getInputFile(args, args.argc - 2) << " does not match. ";
        std::cout << "Expecting " << image.width << " x " << image.height << " but got " << solnImage.width << " x " << solnImage.height << ".\n";
    }
    else // Check solution
    {
        wbInternal::wbImage_save(image, args, "transformed_image.ppm");

        const float tolerance = 1.5f;
        int errCnt = 0;

        for (int i = 0; i < image.width; ++i)
        {
            for (int j = 0; j < image.height; ++j)
            {
                for (int k = 0; k < image.channels; ++k)
                {
                    const int index = (j * image.width + i) * image.channels + k;
                    const float error = fabs(solnImage.data[index] - image.data[index]);

                    if (error > (1.0f / wbInternal::kImageColorLimit * tolerance))
                    {
                        if (errCnt < wbInternal::kErrorReportLimit)
                            std::cout << "Image pixels do not match at position (" << j << ", " << i << ", " << k << "). [" << image.data[index] << ", " <<  solnImage.data[index] << "]\n";

                        ++errCnt;
                    }
                }
            }
        }

        if (!errCnt)
            std::cout << "Solution is correct." << std::endl;
        else
            std::cout << errCnt << " tests failed!" << std::endl;
    }

    wbImage_delete(solnImage);
}
