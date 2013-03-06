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
#include <fstream>
#include <iostream>
#include <iomanip>
#include <list>
#include <sstream>
#include <string>
#include <vector>

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

////
// Logging
////

enum wbLogLevel
{
    OFF,
    FATAL,
    ERROR,
    WARN,
    INFO,
    DEBUG,
    TRACE,
    wbLogLevelNum, // Keep this at the end
};

namespace wbInternal
{
    const char* wbLogLevelStr[] =
    {
        "Off",
        "Fatal",
        "Error",
        "Warn",
        "Info",
        "Debug",
        "Trace",
        "***InvalidLogLevel***", // Keep this at the end
    };

    const char* wbLogLevelToStr(const wbLogLevel level)
    {
        assert(level >= OFF && level <= TRACE);
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
        std::cout << wbInternal::wbLogLevelToStr(static_cast<wbLogLevel>(level)) << " "; \
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
    assert(argNum >= 0 && argNum < (argInfo.argc - 1));
    return argInfo.argv[argNum + 1];
}

// For assignments MP1, MP4 & MP5
float* wbImport(const char* fName, int* numElements)
{
    std::ifstream inFile(fName);

    if (!inFile.is_open())
    {
        std::cerr << "Error opening input file " << fName << ". " << std::strerror(errno) << std::endl;
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

    for (int i = 0; i < *numElements; ++i)
    {
        fBuf[i] = fVec[i];
    }

    return fBuf;
}

namespace wbInternal
{
    float* wbParseCSV(const char* fName, int* numRows, int* numCols)
    {
        std::ifstream inFile(fName);

        if (!inFile.is_open())
        {
            std::cout << "Error opening input file: " << fName << "!\n";
            exit(EXIT_FAILURE);
        }

        std::vector<float>* fVec = new std::vector<float>();
        std::string rowStr;
        (*numRows) = 0;

        while(getline(inFile, rowStr, '\n'))
        {
            (*numRows)++;
            std::istringstream rowStream(rowStr);
            std::string cellStr;
            (*numCols) = 0;

            while(getline(rowStream, cellStr, ','))
            {
                (*numCols)++;
                fVec->push_back(atof(cellStr.c_str()));
            }
        }

        inFile.close();

        return &*fVec->begin();
    }
} // namespace wbInternal

// For assignments MP2, MP3 & MP6
float* wbImport(const char* fName, int* numRows, int* numCols)
{     
    std::string fNameStr = fName;

    if(fNameStr.substr(fNameStr.find_last_of(".") + 1) == "csv")
    {
        return wbInternal::wbParseCSV(fName, numRows, numCols);
    }

    std::ifstream inFile(fName);

    if (!inFile.is_open())
    {
        std::cout << "Error opening input file: " << fName << "!\n";
        exit(EXIT_FAILURE);
    }

    inFile >> *numRows;
    inFile >> *numCols;

    std::string sVal;
    float fVal;
    std::vector<float> fVec;
    
    while (inFile >> sVal)
    {
        std::istringstream iss(sVal);
        iss >> fVal;
        fVec.push_back(fVal);
    }

    inFile.close();

    int numElements = *numRows * *numCols;

    if (static_cast<int>(fVec.size()) != numElements)
    {
        std::cout << "Error reading matrix content for a " << *numRows << " * " << *numCols << "matrix!\n";
        exit(EXIT_FAILURE);
    }

    float* fBuf = (float*) malloc(numElements * sizeof(float));

    if (!fBuf)
    {
        std::cout << "Unable to allocate memory for array of size " << numElements * sizeof(float) <<" bytes";
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i)
    {
        fBuf[i] = fVec[i];
    }

    return fBuf;
}

struct wbImage_t
{
    int width;
    int height;
    int channels;
    float* data;
    unsigned char* rawData;
    
    wbImage_t(int imageWidth = 0, int imageHeight = 0, int imageChannels = 0) : width(imageWidth), height(imageHeight), channels(imageChannels)
    {
        int numElements = width * height * channels;
        data = new float[numElements];
        rawData = new unsigned char[numElements];
    }
};

// For assignment MP6
wbImage_t wbImport(char* fName)
{
    wbImage_t image;
    image.channels = 3;

    std::ifstream inFile;
    inFile.open(fName, std::ios::binary);
    if (inFile.is_open())
    {
        char magic[2];
        inFile.read(magic, 2);

        if (magic[0] != 'P' || magic[1] !='6')
        {
            std::cout << "expected 'P6' but got " << magic[0] << magic[1] << std::endl;
            exit(EXIT_FAILURE);
        }

        char tmp = inFile.peek();
        while (isspace(tmp))
        {
            inFile.read(&tmp, 1);
            tmp = inFile.peek();
        }

        // Filter image comments
        if (tmp == '#')
        {
            inFile.read(&tmp, 1);
            tmp = inFile.peek();
            while (tmp != '\n')
            {
                inFile.read(&tmp, 1);
                tmp = inFile.peek();
            }
        }

        // Get rid of whitespaces
        while (isspace(tmp))
        {
            inFile.read(&tmp, 1);
            tmp = inFile.peek();
        }

        // Read dimensions (TODO add error checking)
        char widthStr[64], heightStr[64], numColorsStr[64], *p;
        p = widthStr;
        if(isdigit(tmp))
        {
            while(isdigit(*p = inFile.get()))
            {
                p++;
            }
            *p = '\0';
            image.width = atoi(widthStr);
            std::cout << "Width: " << image.width << std::endl;
            p = heightStr;
            while(isdigit(*p = inFile.get()))
            {
                p++;
            }
            *p = '\0';
            image.height = atoi(heightStr);
            std::cout << "Height: " << image.height << std::endl;
            p = numColorsStr;
            while(isdigit(*p = inFile.get()))
            { 
                p++;
            }
            *p = '\0';
            int numColors = atoi(numColorsStr);
            std::cout << "Num colors: " << numColors << std::endl;
            if (numColors != 255)
            {
                std::cout << "the number of colors should be 255, but got " << numColors << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            std::cout << "error - cannot read dimensions" << std::endl;
        }

        int numElements = image.width * image.height * image.channels;
        unsigned char* rawData = new unsigned char[numElements];
        inFile.read((char*)rawData, numElements);
        float* data = new float[numElements];
        for (int i = 0; i < numElements; i++)
        {
            data[i] = 1.0 * rawData[i] / 255.0f;
        }
        image.rawData = rawData;
        image.data = data;
        inFile.close();
    }
    else
    {
         std::cout << "cannot open file " << fName;
         exit(EXIT_FAILURE);
    }

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

wbImage_t wbImage_new(int imageWidth, int imageHeight, int imageChannels)
{
    wbImage_t image(imageWidth, imageHeight, imageChannels);
    return image;
}

void wbImage_delete(wbImage_t& image)
{
    delete[] image.data;
    delete[] image.rawData;
}

////
// Timer
////

// Namespace because Windows.h causes errors
namespace wbInternal
{
#if defined(_WIN32)
    #include <Windows.h>

    // CudaTimer class from: https://bitbucket.org/ashwin/cudatimer
    class CudaTimer
    {
    private:
        double        timerResolution;
        LARGE_INTEGER startTime;
        LARGE_INTEGER endTime;

    public:
        CudaTimer::CudaTimer()
        {
            LARGE_INTEGER freq;
            QueryPerformanceFrequency(&freq);
            timerResolution = 1.0 / freq.QuadPart;
        }

        void start()
        {
            cudaDeviceSynchronize();
            QueryPerformanceCounter(&startTime);
        }

        void stop()
        {
            cudaDeviceSynchronize();
            QueryPerformanceCounter(&endTime);
        }

        double value()
        {
            return (endTime.QuadPart - startTime.QuadPart) * timerResolution * 1000;
        }
    };
#elif defined(__APPLE__)
    #include <mach/mach_time.h>

    class CudaTimer
    {
    private:
        uint64_t startTime;
        uint64_t endTime;

    public:
        void start()
        {
            cudaDeviceSynchronize();
            startTime = mach_absolute_time();
        }

        void stop()
        {
            cudaDeviceSynchronize();
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

        #if !defined(MSEC_PER_SEC)
            #define MSEC_PER_SEC 1000L;
        #endif
        #if !defined(NSEC_PER_SEC)
            #define NSEC_PER_SEC 1000000000L;
        #endif

    #endif

    class CudaTimer
    {
    private:
        long long startTime;
        long long endTime;

        long long getTime()
        {
            long long time;
        #if defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0

            struct timespec ts;

            if (0 == clock_gettime(CLOCK_REALTIME, &ts))
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
                time += tv.tv_usec * MSEC_PER_SEC;
            }
        #endif

            return time;
        }

    public:
        void start()
        {
            cudaDeviceSynchronize();
            startTime = getTime();
        }

        void stop()
        {
            cudaDeviceSynchronize();
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
    wbTimeTypeNum, // Keep this at the end
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
        wbTimeType            type;
        std::string           message;
        wbInternal::CudaTimer timer;

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
    assert(timeType >= Generic && timeType < wbTimeTypeNum);

    wbInternal::CudaTimer timer;
    timer.start();

    wbInternal::wbTimerInfo timerInfo = { timeType, timeMessage, timer };

    wbInternal::timerInfoList.push_front(timerInfo);
}

void wbTime_stop(const wbTimeType timeType, const std::string timeMessage)
{
    assert(timeType >= Generic && timeType < wbTimeTypeNum);

    const wbInternal::wbTimerInfo searchInfo = { timeType, timeMessage };
    const wbInternal::wbTimerInfoList::iterator iter = std::find( wbInternal::timerInfoList.begin(), wbInternal::timerInfoList.end(), searchInfo );

    wbInternal::wbTimerInfo& timerInfo = *iter;

    assert(searchInfo == timerInfo);

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

// For assignments MP1, MP4 & MP5
template < typename T, typename S >
void wbSolution(const wbArg_t args, const T& t, const S& s)
{
    int solnItems;
    float *soln = (float *) wbImport(wbArg_getInputFile(args, args.argc - 2), &solnItems);

    if (solnItems != s)
    {
        std::cout << "Number of elements in solution file does not match. ";
        std::cout << "Expecting " << s << " but got " << solnItems << ".\n";
    }
    else // Check solution
    {
        int errCnt = 0;

        for (int item = 0; item < solnItems; item++)
        {
            if (!wbInternal::wbFPCloseEnough(soln[item], t[item]))
            {
                std::cout << "The solution did not match the expected result at element " << item << ". ";
                std::cout << "Expecting " << soln[item] << " but got " << t[item] << ".\n";
                errCnt++;
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
void wbSolution(const wbArg_t args, const T& t, const S& s, const U& u)
{
    int solnRows, solnColumns;
    float *soln = (float *) wbImport(wbArg_getInputFile(args, 2), &solnRows, &solnColumns);

    if (solnRows != s || solnColumns != u)
    {
        std::cout << "Size of solution file does not match. ";
        std::cout << "Expecting " << solnRows << " x " << solnColumns << " but got " << s << " x " << u << ".\n";
    }
    else // Check solution
    {
        int errCnt = 0;

        for (int row = 0; row < solnRows; row++)
        {
            for (int col = 0; col < solnColumns; col++)
            {
                float expected = *(soln + row * solnColumns + col);
                float result = *(t + row * solnColumns + col);

                if (!wbInternal::wbFPCloseEnough(expected, result))
                {
                    std::cout << "The solution did not match the expected results at column " << col << " and row " << row << "). ";
                    std::cout << "Expecting " << expected << " but got " << result << ".\n";
                    errCnt++;
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
    void wbImage_save(wbImage_t& image, char* fName)
    {
        std::ofstream outFile(fName, std::ios::binary);
        char buffer[64];
        std::string magic = "P6\n";
        outFile.write(magic.c_str(), magic.size());
        std::string comment  =  "# image generated by applying convolution\n";
        outFile.write(comment.c_str(), comment.size());
        sprintf(buffer,"%d", image.width);
        outFile.write(buffer, strlen(buffer));
        buffer[0] = ' ';
        outFile.write(buffer, 1);
        sprintf(buffer,"%d", image.height);
        outFile.write(buffer, strlen(buffer));
        buffer[0] = '\n';
        outFile.write(buffer, 1);
        std::string colors = "255\n";
        outFile.write(colors.c_str(), colors.size());

        int numElements = image.width * image.height * image.channels;
        unsigned char* rawData = new unsigned char[numElements];
        for (int i = 0; i < numElements; i++)
        {
            rawData[i] =  ceil(image.data[i] * 255);
        }

        outFile.write((char*)rawData, numElements);
        delete[] rawData;
        outFile.close();
    }
} // namespace wbInternal

// For assignment MP6
void wbSolution(wbArg_t args, wbImage_t image)
{
    wbInternal::wbImage_save(image, "convoluted.ppm");
    char* solutionFile = wbArg_getInputFile(args, 2);
    wbImage_t solnImage = wbImport(solutionFile);
    if (image.width != solnImage.width)
    {
        std::cout << "width is incorrect: expected " << solnImage.width << " but got " << image.width << std::endl;
        exit(EXIT_FAILURE);
    }

    if (image.height != solnImage.height)
    {
       std::cout << "height is incorrect: expected " << solnImage.height << " but got " << image.height << std::endl;
       exit(EXIT_FAILURE);
    }

    int channels = 3;
    for (int i = 0; i < image.width; ++i)
    {
        for (int j = 0; j < image.height; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                int index = (j * image.width + i) * channels + k;
                double scaled = ((double)image.data[index]) * 255.0f;
                double decimalPart = scaled - floor(scaled);
                // If true, don't know how to round, too close to xxx.5
                bool ambiguous = fabs(decimalPart - 0.5) < 0.0001;

                int colorValue = int(((double)image.data[index]) * 255.0f + 0.5);
                double error = abs(colorValue - solnImage.rawData[index]);
                if ( !(error == 0) && !(ambiguous && error <= 1) )
                {
                    std::cout << "data in position [" << i << " " << j << " " << k << "]  (array index: " << index << ") is wrong, expected " <<  (int)solnImage.rawData[index] << " but got " << colorValue << "  (float value is " << image.data[index] << ")" <<std::endl;
                    std::cout << "decimalPart: " << decimalPart << ", ambiguous: " << ambiguous << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
    
    std::cout << "Solution is correct!" << std::endl;  
}
