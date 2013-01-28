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
// NVCC does not support C++11 variadic template yet

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
}

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
float* wbImport(const char* fname, int* itemNum)
{
    // Open file

    std::ifstream inFile(fname);

    if (!inFile.is_open())
    {
        std::cout << "Error opening input file: " << fname << " !\n";
        exit(EXIT_FAILURE);
    }

    // Read from file

    inFile >> *itemNum;

    float* fBuf = (float*) malloc(*itemNum * sizeof(float));

    if (!fBuf)
    {
        std::cout << "Unable to allocate memory for array of size " << *itemNum * sizeof(float) <<" bytes";
        exit(EXIT_FAILURE);
    }

    std::string sval;

    for (int idx = 0; idx < *itemNum && inFile >> sval; ++idx)
    {
        std::istringstream iss(sval);
        iss >> fBuf[idx];
    }

    inFile.close();

    return fBuf;
}

// For assignments MP2 & MP3
float* wbImport(const char* fname, int* numRows, int* numCols)
{
    // Open file

    std::ifstream inFile(fname);

    if (!inFile.is_open())
    {
        std::cout << "Error opening input file: " << fname << " !\n";
        exit(EXIT_FAILURE);
    }

    // Read in matrix dimensions

    inFile >> *numRows;
    inFile >> *numCols;

    std::string sval;
    float fval;
    std::vector<float> fVec;

    // Read file to vector
    
    while (inFile >> sval)
    {
        std::istringstream iss(sval);
        iss >> fval;
        fVec.push_back(fval);
    }

    inFile.close();

    int itemNum = *numRows * *numCols;

    if (static_cast<int>(fVec.size()) != itemNum)
    {
        std::cout << "Error reading matrix content for a " << *numRows << " * " << *numCols << "matrix!\n";
        exit(EXIT_FAILURE);
    }

    // Vector to malloc memory

    float* fBuf = (float*) malloc(itemNum * sizeof(float));

    if (!fBuf)
    {
        std::cout << "Unable to allocate memory for array of size " << itemNum * sizeof(float) <<" bytes";
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < itemNum; ++i)
    {
        fBuf[i] = fVec[i];
    }

    return fBuf;
}

//
//
//  For MP6
//
//

struct wbImage_t 
{
    int  _imageWidth;  
    int  _imageHeight;
    int  _imageChannels;
    float* _data;
    unsigned char* _rawData;
    
    wbImage_t(int imageWidth = 0, int imageHeight = 0, int imageChannels = 0) :_imageWidth(imageWidth), _imageHeight(imageHeight), _imageChannels(imageChannels)
    {
        int dataSize = _imageWidth * _imageHeight * _imageChannels;
        _data = new float[dataSize];
	_rawData = new unsigned char[dataSize];
    }
};


wbImage_t wbImport(char* inputFile) 
{     
    wbImage_t image;
    image._imageChannels = 3;  
    
    std::ifstream fileInput;
    fileInput.open(inputFile, std::ios::binary);
    if (fileInput.is_open()) {
        char magic[2];
        fileInput.read(magic, 2);
        if (magic[0] != 'P' || magic[1] !='6') {
            std::cout << "expected 'P6' but got " << magic[0] << magic[1] << std::endl;
            exit(1);
        }
        char tmp = fileInput.peek();
        while (isspace(tmp)) {
            fileInput.read(&tmp, 1);
            tmp = fileInput.peek();
        }
        // filter image comments
        if (tmp == '#') {
            fileInput.read(&tmp, 1);
            tmp = fileInput.peek();
            while (tmp != '\n') {
                fileInput.read(&tmp, 1);
                tmp = fileInput.peek();
            }
        } 
        // get rid of whitespaces
        while (isspace(tmp)) {
            fileInput.read(&tmp, 1);
            tmp = fileInput.peek();
        }
        
        //read dimensions (TODO add error checking)
        char widthStr[64], heightStr[64], numColorsStr[64], *p; 
        p = widthStr;                    
        if(isdigit(tmp)) {
            while(isdigit(*p = fileInput.get())) { 
                p++;   
            }       
            *p = '\0';           
            image._imageWidth = atoi(widthStr);
            std::cout << "Width: " << image._imageWidth << std::endl;
            p = heightStr;
            while(isdigit(*p = fileInput.get())) { 
                p++;   
            }      
            *p = '\0';
            image._imageHeight = atoi(heightStr);
            std::cout << "Height: " << image._imageHeight << std::endl;
            p = numColorsStr;
            while(isdigit(*p = fileInput.get())) { 
                p++; 
            }
            *p = '\0';
            int numColors = atoi(numColorsStr);
            std::cout << "Num colors: " << numColors << std::endl;
            if (numColors != 255) {
                std::cout << "the number of colors should 255, but got " << numColors << std::endl;
                exit(1);
            }    
        } else  {
            std::cout << "error - cannot read dimensions" << std::endl;
        }
            
        int dataSize = image._imageWidth*image._imageHeight*image._imageChannels;
        unsigned char* data = new unsigned char[dataSize];
        fileInput.read((char*)data, dataSize);
        float* floatData = new float[dataSize];
        for (int i = 0; i < dataSize; i++) {
            floatData[i] = 1.0*data[i]/255.0f;
        }
        image._rawData = data;
        image._data = floatData;
        fileInput.close();
    } else  {
         std::cout << "cannot open file " << inputFile;
         exit(1);
    } 
    return image;
}  

int wbImage_getWidth(const wbImage_t& image)
{
    return image._imageWidth;
}

int wbImage_getHeight(const wbImage_t& image)
{
    return image._imageHeight;
}

int wbImage_getChannels(const wbImage_t& image)
{
    return image._imageChannels;
}

float* wbImage_getData(const wbImage_t& image)
{
     return image._data;
}

wbImage_t wbImage_new(int imageWidth, int imageHeight, int imageChannels)
{
    wbImage_t image(imageWidth, imageHeight, imageChannels);
    return image;
}  

void wbImage_delete(wbImage_t& image)
{
    delete[] image._data;
    delete[] image._rawData;
}

void wbImage_save(wbImage_t& image, char* outputfile) {
    std::ofstream outputFile(outputfile, std::ios::binary);
    char buffer[64];
    std::string magic = "P6\n";
    outputFile.write(magic.c_str(), magic.size());
    std::string comment  =  "# image generated by applying convolution\n";
    outputFile.write(comment.c_str(), comment.size());
    //write dimensions
    sprintf(buffer,"%d", image._imageWidth);
    outputFile.write(buffer, strlen(buffer));
    buffer[0] = ' ';
    outputFile.write(buffer, 1);
    sprintf(buffer,"%d", image._imageHeight);
    outputFile.write(buffer, strlen(buffer));
    buffer[0] = '\n';
    outputFile.write(buffer, 1);
    std::string colors = "255\n";
    outputFile.write(colors.c_str(), colors.size());
    
    int dataSize = image._imageWidth*image._imageHeight*image._imageChannels;
    unsigned char* rgbData = new unsigned char[dataSize];
    for (int i = 0; i < dataSize; i++) {
        rgbData[i] =  ceil(image._data[i] * 255);
    }
    outputFile.write((char*)rgbData, dataSize); 
    delete[] rgbData;         
    outputFile.close(); 
}  

void wbSolution(wbArg_t arg, wbImage_t image) {
    wbImage_save(image, "convoluted.ppm");  
    char* solutionFile = wbArg_getInputFile(arg, 2);
    wbImage_t solutionImage = wbImport(solutionFile);
    if (image._imageWidth != solutionImage._imageWidth) {
        std::cout << "width is incorrect: expected " << solutionImage._imageWidth << " but got " << image._imageWidth << std::endl;
        exit(1);
    } 
    if (image._imageHeight != solutionImage._imageHeight) {
       std::cout << "height is incorrect: expected " << solutionImage._imageHeight << " but got " << image._imageHeight << std::endl;
       exit(1);
    }
    int channels = 3;
    for (int i = 0; i < image._imageWidth; ++i)
        for (int j = 0; j < image._imageHeight; ++j)
            for (int k = 0; k < 3; ++k) {
                int index = ( j*image._imageWidth + i )*channels + k; 
		 double tmp = ((double)image._data[index])*255.0f +0.5;
		 if (abs(int(tmp) - solutionImage._rawData[index]) > 1) { 
                    std::cout << "data in position [" << i << " " << j << " " << k << "]  (array index: " << index << ") is wrong, expected " <<  (int)solutionImage._rawData[index] << " but got " << int(tmp) << "  (float value is " << image._data[index] << ")" <<std::endl;
                    exit(1);
                }
            }
    std::cout << "Solution is correct!" << std::endl;  
}   

//
//
//  MP6 End
//
//


////
// Timer
////

// Namespace because Windows.h causes errors
namespace wbInternal
{
#if defined (_WIN32)
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
            return;
        }

        void start()
        {
            cudaDeviceSynchronize();
            QueryPerformanceCounter(&startTime);
            return;
        }

        void stop()
        {
            cudaDeviceSynchronize();
            QueryPerformanceCounter(&endTime);
            return;
        }

        double value()
        {
            return (endTime.QuadPart - startTime.QuadPart) * timerResolution * 1000;
        }
    };
#elif defined (__APPLE__)
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

            return ((double) endTime - startTime) * (tb.numer / tb.denom) / 1000000000ULL;
        }
    };
#else
    #if defined(_POSIX_TIMERS) && _POSIX_TIMERS > 0
        #include<time.h>
    #else
        #include<sys/time.h>
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

            if ( 0 == clock_gettime(CLOCK_REALTIME, &ts) )
            {
                time  = 1000000000LL; // seconds->nanonseconds
                time *= ts.tv_sec;
                time += ts.tv_nsec;
            }
        #else
            struct timeval tv;

            if ( 0 == gettimeofday(&tv, NULL) )
            {
                time  = 1000000000LL; // seconds->nanonseconds
                time *= tv.tv_sec;
                time += tv.tv_usec * 1000; // ms->ns
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
            return ((double) endTime - startTime) / 1000000000LL;
        }
    };
#endif
}

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
}

void wbTime_start(const wbTimeType timeType, const std::string timeMessage)
{
    assert(timeType >= Generic && timeType < wbTimeTypeNum);

    wbInternal::CudaTimer timer;
    timer.start();

    wbInternal::wbTimerInfo timerInfo = { timeType, timeMessage, timer };

    wbInternal::timerInfoList.push_front(timerInfo);

    return;
}

void wbTime_stop(const wbTimeType timeType, const std::string timeMessage)
{
    assert(timeType >= Generic && timeType < wbTimeTypeNum);

    // Find timer

    const wbInternal::wbTimerInfo searchInfo = { timeType, timeMessage };
    const wbInternal::wbTimerInfoList::iterator iter = std::find( wbInternal::timerInfoList.begin(), wbInternal::timerInfoList.end(), searchInfo );

    wbInternal::wbTimerInfo& timerInfo = *iter;

    assert(searchInfo == timerInfo);

    // Stop timer and print time

    timerInfo.timer.stop();

    std::cout << "[" << wbInternal::wbTimeTypeToStr( timerInfo.type ) << "] ";
    std::cout << std::fixed << std::setprecision(9) << timerInfo.timer.value() << " ";
    std::cout << timerInfo.message << std::endl;

    // Delete timer from list

    wbInternal::timerInfoList.erase(iter);

    return;
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
}

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

    return;
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

    return;
}
