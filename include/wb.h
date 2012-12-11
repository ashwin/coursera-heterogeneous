///////////////////////////////////////////////////////////////////////////////
//
// wb.h: Header file for Heterogeneous Parallel Programming course (Coursera)
//
// Copyright (c) 2012 Ashwin Nanjappa
// Copyright (c) 2012 Greg Bowyer
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

////
// Headers
////

// C++
#include <algorithm>
#include <cassert>
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

const char* _wbLogLevelStr[] =
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

const char* _wbLogLevelToStr(wbLogLevel level)
{
    assert(level >= OFF && level <= TRACE);
    return _wbLogLevelStr[level];
}

//-----------------------------------------------------------------------------
// Begin: Ugly C++03 hack
// NVCC does not support C++11 variadic template yet

template<typename First>
inline void _wbLog(First const& first)
{
    std::cout << first;
}

template<typename First, typename Second>
inline void _wbLog(First const& first, Second const& second)
{
    std::cout << first << second;
}

template<typename First, typename Second, typename Third>
inline void _wbLog(First const& first, Second const& second, Third const& third)
{
    std::cout << first << second << third;
}

template<typename First, typename Second, typename Third, typename Fourth>
inline void _wbLog(First const& first, Second const& second, Third const& third,
        Fourth const& fourth)
{
    std::cout << first << second << third << fourth;
}

template<typename First, typename Second, typename Third, typename Fourth,
    typename Fifth>
inline void _wbLog(First const& first, Second const& second, Third const& third,
        Fourth const& fourth, Fifth const& fifth)
{
    std::cout << first << second << third << fourth << fifth;
}

template<typename First, typename Second, typename Third, typename Fourth,
    typename Fifth, typename Sixth>
inline void _wbLog(First const& first, Second const& second, Third const& third,
        Fourth const& fourth, Fifth const& fifth, Sixth const& sixth)
{
    std::cout << first << second << third << fourth << fifth << sixth;
}

template<typename First, typename Second, typename Third, typename Fourth,
    typename Fifth, typename Sixth, typename Seventh>
inline void _wbLog(First const& first, Second const& second, Third const& third,
        Fourth const& fourth, Fifth const& fifth, Sixth const& sixth,
        Seventh const& seventh)
{
    std::cout << first << second << third << fourth << fifth << sixth << seventh;
}

template<typename First, typename Second, typename Third, typename Fourth,
    typename Fifth, typename Sixth, typename Seventh, typename Eighth>
inline void _wbLog(First const& first, Second const& second, Third const& third,
        Fourth const& fourth, Fifth const& fifth, Sixth const& sixth,
        Seventh const& seventh, Eighth const& eighth)
{
    std::cout << first << second << third << fourth << fifth << sixth
        << seventh << eighth;
}

template<typename First, typename Second, typename Third, typename Fourth,
    typename Fifth, typename Sixth, typename Seventh, typename Eighth,
    typename Ninth>
inline void _wbLog(First const& first, Second const& second, Third const& third,
        Fourth const& fourth, Fifth const& fifth, Sixth const& sixth,
        Seventh const& seventh, Eighth const& eighth, Ninth const& ninth)
{
    std::cout << first << second << third << fourth << fifth << sixth
        << seventh << eighth << ninth;
}

template<typename First, typename Second, typename Third, typename Fourth,
    typename Fifth, typename Sixth, typename Seventh, typename Eighth,
    typename Ninth, typename Tenth>
inline void _wbLog(First const& first, Second const& second, Third const& third,
        Fourth const& fourth, Fifth const& fifth, Sixth const& sixth,
        Seventh const& seventh, Eighth const& eighth, Ninth const& ninth,
        Tenth const& tenth)
{
    std::cout << first << second << third << fourth << fifth << sixth
        << seventh << eighth << ninth << tenth;
}

template<typename First, typename Second, typename Third, typename Fourth,
    typename Fifth, typename Sixth, typename Seventh, typename Eighth,
    typename Ninth, typename Tenth, typename Eleventh>
inline void _wbLog(First const& first, Second const& second, Third const& third,
        Fourth const& fourth, Fifth const& fifth, Sixth const& sixth,
        Seventh const& seventh, Eighth const& eighth, Ninth const& ninth,
        Tenth const& tenth, Eleventh const& eleventh)
{
    std::cout << first << second << third << fourth << fifth << sixth
        << seventh << eighth << ninth << tenth << eleventh;
}

template<typename First, typename Second, typename Third, typename Fourth,
    typename Fifth, typename Sixth, typename Seventh, typename Eighth,
    typename Ninth, typename Tenth, typename Eleventh, typename Twelfth>
inline void _wbLog(First const& first, Second const& second, Third const& third,
        Fourth const& fourth, Fifth const& fifth, Sixth const& sixth,
        Seventh const& seventh, Eighth const& eighth, Ninth const& ninth,
        Tenth const& tenth, Eleventh const& eleventh, Twelfth const& twelfth)
{
    std::cout << first << second << third << fourth << fifth << sixth
        << seventh << eighth << ninth << tenth << eleventh << twelfth;
}

// End: Ugly C++03 hack
//-----------------------------------------------------------------------------

#define wbLog(level, ...) \
    do { \
        std::cout << _wbLogLevelToStr(level) << " "; \
        std::cout << __FUNCTION__ << "::" << __LINE__ << " "; \
        _wbLog(__VA_ARGS__); \
        std::cout << std::endl; \
    } while (0)

////
// Input arguments
////

struct wbArg_t
{
    int    argc;
    char** argv;
};

wbArg_t wbArg_read(int argc, char** argv)
{
    wbArg_t argInfo = { argc, argv };
    return argInfo;
}

char* wbArg_getInputFile(wbArg_t argInfo, int argNum)
{
    assert(argNum >= 0 && argNum < (argInfo.argc - 1));
    return argInfo.argv[argNum + 1];
}

// For assignment MP1
float* wbImport(char* fname, int* itemNum)
{
    // Open file

    std::ifstream inFile(fname);

    if (!inFile)
    {
        std::cout << "Error opening input file: " << fname << " !\n";
        exit(1);
    }

    // Read file to vector

    std::string sval;
    float fval;
    std::vector<float> fVec;

    while (inFile >> sval)
    {
        std::istringstream iss(sval);
        iss >> fval;
        fVec.push_back(fval );
    }

    // Vector to malloc memory

    *itemNum = fVec.size();

    float* fBuf = (float*) malloc(*itemNum * sizeof(float));

    for (int i = 0; i < *itemNum; ++i)
    {
        fBuf[i] = fVec[i];
    }

    return fBuf;
}

// For assignment MP2
float* wbImport(char* fname, int* numRows, int* numCols)
{
    // Open file

    std::ifstream inFile(fname);

    if (!inFile)
    {
        std::cout << "Error opening input file: " << fname << " !\n";
        exit(1);
    }

    // Read file to vector

    std::string sval;
    float fval;
    std::vector<float> fVec;
    int itemNum = 0;

    // Read in matrix dimensions
    inFile >> *numRows;
    inFile >> *numCols;

    while (inFile >> sval)
    {
        std::istringstream iss(sval);
        iss >> fval;
        fVec.push_back(fval );
    }

    // Vector to malloc memory

    if (fVec.size() != (*numRows * *numCols))
    {
        std::cout << "Error reading matrix content for a " << *numRows << " * " << *numCols << "matrix!\n";
        exit(1);
    }

    itemNum = *numRows * *numCols;

    float* fBuf = (float*) malloc(itemNum * sizeof(float));

    for (int i = 0; i < itemNum; ++i)
    {
        fBuf[i] = fVec[i];
    }

    return fBuf;
}

////
// Timer
////

// Namespace because windows.h causes errors
namespace CudaTimerNS
{
#if defined (_WIN32)
    #include <Windows.h>

    // CudaTimer class from: https://bitbucket.org/ashwin/cudatimer
    class CudaTimer
    {
    private:
        double        _freq;
        LARGE_INTEGER _time1;
        LARGE_INTEGER _time2;

    public:
        CudaTimer::CudaTimer()
        {
            LARGE_INTEGER freq;
            QueryPerformanceFrequency(&freq);
            _freq = 1.0 / freq.QuadPart;
            return;
        }

        void start()
        {
            cudaDeviceSynchronize();
            QueryPerformanceCounter(&_time1);
            return;
        }

        void stop()
        {
            cudaDeviceSynchronize();
            QueryPerformanceCounter(&_time2);
            return;
        }

        double value()
        {
            return (_time2.QuadPart - _time1.QuadPart) * _freq * 1000;
        }
    };
#else
    #include <time.h>

    class CudaTimer
    {
    private:
        long long _start;
        long long _end;

    public:
        void start()
        {
            struct timespec sp;

            if (0 == clock_gettime(CLOCK_REALTIME,&sp))
            {
                _start  = 1000000000LL; // seconds->nanonseconds
                _start *= sp.tv_sec;
                _start += sp.tv_nsec;
            }
        }

        void stop()
        {
            struct timespec sp;

            if (0 == clock_gettime(CLOCK_REALTIME,&sp))
            {
                _end  = 1000000000LL; // seconds->nanonseconds
                _end *= sp.tv_sec;
                _end += sp.tv_nsec;
            }
        }

        double value()
        {
            return ((double) _end - (double) _start) / 1000000000LL;
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

const char* wbTimeTypeStr[] =
{
    "Generic",
    "GPU    ",
    "Compute",
    "Copy   ",
    "***InvalidTimeType***", // Keep this at the end
};

const char* wbTimeTypeToStr(wbTimeType t)
{
    assert(t >= Generic && t < wbTimeTypeNum);
    return wbTimeTypeStr[t];
}

struct wbTimerInfo
{
    wbTimeType             type;
    std::string            name;
    CudaTimerNS::CudaTimer timer;

    bool operator == (const wbTimerInfo& t2) const
    {
        return (type == t2.type && (0 == name.compare(t2.name)));
    }
};

typedef std::list< wbTimerInfo> wbTimerInfoList;
wbTimerInfoList gTimerInfoList;

void wbTime_start(wbTimeType timeType, const std::string timeStar)
{
    CudaTimerNS::CudaTimer timer;
    timer.start();

    wbTimerInfo tInfo = { timeType, timeStar, timer };

    gTimerInfoList.push_front(tInfo);

    return;
}

void wbTime_stop(wbTimeType timeType, const std::string timeStar)
{
    // Find timer

    const wbTimerInfo searchInfo         = { timeType, timeStar };
    const wbTimerInfoList::iterator iter = std::find( gTimerInfoList.begin(), gTimerInfoList.end(), searchInfo );

    // Stop timer and print time

    wbTimerInfo& timerInfo = *iter;

    timerInfo.timer.stop();

    std::cout << "[" << wbTimeTypeToStr( timerInfo.type ) << "] ";
    std::cout << std::fixed << std::setprecision(10) << timerInfo.timer.value() << " ";
    std::cout << timerInfo.name << std::endl;

    // Delete timer from list
    gTimerInfoList.erase(iter);

    return;
}

////
// Solution
////

template < typename T, typename S >
void wbSolution(wbArg_t args, const T& t, const S& s)
{
    return;
}

template < typename T, typename S, typename U >
void wbSolution(wbArg_t args, const T& t, const S& s, const U& u)
{
    return;
}