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

#ifndef __WB_H__

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

////
// Logging
////

enum wbLogLevel {
    OFF,
    FATAL,
    ERROR,
    WARN,
    INFO,
    DEBUG,
    TRACE,
    wbLogLevelNum, //*** Keep this at the end
};

const char* _wbLogLevelStr[] = {
    "Off",
    "Fatal",
    "Error",
    "Warn",
    "Info",
    "Debug",
    "Trace",
    "***InvalidLogLevel***", //*** Keep this at the end
};

const char* _wbLogLevelToStr(wbLogLevel level) {
    assert(level >= OFF && level <= TRACE);
    return _wbLogLevelStr[level];
}

template<typename T>
inline void _wbLog(T const& val) {
    std::cout << val << " ";
}

template<typename First, typename ... Rest>
inline void _wbLog(First const& first, Rest const&... rest) {
    std::cout << first << " ";
    _wbLog(rest ...);
}
#define wbLog(level, ...) \
    do { \
        std::cout << _wbLogLevelToStr(level) << " "; \
        std::cout << __func__ << "::" << __LINE__ << " "; \
        _wbLog(__VA_ARGS__); \
        std::cout << std::endl; \
    } while (0)

////
// Input arguments
////

struct wbArg_t {
    int    argc;
    char** argv;
};

wbArg_t wbArg_read(int argc, char** argv) {
    wbArg_t argInfo = { argc, argv };
    return argInfo;
}

char* wbArg_getInputFile(wbArg_t argInfo, int argNum) {
    assert(argNum >= 0 && argNum < (argInfo.argc - 1));
    return argInfo.argv[argNum + 1];
}

float* wbImport(char* fname, int* itemNum) {
    // Open file

    std::ifstream inFile(fname);

    if (!inFile) {
        std::cout << "Error opening input file: " << fname << " !\n";
        exit(1);
    }

    // Read file to vector
    std::string sval;
    float fval;
    std::vector<float> fVec;

    while (inFile >> sval) {
        std::istringstream iss(sval);
        iss >> fval;
        fVec.push_back(fval );
    }

    // Vector to malloc memory
    *itemNum = fVec.size();

    float* fBuf = (float*) malloc(*itemNum * sizeof(float));

    for (int i = 0; i < *itemNum; ++i) {
        fBuf[i] = fVec[i];
    }

    return fBuf;
}

////
// Timer
////

// Namespace because windows.h causes errors
namespace CudaTimerNS {
    // CudaTimer class from: https://bitbucket.org/ashwin/cudatimer

    #include <Windows.h>

    class CudaTimer
    {
    private:
        double        _freq;
        LARGE_INTEGER _time1;
        LARGE_INTEGER _time2;

    public:
        CudaTimer::CudaTimer() {
            LARGE_INTEGER freq;
            QueryPerformanceFrequency(&freq);
            _freq = 1.0 / freq.QuadPart;
            return;    
        }

        void start() {
            cudaDeviceSynchronize();
            QueryPerformanceCounter( &_time1 );
            return;
        }

        void stop() {
            cudaDeviceSynchronize();
            QueryPerformanceCounter( &_time2 );
            return;
        }

        double value() {
            return (_time2.QuadPart - _time1.QuadPart) * _freq * 1000;
        }
    };
}

enum wbTimeType {
    Generic,
    GPU,
    Compute,
    Copy,
    wbTimeTypeNum, // Keep this at the end
};

const char* wbTimeTypeStr[] = {
    "Generic",
    "GPU    ",
    "Compute",
    "Copy   ",
    "***Invalid***",
};

const char* wbTimeTypeToStr(wbTimeType t) {
    assert(t >= Generic && t < wbTimeTypeNum);
    return wbTimeTypeStr[t];
}

struct wbTimerInfo {
    wbTimeType             type;
    std::string            name;
    CudaTimerNS::CudaTimer timer;

    bool operator == (const wbTimerInfo& t2) const {
        return (type == t2.type && (0 == name.compare(t2.name)));
    }
};

typedef std::list< wbTimerInfo> wbTimerInfoList;
wbTimerInfoList gTimerInfoList;

void wbTime_start(wbTimeType timeType, const std::string timeStar) {
    CudaTimerNS::CudaTimer timer;
    timer.start();

    wbTimerInfo tInfo = { timeType, timeStar, timer };

    gTimerInfoList.push_front(tInfo);
}

void wbTime_stop(wbTimeType timeType, const std::string timeStar) {
    // Find timer

    const wbTimerInfo searchInfo         = { timeType, timeStar };
    const wbTimerInfoList::iterator iter = std::find( gTimerInfoList.begin(), gTimerInfoList.end(), searchInfo );

    // Stop timer and print time

    wbTimerInfo& timerInfo = *iter;

    timerInfo.timer.stop();
    
    std::cout << "[" << wbTimeTypeToStr( timerInfo.type ) << "] ";
    std::cout << timerInfo.timer.value() << " ";
    std::cout << timerInfo.name << std::endl;

    // Delete timer from list
    gTimerInfoList.erase(iter);
}

////
// Solution
////

template < typename T, typename S >
void wbSolution(wbArg_t args, const T& t, const S& s) {
    return;
}

#define __WB_H_
#endif
