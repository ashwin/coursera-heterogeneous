///////////////////////////////////////////////////////////////////////////////
//
// wb.h: Header file for Heterogeneous Parallel Programming course (Coursera)
//
// Copyright (c) 2012 Ashwin Nanjappa
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

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>
#include "cudatimer.h"

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

const char* _wbLogLevelToStr( wbLogLevel level )
{
    assert( level >= OFF && level <= TRACE );

    return _wbLogLevelStr[ level ];
}

inline void _wbLog( wbLogLevel level, const char* logStr )
{
    std::cout << _wbLogLevelToStr( level ) << " ";
    std::cout << logStr;

    return;
}

inline void wbLog( wbLogLevel level, const char* logStr )
{
    _wbLog( level, logStr );
    std::cout << std::endl;

    return;
}

template < typename T >
inline void wbLog( wbLogLevel level, const char* logStr, T val )
{
    _wbLog( level, logStr );
    std::cout << val << std::endl;

    return;
}

////
// Input arguments
////

struct wbArg_t
{
    int    argc;
    char** argv;
};

wbArg_t wbArg_read( int argc, char** argv )
{
    wbArg_t argInfo = { argc, argv };
    return argInfo;
}

char* wbArg_getInputFile( wbArg_t argInfo, int argNum )
{
    assert( argNum >= 0 && argNum < ( argInfo.argc - 1 ) );

    return argInfo.argv[ argNum + 1 ];
}

float* wbImport( char* fname, int* itemNum )
{
    // Open file

    std::ifstream inFile( fname );

    if ( !inFile )
    {
        std::cout << "Error opening input file: " << fname << " !\n";
        exit( 1 );
    }

    // Read file to vector

    std::string sval;
    float fval;
    std::vector< float > fVec;

    while ( inFile >> sval )
    {
        std::istringstream iss( sval );

        iss >> fval;

        fVec.push_back( fval );
    }

    // Vector to malloc memory

    *itemNum = fVec.size();

    float* fBuf = ( float* ) malloc( *itemNum * sizeof( float ) );

    for ( int i = 0; i < *itemNum; ++i )
        fBuf[i] = fVec[i];

    return fBuf;
}


enum wbTimeType
{
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

const char* wbTimeTypeToStr( wbTimeType t )
{
    assert( t >= Generic && t < wbTimeTypeNum );

    return wbTimeTypeStr[t];
}

struct wbTimerInfo
{
    wbTimeType             type;
    std::string            name;
    CudaTimerNS::CudaTimer timer;

    bool operator == ( const wbTimerInfo& t2 ) const
    {
        return ( type == t2.type && ( 0 == name.compare( t2.name ) ) );
    }
};

typedef std::list< wbTimerInfo> wbTimerInfoList;

wbTimerInfoList gTimerInfoList;

void wbTime_start( wbTimeType timeType, const std::string timeStar )
{
    CudaTimerNS::CudaTimer timer;
    timer.start();

    wbTimerInfo tInfo = { timeType, timeStar, timer };

    gTimerInfoList.push_front( tInfo );

    return;
}

void wbTime_stop( wbTimeType timeType, const std::string timeStar )
{
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

    gTimerInfoList.erase( iter );

    return;
}

////
// Solution
////

template < typename T, typename S >
void wbSolution( wbArg_t args, const T& t, const S& s )
{
    return;
}

///////////////////////////////////////////////////////////////////////////////
