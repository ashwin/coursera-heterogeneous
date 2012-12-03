// Author:   Ondrej Platek, Copyright 2012, code is without any warranty!
// Created:  10:30:39 03/12/2012
#include <Windows.h>
#include "cudatimer.h"

// Namespace because windows.h causes errors
namespace CudaTimerNS
{
    CudaTimer::CudaTimer()
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        freq_ = 1.0 / freq.QuadPart;
        return;    
    }

    unsigned long CudaTimer::getTime()
    {
        LARGE_INTEGER tmp;
        QueryPerformanceCounter(&tmp);
        return tmp.QuadPart;
    }

}
