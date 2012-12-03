// Author:   Ondrej Platek, Copyright 2012, code is without any warranty!
// Created:  10:25:11 03/12/2012
////
// Timer
////
#ifndef CUDATIMER_H
#define CUDATIMER_H

// TODO
// do not need it #include <cutil.h> // -I${CUDA}/sdk/C/common/inc
// #if CUDART_VERSION >= 4000
// #define CUT_DEVICE_SYNCHRONIZE( )   cudaDeviceSynchronize();
// #else
// #define CUT_DEVICE_SYNCHRONIZE( )   cudaThreadSynchronize();
// #endif

// #include <cuda.h>
// Namespace because windows.h causes errors
namespace CudaTimerNS
{
    // CudaTimer class from: https://github.com/oplatek/coursera-heterogeneous,
    // Inspired by https://bitbucket.org/ashwin/cudatimer

    class CudaTimer
    {
    private:
        double        freq_;
        // On Windows time{1,2}_ are LARGE_INTEGER.QuadPart fields
        // On Linux time{1,2}_ are timespec.tv_nsec fields
        unsigned long time1_;
        unsigned long time2_;
        unsigned long getTime();
    public:
        CudaTimer();
        void start()
        {
            // CUT_DEVICE_SYNCHRONIZE // TODO
            time1_ = getTime();
            return;
        }

        void stop()
        {
            // CUT_DEVICE_SYNCHRONIZE // TODO
            time2_ = getTime();
            return;
        }
        double value()
        {
            return (time2_ - time1_) * freq_ * 1000;
        }
    };
}
#endif
