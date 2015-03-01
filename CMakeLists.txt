#
# WebGPU Machine Problems 2015
#

# mp0 - Device Query
# mp1 - Vector Addition
# mp2 - Basic Matrix-Matrix Multiplication
# mp3 - Tiled Matrix-Matrix Multiplication
# mp6 - Image Convolution
# mp4 - List Reduction
# mp5 - List Scan
# mp7 - Vector Addition (OpenCL)
# mp11 - Histograms
# mp12 - Vector Addition with Streams

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(libwb)


#
# Settings common to CUDA / OpenCL
#

# Include project header
include_directories(${PROJECT_SOURCE_DIR}/include)

# Set warning levels for host compilation; VS defaults to /W3
if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif ()

if (CMAKE_GENERATOR MATCHES "Xcode")
    set(CMAKE_XCODE_ATTRIBUTE_ARCHS $(ARCHS_STANDARD))
    set(CMAKE_XCODE_ATTRIBUTE_VALID_ARCHS x86_64)
endif ()


#
# Settings for NVidia CUDA-based MPs
#

find_package(CUDA)

if (CUDA_FOUND)
    # For some MPs it may be necessary to specify the compute capability of your
    # NVidia GPU. In that case, simply uncomment the following two lines that start
    # with 'set' and change the value of COMPUTE_CAPABILITY to one of the following
    # hardware architectures: Tesla-class '11', '12', or '13'; Fermi-class '20'
    # or '21'; Kepler-class '30', '32', '35' or '37'; or Maxwell-class '50' or '52'
    set(COMPUTE_CAPABILITY 11)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--gpu-architecture;sm_${COMPUTE_CAPABILITY})

    # Select the correct host compiler on OS X
    if (APPLE)
        if (${CMAKE_SYSTEM_VERSION} VERSION_GREATER 11.0)
            # OS X 10.8+
            find_program(libwb_CXX_COMPILER NAMES clang PATHS ENV PATH)
            if (NOT libwb_CXX_COMPILER)
                message(FATAL_ERROR "Could not find host compiler 'clang'")
            endif ()
            set(CUDA_HOST_COMPILER ${libwb_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC")
            if (CUDA_VERSION VERSION_LESS 7.0)
                # CUDA Toolkits 5.5-6.5 are incompatible with LLVM's C++ standard library
                set(libwb_STANDARD_LIBRARY -stdlib=libstdc++)
                if (CMAKE_GENERATOR MATCHES "Xcode")
                    set(libwb_XCODE_ATTRIBUTES XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY libstdc++)
                endif ()
            endif ()
        endif ()
        # Suppress warnings related to the CUDA headers
        set(libwb_NVCC_OPTIONS --compiler-options -isystem,${CUDA_INCLUDE_DIRS})
    endif ()

    # Set additional linker options on UNIX platforms
    if (UNIX)
        set(libwb_LINK_LIBRARIES -lm)
        if (NOT APPLE)
            include(CheckLibraryExists)
            check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)
            if (HAVE_CLOCK_GETTIME)
                list(APPEND libwb_LINK_LIBRARIES -lrt)
            endif ()
        endif ()
    endif ()

    set(EXECUTABLES mp0 mp1 mp2 mp3 mp6 mp4 mp5 mp11 mp12)
    foreach (TARGET ${EXECUTABLES})
        CUDA_ADD_EXECUTABLE(
            ${TARGET}
            ${PROJECT_SOURCE_DIR}/src/${TARGET}.cu
            OPTIONS ${libwb_NVCC_OPTIONS}
        )
        if (DEFINED libwb_STANDARD_LIBRARY)
            set_target_properties(
                ${TARGET}
                PROPERTIES
                COMPILE_FLAGS ${libwb_STANDARD_LIBRARY}
                LINK_FLAGS ${libwb_STANDARD_LIBRARY}
            )
        endif ()
        if (DEFINED libwb_XCODE_ATTRIBUTES)
            set_target_properties(
                ${TARGET}
                PROPERTIES ${libwb_XCODE_ATTRIBUTES}
            )
        endif ()
        if (UNIX)
            target_link_libraries(
                ${TARGET}
                ${libwb_LINK_LIBRARIES}
            )
        endif ()
    endforeach ()
endif ()


#
# Implement a minimal FindOpenCL.cmake
#

find_path(
    OpenCL_INCLUDE_DIR
    NAMES
        OpenCL/cl.h CL/cl.h
    PATHS
        ENV OPENCL_ROOT
        ENV AMDAPPSDKROOT
        ENV INTELOCLSDKROOT
        ENV CUDA_PATH
        ENV NVSDKCOMPUTE_ROOT
    PATH_SUFFIXES
        include
        OpenCL/common/inc
)

if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    find_library(
        OpenCL_LIBRARY
        NAMES
            OpenCL
        PATHS
            ENV OPENCL_ROOT
            ENV AMDAPPSDKROOT
            ENV INTELOCLSDKROOT
            ENV CUDA_PATH
            ENV NVSDKCOMPUTE_ROOT
        PATH_SUFFIXES
            lib/x86
            lib/Win32
    )
elseif (CMAKE_SIZEOF_VOID_P EQUAL 8)
    find_library(
        OpenCL_LIBRARY
        NAMES
            OpenCL
        PATHS
            ENV OPENCL_ROOT
            ENV AMDAPPSDKROOT
            ENV INTELOCLSDKROOT
            ENV CUDA_PATH
            ENV NVSDKCOMPUTE_ROOT
        PATH_SUFFIXES
            lib/x86_64
            lib/x64
    )
endif ()

set(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    OpenCL
    DEFAULT_MSG
    OpenCL_LIBRARY
    OpenCL_INCLUDE_DIR
)
mark_as_advanced(
    OpenCL_INCLUDE_DIR
    OpenCL_LIBRARY
)


#
# Settings for OpenCL MPs
#

# TODO


#
# CTEST TESTING
#

# To test all MPs against the official datasets type 'ctest -V' at the
# terminal; to test a specific MP type 'ctest -L mpX -V'
enable_testing()

if (CUDA_FOUND)
    add_test(
        NAME mp0
        COMMAND mp0
    )
    set_tests_properties(
        mp0
        PROPERTIES FAIL_REGULAR_EXPRESSION "No CUDA GPU has been detected"
        LABELS mp0
    )

    # CUDA-based MPs
    set(TESTS mp1 mp2 mp3 mp6 mp4 mp5 mp11 mp12)
endif ()

set(DATASETS 9)

if (CUDA_FOUND)
    foreach (TEST ${TESTS})
        set(PATH tests/${TEST})
        foreach (i RANGE 0 ${DATASETS})
            if ("mp1 mp2 mp3 mp12" MATCHES ${TEST})
                set(TEST_FILES ${PATH}/${i}/input0.raw ${PATH}/${i}/input1.raw ${PATH}/${i}/output.raw)
            elseif (mp4 STREQUAL ${TEST})
                set(TEST_FILES ${PATH}/${i}/input0.raw ${PATH}/${i}/output.raw)
            elseif (mp5 STREQUAL ${TEST})
                set(TEST_FILES ${PATH}/${i}/input.raw ${PATH}/${i}/output.raw)
            elseif (mp6 STREQUAL ${TEST})
                set(TEST_FILES ${PATH}/${i}/input0.ppm ${PATH}/${i}/input1.csv ${PATH}/${i}/output.ppm)
            elseif (mp11 STREQUAL ${TEST})
                set(TEST_FILES ${PATH}/${i}/input.ppm ${PATH}/${i}/output.ppm)
            else ()
                message(FATAL_ERROR "Unknown test target '${TEST}'")
            endif ()
            set(TEST_LABEL ${TEST}-${i})
            add_test(
                NAME ${TEST_LABEL}
                COMMAND ${TEST} ${TEST_FILES}
            )
            set_tests_properties(
                ${TEST_LABEL}
                PROPERTIES PASS_REGULAR_EXPRESSION "Solution is correct"
                LABELS ${TEST_LABEL}
            )
        endforeach ()
    endforeach ()
endif ()
