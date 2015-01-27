# WebGPU Machine Problems 2015
# mp0 - Device Query
# mp1 - Vector Addition
# mp2 - Basic Matrix-Matrix Multiplication
# mp3 - Tiled Matrix-Matrix Multiplication
# mp4 - Image Convolution
# mp5 - List Reduction
# mp6 - List Scan

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(libwb)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

# To specify the compute capability of your nVidia GPU uncomment the following
# lines and set COMPUTE_CAPABILITY to one of the following architectures:
# Tesla class 11, 12, or 13; Fermi class 20 or 21; Kepler class 30, 32, or 35;
# Maxwell class 50 or 52
# set(COMPUTE_CAPABILITY 35)
# set(CUDA_NVCC_FLAGS --generate-code;arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY})


# Set warning levels for host compilation
if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif ()

# Select the correct host compiler on OS X
if (APPLE)
    if (${CMAKE_SYSTEM_VERSION} LESS 12.0)
        # OS X 10.7 or less
        find_program(GCC_EXECUTABLE NAMES g++ PATHS ENV PATH)
        if (NOT GCC_EXECUTABLE)
            message(FATAL_ERROR "Could not find host compiler 'g++'")
        endif()
        set(CUDA_HOST_COMPILER ${GCC_EXECUTABLE})
    else ()
        # OS X 10.8+
        find_program(CLANG_EXECUTABLE NAMES clang++ PATHS ENV PATH)
        if (NOT CLANG_EXECUTABLE)
            message(FATAL_ERROR "Could not find host compiler 'clang++'")
        endif ()
        set(CUDA_HOST_COMPILER ${CLANG_EXECUTABLE})
        set(CMAKE_CXX_FLAGS    "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -isystem ${CUDA_INCLUDE_DIRS}")
    endif ()
endif ()

# Set the correct linker options on UNIX platforms
if (UNIX)
    set(LINK_LIBRARIES "-lm")
    if (APPLE)
        find_library(CUDA_LIBRARY NAMES CUDA)
        get_filename_component(FRAMEWORK_DIRECTORY ${CUDA_LIBRARY} DIRECTORY)
        set(LINK_LIBRARIES "${LINK_LIBRARIES} -F${FRAMEWORK_DIRECTORY} -framework CUDA")
    else ()
        include(CheckLibraryExists)
        check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)
        if (HAVE_CLOCK_GETTIME)
            set(LINK_LIBRARIES "${LINK_LIBRARIES} -lrt")
        endif ()
    endif ()
endif ()

set(EXECUTABLES mp0 mp1 mp2 mp3 mp4 mp5 mp6)
foreach (EXECUTABLE ${EXECUTABLES})
    CUDA_ADD_EXECUTABLE(${EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/${EXECUTABLE}.cu)
    if (UNIX)
        target_link_libraries(${EXECUTABLE} ${LINK_LIBRARIES})
    endif ()
endforeach ()


# CTEST TESTING
#
# To test all MPs against the official datasets type 'ctest -V' at the
# terminal. To test a specific MP type 'ctest -L mpX -V'
enable_testing()

add_test(
    NAME mp0
    COMMAND mp0
)
set_tests_properties(
    mp0
    PROPERTIES FAIL_REGULAR_EXPRESSION "No CUDA GPU has been detected"
    LABELS mp0
)

set(DATASETS 9)

set(TESTS mp1 mp2 mp3 mp4 mp5 mp6)
foreach (TEST ${TESTS})
    set(PATH tests/${TEST})
    foreach (i RANGE 0 ${DATASETS})
        if (${TEST} STREQUAL mp1 OR ${TEST} STREQUAL mp2 OR ${TEST} STREQUAL mp3)
            set(TEST_FILES ${PATH}/${i}/input0.raw ${PATH}/${i}/input1.raw ${PATH}/${i}/output.raw)
        elseif (${TEST} STREQUAL mp4)
            set(TEST_FILES ${PATH}/${i}/input0.ppm ${PATH}/${i}/input1.csv ${PATH}/${i}/output.ppm)
        elseif (${TEST} STREQUAL mp5)
            set(TEST_FILES ${PATH}/${i}/input0.raw ${PATH}/${i}/output.raw)
        elseif (${TEST} STREQUAL mp6)
            set(TEST_FILES ${PATH}/${i}/input.raw ${PATH}/${i}/output.raw)
        else ()
            message(FATAL_ERROR "Unknown test target '${TEST}' specified")
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
