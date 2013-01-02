CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CUDA_VERBOSE_BUILD ON)

find_package(CUDA QUIET REQUIRED)
# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

set(BUILD_SHARED_LIBS ON)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(CMAKE_EXE_LINKER_FLAGS "-lrt")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Linux") 

INCLUDE_DIRECTORIES(./include)

CUDA_ADD_EXECUTABLE(cudadiag
    src/cudadiag.cu)

CUDA_ADD_EXECUTABLE(mp0 
    src/mp0.cc)

CUDA_ADD_EXECUTABLE(mp1 
    src/mp1.cu)

CUDA_ADD_EXECUTABLE(mp2 
    src/mp2.cu)

CUDA_ADD_EXECUTABLE(mp3 
    src/mp3.cu)
