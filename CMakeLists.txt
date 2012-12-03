CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CUDA_VERBOSE_BUILD ON)

find_package(CUDA QUIET REQUIRED)
# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# You can set BUILD_SHARED_LIBS or you can pass STATIC, SHARED, or MODULE to
# CUDA_ADD_LIBRARY.  Remember that BUILD_SHARED_LIBS is only respected by
# CUDA_ADD_LIBRARY.  If you use CUDA_COMPILE or CUDA_WRAP_SRCS you must specify
# SHARED or STATIC as a parameter.
#  -arch sm_13
set(BUILD_SHARED_LIBS ON)

#INCLUDE_DIRECTORIES(./inc)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -gdwarf-2 -fno-omit-frame-pointer --std=c++0x -Weffc++ -Wall")
set(CMAKE_CSS_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} --std=c++0x -Weffc++ -Wall")

CUDA_ADD_LIBRARY(wb 
    include/wb.h
    STATIC)
